#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2ma_train.py — cooperative, constrained multi-agent trainer (feasibility prototype).

Reuses the Paper-2 controller wholesale: the shared policy is a BranchedQRDQN with CVaR-greedy action
selection and quantile-Huber / n-step Double-DQN learning (paper2_controller.Agent). The multi-agent
extension is exactly two things:

  1. PARAMETER SHARING (cooperative CTDE-lite): all M agents act through ONE shared network from their
     own local observation, and every agent's trajectory is a training sequence for that shared net.
  2. PER-AGENT RELIABILITY CONSTRAINT (the reward-trap fix): the environment reward is pure efficiency
     (availability minus power/spectrum/handover cost) with NO RLF penalty and NO episode truncation.
     Reliability enters only through a Lagrangian: the training reward is r_eff - lambda * rlf, and
     lambda is raised or lowered on a slow timescale to hold the mean sustained-RLF at a target epsilon.
     This sets the reliability price explicitly, instead of inheriting it from a truncation accident.

Feasibility question this answers: at the calibrated congested regime (4 gNBs, capacity 4, 16 agents)
where the decentralized reactive rule fails ~15% (worst agent ~41%), can a learned coordinator do
better? If yes, the paper's core claim holds on real data; if no, that is a finding too.

INTEGRITY: standard algorithms on the real-geometry environment; nothing fabricated. Feed-forward is
used (the single-UE study found LSTM unnecessary once the geometry state is explicit).
"""
import os, sys, time, argparse
import inspect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pandas as pd
import paper2ma_env as MA
import paper2_controller as C
from paper2_per import SequencePER

BRANCHES = C.BRANCHES
NSTEP = C.NSTEP


def batched_greedy(net, obs_MB, alpha):
    """One forward pass for all M agents. obs_MB: (M, obs) -> actions (M, 3)."""
    x = torch.tensor(obs_MB, dtype=torch.float32).unsqueeze(1)      # (M,1,obs)
    with torch.no_grad():
        qs, _ = net(x)
    acts = [g[:, 0].cpu().numpy() for g in C.greedy_actions(qs, alpha)]
    return np.stack(acts, axis=1)                                   # (M,3)


def sequential_select(env, agent, base_obs, eps):
    """Autoregressive (symmetry-breaking) action selection.

    Agents commit ONE AT A TIME in a fixed order. Each agent sees the per-tower load implied by the
    handover choices already committed by earlier agents THIS step, so when one agent flees to a clear
    tower the next sees that tower fill and picks elsewhere. This breaks the herding that made the
    simultaneous shared policy collapse (all agents flee to the same tower and re-crowd it). The load
    features occupy the last n_gnb slots of the observation, so only those are rewritten per agent;
    the rest of each agent's observation is the snapshot from the previous env step.

    Returns (acts (M,3), adj_obs (M,D)) where adj_obs[i] is the observation agent i actually acted on
    (stored for learning so the policy trains on what it conditioned on)."""
    M, ng, D = env.M, env.n_gnb, base_obs.shape[1]
    cap = float(env.gnb_cap) if env.gnb_cap else float(M)
    fr = env.frames[env.t]
    intended = env.serv.copy()                     # where each agent will serve after this step's HOs
    acts = np.zeros((M, 3), dtype=int)
    adj = base_obs.copy()
    for i in range(M):                             # fixed commit order
        if not fr['active'][i]:
            continue
        occ = np.zeros(ng)
        for j in range(M):
            if fr['active'][j]:
                occ[intended[j]] += 1
        adj[i, D - ng:D] = np.minimum(occ / cap, 2.0)     # running load from prior commitments
        if eps > 0.0 and np.random.random() < eps:
            a = np.random.randint([2, 3, 3])
        else:
            a = batched_greedy(agent.net, adj[i:i + 1], agent.alpha)[0]
        acts[i] = a
        if int(a[0]) == 1 and env._ho_timer[i] == 0:      # will initiate a handover -> update intent
            g = env.serv[i]
            cand = int(np.argmax([env._desired_dbm(i, gg) if gg != g else -1e9 for gg in range(ng)]))
            if cand != g:
                intended[i] = cand
    return acts, adj


def boltzmann_select(env, agent, base_obs, temp):
    """Sequential selection but each branch samples from a Boltzmann(softmax) policy over its CVaR
    action-scores at temperature `temp`, instead of argmax. This IS the stochastic (max-entropy)
    policy — no separate epsilon needed. Returns (acts, adj_obs) like sequential_select."""
    M, ng, D = env.M, env.n_gnb, base_obs.shape[1]
    cap = float(env.gnb_cap) if env.gnb_cap else float(M)
    fr = env.frames[env.t]
    intended = env.serv.copy()
    acts = np.zeros((M, 3), dtype=int); adj = base_obs.copy()
    for i in range(M):
        if not fr['active'][i]:
            continue
        occ = np.zeros(ng)
        for j in range(M):
            if fr['active'][j]:
                occ[intended[j]] += 1
        adj[i, D - ng:D] = np.minimum(occ / cap, 2.0)
        x = torch.tensor(adj[i:i + 1], dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            qs, _ = agent.net(x)
        a = []
        for q in qs:
            sc = C.cvar_scores(q, agent.alpha)[0, 0]                 # (nb,)
            p = torch.softmax(sc / temp, dim=-1).numpy(); p = p / p.sum()
            a.append(int(np.random.choice(len(p), p=p)))
        acts[i] = a
        if a[0] == 1 and env._ho_timer[i] == 0:
            g = env.serv[i]
            cand = int(np.argmax([env._desired_dbm(i, gg) if gg != g else -1e9 for gg in range(ng)]))
            if cand != g:
                intended[i] = cand
    return acts, adj


def maxent_seq_loss(agent, ep, temp):
    """Soft (max-entropy) quantile target: bootstrap from the POLICY-AVERAGED next quantiles plus an
    entropy bonus, instead of the greedy argmax. This keeps the learned values consistent with a
    STOCHASTIC policy, so the Q-scores never sharpen into the over-confident spike that makes every
    agent pick the same tower (the herding failure the pilot isolated)."""
    dev = agent.device
    obs = torch.tensor(ep['obs'], dtype=torch.float32, device=dev).unsqueeze(0)
    acts = torch.tensor(ep['acts'], dtype=torch.long, device=dev)
    rews = torch.tensor(ep['rews'], dtype=torch.float32, device=dev)
    term = bool(ep.get('term', True)); T = acts.shape[0]
    qs_all, _ = agent.net(obs)
    with torch.no_grad():
        qs_tgt, _ = agent.tgt(obs)
        qs_on, _ = agent.net(obs)
    loss, td = 0.0, 0.0
    for i, nb in enumerate(BRANCHES):
        pred = qs_all[i][0, torch.arange(T), acts[:, i]]
        tgt = torch.zeros(T, C.N_QUANT, device=dev)

        def soft_next(p):
            sc = C.cvar_scores(qs_on[i][0, p:p + 1], agent.alpha)[0]      # (nb,)
            pi = torch.softmax(sc / temp, dim=-1)                          # (nb,)
            nq = (pi.unsqueeze(-1) * qs_tgt[i][0, p]).sum(0)               # (nq,) policy-averaged
            ent = -(pi * pi.clamp_min(1e-8).log()).sum()
            return nq + temp * ent                                        # entropy-shifted value
        for t in range(T):
            R = 0.0; g = 1.0; k = 0
            while k < NSTEP and t + k < T:
                R += g * rews[t + k].item(); g *= C.GAMMA; k += 1
            if t + k < T:
                tgt[t] = R + g * soft_next(t + k)
            elif not term:
                tgt[t] = R + g * soft_next(T)
            else:
                tgt[t] = R
        tgt = tgt.detach()
        loss = loss + C.quantile_huber(pred, tgt, agent.taus)
        td = td + float((tgt.mean(dim=1) - pred.detach().mean(dim=1)).abs().mean())
    return loss, td / len(BRANCHES)


def maxent_learn(agent, episodes, is_w, temp):
    """One optimizer step with the max-entropy soft targets (mirrors Agent.learn)."""
    agent.net.train()
    is_w = np.asarray(is_w, dtype=np.float64); wsum = float(is_w.sum()) or 1.0
    agent.opt.zero_grad()
    total, tds = 0.0, []
    for ep, w in zip(episodes, is_w):
        l, td = maxent_seq_loss(agent, ep, temp)
        (float(w) / wsum * l).backward()
        total += float(l.detach()) * float(w) / wsum; tds.append(td)
    torch.nn.utils.clip_grad_norm_(agent.net.parameters(), 10.0)
    agent.opt.step()
    return total, np.array(tds, dtype=np.float64)


def pivotal_index(env):
    """The agent the order statistic is currently resting on.

    The team objective is the worst served outcome. Its gradient with respect to the outcome vector
    is ONE-HOT - supported on the single agent at rank k - or ZERO when there is a tie at rank k.
    This returns that agent. Unadmitted agents score 0, which is the atom the tail sits on; ranking
    by SINR alone would pick an admitted agent with poor signal while the objective rests on an
    unserved one. Returns None when the period is fully inert (a tie at the bottom), in which case
    no agent can move the objective and no one should be credited for it.
    """
    fr = env.frames[env.t]
    active = [i for i in range(env.M) if fr['active'][i]]
    if not active:
        return None
    out = {}
    for i in active:
        if not env._admitted[i]:
            out[i] = 0.0
        else:
            sd = env._sinr_db(i)
            out[i] = 0.0 if sd < MA.SINR_OUT_DB else sd
    lo = min(out.values())
    at_bottom = [i for i in active if out[i] <= lo + 1e-12]
    if len(at_bottom) > 1:
        return None                     # tie at the bottom: the gradient is zero for everyone
    return at_bottom[0]


def credit_vector(env, rew, mode, act_mask):
    """Redistribute the SAME total efficiency signal under three rules, so the comparison isolates
    the credit rule and not the amount of reward in the system.

      peragent : credit_i = rew_i                      (what the trainer does today - additive)
      team     : credit_i = sum(rew)/n_active          (the equal additive share)
      pivotal  : credit_i = sum(rew) if i is pivotal else 0   (the one-hot gradient made a rule)
    """
    if mode == 'peragent':
        return rew
    idx = [i for i in range(env.M) if act_mask[i]]
    if not idx:
        return rew
    tot = float(np.sum([rew[i] for i in idx]))
    out = np.zeros(env.M, dtype=float)
    if mode == 'team':
        for i in idx:
            out[i] = tot / len(idx)
        return out
    piv = pivotal_index(env)
    if piv is not None and act_mask[piv]:
        out[piv] = tot
    return out                          # fully-inert period: everyone gets 0, by construction


def ma_rollout(env, agent, lam, eps, seed=None, buf=None, temp=0.0, credit='peragent'):
    """One multi-agent episode. Adds one training sequence per full-episode-active agent to `buf`
    (constrained reward r_eff - lam*rlf). Returns per-agent (rlf_steps, ok_steps, active_steps)."""
    obs, info = env.reset(seed=seed)
    M = env.M
    O = [[] for _ in range(M)]; Ac = [[] for _ in range(M)]; Re = [[] for _ in range(M)]
    rlf_s = np.zeros(M); ok_s = np.zeros(M); act_s = np.zeros(M); seen = np.zeros(M)
    total = len(env.frames) - 1
    done = False
    while not done:
        act_mask = env.frames[env.t]['active'].copy()       # active at the decision frame
        if temp > 0.0:
            acts, adj = boltzmann_select(env, agent, obs, temp)     # max-entropy stochastic policy
        else:
            acts, adj = sequential_select(env, agent, obs, eps)
        obs, rew, term, done, info = env.step(acts)
        cred = credit_vector(env, rew, credit, act_mask)
        for i in range(M):
            if not act_mask[i]:
                continue
            Ac[i].append(acts[i].copy())
            # the reliability price stays per-agent; only the EFFICIENCY credit is redistributed,
            # so the comparison isolates the credit rule from the constraint.
            Re[i].append(float(cred[i]) - lam * float(info['rlf'][i]))
            O[i].append(adj[i].copy())                       # store the obs the agent acted on
            rlf_s[i] += float(info['rlf'][i]); ok_s[i] += float(info['link_ok'][i])
            act_s[i] += 1.0; seen[i] += 1.0
    if buf is not None:
        for i in range(M):
            if int(seen[i]) == total and len(Ac[i]) >= NSTEP + 1:      # active every step -> clean sequence
                seq_obs = np.array(O[i] + [obs[i]], dtype=np.float32)   # T acted-on obs + final obs
                ep = dict(obs=seq_obs, acts=np.array(Ac[i]),
                          rews=np.array(Re[i], dtype=np.float32), term=False)
                buf.add(ep)
    return rlf_s, ok_s, act_s


def ma_evaluate(env, agent, n, seed_base=C.EVAL_SEED_BASE, temp=0.0):
    """Evaluation over n episodes. With temp>0 the DEPLOYED policy is the stochastic Boltzmann policy
    (a legitimate stochastic controller, not injected noise); else greedy. Per-agent RLF summary."""
    per_rlf = []; worst = []; ret = []; ok = []
    for k in range(n):
        obs, info = env.reset(seed=seed_base + k)
        M = env.M
        rlf_s = np.zeros(M); act_s = np.zeros(M); ok_s = np.zeros(M); R = 0.0
        done = False
        while not done:
            if temp > 0.0:
                acts, _ = boltzmann_select(env, agent, obs, temp)
            else:
                acts, _ = sequential_select(env, agent, obs, 0.0)
            obs, rew, term, done, info = env.step(acts)
            for i in range(M):
                if not info['active'][i]:
                    continue
                rlf_s[i] += float(info['rlf'][i]); ok_s[i] += float(info['link_ok'][i]); act_s[i] += 1.0
                R += float(rew[i])
        a = np.maximum(act_s, 1.0)
        pr = rlf_s / a
        per_rlf.append(float((rlf_s.sum()) / a.sum()))
        worst.append(float(pr.max()))
        ok.append(float(ok_s.sum() / a.sum()))
        ret.append(R / M)
    return dict(mean_rlf=float(np.mean(per_rlf)), worst_rlf=float(np.mean(worst)),
                link_ok=float(np.mean(ok)), ret_per_agent=float(np.mean(ret)), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--agents', type=int, default=16)
    ap.add_argument('--gnb_cap', type=int, default=4)
    ap.add_argument('--subbands', type=int, default=3)
    ap.add_argument('--episodes', type=int, default=400)
    ap.add_argument('--alpha', type=float, default=C.CVAR_ALPHA)
    ap.add_argument('--eps_target', type=float, default=0.10, help='reliability constraint: target WORST-agent RLF (the metric the mean-constrained learner lost on)')
    ap.add_argument('--lam0', type=float, default=2.0, help='initial Lagrangian multiplier on RLF')
    ap.add_argument('--lam_lr', type=float, default=0.5, help='dual step size (gentle: prevents runaway)')
    ap.add_argument('--lam_max', type=float, default=20.0, help='cap on the multiplier so an infeasible target cannot explode it')
    ap.add_argument('--w_cong', type=float, default=0.0, help='congestion (difference-reward) penalty weight. NOT SUPPORTED by the current paper2ma_env (removed in the 23 Aug 2026 gnb_cap rewrite); any non-zero value is refused rather than silently ignored. Kept so pre-23-Aug envs still work.')
    ap.add_argument('--temp', type=float, default=0.0, help='max-entropy temperature: >0 trains AND deploys a stochastic Boltzmann policy with soft targets (the anti-herding fix); 0 = greedy')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--updates_per_ep', type=int, default=3)
    ap.add_argument('--eval_n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--credit', default='peragent', choices=('peragent', 'team', 'pivotal'),
                    help="efficiency-credit rule. 'peragent' is the trainer's existing additive "
                         "behaviour and is the default, so results produced without this flag are "
                         "unchanged. 'pivotal' concentrates the whole team signal on the agent the "
                         "order statistic rests on, and gives everyone zero in a fully-inert period.")
    ap.add_argument('--ckpt_every', type=int, default=0,
                    help='save an intermediate checkpoint every N episodes (0 = off). '
                         'Purely additive and OFF by default: with 0 the run consumes no extra '
                         'RNG and behaves bit-identically to the submitted Paper-2 campaign. '
                         'Needed for D3 (design drift across checkpoints) and D4b.')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    pool = pd.read_csv(a.pool)
    # paper2ma_env.py was rewritten on 23 Aug 2026: it gained gnb_cap and the finite-admission
    # coupling, and the older w_cong reward penalty was dropped. This trainer was not updated and
    # passed w_cong= unconditionally, which made it un-runnable against the current env. Pass the
    # argument only if the env still accepts it. If the env does NOT accept it and a non-zero
    # penalty was requested, stop: silently dropping it would run a different experiment from the
    # one asked for. At the default of 0.0 the two are semantically identical (no penalty).
    _env_kw = dict(n_agents=a.agents, n_subbands=a.subbands, gnb_cap=a.gnb_cap,
                   seed=a.seed, split='train', compute_anticipation=True)
    if 'w_cong' in inspect.signature(MA.MultiAgentRSUBlockageEnv.__init__).parameters:
        _env_kw['w_cong'] = a.w_cong
    elif a.w_cong:
        raise SystemExit(
            f"--w_cong {a.w_cong} was requested, but this paper2ma_env.MultiAgentRSUBlockageEnv "
            "has no w_cong parameter (it was removed in the 23 Aug 2026 rewrite that introduced "
            "gnb_cap). Refusing to run: dropping the penalty would silently change the experiment. "
            "Use --w_cong 0 for the admission-coupled env, or restore the pre-23-Aug env.")
    env = MA.MultiAgentRSUBlockageEnv(pool, **_env_kw)
    agent = C.Agent(obs_dim=env.observation_space.shape[0], alpha=a.alpha, use_lstm=False)
    buf = SequencePER(capacity=800, alpha=0.6, beta0=0.4, beta_steps=max(a.episodes, 1),
                      rng=np.random.default_rng(a.seed + 777))
    if a.smoke:
        a.episodes = 8; a.warmup = 4; a.eval_n = 4
    lam = a.lam0
    print(f"credit rule: {a.credit}")
    print(f"regime: {a.agents} agents, {env.n_gnb} gNBs, cap {a.gnb_cap}, {a.subbands} sub-bands | "
          f"CVaR{a.alpha} | constraint target RLF <= {a.eps_target:.1%} | {len(env.scene_ids)} train scenes",
          flush=True)
    hist_rlf, t0 = [], time.time()
    sync_every = max(a.episodes // 8, 1)
    hist_mean = []
    for ep in range(a.episodes):
        eps = max(0.05, 1.0 - ep / max(a.episodes * 0.6, 1))
        rlf_s, ok_s, act_s = ma_rollout(env, agent, lam, eps, buf=buf, temp=a.temp, credit=a.credit)
        per = rlf_s / np.maximum(act_s, 1.0)
        worst = float(per.max()); mean_rlf = float(rlf_s.sum() / max(act_s.sum(), 1))
        hist_rlf.append(worst); hist_mean.append(mean_rlf)
        if len(buf) >= a.warmup:
            for _ in range(a.updates_per_ep):
                eps_b, idx, w = buf.sample(a.batch)
                loss, td = (maxent_learn(agent, eps_b, w, a.temp) if a.temp > 0.0
                            else agent.learn(eps_b, w))
                buf.update(idx, td)
            buf.anneal()
        # constraint driven by the WORST-agent RLF (clamped dual ascent): raise the reliability price
        # until even the worst-off agent meets the target, focusing the pressure where the mean-
        # constrained learner failed. lambda stays bounded so an infeasible target cannot explode it.
        recent = float(np.mean(hist_rlf[-25:]))
        lam = float(np.clip(lam + a.lam_lr * (recent - a.eps_target), 0.0, a.lam_max))
        # D3/D4b: intermediate checkpoints. Guarded on --ckpt_every so the default path is
        # untouched; torch.save does not draw from any RNG, so enabling it cannot shift a run.
        if a.ckpt_every and a.out and (ep + 1) % a.ckpt_every == 0:
            torch.save(agent.net.state_dict(), f'{a.out}_ep{ep + 1:04d}.pt')
        if (ep + 1) % sync_every == 0:
            agent.sync()
            print(f"  ep {ep+1:4d} | eps {eps:.2f} | recent worst {recent*100:5.2f}% | "
                  f"mean {float(np.mean(hist_mean[-25:]))*100:5.2f}% | lambda {lam:6.2f} | "
                  f"{time.time()-t0:5.0f}s", flush=True)
    ev = ma_evaluate(env, agent, a.eval_n, temp=a.temp)
    print(f"\nLEARNED (greedy, {ev['n']} eval ep): mean RLF {ev['mean_rlf']*100:.2f}%  "
          f"worst-agent RLF {ev['worst_rlf']*100:.2f}%  link-OK {ev['link_ok']*100:.1f}%  "
          f"return/agent {ev['ret_per_agent']:.2f}")
    print(f"[reference reactive-greedy on this pool/regime: measure with paper2ma_calibrate; "
          f"full-300 4gNB/cap4/M16 = 9.8% mean / 17.8% worst]")
    if a.out:
        torch.save(agent.net.state_dict(), a.out + '.pt')
        pd.DataFrame({'episode': range(len(hist_rlf)), 'train_rlf': hist_rlf}).to_csv(a.out + '_train.csv', index=False)
        pd.DataFrame([ev]).to_csv(a.out + '_eval.csv', index=False)
        print(f"saved -> {a.out}.pt / _train.csv / _eval.csv")


if __name__ == '__main__':
    main()
