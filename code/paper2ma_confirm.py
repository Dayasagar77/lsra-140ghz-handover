#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2ma_confirm.py — confirmatory evaluation harness for the multi-agent MaxEnt controller.

Rigorous, PAIRED, same-seed comparison on ONE locked metric (per-episode worst-agent RLF, plus mean),
with bootstrap confidence intervals, on TRAIN and HELD-OUT scenes. Baselines span weak-to-oracle so
the win is defensible:
  * random            - uniform actions (a spreading floor)
  * reactive rule     - max power, round-robin band, reactive best-signal handover (the herding rule)
  * load-aware rule   - reactive, but hands over to the LEAST-LOADED of the top-2 signal towers
  * Hungarian oracle  - per-step capacity-constrained optimal agent->tower assignment (upper bound)
  * MaxEnt model(s)   - the trained Boltzmann policy at a deploy temperature (one per training seed)

Metric lock: per-episode worst-agent = mean over episodes of (max over agents of that agent's RLF
rate in the episode). This is the meaningful "worst-off agent" experience and is used identically for
every policy. INTEGRITY: same seeds for every policy; nothing tuned to a target; CIs reported.
"""
import sys, argparse, glob
sys.path.insert(0, '/root/work/paper2')
import numpy as np, torch, pandas as pd
import paper2ma_env as MA, paper2_controller as C, paper2ma_train as TT
from scipy.optimize import linear_sum_assignment


def desired(env, i, g, pt):
    a = env.frames[env.t]['ag'][i]
    return pt + MA.G_SERV - MA.CH.ci_pathloss_db(max(a['dist'][g], 1.0), True) - a['depth'][g]


def occ_vec(env):
    o = np.zeros(env.n_gnb); fr = env.frames[env.t]
    for j in range(env.M):
        if fr['active'][j]:
            o[env.serv[j]] += 1
    return o


# ---- policies: each returns (action (M,3), ho_targets (M,) or None) ----
def pol_random(env, obs, sinr, rng, ag, temp):
    return np.stack([rng.integers([2, 3, 3]) for _ in range(env.M)]), None


def pol_reactive(env, obs, sinr, rng, ag, temp):
    M = env.M; a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; fr = env.frames[env.t]
    for i in range(M):
        if fr['active'][i]:
            a[i, 0] = 1 if sinr[i] < MA.SINR_OUT_DB else 0
    return a, None


def pol_loadaware(env, obs, sinr, rng, ag, temp):
    M = env.M; a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; fr = env.frames[env.t]
    occ = occ_vec(env); tgt = -np.ones(M, int)
    for i in range(M):
        if not fr['active'][i]:
            continue
        if sinr[i] < MA.SINR_OUT_DB:
            a[i, 0] = 1
            ds = [(desired(env, i, g, MA.PT_MAX) if g != env.serv[i] else -1e9) for g in range(env.n_gnb)]
            top = list(np.argsort(ds)[::-1][:2])                 # two best-signal alternatives
            tgt[i] = int(min(top, key=lambda g: occ[g]))         # ... the less crowded of them
        else:
            a[i, 0] = 0
    return a, tgt


def pol_hungarian(env, obs, sinr, rng, ag, temp):
    M = env.M; fr = env.frames[env.t]; ng = env.n_gnb; cap = int(env.gnb_cap or M)
    active = [i for i in range(M) if fr['active'][i]]
    slot_tower = [g for g in range(ng) for _ in range(cap)]
    NS = len(slot_tower)
    cost = np.full((len(active), NS), 1e6)
    for r, i in enumerate(active):
        for s in range(NS):
            cost[r, s] = -desired(env, i, slot_tower[s], MA.PT_MAX)
    a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; tgt = -np.ones(M, int)
    if len(active):
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            tgt[active[r]] = slot_tower[c]; a[active[r], 0] = 1
    return a, tgt


def pol_maxent(env, obs, sinr, rng, ag, temp):
    acts, _ = TT.boltzmann_select(env, ag, obs, temp)
    return acts, None


def run_eval(env, policy_fn, seeds, ag=None, temp=0.0):
    """Returns per-episode arrays: worst-agent RLF, mean RLF, link-ok. Round-robin band init for the
    rule baselines is applied every reset (harmless for the model, which controls its own band)."""
    pw, pm, po = [], [], []
    rng = np.random.default_rng(20260726)
    for s in seeds:
        obs, info = env.reset(seed=s); M = env.M
        env.sb = np.array([i % env.n_sb for i in range(M)], dtype=int)
        sinr = np.array([env._sinr_db(i) for i in range(M)])
        rlf = np.zeros(M); act = np.zeros(M); ok = np.zeros(M); done = False
        while not done:
            a, tgt = policy_fn(env, obs, sinr, rng, ag, temp)
            obs, rew, term, done, info = env.step(a, ho_targets=tgt)
            for i in range(M):
                if info['active'][i]:
                    sinr[i] = obs[i, 0] * 30.0
                    rlf[i] += info['rlf'][i]; ok[i] += info['link_ok'][i]; act[i] += 1
        aa = np.maximum(act, 1)
        pw.append(float((rlf / aa).max())); pm.append(float(rlf.sum() / aa.sum())); po.append(float(ok.sum() / aa.sum()))
    return np.array(pw), np.array(pm), np.array(po)


def ci(x):
    m = float(x.mean())
    bs = np.array([np.random.default_rng(k).choice(x, len(x)).mean() for k in range(1500)])
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate_split(pool, split, seeds, model_paths, temp):
    env = MA.MultiAgentRSUBlockageEnv(pool, n_agents=16, n_subbands=3, gnb_cap=4, seed=0,
                                      split=split, compute_anticipation=True)
    D = env.observation_space.shape[0]
    rows = []
    for name, fn in [('random', pol_random), ('reactive-rule', pol_reactive),
                     ('load-aware-rule', pol_loadaware), ('Hungarian-oracle', pol_hungarian)]:
        pw, pm, po = run_eval(env, fn, seeds)
        rows.append((name, ci(pw), ci(pm), po.mean()))
    # MaxEnt models (one per training seed) at the deploy temperature; aggregate per-episode across seeds
    all_w, all_m, all_o = [], [], []
    for mp in model_paths:
        ag = C.Agent(obs_dim=D, use_lstm=False); ag.net.load_state_dict(torch.load(mp)); ag.net.eval()
        pw, pm, po = run_eval(env, pol_maxent, seeds, ag=ag, temp=temp)
        all_w.append(pw); all_m.append(pm); all_o.append(po.mean())
    if model_paths:
        W = np.concatenate(all_w); Mn = np.concatenate(all_m)
        rows.append((f'MaxEnt(t{temp:g}, {len(model_paths)} seeds)', ci(W), ci(Mn), float(np.mean(all_o))))
    return env, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='/root/work/paper2/results/dair_geometry_pool_full300.csv')
    ap.add_argument('--models', default='models/ma_maxent_full300_s*.pt')
    ap.add_argument('--temp', type=float, default=4.0)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    a = ap.parse_args()
    pool = pd.read_csv(a.pool)
    model_paths = sorted(glob.glob(a.models))
    print(f'MaxEnt models: {[m.split("/")[-1] for m in model_paths]} | deploy temp {a.temp} | {a.n} episodes/split')
    print('metric = per-episode worst-agent RLF% (primary) and mean RLF%, paired same seeds, 95% CI\n')
    for split in a.splits:
        seeds = [C.EVAL_SEED_BASE + k for k in range(a.n)]
        env, rows = evaluate_split(pool, split, seeds, model_paths, a.temp)
        print(f'=== {split.upper()} scenes ({len(env.scene_ids)}) ===')
        print(f'{"policy":<26} {"worst-agent RLF%":>26} {"mean RLF%":>22} {"linkOK%":>8}')
        for name, (mw, lw, hw), (mm, lm, hm), o in rows:
            print(f'{name:<26} {mw*100:6.2f} [{lw*100:5.2f},{hw*100:5.2f}]   {mm*100:6.2f} [{lm*100:5.2f},{hm*100:5.2f}]   {o*100:6.1f}')
        print()


if __name__ == '__main__':
    main()
