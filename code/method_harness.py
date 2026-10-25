#!/usr/bin/env python3
"""Evaluation harness for the method test (20 Oct 2026).

Identical to remeasure_ma.run_eval_log (CF.run_eval's loop plus read-only logging of failure causes
and handover starts), with one extra read-only measure:

  land_over: after all the handovers started in a step, the landing load of every station (cars in
  flight counted where they will land, the others where they are); land_over adds up the excess
  over capacity, step by step. It includes excess that was already there before the step's decisions
  (for example a car that appeared at a station with room now but full at landing).
  dec_over: only the excess the step's decisions created, i.e. the rise of each station's landing
  load above max(capacity, its load before the decisions). A planner that respects the landing state
  never creates any.

The sequential 'blind start' count of the 20 Oct harness is kept for continuity, but it processes
the starts of one step one at a time, so a simultaneous swap can be counted as blind; land_over
does not have that problem.
"""
import numpy as np
import remeasure_ma as RM

CAUSES = RM.CAUSES


def run_eval_m(env, policy_fn, seeds):
    pw, pm, po, rows = [], [], [], []
    rng = np.random.default_rng(20260726)
    cap = int(env.gnb_cap)
    for s in seeds:
        obs, info0 = env.reset(seed=s); M = env.M
        env.sb = np.array([i % env.n_sb for i in range(M)], dtype=int)
        sinr = np.array([env._sinr_db(i) for i in range(M)])
        rlf = np.zeros(M); act = np.zeros(M); ok = np.zeros(M); done = False
        G = env.n_gnb
        cause = np.zeros((M, len(CAUSES)))
        prev_active = env.frames[0]['active'].copy()
        prev_serv = env.serv.copy()
        last_land = np.full(G, -10 ** 9); last_app = np.full(G, -10 ** 9)
        self_app = np.full(M, -10 ** 9)
        n_start = 0; n_blind = 0; inflight_sum = 0.0; n_frames = 0
        land_over = 0; steps_over = 0; dec_over = 0
        while not done:
            act_pre = env.frames[env.t]['active'].copy()
            serv_pre = env.serv.copy(); tim_pre = env._ho_timer.copy(); tgt_pre = env._ho_target.copy()
            a, tgt = policy_fn(env, obs, sinr, rng, None, 0.0)
            obs, rew, term, done, info = env.step(a, ho_targets=tgt)
            for i in range(M):
                if info['active'][i]:
                    sinr[i] = obs[i, 0] * 30.0
                    rlf[i] += info['rlf'][i]; ok[i] += info['link_ok'][i]; act[i] += 1
            tau = env.t
            act_now = info['active']; serv_now = info['serv']
            started = np.where(info['ho'])[0]
            land = np.zeros(G)
            for j in range(M):
                if act_pre[j]:
                    land[tgt_pre[j] if tim_pre[j] > 0 else serv_pre[j]] += 1
            land_after = land.copy(); land_before = land.copy()
            for j in started:
                g_from = int(serv_pre[j]); g_to = int(env._ho_target[j])
                n_start += 1
                if land[g_to] >= cap:
                    n_blind += 1
                land[g_to] += 1; land[g_from] -= 1
                land_after[g_to] += 1; land_after[g_from] -= 1
            exc = int(np.maximum(land_after - cap, 0).sum())
            land_over += exc; steps_over += int(exc > 0)
            dec_over += int(np.maximum(land_after - np.maximum(land_before, cap), 0).sum())
            for j in range(M):
                if act_now[j] and not prev_active[j]:
                    last_app[serv_now[j]] = tau; self_app[j] = tau
                if act_now[j] and serv_now[j] != prev_serv[j]:
                    last_land[serv_now[j]] = tau
            for i in range(M):
                if not (act_now[i] and info['rlf'][i]):
                    continue
                tau0 = tau - int(env._consec[i]) + 1
                g = int(serv_now[i])
                if not info['admitted'][i]:
                    if self_app[i] >= tau0 or last_app[g] >= tau0 or tau0 <= 1:
                        k = 0
                    elif last_land[g] >= tau0:
                        k = 1
                    else:
                        k = 2
                else:
                    k = 3 if env._ho_timer[i] > 0 else 4
                cause[i, k] += 1
            inflight_sum += float(np.sum((env._ho_timer > 0) & act_now)); n_frames += 1
            prev_active = act_now.copy(); prev_serv = serv_now.copy()
        assert np.array_equal(cause.sum(axis=1), rlf), 'causes do not add up'
        aa = np.maximum(act, 1)
        w = float((rlf / aa).max()); m = float(rlf.sum() / aa.sum()); o = float(ok.sum() / aa.sum())
        pw.append(w); pm.append(m); po.append(o)
        wi = int(np.argmax(rlf / aa))
        row = dict(seed=int(s), scene=str(info0['scene']), worst=w, mean=m, ok=o, worst_car=wi,
                   worst_act=float(act[wi]), worst_rlf=float(rlf[wi]), p_act=float(act.sum()),
                   p_rlf=float(rlf.sum()), starts=n_start, blind=n_blind,
                   inflight=inflight_sum / max(n_frames, 1), land_over=land_over, steps_over=steps_over,
                   dec_over=dec_over)
        for k, c in enumerate(CAUSES):
            row[f'w_{c}'] = float(cause[wi, k]); row[f'p_{c}'] = float(cause[:, k].sum())
        rows.append(row)
    return np.array(pw), np.array(pm), np.array(po), rows
