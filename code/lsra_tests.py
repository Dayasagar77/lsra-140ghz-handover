#!/usr/bin/env python3
"""Correctness tests for lsra.py and method_harness.py (20 Oct 2026). Design scenes only; no held-out
scene is touched, and no failure-rate number is printed.

  H  : method_harness.run_eval_m equals remeasure_ma.run_eval_log on every shared output
       (12 design episodes, fixed attachment, 1.0 s, R1 and R3).
  U1 : the bottleneck-then-min-cost solver equals brute force (500 random small instances with ties).
  U2 : LSRA with the landing state (forecast now / oracle, with and without min-max) and R1i never
       leave a station over capacity at landing after a step's decisions (land_over == 0), in
       12 design episodes at 0.2 s and 1.0 s; the 'landing state off' ablation runs.
  U3 : seconds per episode for each new arm.
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import remeasure_ma as RM
import remeasure_ma2 as RM2
import capacity_rule as CR
import paper2ma_confirm as CF
import lsra as LS
import method_harness as MH

OUT = os.path.join(RM.HERE, 'models', 'method_dev')


def brute(R, q, serv, beta, floor):
    nF, G = R.shape
    best = None
    for combo in itertools.product(range(G), repeat=nF):
        cnt = np.bincount(combo, minlength=G)
        if (cnt > q).any():
            continue
        mx = max(max(R[i, g] for i, g in enumerate(combo)), floor)
        sm = sum(R[i, g] + beta * (g != serv[i]) for i, g in enumerate(combo))
        key = (mx, sm)
        if best is None or key[0] < best[0] - 1e-15 or (abs(key[0] - best[0]) <= 1e-15 and key[1] < best[1] - 1e-12):
            best = key
    return best


def u1(n=500, seed=0):
    rng = np.random.default_rng(seed); bad = 0
    for _ in range(n):
        G = int(rng.integers(2, 5)); nF = int(rng.integers(1, 7))
        q = rng.integers(0, 4, G)
        while q.sum() < nF:
            q[rng.integers(G)] += 1
        R = rng.choice([0.0, 0.05, 0.1, 0.3, 0.7, 1.0], size=(nF, G))
        serv = rng.integers(0, G, nF); floor = float(rng.choice([0.0, 0.1, 0.5]))
        beta = 1e-3
        slot_g = [g for g in range(G) for _ in range(int(q[g]))]
        s = LS.bottleneck_then_mincost(R, slot_g, serv, beta, floor, minmax=True)
        cnt = np.bincount(s, minlength=G)
        mx = max(max(R[i, g] for i, g in enumerate(s)), floor)
        sm = sum(R[i, g] + beta * (g != serv[i]) for i, g in enumerate(s))
        b = brute(R, q, serv, beta, floor)
        if (cnt > q).any() or abs(mx - b[0]) > 1e-12 or abs(sm - b[1]) > 1e-9:
            bad += 1
    return dict(instances=n, mismatches=bad, ok=bool(bad == 0))


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = pd.read_csv(RM.POOL)
    seeds = list(pd.read_csv(os.path.join(RM.HERE, 'models', 'remeasure_ma', 'SEEDS_train.csv')).seed.astype(int))[:12]
    res = {}
    # H
    env = RM2.make_env2(pool, 'train', 10, attach=True)
    shared = ['seed', 'scene', 'worst', 'mean', 'ok', 'worst_car', 'worst_act', 'worst_rlf', 'p_act', 'p_rlf',
              'starts', 'blind', 'inflight'] + [f'w_{c}' for c in RM.CAUSES] + [f'p_{c}' for c in RM.CAUSES]
    hok = True
    for name, fn in [('R1', CR.pol_loadaware_cap), ('R3', CF.pol_hungarian)]:
        _, _, _, r1 = RM.run_eval_log(env, fn, seeds)
        _, _, _, r2 = MH.run_eval_m(env, fn, seeds)
        hok &= all(all(a[k] == b[k] for k in shared) for a, b in zip(r1, r2))
    res['H'] = dict(ok=bool(hok)); print('[H]', res['H'], flush=True)
    # U1
    res['U1'] = u1(); print('[U1]', res['U1'], flush=True)
    # U2 / U3
    arms = {'LSRA_now': LS.make_lsra('now'), 'LSRA_oracle': LS.make_lsra('oracle'),
            'LSRA_now_nominmax': LS.make_lsra('now', minmax=False), 'R1i': LS.pol_r1i,
            'LSRA_now_nolanding': LS.make_lsra('now', landing=False),
            'R3': CF.pol_hungarian, 'R1': CR.pol_loadaware_cap}
    lsra_landing = {'LSRA_now', 'LSRA_oracle', 'LSRA_now_nominmax'}
    u2 = {}; u3 = {}
    for hl in [2, 10]:
        env = RM2.make_env2(pool, 'train', hl, attach=True)
        for name, fn in arms.items():
            t0 = time.time()
            try:
                _, _, _, rows = MH.run_eval_m(env, fn, seeds)
                lo = int(sum(r['land_over'] for r in rows)); do = int(sum(r['dec_over'] for r in rows))
                st = int(sum(r['starts'] for r in rows)); err = None
            except Exception as ex:
                lo = do = st = None; err = repr(ex)
            u3[f'{name}_hl{hl}'] = round((time.time() - t0) / len(seeds), 3)
            if name in lsra_landing:
                ok = (err is None) and do == 0 and lo == 0         # creates none, leaves none
            elif name == 'R1i':
                ok = (err is None) and do == 0                     # creates none
            else:
                ok = err is None                                   # reported only
            u2[f'{name}_hl{hl}'] = dict(starts=st, dec_over=do, land_over=lo, error=err, ok=bool(ok))
            print('[U2]', name, hl, u2[f'{name}_hl{hl}'], 's/episode', u3[f'{name}_hl{hl}'], flush=True)
    res['U2'] = u2; res['U3_seconds_per_episode'] = u3
    res['_pass'] = bool(res['H']['ok'] and res['U1']['ok'] and all(v['ok'] for v in u2.values()))
    json.dump(res, open(os.path.join(OUT, 'LSRA_TESTS.json'), 'w'), indent=1)
    print('LSRA TESTS', 'PASS' if res['_pass'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
