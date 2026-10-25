#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage16 -- can one `if` close the assignment gap? Pre-registered in
stage16/PREREG_CAPACITY_RULE_20260914.md, frozen before this ran.

R0 and R3 come verbatim from paper2ma_confirm.py; R2 is imported from stage15's assign_value.py so
it is bit-identical to Q1. R1 is R0 with exactly one change to its candidate rule.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import paper2ma_env as MA, paper2_controller as C
import paper2ma_confirm as CF
import assign_value as AV          # Q1 reused verbatim; also run_eval4 / paired_ci / make_env

BOOT = AV.BOOT
_S = {'trig': 0, 'exposed': 0, 'held': 0}


def pol_loadaware_cap(env, obs, sinr, rng, ag, temp):
    """R1: pol_loadaware, except a tower at or above gnb_cap is not a candidate.

    Identical trigger (sinr < SINR_OUT_DB), identical scoring (CF.desired at PT_MAX), identical
    'less crowded of the two best alternatives' tie-break. Only the definition of 'alternative'
    changes: not the current tower AND not already full. With no alternative left, the agent holds.
    """
    M = env.M; a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; fr = env.frames[env.t]
    occ = CF.occ_vec(env); cap = int(env.gnb_cap or M); tgt = -np.ones(M, int)
    for i in range(M):
        if not fr['active'][i]:
            continue
        if sinr[i] >= MA.SINR_OUT_DB:
            a[i, 0] = 0
            continue
        _S['trig'] += 1
        others = [g for g in range(env.n_gnb) if g != int(env.serv[i])]
        if any(occ[g] >= cap for g in others):
            _S['exposed'] += 1
        feas = [g for g in others if occ[g] < cap]
        if not feas:
            a[i, 0] = 0; _S['held'] += 1
            continue
        a[i, 0] = 1
        top = sorted(feas, key=lambda g: -CF.desired(env, i, g, MA.PT_MAX))[:2]
        tgt[i] = int(min(top, key=lambda g: occ[g]))
    return a, tgt


LADDER = [('R0_loadaware', CF.pol_loadaware), ('R1_loadaware_cap', pol_loadaware_cap),
          ('R2_hung_restricted', AV.pol_hung_restricted), ('R3_hung_full', CF.pol_hungarian)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--ho_lats', type=int, nargs='+', default=[2, 5, 10, 15])
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='CAPACITY_RULE.json')
    a = ap.parse_args()
    pool = pd.read_csv(a.pool)
    seeds = [C.EVAL_SEED_BASE + k for k in range(a.n)]
    print(f'stage16 | M=16 cap=4 | n={a.n} | split={a.split} | hl={a.ho_lats}', flush=True)

    env = AV.make_env(pool, 'test', 5)
    e1, f1, g1 = CF.run_eval(env, CF.pol_loadaware, seeds[:12])
    env = AV.make_env(pool, 'test', 5)
    e2, f2, g2, _, _ = AV.run_eval4(env, CF.pol_loadaware, seeds[:12])
    eq = bool(np.array_equal(e1, e2) and np.array_equal(f1, f2) and np.array_equal(g1, g2))
    env = AV.make_env(pool, 'test', 5)
    pw, _, _, _, _ = AV.run_eval4(env, CF.pol_hungarian, seeds)
    got = float(pw.mean()) * 100.0
    delta = abs(got - AV.IDENTITY_TARGET); idok = delta < 1e-9
    print(f'[equivalence] {eq} | [identity] {got!r} delta {delta:.3e} PASS={idok}', flush=True)
    out = {'_gates': {'equivalence': eq, 'identity_got': got, 'identity_delta': delta,
                      'identity_pass': idok},
           '_config': {'n': a.n, 'split': a.split, 'ho_lats': a.ho_lats, 'pool': a.pool,
                       'boot': BOOT, 'agents': 16, 'cap': 4, 'n_subbands': 3}}
    if not (eq and idok):
        out['_verdict'] = {'outcome': 'HALTED -- gate failure'}
        json.dump(out, open(a.out, 'w'), indent=1); print('ABORT: gate failed.'); sys.exit(2)

    for hl in a.ho_lats:
        t0 = time.time(); env = AV.make_env(pool, a.split, hl); cell = {}; pe = {}
        for k in _S:
            _S[k] = 0
        for k in AV._STATS:
            AV._STATS[k] = 0
        for name, fn in LADDER:
            w, m_, o_, pooled, hor = AV.run_eval4(env, fn, seeds)
            pe[name] = w
            mm, lo, hi = AV.paired_ci(w)
            cell[name] = dict(worst=mm * 100, worst_ci=[lo * 100, hi * 100],
                              mean_rlf=float(m_.mean()) * 100, link_ok=float(o_.mean()) * 100,
                              pooled_worst=pooled * 100, ho_per_step=hor)
        W = {k: cell[k]['worst'] for k, _ in LADDER}
        P = {k: cell[k]['pooled_worst'] for k, _ in LADDER}
        resid = W['R1_loadaware_cap'] - W['R2_hung_restricted']
        _, lo_r, hi_r = AV.paired_ci((pe['R1_loadaware_cap'] - pe['R2_hung_restricted']) * 100.0, seed=1)
        cheap = W['R0_loadaware'] - W['R1_loadaware_cap']
        _, lo_c, hi_c = AV.paired_ci((pe['R0_loadaware'] - pe['R1_loadaware_cap']) * 100.0, seed=2)
        total = W['R0_loadaware'] - W['R2_hung_restricted']
        capture = cheap / total if total != 0 else float('nan')
        expo = _S['exposed'] / max(_S['trig'], 1)
        hold = _S['held'] / max(_S['trig'], 1)
        resid_p = P['R1_loadaware_cap'] - P['R2_hung_restricted']
        F0 = bool(expo > 0.10)
        F2 = bool(resid >= 10.0 and lo_r > 0.0)
        cell['_cap'] = dict(total_assign_gap=total, cheap_fix=cheap, cheap_fix_ci=[lo_c, hi_c],
                            residual=resid, residual_ci=[lo_r, hi_r], residual_pooled=resid_p,
                            capture=capture, exposure=expo, hold_rate=hold,
                            F0=F0, F2=F2, SURVIVES=bool(F0 and F2),
                            primary=bool(hl == 2), wall_s=round(time.time() - t0, 1))
        out[f'{a.split}_hl{hl}'] = cell
        print(f'[hl={hl}] R0 {W["R0_loadaware"]:.2f} R1 {W["R1_loadaware_cap"]:.2f} '
              f'R2 {W["R2_hung_restricted"]:.2f} R3 {W["R3_hung_full"]:.2f} | '
              f'cheap {cheap:+.2f} [{lo_c:+.2f},{hi_c:+.2f}] | residual {resid:+.2f} '
              f'[{lo_r:+.2f},{hi_r:+.2f}] (pooled {resid_p:+.2f}) | capture {capture:.3f} | '
              f'exposure {expo:.3f} hold {hold:.3f} | HO/step R0 {cell["R0_loadaware"]["ho_per_step"]:.4f} '
              f'R1 {cell["R1_loadaware_cap"]["ho_per_step"]:.4f} | F0={F0} F2={F2} | '
              f'{cell["_cap"]["wall_s"]}s', flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)

    p = out.get(f'{a.split}_hl2', {}).get('_cap')
    if p:
        out['_verdict'] = {k: p[k] for k in ('cheap_fix', 'residual', 'residual_ci',
                                             'residual_pooled', 'capture', 'exposure',
                                             'hold_rate', 'F0', 'F2', 'SURVIVES')}
        out['_verdict']['primary_cell'] = f'{a.split}_hl2'
        out['_verdict']['outcome'] = (
            'CANDIDATE 32 SURVIVES -- residual joint-matching value remains after the capacity check'
            if p['SURVIVES'] else
            'CANDIDATE 32 CLOSES -- a capacity-aware hand rule accounts for the assignment gap')
        json.dump(out, open(a.out, 'w'), indent=1)
        print('\nVERDICT:', json.dumps(out['_verdict'], indent=1), flush=True)


if __name__ == '__main__':
    main()
