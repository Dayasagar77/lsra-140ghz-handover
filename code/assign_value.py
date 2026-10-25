#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage15 -- the value of global optimal assignment at matched timing.

Pre-registered in stage15/PREREG_ASSIGNMENT_VALUE_20260914.md, frozen before this ran.
Q0 and Q2 are imported unmodified from paper2ma_confirm.py (26 Jul). Q1 applies the Hungarian
solve to exactly Q0's decision set, on exactly Q0's desired() scores.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
import paper2ma_env as MA, paper2_controller as C
import paper2ma_confirm as CF

BOOT = 10000
IDENTITY_TARGET = 8.337717529555423
BIG = 1e6
_STATS = {'mover_mismatch': 0, 'mover_steps': 0, 'unplaceable': 0}


def movers_of(env, sinr):
    """pol_loadaware's trigger, verbatim: active and sinr below the outage threshold."""
    fr = env.frames[env.t]
    return [i for i in range(env.M) if fr['active'][i] and sinr[i] < MA.SINR_OUT_DB]


def pol_hung_restricted(env, obs, sinr, rng, ag, temp):
    """Q1: global capacity-constrained optimal matching of exactly Q0's movers.

    Non-movers hold their tower and consume one slot of its capacity. Movers are matched to the
    remaining slots on CF.desired(), each with its own tower excluded (Q0's must-move discipline).
    A mover with no feasible slot does not move -- recorded as 'unplaceable'.
    """
    M = env.M; fr = env.frames[env.t]; ng = env.n_gnb; cap = int(env.gnb_cap or M)
    a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; tgt = -np.ones(M, int)
    mv = movers_of(env, sinr); mvset = set(mv)
    # every active agent starts as a non-mover; only a successfully matched mover gets a[i,0]=1
    for i in range(M):
        if fr['active'][i]:
            a[i, 0] = 0
    rem = np.full(ng, cap, dtype=int)
    for i in range(M):
        if fr['active'][i] and i not in mvset:
            rem[env.serv[i]] -= 1
    slot_tower = [g for g in range(ng) for _ in range(max(int(rem[g]), 0))]
    NS = len(slot_tower)
    _STATS['mover_steps'] += len(mv)
    if mv and NS:
        cost = np.full((len(mv), NS), BIG)
        for r, i in enumerate(mv):
            for s in range(NS):
                g = slot_tower[s]
                cost[r, s] = BIG if g == int(env.serv[i]) else -CF.desired(env, i, g, MA.PT_MAX)
        ri, ci = linear_sum_assignment(cost)
        placed = set()
        for r, c in zip(ri, ci):
            if cost[r, c] < BIG:
                tgt[mv[r]] = slot_tower[c]; a[mv[r], 0] = 1; placed.add(r)
        _STATS['unplaceable'] += len(mv) - len(placed)
    else:
        _STATS['unplaceable'] += len(mv)
    return a, tgt


def pol_q0_checked(env, obs, sinr, rng, ag, temp):
    """Q0 verbatim, plus the E0 structural check that its mover set equals movers_of()."""
    a, tgt = CF.pol_loadaware(env, obs, sinr, rng, ag, temp)
    fr = env.frames[env.t]
    q0 = set(i for i in range(env.M) if fr['active'][i] and int(a[i, 0]) == 1)
    if q0 != set(movers_of(env, sinr)):
        _STATS['mover_mismatch'] += 1
    return a, tgt


LADDER = [('Q0_loadaware', pol_q0_checked), ('Q1_hung_restricted', pol_hung_restricted),
          ('Q2_hung_full', CF.pol_hungarian)]


def run_eval4(env, policy_fn, seeds):
    pw, pm, po = [], [], []
    tot_rlf = None; tot_act = None; n_ho = 0.0; n_step = 0.0
    rng = np.random.default_rng(20260726)
    for s in seeds:
        obs, info = env.reset(seed=s); M = env.M
        env.sb = np.array([i % env.n_sb for i in range(M)], dtype=int)
        sinr = np.array([env._sinr_db(i) for i in range(M)])
        rlf = np.zeros(M); act = np.zeros(M); ok = np.zeros(M); done = False
        while not done:
            a, tgt = policy_fn(env, obs, sinr, rng, None, 0.0)
            obs, rew, term, done, info = env.step(a, ho_targets=tgt)
            n_ho += float(np.sum(info['ho']))
            for i in range(M):
                if info['active'][i]:
                    sinr[i] = obs[i, 0] * 30.0
                    rlf[i] += info['rlf'][i]; ok[i] += info['link_ok'][i]; act[i] += 1
        aa = np.maximum(act, 1)
        pw.append(float((rlf / aa).max())); pm.append(float(rlf.sum() / aa.sum()))
        po.append(float(ok.sum() / aa.sum()))
        if tot_rlf is None:
            tot_rlf = np.zeros(M); tot_act = np.zeros(M)
        tot_rlf += rlf; tot_act += act; n_step += float(act.sum())
    pooled = float((tot_rlf / np.maximum(tot_act, 1)).max())
    return np.array(pw), np.array(pm), np.array(po), pooled, n_ho / max(n_step, 1.0)


def paired_ci(d, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def make_env(pool, split, hl):
    return MA.MultiAgentRSUBlockageEnv(pool, n_agents=16, n_subbands=3, gnb_cap=4, seed=0,
                                       split=split, compute_anticipation=True, ho_latency=hl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--ho_lats', type=int, nargs='+', default=[2, 5, 10, 15])
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='ASSIGN_VALUE.json')
    a = ap.parse_args()
    pool = pd.read_csv(a.pool)
    seeds = [C.EVAL_SEED_BASE + k for k in range(a.n)]
    print(f'stage15 | M=16 cap=4 | n={a.n} | split={a.split} | hl={a.ho_lats}', flush=True)

    env = make_env(pool, 'test', 5)
    e1, f1, g1 = CF.run_eval(env, CF.pol_loadaware, seeds[:12])
    env = make_env(pool, 'test', 5)
    e2, f2, g2, _, _ = run_eval4(env, CF.pol_loadaware, seeds[:12])
    eq = bool(np.array_equal(e1, e2) and np.array_equal(f1, f2) and np.array_equal(g1, g2))
    print(f'[equivalence] run_eval4 == CF.run_eval: {eq}', flush=True)
    env = make_env(pool, 'test', 5)
    pw, _, _, _, _ = run_eval4(env, CF.pol_hungarian, seeds)
    got = float(pw.mean()) * 100.0; delta = abs(got - IDENTITY_TARGET); idok = delta < 1e-9
    print(f'[identity] Hungarian test/hl=5 -> {got!r} | delta {delta:.3e} | PASS={idok}', flush=True)
    out = {'_gates': {'equivalence': eq, 'identity_got': got, 'identity_delta': delta,
                      'identity_pass': idok},
           '_config': {'n': a.n, 'split': a.split, 'ho_lats': a.ho_lats, 'pool': a.pool,
                       'boot': BOOT, 'agents': 16, 'cap': 4, 'n_subbands': 3}}
    if not (eq and idok):
        out['_verdict'] = {'outcome': 'HALTED -- gate failure'}
        json.dump(out, open(a.out, 'w'), indent=1); print('ABORT: gate failed.'); sys.exit(2)

    for hl in a.ho_lats:
        t0 = time.time(); env = make_env(pool, a.split, hl); cell = {}; pe = {}
        for k in _STATS:
            _STATS[k] = 0
        for name, fn in LADDER:
            pw, pm, po, pooled, hor = run_eval4(env, fn, seeds)
            pe[name] = pw
            m, lo, hi = paired_ci(pw)
            cell[name] = dict(worst=m * 100, worst_ci=[lo * 100, hi * 100],
                              mean_rlf=float(pm.mean()) * 100, link_ok=float(po.mean()) * 100,
                              pooled_worst=pooled * 100, ho_per_step=hor)
        W = {k: cell[k]['worst'] for k, _ in LADDER}
        P = {k: cell[k]['pooled_worst'] for k, _ in LADDER}
        d_as = W['Q0_loadaware'] - W['Q1_hung_restricted']
        d_sc = W['Q1_hung_restricted'] - W['Q2_hung_full']
        _, lo_a, hi_a = paired_ci((pe['Q0_loadaware'] - pe['Q1_hung_restricted']) * 100.0, seed=1)
        _, lo_s, hi_s = paired_ci((pe['Q1_hung_restricted'] - pe['Q2_hung_full']) * 100.0, seed=2)
        d_as_p = P['Q0_loadaware'] - P['Q1_hung_restricted']
        ho_ratio = cell['Q1_hung_restricted']['ho_per_step'] / max(cell['Q0_loadaware']['ho_per_step'], 1e-12)
        unpl = _STATS['unplaceable'] / max(_STATS['mover_steps'], 1)
        E0 = bool(_STATS['mover_mismatch'] == 0)
        E1 = bool(d_as >= 10.0 and lo_a > 0.0)
        E2 = bool(d_as_p >= 10.0)
        E3 = bool(ho_ratio <= 2.0)
        cell['_assign'] = dict(d_assignment=d_as, d_assignment_ci=[lo_a, hi_a],
                               d_scope=d_sc, d_scope_ci=[lo_s, hi_s],
                               d_assignment_pooled=d_as_p, ho_ratio_Q1_over_Q0=ho_ratio,
                               unplaceable_rate=unpl, mover_mismatch=_STATS['mover_mismatch'],
                               E0=E0, E1=E1, E2=E2, E3=E3,
                               MOTIVATED=bool(E0 and E1 and E2 and E3),
                               primary=bool(hl == 2), wall_s=round(time.time() - t0, 1))
        out[f'{a.split}_hl{hl}'] = cell
        print(f'[hl={hl}] Q0 {W["Q0_loadaware"]:.2f} Q1 {W["Q1_hung_restricted"]:.2f} '
              f'Q2 {W["Q2_hung_full"]:.2f} | assign {d_as:+.2f} [{lo_a:+.2f},{hi_a:+.2f}] '
              f'(pooled {d_as_p:+.2f}) | scope {d_sc:+.2f} [{lo_s:+.2f},{hi_s:+.2f}] | '
              f'HOratio {ho_ratio:.2f} unplaceable {unpl:.3f} mismatch {_STATS["mover_mismatch"]} | '
              f'E0={E0} E1={E1} E2={E2} E3={E3} | {cell["_assign"]["wall_s"]}s', flush=True)
        json.dump(out, open(a.out, 'w'), indent=1)
        if not E0:
            out['_verdict'] = {'outcome': 'HALTED -- E0 mover-set mismatch; ladder confounded'}
            json.dump(out, open(a.out, 'w'), indent=1); print('ABORT: E0 failed.'); sys.exit(3)

    p = out.get(f'{a.split}_hl2', {}).get('_assign')
    if p:
        out['_verdict'] = {k: p[k] for k in ('d_assignment', 'd_assignment_ci', 'd_assignment_pooled',
                                             'd_scope', 'ho_ratio_Q1_over_Q0', 'unplaceable_rate',
                                             'E0', 'E1', 'E2', 'E3', 'MOTIVATED')}
        out['_verdict']['primary_cell'] = f'{a.split}_hl2'
        out['_verdict']['outcome'] = ('CANDIDATE 32 MOTIVATED AND SIZED' if p['MOTIVATED']
                                      else 'CANDIDATE 32 NOT MOTIVATED -- assignment value below bar')
        json.dump(out, open(a.out, 'w'), indent=1)
        print('\nVERDICT:', json.dumps(out['_verdict'], indent=1), flush=True)


if __name__ == '__main__':
    main()
