#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2ma_antic_test.py — the decisive test: does ANTICIPATION beat REACTION?

Same optimal dispatcher (Hungarian) and same decentralized rule (load-aware), each run TWO ways that
differ ONLY in the information they use to choose a tower:
  * REACTIVE   : the tower with the best signal RIGHT NOW.
  * ANTICIPATORY: the same, but a tower predicted to be blocked within WINDOW seconds is penalised by
    its projected block depth -- so a car is not sent to a tower that is about to go dark, even if it
    looks best this instant. The prediction is the geometric time-to-blockage / projected-depth from
    anticipate.py (constant-velocity forward projection + ray-cast), already precomputed per (agent,
    tower) in the env.

If the anticipatory variants keep more cars connected than the reactive ones, on the SAME scenes, the
value of anticipation is proven cleanly -- everything else held identical. That is the paper's spine.
Metric: per-episode worst-agent RLF% (primary) + mean RLF%, paired same seeds, bootstrap 95% CI,
TRAIN and HELD-OUT scenes.
"""
import sys, argparse
sys.path.insert(0, '/root/work/paper2')
import numpy as np, pandas as pd
import paper2ma_env as MA, paper2_controller as C
import paper2ma_confirm as CF
from scipy.optimize import linear_sum_assignment

WINDOW = 0.5   # seconds: treat a tower as 'about to be blocked' if predicted block is within this


def adj_desired(env, i, g, pt):
    """Anticipatory desirability: current signal minus the projected block depth if the tower is
    predicted to be blocked within WINDOW seconds."""
    a = env.frames[env.t]['ag'][i]
    d = pt + MA.G_SERV - MA.CH.ci_pathloss_db(max(a['dist'][g], 1.0), True) - a['depth'][g]
    if a['ttb'][g] <= WINDOW:
        d -= a['projd'][g]
    return d


def hungarian_generic(env, signal_fn):
    M = env.M; fr = env.frames[env.t]; ng = env.n_gnb; cap = int(env.gnb_cap or M)
    active = [i for i in range(M) if fr['active'][i]]
    slot_tower = [g for g in range(ng) for _ in range(cap)]; NS = len(slot_tower)
    cost = np.full((len(active), NS), 1e6)
    for r, i in enumerate(active):
        for s in range(NS):
            cost[r, s] = -signal_fn(env, i, slot_tower[s], MA.PT_MAX)
    a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; tgt = -np.ones(M, int)
    if active:
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            tgt[active[r]] = slot_tower[c]; a[active[r], 0] = 1
    return a, tgt


def loadaware_generic(env, sinr, signal_fn):
    M = env.M; a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; fr = env.frames[env.t]
    occ = CF.occ_vec(env); tgt = -np.ones(M, int)
    for i in range(M):
        if not fr['active'][i]:
            continue
        if sinr[i] < MA.SINR_OUT_DB:
            a[i, 0] = 1
            ds = [(signal_fn(env, i, g, MA.PT_MAX) if g != env.serv[i] else -1e9) for g in range(env.n_gnb)]
            top = list(np.argsort(ds)[::-1][:2])
            tgt[i] = int(min(top, key=lambda g: occ[g]))
        else:
            a[i, 0] = 0
    return a, tgt


def oracle_desired(env, i, g, pt):
    """PERFECT foresight: the ACTUAL signal at the frame the handover would complete (t + ho_lat),
    read from the real precomputed future geometry -- the ceiling on any predictor's value."""
    tf = min(env.t + env.ho_lat, len(env.frames) - 1)
    a = env.frames[tf]['ag'][i]
    if a is None:
        a = env.frames[env.t]['ag'][i]
    return pt + MA.G_SERV - MA.CH.ci_pathloss_db(max(a['dist'][g], 1.0), True) - a['depth'][g]


def pol_hung_react(env, obs, sinr, rng, ag, temp): return hungarian_generic(env, CF.desired)
def pol_hung_antic(env, obs, sinr, rng, ag, temp): return hungarian_generic(env, adj_desired)
def pol_hung_oracle(env, obs, sinr, rng, ag, temp): return hungarian_generic(env, oracle_desired)
def pol_la_react(env, obs, sinr, rng, ag, temp):  return loadaware_generic(env, sinr, CF.desired)
def pol_la_antic(env, obs, sinr, rng, ag, temp):  return loadaware_generic(env, sinr, adj_desired)


def main():
    global WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='/root/work/paper2/results/dair_geometry_pool_full300.csv')
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--cap', type=int, default=4)
    ap.add_argument('--agents', type=int, default=16)
    ap.add_argument('--split', default='train')
    ap.add_argument('--ho_lats', type=int, nargs='+', default=[2, 5, 10])
    a = ap.parse_args()
    pool = pd.read_csv(a.pool)
    print(f'ANTICIPATION vs REACTION across handover latency | window {WINDOW}s | cap {a.cap} | '
          f'{a.n} eps | {a.split} scenes | metric = per-episode worst-agent RLF% (+mean), 95% CI')
    seeds = [C.EVAL_SEED_BASE + k for k in range(a.n)]
    for hl in a.ho_lats:
        WINDOW = hl * 0.1 + 0.5          # foresight must cover the handover time + margin, else it is too late
        env = MA.MultiAgentRSUBlockageEnv(pool, n_agents=a.agents, n_subbands=3, gnb_cap=a.cap, seed=0,
                                          split=a.split, compute_anticipation=True, ho_latency=hl)
        print(f'\n=== handover latency = {hl} steps ({hl*0.1:.1f}s), foresight window = {WINDOW:.1f}s ===')
        print(f'{"policy":<26} {"worst-agent RLF%":>24} {"mean RLF%":>22}')
        res = {}
        for name, fn in [('Hungarian  REACTIVE', pol_hung_react),
                         ('Hungarian  ANTICIPATORY(cv)', pol_hung_antic),
                         ('Hungarian  ORACLE-future', pol_hung_oracle)]:
            pw, pm, po = CF.run_eval(env, fn, seeds)
            mw, lw, hw = CF.ci(pw); mm, lm, hm = CF.ci(pm)
            res[name] = mw * 100
            print(f'{name:<28} {mw*100:6.2f} [{lw*100:5.2f},{hw*100:5.2f}]   {mm*100:6.2f} [{lm*100:5.2f},{hm*100:5.2f}]')
        print(f'  -> worst-agent gain vs REACTIVE: constant-velocity {res["Hungarian  REACTIVE"]-res["Hungarian  ANTICIPATORY(cv)"]:+.2f} pp | '
              f'PERFECT foresight {res["Hungarian  REACTIVE"]-res["Hungarian  ORACLE-future"]:+.2f} pp')


if __name__ == '__main__':
    main()
