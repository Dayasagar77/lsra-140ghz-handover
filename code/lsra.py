#!/usr/bin/env python3
"""Landing-state rotation assignment (LSRA) and the one-line fix R1i, for paper2ma_env (20 Oct 2026).

Method (maths note MATH_LandingStateRotationAssignment_DRAFT_v1_20261020.md, section 4). Every step:
  1. landing state: a car in flight is pinned to the station it will land on; the others are free;
     quota left at each station q_g = C - (pinned cars landing at g);
  2. risk after landing: for each car and station, the forecast probability of failure, averaged over
     the window [t+L+1, t+2L+1] (H = L+1 steps, until the earliest possible correction lands);
     p = Phi((SINR_out - mean SINR) / sigma), with the environment's own channel model at mean
     shadow fading and interference from every present car on the same sub-band;
       forecast='now'    : the current geometry, held over the window (deployable, no prediction);
       forecast='oracle' : the true future geometry in the window (reference only);
  3. decision: for the free cars, minimise the worst car's risk (counting the pinned cars' fixed
     risks), then the total risk plus beta per move, subject to every station's quota (exact:
     threshold search with assignment feasibility, then a min-cost assignment);
  4. act: free cars whose station changes start their handovers now; all land together.

Ablations: landing=False plans on the current serving stations with every car movable (in-flight
cars cannot actually move, as in R3); minmax=False keeps only the min-cost stage.

R1i = R1 (capacity_rule.pol_loadaware_cap) that counts cars on their way: the load of a station is
its landing load, and each decision is counted for the cars decided after it in the same step.

Power and sub-band are kept as in the other rules (full power, sub-band unchanged).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment as LSA
from scipy.special import ndtr
import paper2ma_env as MA
import paper2ma_confirm as CF

BIG = 1e9


def mean_sinr_db(env, fr):
    """Mean SINR (dB) of every agent on every station in frame fr (shadow fading at its mean)."""
    M, G = env.M, env.n_gnb
    act = np.asarray(fr['active'], dtype=bool)
    D = np.ones((M, G)); DEP = np.zeros((M, G))
    for j in range(M):
        a = fr['ag'][j]
        if a is not None:
            D[j] = a['dist']; DEP[j] = a['depth']
    PL = MA.CH.ci_pathloss_db(np.maximum(D, 1.0), True)
    pt = np.asarray(env.pt, dtype=float)[:, None]
    pdes = 10 ** ((pt + MA.G_SERV - PL - DEP) / 10.0)
    pside = 10 ** ((pt + MA.G_SIDE - PL - DEP) / 10.0) * act[:, None]
    sb = np.asarray(env.sb, dtype=int)
    band = np.zeros((env.n_sb, G))
    for b in range(env.n_sb):
        band[b] = pside[sb == b].sum(axis=0)
    I = band[sb] - pside + np.asarray(fr['bg'], dtype=float)[None, :]
    return 10 * np.log10(pdes / (MA.N0_MW + I)), DEP, act


def risk_frame(env, fr):
    sinr, dep, act = mean_sinr_db(env, fr)
    sig = np.where(dep > 0, MA.CH.SIG_NLOS, MA.CH.SIG_LOS)
    p = ndtr((MA.SINR_OUT_DB - sinr) / sig)
    p[~act] = 0.0
    return p


def window_risk(env, forecast):
    t, L, F = env.t, int(env.ho_lat), len(env.frames)
    if forecast == 'now':
        frs = [env.frames[t]]
    elif forecast == 'oracle':
        lo, hi = t + L + 1, min(t + 2 * L + 1, F - 1)
        frs = [env.frames[k] for k in range(lo, hi + 1)] if lo <= hi else [env.frames[F - 1]]
    else:
        raise ValueError(forecast)
    r = np.zeros((env.M, env.n_gnb))
    for f in frs:
        r += risk_frame(env, f)
    return r / len(frs)


def bottleneck_then_mincost(R, slot_g, serv, beta, floor, minmax=True):
    """R: (nF, G) risks of the free cars; slot_g: station of each available slot (len >= nF);
    serv: current station of each free car; floor: largest fixed risk among pinned cars.
    Returns one station per free car: lexicographically minimal (max risk, then total risk + beta*moves)."""
    nF = R.shape[0]
    if nF == 0:
        return []
    slot_g = np.asarray(slot_g, dtype=int)
    RS = R[:, slot_g]
    if minmax:
        vals = np.unique(RS)

        def feasible(th):
            c = (RS > th).astype(float)
            r, k = LSA(c)
            return c[r, k].sum() == 0

        lo, hi = 0, len(vals) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if feasible(vals[mid]):
                hi = mid
            else:
                lo = mid + 1
        th = max(float(vals[lo]), float(floor))
    else:
        th = np.inf
    move = (slot_g[None, :] != np.asarray(serv, dtype=int)[:, None]).astype(float)
    C2 = np.where(RS <= th, RS + beta * move, BIG)
    r, k = LSA(C2)
    out = [None] * nF
    for a, b in zip(r, k):
        out[a] = int(slot_g[b])
    return out


def make_lsra(forecast='now', landing=True, minmax=True, beta=1e-3):
    def pol(env, obs, sinr, rng, ag, temp):
        M, G, cap = env.M, env.n_gnb, int(env.gnb_cap)
        fr = env.frames[env.t]; act = fr['active']
        a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; tgt = -np.ones(M, int)
        if landing:
            pinned = [i for i in range(M) if act[i] and env._ho_timer[i] > 0]
            free = [i for i in range(M) if act[i] and env._ho_timer[i] == 0]
        else:
            pinned = []
            free = [i for i in range(M) if act[i]]
        risk = window_risk(env, forecast)
        q = np.full(G, cap, dtype=int); floor = 0.0
        for i in pinned:
            g = int(env._ho_target[i]); q[g] -= 1; floor = max(floor, float(risk[i, g]))
        if (q < 0).any():
            raise RuntimeError('pinned cars exceed a station capacity at landing')
        slot_g = [g for g in range(G) for _ in range(int(q[g]))]
        if len(slot_g) < len(free):
            raise RuntimeError('more free cars than places at landing')
        serv = [int(env.serv[i]) for i in free]
        s = bottleneck_then_mincost(risk[free], slot_g, serv, beta, floor, minmax=minmax)
        for i, g in zip(free, s):
            tgt[i] = g
            a[i, 0] = int(g != int(env.serv[i]))
        for i in pinned:
            a[i, 0] = 0
        return a, tgt
    return pol


def pol_r1i(env, obs, sinr, rng, ag, temp):
    """R1 that counts cars on their way (landing load), updated decision by decision."""
    M = env.M; a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1; fr = env.frames[env.t]
    cap = int(env.gnb_cap); tgt = -np.ones(M, int)
    occ = np.zeros(env.n_gnb, dtype=int)
    for j in range(M):
        if fr['active'][j]:
            occ[int(env._ho_target[j]) if env._ho_timer[j] > 0 else int(env.serv[j])] += 1
    for i in range(M):
        if not fr['active'][i]:
            continue
        if env._ho_timer[i] > 0 or sinr[i] >= MA.SINR_OUT_DB:
            a[i, 0] = 0
            continue
        others = [g for g in range(env.n_gnb) if g != int(env.serv[i])]
        feas = [g for g in others if occ[g] < cap]
        if not feas:
            a[i, 0] = 0
            continue
        a[i, 0] = 1
        top = sorted(feas, key=lambda g: -CF.desired(env, i, g, MA.PT_MAX))[:2]
        tgt[i] = int(min(top, key=lambda g: occ[g]))
        occ[tgt[i]] += 1; occ[int(env.serv[i])] -= 1
    return a, tgt
