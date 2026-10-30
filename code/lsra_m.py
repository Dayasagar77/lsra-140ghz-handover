#!/usr/bin/env python3
"""lsra_m.py -- LSRA fed only measurement reports (28 Oct 2026).

Plan: PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation (28 Oct 2026).

LSRA (lsra.make_lsra('now', landing=True, minmax=False, beta=0.05)) computes each vehicle's risk on each
station from the geometry: the mean received power p + G_s - PL - blockage, the side-lobe power of every other
vehicle on the same sub-band, and sigma = 8.2 dB on blocked links, 2.7 dB otherwise. LSRA-M keeps the planner
and replaces only these inputs by what a network measures:
  * F[i, g]: the received power of vehicle i from station g in each slot, with that slot's shadowing draw
    (a3_rule.measured_dbm, the quantity A3 uses), smoothed by the 3GPP layer-3 filter with K = 4 (the default
    filter coefficient), restarted when a vehicle (re)appears;
  * the side-lobe power of vehicle j at station g: F[j, g] - G_s + G_sl;
  * sigma = 8.2 dB on every link (no geometry says which links are blocked).
risk[i, g] = Phi((gamma_out - SINR_est[i, g]) / sigma), with SINR_est built exactly as lsra.mean_sinr_db does.
Gate mode (G2): source = 'mean', k = 0, sigma_rule = 'depth' reproduces LSRA's risk (to rounding) and decisions.
"""
import numpy as np
from scipy.special import ndtr
import paper2ma_env as MA
import lsra as LS
import a3_rule as A3
import a3_general as AG


def risk_from_power(env, fr, F, sigma_rule='fixed', sigma=8.2):
    """F: (M, G) estimated received power in dBm (rows of absent vehicles are ignored)."""
    M, G = env.M, env.n_gnb
    act = np.asarray(fr['active'], dtype=bool)
    pdes = 10 ** (F / 10.0)
    pside = 10 ** ((F - MA.G_SERV + MA.G_SIDE) / 10.0) * act[:, None]
    sb = np.asarray(env.sb, dtype=int)
    band = np.zeros((env.n_sb, G))
    for b in range(env.n_sb):
        band[b] = pside[sb == b].sum(axis=0)
    I = band[sb] - pside + np.asarray(fr['bg'], dtype=float)[None, :]
    sinr = 10 * np.log10(pdes / (MA.N0_MW + I))
    if sigma_rule == 'depth':
        DEP = np.zeros((M, G))
        for j in range(M):
            a = fr['ag'][j]
            if a is not None:
                DEP[j] = a['depth']
        sig = np.where(DEP > 0, MA.CH.SIG_NLOS, MA.CH.SIG_LOS)
    else:
        sig = np.full((M, G), float(sigma))
    p = ndtr((MA.SINR_OUT_DB - sinr) / sig)
    p[~act] = 0.0
    return p


def lsra_decide(env, risk, beta):
    """lsra.make_lsra's decision lines (landing=True, minmax=False) with a given risk matrix."""
    M, G, cap = env.M, env.n_gnb, int(env.gnb_cap)
    fr = env.frames[env.t]
    act = fr['active']
    a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1
    tgt = -np.ones(M, int)
    pinned = [i for i in range(M) if act[i] and env._ho_timer[i] > 0]
    free = [i for i in range(M) if act[i] and env._ho_timer[i] == 0]
    q = np.full(G, cap, dtype=int)
    floor = 0.0
    for i in pinned:
        g = int(env._ho_target[i]); q[g] -= 1; floor = max(floor, float(risk[i, g]))
    if (q < 0).any():
        raise RuntimeError('pinned cars exceed a station capacity at landing')
    slot_g = [g for g in range(G) for _ in range(int(q[g]))]
    if len(slot_g) < len(free):
        raise RuntimeError('more free cars than places at landing')
    serv = [int(env.serv[i]) for i in free]
    s = LS.bottleneck_then_mincost(risk[free], slot_g, serv, beta, floor, minmax=False)
    for i, g in zip(free, s):
        tgt[i] = g
        a[i, 0] = int(g != int(env.serv[i]))
    for i in pinned:
        a[i, 0] = 0
    return a, tgt


class LSRAM:
    """LSRA-M as a policy. Holds the layer-3 filter state; slots of an episode must come in order from t = 0."""

    def __init__(self, k=4, source='measured', sigma_rule='fixed', sigma=8.2, beta=0.05):
        if source not in ('measured', 'mean'):
            raise ValueError(source)
        self.k, self.a = k, A3.alpha_slot(k)
        self.source, self.sigma_rule, self.sigma, self.beta = source, sigma_rule, float(sigma), float(beta)
        self._z = None
        self.last_risk = None

    def _new_episode(self, env):
        self.F = np.zeros((env.M, env.n_gnb))
        self.seen = np.zeros(env.M, dtype=bool)
        self.last_t = -1
        self._z = env._z

    def __call__(self, env, obs, sinr, rng, ag, temp):
        if env.t == 0:
            self._new_episode(env)
        if self._z is not env._z or env.t != self.last_t + 1:
            raise RuntimeError('LSRAM: slots of one episode must be given in order, starting at t = 0')
        self.last_t = env.t
        fr = env.frames[env.t]
        for i in range(env.M):
            if not fr['active'][i]:
                self.seen[i] = False
                continue
            m = A3.measured_dbm(env, i) if self.source == 'measured' else AG.mean_dbm(env, i)
            if not self.seen[i]:
                self.F[i] = m
                self.seen[i] = True
            else:
                self.F[i] = (1.0 - self.a) * self.F[i] + self.a * m
        risk = risk_from_power(env, fr, self.F, self.sigma_rule, self.sigma)
        self.last_risk = risk
        return lsra_decide(env, risk, self.beta)
