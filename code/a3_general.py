#!/usr/bin/env python3
"""a3_general.py -- event A3 with free parameters and optional load balancing (28 Oct 2026).

Plan: PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation (28 Oct 2026).

A3Gen generalises a3_rule.A3Rule (25 Oct 2026, frozen) and is identical to it when
    n_ttt = a3_rule.ttt_slots(TTT), off = A3 offset, k = filter coefficient, source = 'measured', kappa = 0,
which gate G1 checks against the stored A3 episodes. What it adds:
  * source = 'mean': the measurement is the mean received power from the geometry (no shadowing draw), the
    quantity R1-R3 and LSRA use (the equal-information A3);
  * kappa > 0: a cell individual offset towards each neighbour n of the serving station p,
        O_pn = clip(kappa * (N_p - N_n) / C, -clip, clip)  dB,
    N_g = vehicles present and attached to station g in the slot, C = places per station; the entering
    condition becomes F_n + O_pn > F_p + Off and, among the triggered neighbours, the request goes to the one
    with the largest F_n + O_pn (with kappa = 0 this is A3Rule's choice, the largest F_n).
Everything else (layer-3 filter, counts, restarts, in-flight vehicles, full power, fixed sub-band) is A3Rule's.
"""
import numpy as np
import paper2ma_env as MA
import a3_rule as A3


def mean_dbm(env, i):
    """Mean received power (dBm) of vehicle i at every station in the current slot: no shadowing draw."""
    a = env.frames[env.t]['ag'][i]
    dist = np.maximum(np.asarray(a['dist'], dtype=float), 1.0)
    depth = np.asarray(a['depth'], dtype=float)
    pl = MA.CH.ci_pathloss_db(dist, True)
    return float(env.pt[i]) + MA.G_SERV - pl - depth


class A3Gen:
    """Event A3 with parameters (n_ttt slots, off dB, filter k, source, kappa). One object per arm; it holds
    per-episode state and requires the slots of an episode in order from t = 0 (as A3Rule)."""

    def __init__(self, n_ttt, off, k, source='measured', kappa=0.0, clip=10.0):
        if source not in ('measured', 'mean'):
            raise ValueError(source)
        self.n_ttt = int(n_ttt)
        self.off = float(off)
        self.k = k
        self.a = A3.alpha_slot(k)
        self.source = source
        self.kappa = float(kappa)
        self.clip = float(clip)
        self._z = None

    def describe(self):
        return dict(n_ttt=self.n_ttt, off=self.off, k=self.k, source=self.source, kappa=self.kappa)

    def _new_episode(self, env):
        M, G = env.M, env.n_gnb
        self.F = np.zeros((M, G))
        self.cnt = np.zeros((M, G), dtype=int)
        self.p = np.full(M, -1, dtype=int)
        self.seen = np.zeros(M, dtype=bool)
        self.last_t = -1
        self._z = env._z

    def _cio(self, env, fr):
        """O[p, n] in dB for every pair of stations, from the attached loads of this slot."""
        G = env.n_gnb
        if self.kappa == 0.0:
            return np.zeros((G, G))
        N = np.zeros(G)
        for j in range(env.M):
            if fr['active'][j]:
                N[int(env.serv[j])] += 1
        C = float(env.gnb_cap)
        return np.clip(self.kappa * (N[:, None] - N[None, :]) / C, -self.clip, self.clip)

    def __call__(self, env, obs, sinr, rng, ag, temp):
        if env.t == 0:
            self._new_episode(env)
        if self._z is not env._z or env.t != self.last_t + 1:
            raise RuntimeError('A3Gen: slots of one episode must be given in order, starting at t = 0')
        self.last_t = env.t
        M = env.M
        fr = env.frames[env.t]
        a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1
        tgt = -np.ones(M, int)
        O = self._cio(env, fr)
        for i in range(M):
            if not fr['active'][i]:
                self.seen[i] = False
                a[i, 0] = 0
                continue
            m = A3.measured_dbm(env, i) if self.source == 'measured' else mean_dbm(env, i)
            if not self.seen[i]:
                self.F[i] = m
                self.cnt[i] = 0
                self.seen[i] = True
            else:
                self.F[i] = (1.0 - self.a) * self.F[i] + self.a * m
            p = int(env.serv[i])
            if p != self.p[i] or env._ho_timer[i] > 0:
                self.cnt[i] = 0
                self.p[i] = p
            if env._ho_timer[i] > 0:
                a[i, 0] = 0
                continue
            score = self.F[i] + O[p] if self.kappa != 0.0 else self.F[i]
            cond = score > self.F[i, p] + self.off
            cond[p] = False
            self.cnt[i] = np.where(cond, self.cnt[i] + 1, 0)
            trig = np.flatnonzero(self.cnt[i] >= self.n_ttt)
            if trig.size:
                g = int(trig[np.argmax(score[trig])])
                a[i, 0] = 1
                tgt[i] = g
            else:
                a[i, 0] = 0
        return a, tgt


def set_params(set_id):
    """A3Gen parameters equal to TR 36.839 set set_id in a3_rule (gate G1)."""
    ttt, off, k = A3.SETS[int(set_id)]
    return dict(n_ttt=A3.ttt_slots(ttt), off=off, k=k, source='measured', kappa=0.0)


# The tuning grid (plan section 2): n_ttt in slots of 0.1 s = 3GPP TTT 40, 160, 256, 480, 640 ms.
GRID_NTTT = [1, 2, 3, 5, 7]
GRID_OFF = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
GRID_K_MEASURED = [0, 1, 4]
GRID_K_MEAN = [0, 4]
GRID_KAPPA = [0.0, 2.0, 4.0, 8.0, 16.0]


def grid(source):
    ks = GRID_K_MEASURED if source == 'measured' else GRID_K_MEAN
    return [dict(n_ttt=n, off=o, k=k, source=source, kappa=0.0) for n in GRID_NTTT for o in GRID_OFF for k in ks]


def setting_id(p):
    return f"{p['source'][:4]}_n{p['n_ttt']}_o{p['off']:+.0f}_k{p['k']}_ka{p['kappa']:.0f}"
