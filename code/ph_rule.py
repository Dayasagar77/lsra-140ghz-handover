#!/usr/bin/env python3
"""ph_rule.py -- the prediction-based proactive handover of Na et al. (IEEE TVT 2026), adapted to our slots
(28 Oct 2026).

Plan: PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation (28 Oct 2026).

Source: G. Na et al., "Prediction-Based Proactive Handover for Reliable Connectivity in mmWave and Terahertz
Vehicular Networks", IEEE TVT 75(9):20880-20895, 2026, doi 10.1109/TVT.2026.3681181, Section IV-C and
Algorithm 1 (read in full on 28 Oct 2026). Their rule looks at the forecast received power (RSRP) of every
cell over a window W of T_window = 0.5 s and decides:
  Case 1 (serving cell forecast above tau in all of W): among the other cells that stay above tau in all of W,
    whose window average beats the serving cell's by Delta_avg = 3 dB and whose power at the window start beats
    the serving cell's by Delta_init, and if the vehicle has stayed T_min = 0.5 s, hand over to the cell with
    the highest window average.
  Case 2 (serving cell forecast below tau somewhere in W): hand over to the cell with the highest window average
    among those above tau in all of W; if there is none, to the cell that stays above tau longest from the
    window start; in both branches only if its power at the window start beats the serving cell's by Delta_init.
Our adaptation: a handover decided in slot t lands in slot t+L+1, so the window is W = t+L+1 ... t+L+T_w
(T_w = 5 slots = 0.5 s) and a decision is evaluated once per slot. tau = -92 dBm (the paper's outage threshold
at the noise floor). Delta_init = 0 dB (their paper gives no number). The forecast RSRP of station g is the mean
received power p + G_s - PL - blockage loss in each window slot, from
  forecast = 'oracle': the true future slots (the stored geometry; past the last slot, the last slot);
  forecast = 'cv':     the straight-line forecast of cv_forecast.SceneGeom.cv_window (learned-forecast study).
room = True (PH-OR) keeps only stations with room when the vehicle lands, counting the vehicles on their way
and the decisions made earlier in the same slot, in vehicle order (as R1i).
"""
import numpy as np
import paper2ma_env as MA

TAU_DBM = -92.0
D_AVG = 3.0
D_INIT = 0.0
T_WIN = 5
T_MIN = 5


def rsrp_frame(env, fr):
    """Mean received power (dBm), shape (M, G); -inf for vehicles absent from the frame."""
    M, G = env.M, env.n_gnb
    out = np.full((M, G), -np.inf)
    for j in range(M):
        a = fr['ag'][j]
        if a is None:
            continue
        dist = np.maximum(np.asarray(a['dist'], dtype=float), 1.0)
        depth = np.asarray(a['depth'], dtype=float)
        out[j] = float(env.pt[j]) + MA.G_SERV - MA.CH.ci_pathloss_db(dist, True) - depth
    return out


def oracle_window(env, t_win=T_WIN):
    t, L, F = env.t, int(env.ho_lat), len(env.frames)
    return [env.frames[min(k, F - 1)] for k in range(t + L + 1, t + L + 1 + t_win)]


_CV_CACHE = {}


def cv_window(env, t_win=T_WIN):
    """Straight-line forecast of the window. The forecast depends only on the scene, the slot and L (every seed of
    a scene replays the same trajectories), so it is cached per (scene, t, L, t_win)."""
    import cv_forecast as CVF
    L = int(env.ho_lat)
    key = (env.scene_id, env.t, L, int(t_win))
    if key not in _CV_CACHE:
        _CV_CACHE[key] = CVF.scene_geom(env).cv_window(env.t, L, ks=list(range(L + 1, L + 1 + t_win)))
    return _CV_CACHE[key]


def decide(R, s, tau=TAU_DBM, d_avg=D_AVG, d_init=D_INIT, dwell_ok=True, allowed=None):
    """One vehicle. R: (G, T) forecast powers over the window; s: serving station; allowed: stations that may be
    chosen (None = all). Returns the requested station or -1."""
    G = R.shape[0]
    cands = [c for c in range(G) if c != s and (allowed is None or c in allowed)]
    if not cands:
        return -1
    above = R >= tau
    avg = R.mean(axis=1)
    if above[s].all():                                         # Case 1
        E = [c for c in cands if above[c].all() and avg[c] > avg[s] + d_avg and R[c, 0] > R[s, 0] + d_init]
        if E and dwell_ok:
            return int(max(E, key=lambda c: (avg[c], -c)))
        return -1
    Rel = [c for c in cands if above[c].all()]                 # Case 2
    if Rel:
        c = int(max(Rel, key=lambda c: (avg[c], -c)))
    else:
        def prefix(c):
            n = 0
            while n < R.shape[1] and above[c, n]:
                n += 1
            return n
        c = int(max(cands, key=lambda c: (prefix(c), avg[c], -c)))
    return c if R[c, 0] > R[s, 0] + d_init else -1


class PHRule:
    """Na et al.'s rule as a policy. forecast in ('oracle', 'cv'); room adds the landing-room check."""

    def __init__(self, forecast='oracle', room=False, tau=TAU_DBM, d_avg=D_AVG, d_init=D_INIT,
                 t_win=T_WIN, t_min=T_MIN):
        if forecast not in ('oracle', 'cv'):
            raise ValueError(forecast)
        self.forecast, self.room = forecast, bool(room)
        self.tau, self.d_avg, self.d_init, self.t_win, self.t_min = tau, d_avg, d_init, int(t_win), int(t_min)
        self._z = None

    def _new_episode(self, env):
        self.since = np.zeros(env.M, dtype=int)      # slots since the vehicle was attached or last landed
        self.prev_serv = np.full(env.M, -1, dtype=int)
        self.prev_act = np.zeros(env.M, dtype=bool)
        self.last_t = -1
        self._z = env._z

    def window(self, env):
        frs = oracle_window(env, self.t_win) if self.forecast == 'oracle' else cv_window(env, self.t_win)
        return np.stack([rsrp_frame(env, f) for f in frs], axis=2)       # (M, G, T)

    def __call__(self, env, obs, sinr, rng, ag, temp):
        if env.t == 0:
            self._new_episode(env)
        if self._z is not env._z or env.t != self.last_t + 1:
            raise RuntimeError('PHRule: slots of one episode must be given in order, starting at t = 0')
        self.last_t = env.t
        M, G, cap = env.M, env.n_gnb, int(env.gnb_cap)
        fr = env.frames[env.t]
        act = np.asarray(fr['active'], dtype=bool)
        # dwell: restart on attachment (appearance) and on a landing (serving station changed)
        for i in range(M):
            if not act[i]:
                continue
            if (not self.prev_act[i]) or int(env.serv[i]) != int(self.prev_serv[i]):
                self.since[i] = 0
            else:
                self.since[i] += 1
        self.prev_act = act.copy()
        self.prev_serv = np.asarray(env.serv, dtype=int).copy()
        a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1
        tgt = -np.ones(M, int)
        free = [i for i in range(M) if act[i] and env._ho_timer[i] == 0]
        if not free:
            a[:, 0] = 0
            return a, tgt
        W = self.window(env)
        occ = np.zeros(G, dtype=int)                  # landing load, updated decision by decision (as R1i)
        for j in range(M):
            if act[j]:
                occ[int(env._ho_target[j]) if env._ho_timer[j] > 0 else int(env.serv[j])] += 1
        for i in range(M):
            if i not in free:
                a[i, 0] = 0
                continue
            R = W[i]
            if not np.isfinite(R).all():
                a[i, 0] = 0
                continue
            s = int(env.serv[i])
            allowed = [g for g in range(G) if occ[g] < cap] if self.room else None
            c = decide(R, s, self.tau, self.d_avg, self.d_init, dwell_ok=self.since[i] >= self.t_min,
                       allowed=allowed)
            if c < 0:
                a[i, 0] = 0
                continue
            a[i, 0] = 1
            tgt[i] = c
            occ[c] += 1
            occ[s] -= 1
        return a, tgt
