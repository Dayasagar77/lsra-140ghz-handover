#!/usr/bin/env python3
"""realism_env.py -- the changed worlds of the LSRA realism run (30 Oct 2026).

Plan: PREREG_LSRA_Realism_ChannelBlockageBeamsHandoverFailures_ImperfectCoordination (the FROZEN copy is checked by
realism_check.py before every registered step).

RealismEnv is the delay-jitter environment (paper2ma_env_jitter.JitterEnv, i.e. the admission environment of every
earlier check) with switches that change the world. With every switch off it is that environment exactly (gate G0).
Each switch draws its random numbers from its own generator seeded by the episode seed, so the scene, the vehicles
and the base shadowing draws stay those of the paper (common random numbers). Class attributes are the defaults;
realism_check.configure() sets them on an instance.

  SHADOW = (d_clear, d_blocked) m   S1-S3: shadowing correlated along each vehicle's path (AR(1) in place on _z)
  FADING = True                     F1: small-scale fading on every vehicle-station link, independent per slot
  BLOCK = 'B1' | 'B2' | 'B3'        B1-B3: the blockage rule of every slot changed; frames rebuilt (rebuilt_frames)
  PATTERN = True                    A2: directional interference (3GPP parabolic receive pattern)
  MISALIGN = True                   A3: serving-beam misalignment between pointings
  H1 = True                         H1: a vehicle is unserved in the slot it lands
  H2 = (q_out_db, slots)            H2a/H2b: handover failure (too late, or target below q_out) and re-establishment
  DROP = p                          C2: each admitted move fails to start with probability p
  FAIL = q                          C3: each admitted move fails at landing with probability q (re-establishment)
  JITTER = J, JITTER_LATE           C1: landing spread (JitterEnv); JITTER_LATE draws d from {0..J} only
  set_beam_gain(g)                  A1a/A1b: the serving-beam gain of every module; set_beam_gain(None) restores 15 dBi

The planners are not changed and are not told about any switch: a vehicle being re-established is shown to them
as a vehicle in flight to the station that will serve it.
"""
import math
import os
import pickle
import numpy as np
import paper2ma_env as MA
import paper2ma_env_jitter as PJ
import blockage3d as B
import cv_forecast as CVF

DT = 0.1
FADE_SEED = 202610301
COORD_SEED = 202610302
K_LOS_DB = 9.0                     # F1: Rician K on a clear link (E10 variant V2); Rayleigh on a blocked link
THETA_3DB = 30.0                   # A2, A3: half-power beamwidth (deg), as in E10
MISALIGN_PERIOD = 5                # A3: slots between beam pointings (0.5 s)
MISALIGN_MAX_DB = 30.0             # A3: largest misalignment loss (E10)
B3_CAP_DB = 60.0                   # B3: cap of the summed loss (the paper's single-blocker cap is 40 dB)
B2_TALL_M = 2.5                    # B2: vehicles taller than this block over 1.3 x their length, others 0.75 x
B2_TALL_FACTOR = 1.3
B2_LOW_FACTOR = 0.75
C3_REEST_SLOTS = 2                 # C3: re-establishment after a failure inside a set, as H2a (0.2 s)
G_SERV0 = float(MA.G_SERV)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geom_cache_realism')


def set_beam_gain(g=None):
    """A1: the serving-beam gain read by every module at run time (paper2ma_env's global and MA.G_SERV are the same
    object). None restores the paper's 15 dBi."""
    MA.G_SERV = G_SERV0 if g is None else float(g)
    return MA.G_SERV


# ---------------------------------------------------------------- blockage rules (B1-B3)
def _blockage_loop(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz, near_miss, total, cap_db):
    D = math.hypot(gnb_xy[0] - ue_xy[0], gnb_xy[1] - ue_xy[1])
    lam = B.C_LIGHT / freq_hz
    best = dict(blocked=False, depth_db=0.0, blocker=None, dh=0.0, d1=0.0, d2=0.0, n_cross=0)
    if D < 1e-3:
        return best
    rising = h_gnb >= h_ue
    n_cross = 0
    acc = 0.0
    for k, b in enumerate(boxes):
        tr = B._seg_rect_trange(ue_xy, gnb_xy, b['x'], b['y'], b['length'], b['width'], b['theta'])
        if tr is None:
            continue
        n_cross += 1
        t0, t1 = tr
        t_eval = t0 if rising else t1
        t_eval = min(max(t_eval, 1e-3), 1.0 - 1e-3)
        h_ray = h_ue + t_eval * (h_gnb - h_ue)
        dh = float(b['height']) - h_ray
        if dh <= 0 and not near_miss:
            continue
        d1 = max(D * t_eval, 0.5)
        d2 = max(D * (1.0 - t_eval), 0.5)
        v = dh * math.sqrt(2.0 / lam * (1.0 / d1 + 1.0 / d2))
        if total:
            acc += B.knife_edge_loss_db(v)
            loss = B.knife_edge_loss_db(v)
        else:
            loss = min(B.knife_edge_loss_db(v), cap_db)
        if loss > best['depth_db']:
            best.update(depth_db=loss, blocker=k, dh=dh, d1=d1, d2=d2)
    if total:
        best['depth_db'] = min(acc, cap_db)
    best['n_cross'] = n_cross
    best['blocked'] = best['depth_db'] >= B.BLOCK_THR_DB
    return best


def link_blockage_b1(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz=B.FREQ_HZ, cap_db=B.CAP_DB_DEFAULT):
    """B1: blockage3d.link_blockage with the near-miss part of the knife-edge curve: a blocker whose roof is below the
    ray (dh <= 0) also gives the ITU-R P.526 loss, which is above 0 for v > -0.78. Otherwise line for line the same."""
    return _blockage_loop(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz, True, False, cap_db)


def link_blockage_b3(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz=B.FREQ_HZ, cap_db=B3_CAP_DB):
    """B3: the losses of all crossing blockers whose roof is above the ray, summed in dB (TR 38.901 blockage model B),
    then capped at 60 dB."""
    return _blockage_loop(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz, False, True, cap_db)


def b2_boxes(boxes):
    out = []
    for b in boxes:
        c = dict(b)
        c['length'] = float(b['length']) * (B2_TALL_FACTOR if float(b['height']) > B2_TALL_M else B2_LOW_FACTOR)
        out.append(c)
    return out


def link_blockage_b2(ue_xy, h_ue, gnb_xy, h_gnb, boxes, freq_hz=B.FREQ_HZ, cap_db=B.CAP_DB_DEFAULT):
    """B2: blockage persistence. Every blocker counts with its length x 1.3 if taller than 2.5 m, x 0.75 otherwise
    (fade-causing length, measured at 28 GHz), then the paper's rule."""
    return B.link_blockage(ue_xy, h_ue, gnb_xy, h_gnb, b2_boxes(boxes), freq_hz=freq_hz, cap_db=cap_db)


KERNELS = {None: B.link_blockage, 'B1': link_blockage_b1, 'B2': link_blockage_b2, 'B3': link_blockage_b3}


def frame_geometry_k(fr, agents, gnbs, k_bg, kernel):
    """cv_forecast.frame_geometry (a copy of paper2ma_env._precompute's per-slot code, shown equal to the stored slots
    of all 60 held-out scenes by gate G1d of the baselines run) with the blockage rule passed in."""
    NG = len(gnbs)
    present = {r['agent_id']: r for _, r in fr.iterrows()}
    all_boxes = fr[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
    ag = []
    for aid in agents:
        if aid not in present:
            ag.append(None)
            continue
        v = present[aid]
        boxes = [b for b in all_boxes if not (abs(b['x'] - v['x']) < 1e-9 and abs(b['y'] - v['y']) < 1e-9)]
        dist = np.zeros(NG)
        depth = np.zeros(NG)
        for gi, g in enumerate(gnbs):
            r = kernel((v['x'], v['y']), B.H_UE_DEFAULT, g[:2], g[2], boxes)
            dist[gi] = math.hypot(v['x'] - g[0], v['y'] - g[1])
            depth[gi] = r['depth_db']
        ag.append(dict(dist=dist, depth=depth))
    agent_xy = set((round(present[a]['x'], 3), round(present[a]['y'], 3)) for a in agents if a in present)
    statics = [b for b in all_boxes if (round(b['x'], 3), round(b['y'], 3)) not in agent_xy]
    bg = np.zeros(NG)
    for gi, g in enumerate(gnbs):
        ds = sorted(((math.hypot(b['x'] - g[0], b['y'] - g[1]), b) for b in statics), key=lambda z: z[0])
        acc = 0.0
        for d, b in ds[:k_bg]:
            rr = kernel((b['x'], b['y']), B.H_UE_DEFAULT, g[:2], g[2], [c for c in statics if c is not b])
            acc += 10 ** ((MA.INTERF_PT_DBM + MA.G_SIDE - MA.CH.ci_pathloss_db(max(d, 1.0), True)
                           - rr['depth_db']) / 10.0)
        bg[gi] = acc
    active = np.array([a is not None for a in ag], dtype=bool)
    return dict(ag=ag, bg=bg, active=active)


def cache_path(block, sid):
    return os.path.join(CACHE_DIR, str(block), f'{sid}.pkl')


def build_frames(env, sid, block):
    """Every slot of scene sid rebuilt with the blockage rule of `block`. The anticipation features and the speed,
    which the blockage rule does not touch, are copied from the stored slot."""
    obj = MA.MultiAgentRSUBlockageEnv._precompute(env, sid)
    g = CVF.SceneGeom(env, sid, obj['agents'])
    assert len(g.rows) == len(obj['frames']), 'slot count differs from the stored scene'
    frames = []
    for k, fr in enumerate(obj['frames']):
        nf = frame_geometry_k(g.rows[k], g.agents, g.gnbs, g.k_bg, KERNELS[block])
        assert np.array_equal(nf['active'], fr['active']), 'presence differs from the stored slot'
        for j, a in enumerate(nf['ag']):
            if a is not None:
                old = fr['ag'][j]
                for key in ('ttb', 'projd', 'ncorr', 'clos', 'speed'):
                    a[key] = old[key]
        frames.append(nf)
    return dict(agents=list(obj['agents']), frames=frames)


def rebuilt_frames(env, sid, block):
    path = cache_path(block, sid)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    obj = build_frames(env, sid, block)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.partial'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
    return obj


def scene_azimuths(env, sid, agents):
    """az[t, g, j]: the azimuth (rad) of vehicle j seen from station g in slot t (NaN if absent), from the same rows and
    station positions as the geometry."""
    g = CVF.SceneGeom(env, sid, agents)
    T, G = len(g.rows), len(g.gnbs)
    az = np.full((T, G, env.M), np.nan)
    for k, fr in enumerate(g.rows):
        present = {r['agent_id']: r for _, r in fr.iterrows()}
        for j, aid in enumerate(g.agents):
            if aid is None or aid not in present:
                continue
            v = present[aid]
            for gi, gg in enumerate(g.gnbs):
                az[k, gi, j] = math.atan2(float(v['y']) - gg[1], float(v['x']) - gg[0])
    return az


def angle_deg(a, b):
    """Absolute difference of two azimuths, in degrees, in [0, 180]."""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(d))


def pattern_gain_db(theta_deg):
    """A2: 3GPP parabolic receive pattern towards a co-channel vehicle theta degrees off the served vehicle's direction,
    floored at the paper's side-lobe level: G = G_serv - min(12 (theta/30)^2, G_serv - G_side)."""
    return MA.G_SERV - min(12.0 * (theta_deg / THETA_3DB) ** 2, MA.G_SERV - MA.G_SIDE)


def misalign_loss_db(dtheta_deg):
    """A3 (E10 variant V3): loss of a serving beam pointed dtheta degrees away from the vehicle."""
    return min(12.0 * (dtheta_deg / THETA_3DB) ** 2, MISALIGN_MAX_DB)


def shadow_filter(z, frames, d_clear, d_blocked):
    """S1-S3, in place: z[t, j, g] <- rho z[t-1, j, g] + sqrt(1 - rho^2) z[t, j, g], rho = exp(-v_t dt / d), d chosen by
    the link's state in slot t, v_t vehicle j's speed. A vehicle that (re)appears starts afresh; a stopped vehicle keeps
    its draw (rho = 1). The unit variance of every draw is kept."""
    T, M, G = z.shape
    for t in range(1, T):
        act_now, act_prev = frames[t]['active'], frames[t - 1]['active']
        for j in range(M):
            if not (act_now[j] and act_prev[j]):
                continue
            a = frames[t]['ag'][j]
            v = float(a['speed'])
            for g in range(G):
                d = d_blocked if a['depth'][g] > 0 else d_clear
                rho = math.exp(-v * DT / d)
                z[t, j, g] = rho * z[t - 1, j, g] + math.sqrt(max(0.0, 1.0 - rho * rho)) * z[t, j, g]
    return z


def fading_gains(frames, M, G, seed):
    """F1: |h|^2 per slot, vehicle and station, h = sqrt(K/(K+1)) + sqrt(1/(K+1)) w, w ~ CN(0, 1); K = 9 dB on a clear
    link, 0 (Rayleigh) on a blocked one. E|h|^2 = 1. Its own generator, seeded by the episode seed."""
    T = len(frames)
    r = np.random.default_rng([FADE_SEED, int(seed)])
    w = (r.normal(size=(T, M, G)) + 1j * r.normal(size=(T, M, G))) / math.sqrt(2.0)
    K = np.zeros((T, M, G))
    kl = 10.0 ** (K_LOS_DB / 10.0)
    for t in range(T):
        for j in range(M):
            a = frames[t]['ag'][j]
            if a is not None:
                K[t, j] = np.where(np.asarray(a['depth']) > 0, 0.0, kl)
    h = np.sqrt(K / (K + 1.0)) + np.sqrt(1.0 / (K + 1.0)) * w
    return np.abs(h) ** 2


class RealismEnv(PJ.JitterEnv):
    SHADOW = None
    FADING = False
    BLOCK = None
    PATTERN = False
    MISALIGN = False
    H1 = False
    H2 = None
    DROP = 0.0
    FAIL = 0.0
    JITTER_LATE = False
    DEBUG = False

    def __init__(self, *args, **kwargs):
        self.real_log = []
        self.real_events = []
        self._rcache = {}
        self._az_cache = {}
        self._h1_now = ()
        self._ep_ready = True
        self._pending_seed = None
        self._reest = None
        super().__init__(*args, **kwargs)

    def any_switch(self):
        return bool(self.SHADOW is not None or self.FADING or self.BLOCK is not None or self.PATTERN or self.MISALIGN
                    or self.H1 or self.H2 is not None or self.DROP > 0 or self.FAIL > 0 or self.JITTER_LATE)

    # ---------- per-episode state, made the first time the base reset needs it ----------
    def _precompute(self, sid):
        if self.BLOCK is None:
            return super()._precompute(sid)
        key = (self.BLOCK, sid)
        if key not in self._rcache:
            self._rcache[key] = rebuilt_frames(self, sid, self.BLOCK)
        return self._rcache[key]

    def _ensure_episode(self):
        if self._ep_ready:
            return
        self._ep_ready = True
        seed = self._pending_seed
        if self.any_switch():
            assert seed is not None, 'the realism switches need a seeded reset (common random numbers)'
        F, M, G = len(self.frames), self.M, self.n_gnb
        if self.SHADOW is not None:
            shadow_filter(self._z, self.frames, float(self.SHADOW[0]), float(self.SHADOW[1]))
        self._fade = fading_gains(self.frames, M, G, seed) if self.FADING else None
        if self.PATTERN or self.MISALIGN:
            if self.scene_id not in self._az_cache:
                self._az_cache[self.scene_id] = scene_azimuths(self, self.scene_id, self._precompute(self.scene_id)['agents'])
            self._az = self._az_cache[self.scene_id]
        self._beam_serv = np.full(M, -1, dtype=int)
        self._beam_t0 = np.zeros(M, dtype=int)
        self._beam_last = np.full(M, -10, dtype=int)
        self._reest = np.zeros(M, dtype=bool)
        self._will_fail = np.zeros(M, dtype=bool)
        if self.DROP > 0 or self.FAIL > 0:
            r = np.random.default_rng([COORD_SEED, int(seed)])
            self._u_drop = r.random((F, M))
            self._u_fail = r.random((F, M))

    def reset(self, *, seed=None, options=None):
        self._ep_ready = False
        self._pending_seed = seed
        self._h1_now = ()
        out = super().reset(seed=seed, options=options)
        self._ensure_episode()
        if self.JITTER_LATE and int(self.JITTER) > 0:
            r = np.random.default_rng([PJ.JIT_SEED, int(seed)])
            self._D = r.integers(0, int(self.JITTER) + 1, size=(len(self.frames), self.M))
        self.real_log.append(dict(h1_unserved=0, h2_too_late=0, h2_target=0, reest=0, dropped=0, c3_failed=0))
        return out

    # ---------- the channel (F1, A2, A3) ----------
    def _misalign_db(self, i, g, t):
        if self._beam_serv[i] != g or self._beam_last[i] < t - 1:     # a new station, or service resumed: re-pointed
            self._beam_serv[i] = g
            self._beam_t0[i] = t
        self._beam_last[i] = t
        tp = max(int(self._beam_t0[i]), t - (t % MISALIGN_PERIOD))
        return misalign_loss_db(angle_deg(self._az[t, g, i], self._az[tp, g, i]))

    def _sinr_db(self, i):
        self._ensure_episode()
        if not (self.FADING or self.PATTERN or self.MISALIGN):
            return super()._sinr_db(i)
        fr = self.frames[self.t]
        a = fr['ag'][i]
        if a is None:
            return MA.SINR_OUT_DB - 30.0
        t = self.t
        g = int(self.serv[i])
        lvl = (self.pt[i] + MA.G_SERV - MA.CH.ci_pathloss_db(max(a['dist'][g], 1.0), True)
               - self._z[t, i, g] * MA._sigma(a['depth'][g]) - a['depth'][g])
        if self.MISALIGN:
            lvl -= self._misalign_db(i, g, t)
        p_des = 10 ** (lvl / 10.0)
        if self.FADING:
            p_des *= self._fade[t, i, g]
        inti = float(fr['bg'][g])
        for j in range(self.M):
            if j == i or not fr['active'][j] or self.sb[j] != self.sb[i]:
                continue
            if not self._admitted[j]:
                continue
            aj = fr['ag'][j]
            gain = MA.G_SIDE
            if self.PATTERN:
                gain = pattern_gain_db(angle_deg(self._az[t, g, j], self._az[t, g, i]))
            pj = 10 ** ((self.pt[j] + gain - MA.CH.ci_pathloss_db(max(aj['dist'][g], 1.0), True)
                         - self._z[t, j, g] * MA._sigma(aj['depth'][g]) - aj['depth'][g]) / 10.0)
            if self.FADING:
                pj *= self._fade[t, j, g]
            inti += pj
        return 10 * math.log10(p_des / (MA.N0_MW + inti))

    # ---------- who is served (H1, H2, C3) ----------
    def _compute_admission(self):
        self._ensure_episode()
        adm = super()._compute_admission()
        fr = self.frames[self.t]
        unserved = set(i for i in self._h1_now if fr['active'][i])
        if self._reest is not None:
            unserved |= set(int(i) for i in np.flatnonzero(self._reest & (self._ho_timer > 0)) if fr['active'][i])
        if not unserved:
            return adm
        adm = np.asarray(fr['active'], dtype=bool).copy()          # the base scheduling rule, served vehicles only
        for i in unserved:
            adm[i] = False
        if self.gnb_cap is not None:
            for g in range(self.n_gnb):
                on_g = [i for i in range(self.M) if fr['active'][i] and self.serv[i] == g and i not in unserved]
                if len(on_g) <= self.gnb_cap:
                    continue
                ranked = sorted(on_g, key=lambda i: self._desired_dbm(i, g), reverse=True)
                for i in ranked[self.gnb_cap:]:
                    adm[i] = False
        return adm

    def _start_reest(self, i, slots, kind):
        """Re-establishment: unserved for `slots` slots, then served by the strongest station with a place in the landing
        state (its own reserved place counts). Shown to the planners as a flight to that station."""
        fr = self.frames[self.t]
        cap = int(self.gnb_cap)
        occ = np.zeros(self.n_gnb, dtype=int)
        for j in range(self.M):
            if j != i and fr['active'][j]:
                occ[int(self._ho_target[j]) if self._ho_timer[j] > 0 else int(self.serv[j])] += 1
        order = sorted(range(self.n_gnb), key=lambda g: (-self._desired_dbm(i, g), g))
        g = next(g for g in order if occ[g] < cap)
        self._ho_target[i] = g
        self._ho_timer[i] = int(slots) + 1            # unserved in the next `slots` slots, served from the one after
        self._delay[i] = int(slots) + 1
        self._flown[i] = 0
        self._reest[i] = True
        self._will_fail[i] = False
        rec = self.real_log[-1]
        rec['reest'] += 1
        if self.DEBUG:
            self.real_events.append(('reest', kind, int(i), int(self.t), int(g), [int(x) for x in occ], int(slots)))

    def step(self, action, ho_targets=None):
        self._ensure_episode()
        t0 = self.t
        act0 = np.asarray(self.frames[t0]['active'], dtype=bool).copy()
        landing = [i for i in range(self.M) if act0[i] and self._ho_timer[i] == 1]
        normal_landing = [i for i in landing if not self._reest[i]]
        reest_landing = [i for i in landing if self._reest[i]]
        self._h1_now = tuple(normal_landing) if self.H1 else ()
        out = super().step(action, ho_targets)
        h1 = self._h1_now
        self._h1_now = ()
        obs, rew, term, done, info = out
        rec = self.real_log[-1]
        act = np.asarray(self.frames[self.t]['active'], dtype=bool)
        rec['h1_unserved'] += sum(1 for i in h1 if act[i])
        for i in reest_landing:
            self._reest[i] = False
            if self.DEBUG:
                self.real_events.append(('reest_landed', int(i), int(self.t), int(self.serv[i])))
        if self.DROP > 0:                                     # C2: admitted moves that fail to start
            for i in np.flatnonzero(info['ho']):
                if self._u_drop[t0, i] < self.DROP:
                    self._ho_timer[i] = 0
                    self._ho_target[i] = self.serv[i]
                    info['ho'][i] = False
                    rec['dropped'] += 1
        if self.FAIL > 0:                                     # C3: decided when the move starts
            for i in np.flatnonzero(info['ho']):
                self._will_fail[i] = bool(self._u_fail[t0, i] < self.FAIL)
        for i in normal_landing:                              # landings of ordinary handovers
            if not act[i]:
                self._will_fail[i] = False
                continue
            if self.FAIL > 0 and self._will_fail[i]:
                rec['c3_failed'] += 1
                self._start_reest(i, self.H2[1] if self.H2 is not None else C3_REEST_SLOTS, 'c3')
            elif self.H2 is not None and info['admitted'][i] and self._sinr_db(i) < float(self.H2[0]):
                rec['h2_target'] += 1
                self._start_reest(i, self.H2[1], 'target')
            self._will_fail[i] = False
        if self.H2 is not None:                               # too late: RLF on the source while in flight
            for i in range(self.M):
                if act[i] and self._ho_timer[i] > 0 and not self._reest[i] and info['rlf'][i]:
                    rec['h2_too_late'] += 1
                    self._start_reest(i, self.H2[1], 'too_late')
        idle = self._ho_timer == 0                            # a vehicle not in flight (landed, or re-attached after
        self._reest[idle] = False                             # leaving the scene) carries no failure state
        self._will_fail[idle] = False
        return obs, rew, term, done, info
