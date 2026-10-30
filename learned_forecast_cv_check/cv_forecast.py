#!/usr/bin/env python3
"""Constant-velocity look-ahead for LSRA: the first cheap check of the learned-forecast paper (27 Oct 2026).

Plan: PREREG_LF_ConstantVelocityLookahead_DesignScenes (section 1). Nothing in Paper 2's code is changed.
This module imports Paper 2's environment constants (paper2ma_env), its blockage kernel (blockage3d) and
LSRA's own risk model and solver (lsra.risk_frame, lsra.bottleneck_then_mincost).

Forecast 'cv' at decision slot t, for the window tau = t+L+1 ... t+2L+1 (H = L+1 slots):
  * every vehicle present in slot t (controlled or not) moves straight on at its velocity
        v = (p(t) - p(t-m)) / (m * DT),  m = min(5, earlier consecutive slots in which it was present),
    with v = 0 when m = 0. Only slots up to t are read;
  * its box keeps its heading and size; vehicles absent in slot t are absent from every forecast slot;
  * the geometry of each forecast slot (distances, knife-edge depths, background interference) comes from
    frame_geometry(), a copy of the per-slot code of paper2ma_env.MultiAgentRSUBlockageEnv._precompute
    (paper2ma_env.py md5 42c9dba3..., lines 144-185) without the anticipation features, which LSRA does not
    read. One deliberate difference: the background interferers are sorted by distance with the distance as
    the key, so an exact tie cannot raise (the original compares the box dicts on a tie and would raise
    TypeError); with no exact tie the order is the same. Gate G2a checks frame_geometry against the
    environment's own stored slots, bit for bit.

make_lsra_fc() is lsra.make_lsra(forecast, landing=True, minmax=False, beta) with only the source of the
window's slots replaced (frames_fn). Gates G1 and G2b check it against lsra's own 'now' and 'oracle' arms.
"""
import math
import numpy as np
import paper2ma_env as MA
import blockage3d as B
import lsra as LS

DT = 0.1          # slot length (s); every design scene has 100 slots exactly 0.1 s apart
M_VEL = 5         # at most 5 earlier slots (0.5 s) for the velocity; fixed in the plan, not tuned


# ------------------------------------------------------------------ geometry of one slot
def frame_geometry(fr, agents, gnbs, k_bg):
    """One slot's geometry, as paper2ma_env._precompute builds it (without the anticipation features).
    fr: the slot's vehicle rows (DataFrame with agent_id, x, y, length, width, height, theta).
    agents: the scene's controlled vehicles (ids, None for padding); gnbs: [(x, y, h)]; k_bg: interferers."""
    NG = len(gnbs)
    present = {r['agent_id']: r for _, r in fr.iterrows()}
    all_boxes = fr[['x', 'y', 'length', 'width', 'height', 'theta']].to_dict('records')
    ag = []
    for aid in agents:
        if aid not in present:
            ag.append(None)
            continue
        v = present[aid]
        boxes = [b for b in all_boxes
                 if not (abs(b['x'] - v['x']) < 1e-9 and abs(b['y'] - v['y']) < 1e-9)]
        dist = np.zeros(NG)
        depth = np.zeros(NG)
        for gi, g in enumerate(gnbs):
            r = B.link_blockage((v['x'], v['y']), B.H_UE_DEFAULT, g[:2], g[2], boxes)
            dist[gi] = math.hypot(v['x'] - g[0], v['y'] - g[1])
            depth[gi] = r['depth_db']
        ag.append(dict(dist=dist, depth=depth))
    agent_xy = set((round(present[a]['x'], 3), round(present[a]['y'], 3))
                   for a in agents if a in present)
    statics = [b for b in all_boxes if (round(b['x'], 3), round(b['y'], 3)) not in agent_xy]
    bg = np.zeros(NG)
    for gi, g in enumerate(gnbs):
        ds = sorted(((math.hypot(b['x'] - g[0], b['y'] - g[1]), b) for b in statics), key=lambda z: z[0])
        acc = 0.0
        for d, b in ds[:k_bg]:
            rr = B.link_blockage((b['x'], b['y']), B.H_UE_DEFAULT, g[:2], g[2],
                                 [c for c in statics if c is not b])
            acc += 10 ** ((MA.INTERF_PT_DBM + MA.G_SIDE - MA.CH.ci_pathloss_db(max(d, 1.0), True)
                           - rr['depth_db']) / 10.0)
        bg[gi] = acc
    active = np.array([a is not None for a in ag], dtype=bool)
    return dict(ag=ag, bg=bg, active=active)


# ------------------------------------------------------------------ constant-velocity extrapolation
def velocities(pos, t, cur_ids, cur_x, cur_y, m_max=M_VEL):
    """Velocity of every vehicle of slot t from its own positions in slots t-m ... t only.
    pos: list of dicts {agent_id: (x, y)}, one per slot; only pos[0..t] is read.
    Returns vx, vy (m/s) and m (number of earlier slots used) per vehicle of slot t."""
    n = len(cur_ids)
    vx = np.zeros(n)
    vy = np.zeros(n)
    mm = np.zeros(n, dtype=int)
    for j in range(n):
        aid = cur_ids[j]
        m = 0
        prev = None
        for k in range(1, m_max + 1):
            if t - k < 0:
                break
            p = pos[t - k].get(aid)
            if p is None:
                break
            m = k
            prev = p
        if m > 0:
            vx[j] = (float(cur_x[j]) - float(prev[0])) / (m * DT)
            vy[j] = (float(cur_y[j]) - float(prev[1])) / (m * DT)
        mm[j] = m
    return vx, vy, mm


def moved_rows(cur, vx, vy, k):
    """The rows of slot t with every box moved k slots ahead at its velocity; nothing else changes."""
    out = cur.copy()
    out['x'] = cur['x'].to_numpy(dtype=float) + vx * (k * DT)
    out['y'] = cur['y'].to_numpy(dtype=float) + vy * (k * DT)
    return out


# ------------------------------------------------------------------ per-scene data
class SceneGeom:
    """The recorded rows of one scene, in the environment's slot order, with what frame_geometry needs.
    agent_strs: the stored scene's controlled vehicles as text (the 'agents' entry of the geometry cache),
    so the vehicle order is exactly the one of the stored slots."""

    def __init__(self, env, sid, agent_strs):
        sdf = env.pool[sid]
        self.sid = sid
        ids = {str(a): a for a in sdf['agent_id'].unique()}
        self.agents = [None if s == 'None' else ids[s] for s in agent_strs]
        self.gnbs = [B.scene_gnb(sdf, B.H_GNB_DEFAULT, mode=m) for m in env.gnb_modes]
        self.k_bg = int(env.k_bg)
        self.rows = [fr for _, fr in sdf.groupby('timestamp')]
        self.pos = [dict(zip(fr['agent_id'].tolist(), zip(fr['x'].tolist(), fr['y'].tolist())))
                    for fr in self.rows]
        self._rec = {}

    def recorded(self, k):
        if k not in self._rec:
            self._rec[k] = frame_geometry(self.rows[k], self.agents, self.gnbs, self.k_bg)
        return self._rec[k]

    def cv_window(self, t, L, zero=False, ks=None):
        cur = self.rows[t]
        ids = cur['agent_id'].tolist()
        vx, vy, _ = velocities(self.pos, t, ids, cur['x'].tolist(), cur['y'].tolist())
        if zero:
            vx = np.zeros_like(vx)
            vy = np.zeros_like(vy)
        if ks is None:
            ks = range(L + 1, 2 * L + 2)
        return [frame_geometry(moved_rows(cur, vx, vy, k), self.agents, self.gnbs, self.k_bg) for k in ks]


_SCENES = {}
_CV = {}


def scene_geom(env):
    sid = env.scene_id
    if sid not in _SCENES:
        g = SceneGeom(env, sid, env.agent_ids)
        assert len(g.rows) == len(env.frames), 'slot count differs from the stored scene'
        _SCENES[sid] = g
    return _SCENES[sid]


# ------------------------------------------------------------------ window providers
def provider_cv(env):
    """The constant-velocity forecast of the window [t+L+1, t+2L+1] (the arm CV)."""
    t, L = env.t, int(env.ho_lat)
    key = (env.scene_id, t, L)
    if key not in _CV:
        _CV[key] = scene_geom(env).cv_window(t, L)
    return _CV[key]


def provider_zero_now(env):
    """Gate G1: the CV path with every velocity set to 0 and the window reduced to slot t."""
    return scene_geom(env).cv_window(env.t, int(env.ho_lat), zero=True, ks=[0])


def provider_recorded_oracle(env):
    """Gate G2b: the recorded slots of the oracle's window, rebuilt by frame_geometry,
    with lsra.window_risk's end-of-episode rule."""
    t, L, F = env.t, int(env.ho_lat), len(env.frames)
    lo, hi = t + L + 1, min(t + 2 * L + 1, F - 1)
    ks = list(range(lo, hi + 1)) if lo <= hi else [F - 1]
    g = scene_geom(env)
    return [g.recorded(k) for k in ks]


# ------------------------------------------------------------------ LSRA with a given window
def make_lsra_fc(frames_fn, beta=0.05):
    """lsra.make_lsra(forecast, landing=True, minmax=False, beta) with the window's slots from frames_fn(env).
    The lines below follow lsra.make_lsra's policy and lsra.window_risk one for one."""
    def pol(env, obs, sinr, rng, ag, temp):
        M, G, cap = env.M, env.n_gnb, int(env.gnb_cap)
        fr = env.frames[env.t]
        act = fr['active']
        a = np.ones((M, 3), int)
        a[:, 1] = 2
        a[:, 2] = 1
        tgt = -np.ones(M, int)
        pinned = [i for i in range(M) if act[i] and env._ho_timer[i] > 0]
        free = [i for i in range(M) if act[i] and env._ho_timer[i] == 0]
        frs = frames_fn(env)
        risk = np.zeros((env.M, env.n_gnb))
        for f in frs:
            risk += LS.risk_frame(env, f)
        risk = risk / len(frs)
        q = np.full(G, cap, dtype=int)
        floor = 0.0
        for i in pinned:
            g = int(env._ho_target[i])
            q[g] -= 1
            floor = max(floor, float(risk[i, g]))
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
    return pol
