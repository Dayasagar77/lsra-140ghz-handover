#!/usr/bin/env python3
"""realism_check.py -- the realism run for the LSRA revision (30 Oct 2026).

Plan: PREREG_LSRA_Realism_ChannelBlockageBeamsHandoverFailures_ImperfectCoordination (the FROZEN copy in this folder is
checked by md5 before every registered step). Daya approved building and testing (30 Oct 2026: 4 places only, fading
added as one more world); the freeze and the run need his OK after the gates.

The worlds and settings are switches of realism_env.RealismEnv. Every planner keeps its settings: LSRA (beta 0.05),
R1i, R2, R3, A3 Set 5, the tuned A3 at its frozen setting (models/baselines_check/TUNED.json) and Na et al.'s rule with
the perfect forecast. The paper's world (V0) is not run again: its cells are the stored episodes of the earlier checks
and of the new-baselines run, paired by seed and scene.

Modes, in this order, each launched by its own .cmd through WMI on the laptop (Windows Python 3.11.9):
  unit                  hand-made tests, no scenes                                             -> UNIT.json
  prep                  the B1-B3 frames of the 60 held-out scenes, built once into geom_cache_realism -> PREP.json
  gate                  identity, geometry, mechanics, every-cell runs and timing; no world's LSRA is compared
                        with a baseline                                                         -> GATE.json
  dry                   the registered analysis on stored V0 cells relabelled as every world and setting;
                        every number that can be checked must equal the earlier results         -> DRY.json
  freeze                md5 of every file the run reads (code, data, stored episodes, frames, plan, records)
  pipeline --worker k   the 163 cells, one job each, taken through claim files; then the analysis, once
Outputs: models/realism_check/.
"""
import os, sys, io, json, time, argparse, hashlib, platform, glob

HERE = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'LEARNED_FORECAST_PAPER', 'cv_check'))
sys.path.insert(0, HERE)
sys.path.append(CV_DIR)
import numpy as np
import pandas as pd
import scipy
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import load_check_c6 as LC6
import a3_check as A3C
import paper2ma_env as MA
import paper2ma_env_jitter as PJ
import paper2ma_confirm as CF
import a3_rule as A3
import a3_general as AG
import ph_rule as PH
import lsra as LS
import cv_forecast as CVF
import baselines_check as BC
import realism_env as RE

OUT_REAL = os.path.join(HERE, 'models', 'realism_check')
OUT_DRY = os.path.join(HERE, 'models', '_dry_rc')
OUT = OUT_REAL
PREREG = 'PREREG_LSRA_Realism_ChannelBlockageBeamsHandoverFailures_ImperfectCoordination_FROZEN_20261030.md'
ADM = {'A0': 'none', 'A1': 'per_request', 'A2': 'coordinated'}
HL = 10
CAP = 4
N_EP = 480
N_GATE = 40
BETA = 0.05
BOOT_SEED0 = 14000
NO_PROGRESS_STOP_S = 3600
POLL_S = 30
TUNED_PATH = os.path.join(HERE, 'models', 'baselines_check', 'TUNED.json')
TUNED_MD5 = '7af6777289b7aeff008ec408f8b38207'
Q_OUT_A = float(MA.SINR_OUT_DB)

WORLDS = {
    'S1': dict(SHADOW=(3.8, 3.8)),
    'S2': dict(SHADOW=(10.0, 13.0)),
    'S3': dict(SHADOW=(50.0, 50.0)),
    'B1': dict(BLOCK='B1'),
    'B2': dict(BLOCK='B2'),
    'B3': dict(BLOCK='B3'),
    'A1a': dict(GAIN=10.0),
    'A1b': dict(GAIN=20.0),
    'A2': dict(PATTERN=True),
    'A3': dict(MISALIGN=True),
    'F1': dict(FADING=True),
    'H1': dict(H1=True),
    'H2a': dict(H2=(Q_OUT_A, 2)),
    'H2b': dict(H2=(-8.0, 5)),
    'ALL': dict(SHADOW=(10.0, 13.0), BLOCK='B2', PATTERN=True, H2=(Q_OUT_A, 2)),
}
COORD = {
    'C1a': dict(JITTER=5),
    'C1b': dict(JITTER=5, JITTER_LATE=True),
    'C2a': dict(DROP=0.05),
    'C2b': dict(DROP=0.20),
    'C3a': dict(FAIL=0.02),
    'C3b': dict(FAIL=0.05),
    'C3c': dict(FAIL=0.10),
    'C4a': dict(HL=11),
    'C4b': dict(HL=12),
}
SEQUENCES = {'C1': ['C1a', 'C1b'], 'C2': ['C2a', 'C2b'], 'C3': ['C3a', 'C3b', 'C3c'], 'C4': ['C4a', 'C4b', 'C4c']}
ARMS = [('LSRA', 'A1'), ('LSRA', 'A2'), ('R2', 'A1'), ('R2', 'A2'), ('A3T', 'A1'), ('A3T', 'A2'),
        ('R1i', 'A2'), ('R3', 'A2'), ('A3S5', 'A2'), ('PHO', 'A2')]
RW1_BASE = ['R1i', 'R2', 'R3', 'A3S5', 'A3T', 'PHO']
COORD_ARMS = [(c, 'LSRA') for c in COORD] + [('C2b', 'R3'), ('C2b', 'A3T'), ('C3c', 'R3'), ('C3c', 'A3T')]
B_BLOCKS = ['B1', 'B2', 'B3']


# ---------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_md5():
    return {f: md5(os.path.join(HERE, f)) for f in ('realism_check.py', 'realism_env.py')}


def status(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)


def write_atomic(rows, path):
    tmp = f'{path}.{os.getpid()}.partial'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


_POOL = {}
_SEEDS = {}


def pool():
    if 'h' not in _POOL:
        _POOL['h'] = pd.read_csv(MHO.POOL_H)
    return _POOL['h']


def seeds_test():
    if 't' not in _SEEDS:
        _SEEDS['t'] = AC.seeds_all()
    return _SEEDS['t']


def tuned():
    assert md5(TUNED_PATH) == TUNED_MD5, 'TUNED.json of the new-baselines run changed'
    return json.load(open(TUNED_PATH))


def policy(arm, A):
    """A fresh policy object (A3T and PHO hold per-episode state)."""
    if arm == 'LSRA':
        return LS.make_lsra('now', landing=True, minmax=False, beta=BETA)
    if arm == 'R1i':
        return LS.pol_r1i
    if arm in ('R2', 'R3'):
        return dict(MHO.arms(BETA))[arm]
    if arm == 'A3S5':
        return A3.A3Rule(5)
    if arm == 'A3T':
        return AG.A3Gen(**tuned()[f'C4_{A}']['A3T'])
    if arm == 'PHO':
        return PH.PHRule('oracle', room=False)
    raise ValueError(arm)


def config(world=None, coord=None):
    cfg = dict(WORLDS[world]) if world else {}
    if coord:
        cfg.update(COORD[coord])
    return cfg


def make_env(world=None, coord=None, A='A2'):
    """The admission environment of every earlier check (load_check_c6.make_env_cap, 4 places), as RealismEnv, with
    the world's or setting's switches. Returns the environment and the beam gain to set (None: the paper's)."""
    cfg = config(world, coord)
    hl = int(cfg.pop('HL', HL))
    gain = cfg.pop('GAIN', None)
    env = RE.RealismEnv(pool(), n_agents=16, n_subbands=3, gnb_cap=CAP, seed=0, split=None,
                        compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = ADM[A]
    for k, v in cfg.items():
        if not hasattr(RE.RealismEnv, k) and k != 'JITTER':
            raise KeyError(k)
        setattr(env, k, v)
    return env, gain


def run_real(env, fn, seeds):
    """baselines_check.run (the quick-return count; admission_check.run_cell) with the realism and jitter records."""
    env.real_log = []
    env.jit_log = []
    rows = BC.run(env, fn, seeds)
    assert len(env.real_log) == len(rows) == len(env.jit_log) == len(seeds), 'one record per episode'
    for r, a, j in zip(rows, env.real_log, env.jit_log):
        r.update(a)
        r.update(jit_starts=j['starts'], jit_min=j['min_delay'], jit_max=j['max_delay'],
                 excess_car_steps=j['excess_car_steps'], flight_mismatch=j['flight_mismatch'])
    return rows


def run_job(world, coord, arm, A, seeds):
    env, gain = make_env(world, coord, A)
    RE.set_beam_gain(gain)
    try:
        rows = run_real(env, policy(arm, A), seeds)
    finally:
        RE.set_beam_gain(None)
    for x in rows:
        x.update(world=world or 'V0', coord=coord or '', arm=arm, adm=A, hl=int(env.ho_lat), cap=CAP)
    return rows


def world_cells():
    return [(w, None, arm, A) for w in WORLDS for arm, A in ARMS]


def coord_cells():
    return [(None, c, arm, 'A2') for c, arm in COORD_ARMS]


def cell_name(w, c, arm, A):
    return f'{w or c}_{arm}_{A}'


def cell_path(w, c, arm, A):
    d = os.path.join(OUT, 'cells')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, cell_name(w, c, arm, A) + '.csv')


def stored_v0(arm, A):
    """The paper's world: the stored episodes of the earlier checks and of the new-baselines run (1.0 s, 4 places)."""
    if arm in ('LSRA', 'R2', 'R1i', 'R3'):
        s = pd.read_csv(A3C.stored_path(A, 4, HL))
        s = s[(s['arm'] == ('METHOD' if arm == 'LSRA' else arm)) & (s['hl'] == HL)].copy()
    elif arm == 'A3S5':
        s = pd.read_csv(A3C.cell_path(A, 4, HL))
        s = s[(s['arm'] == 'A3S5') & (s['hl'] == HL)].copy()
    elif arm in ('A3T', 'PHO'):
        s = pd.read_csv(os.path.join(HERE, 'models', 'baselines_check', 'test', f'{arm}_C4_{A}.csv'))
    else:
        raise ValueError(arm)
    s['scene'] = s['scene'].astype(str)
    return s.sort_values('seed').reset_index(drop=True)


def stored_lsra_hl15():
    s = pd.read_csv(A3C.stored_path('A2', 4, 15))
    s = s[(s['arm'] == 'METHOD') & (s['hl'] == 15)].copy()
    s['scene'] = s['scene'].astype(str)
    return s.sort_values('seed').reset_index(drop=True)


def stored_files():
    """Every stored file the V0 cells and the dry run read."""
    fs = [A3C.stored_path(A, 4, HL) for A in ('A1', 'A2')] + [A3C.stored_path('A2', 4, 15), A3C.cell_path('A2', 4, HL)]
    fs += [os.path.join(HERE, 'models', 'baselines_check', 'test', f'{a}_C4_{A}.csv')
           for a, A in (('A3T', 'A1'), ('A3T', 'A2'), ('PHO', 'A2'))]
    fs += [TUNED_PATH, os.path.join(HERE, 'models', 'baselines_check', 'TUNED_FREEZE.json')]
    fs += [os.path.join(HERE, p) for p in (os.path.join('models', 'admission_check', 'ADMIT_RESULT.json'),
                                           os.path.join('models', 'a3_check', 'A3_RESULT.json'),
                                           os.path.join('models', 'baselines_check', 'BASE_RESULT.json'),
                                           os.path.join('models', 'stored_checks', 'STORED_CHECKS_RESULT.json'))]
    return fs


# ---------------------------------------------------------------- analysis helpers (pure; unit-tested)
def holds_above(ci):
    return bool(ci[0] > 0.0)


def break_even(seq_results):
    """seq_results: [(setting, ci)] in order of growing imperfection. The first setting whose interval does not lie
    above 0, or None if every interval lies above 0."""
    for s, ci in seq_results:
        if not ci[0] > 0.0:
            return s
    return None


# ---------------------------------------------------------------- unit tests (no scenes)
class _FakeEnv:
    pass


def unit_mode():
    path = os.path.join(OUT, 'UNIT.json')
    assert not os.path.exists(path), 'unit tests already ran: archive UNIT.json first'
    rep = {'_versions': versions()}
    rng = np.random.default_rng(20261030)

    # U1: the shadowing filter
    T, M, G = 6, 3, 2
    frames = []
    for t in range(T):
        act = np.array([True, t != 3, True])
        ag = []
        for j in range(M):
            if not act[j]:
                ag.append(None)
            else:
                ag.append(dict(speed=[10.0, 5.0, 0.0][j], depth=np.array([0.0, 7.0 if t >= 2 else 0.0])))
        frames.append(dict(active=act, ag=ag))
    z0 = rng.normal(size=(T, M, G))
    z = RE.shadow_filter(z0.copy(), frames, 10.0, 13.0)
    want = z0.copy()
    for t in range(1, T):
        for j in range(M):
            if not (frames[t]['active'][j] and frames[t - 1]['active'][j]):
                continue
            for g in range(G):
                d = 13.0 if frames[t]['ag'][j]['depth'][g] > 0 else 10.0
                rho = np.exp(-frames[t]['ag'][j]['speed'] * 0.1 / d)
                want[t, j, g] = rho * want[t - 1, j, g] + np.sqrt(1 - rho * rho) * z0[t, j, g]
    rep['U1_filter_recursion'] = bool(np.allclose(z, want, rtol=0, atol=1e-12))
    rep['U1_restart_on_reappearance'] = bool(np.array_equal(z[4, 1], z0[4, 1]) and np.array_equal(z[3, 1], z0[3, 1]))
    rep['U1_stopped_vehicle_keeps_draw'] = bool(np.allclose(z[:, 2, :], np.repeat(z0[0:1, 2, :], T, axis=0), rtol=0, atol=1e-12))
    T2, M2 = 60, 2000
    fr2 = [dict(active=np.ones(M2, dtype=bool), ag=[dict(speed=10.0, depth=np.zeros(1)) for _ in range(M2)])
           for _ in range(T2)]
    zz = RE.shadow_filter(rng.normal(size=(T2, M2, 1)), fr2, 3.8, 3.8)
    rho38 = np.exp(-1.0 / 3.8)
    lag = np.corrcoef(zz[T2 - 1, :, 0], zz[T2 - 2, :, 0])[0, 1]
    rep['U1_unit_variance_and_correlation'] = bool(abs(zz[T2 - 1, :, 0].var() - 1.0) < 0.1 and abs(lag - rho38) < 0.05)
    rep['U1_values'] = dict(var=float(zz[T2 - 1, :, 0].var()), lag1=float(lag), rho=float(rho38))

    # U2: the blockage rules on hand-made boxes (UE at (0, 0) 1.5 m; station at (100, 0) 6 m)
    import blockage3d as B
    ue, gnb = (0.0, 0.0), (100.0, 0.0)

    def box(x, y, L, W, h, th=0.0):
        return dict(x=x, y=y, length=L, width=W, height=h, theta=th)

    h_ray50 = 1.5 + 0.48 * 4.5                                     # the ray where it enters a 4 m box centred at x = 50
    near = box(50.0, 0.0, 4.0, 2.0, h_ray50 - 0.05)
    lam = B.C_LIGHT / B.FREQ_HZ
    v = -0.05 * np.sqrt(2.0 / lam * (1.0 / 48.0 + 1.0 / 52.0))
    r0 = B.link_blockage(ue, 1.5, gnb, 6.0, [near])
    r1 = RE.link_blockage_b1(ue, 1.5, gnb, 6.0, [near])
    rep['U2_b1_near_miss'] = bool(r0['depth_db'] == 0.0 and abs(r1['depth_db'] - B.knife_edge_loss_db(v)) < 1e-9
                                  and r1['depth_db'] > 0.0)
    tall = box(50.0, 0.0, 4.0, 2.0, 4.5)
    rep['U2_b1_same_when_above'] = bool(B.link_blockage(ue, 1.5, gnb, 6.0, [tall]) ==
                                        RE.link_blockage_b1(ue, 1.5, gnb, 6.0, [tall]))
    side_tall = box(50.0, 2.2, 4.0, 2.0, 4.5, np.pi / 2)          # spans y 0.2..4.2: misses; x1.3 spans -0.4..4.8: hits
    rep['U2_b2_tall_longer'] = bool(B.link_blockage(ue, 1.5, gnb, 6.0, [side_tall])['depth_db'] == 0.0
                                    and RE.link_blockage_b2(ue, 1.5, gnb, 6.0, [side_tall])['depth_db'] > 0.0)
    side_low = box(10.0, 1.8, 4.0, 2.0, 2.4, np.pi / 2)            # spans -0.2..3.8: hits; x0.75 spans 0.3..3.3: misses
    rep['U2_b2_low_shorter'] = bool(B.link_blockage(ue, 1.5, gnb, 6.0, [side_low])['depth_db'] > 0.0
                                    and RE.link_blockage_b2(ue, 1.5, gnb, 6.0, [side_low])['depth_db'] == 0.0)
    two = [box(30.0, 0.0, 4.0, 2.0, 4.0), box(70.0, 0.0, 4.0, 2.0, 6.0)]
    l1 = B.link_blockage(ue, 1.5, gnb, 6.0, [two[0]])['depth_db']
    l2 = B.link_blockage(ue, 1.5, gnb, 6.0, [two[1]])['depth_db']
    r3 = RE.link_blockage_b3(ue, 1.5, gnb, 6.0, two)['depth_db']
    rep['U2_b3_sum_and_cap'] = bool(l1 > 0 and l2 > 0 and abs(r3 - min(l1 + l2, 60.0)) < 1e-9
                                    and B.link_blockage(ue, 1.5, gnb, 6.0, two)['depth_db'] == max(l1, l2))
    rep['U2_values'] = dict(near_miss_db=float(r1['depth_db']), b3_two=float(r3), one=float(l1), other=float(l2))

    # U3: the beam gain switch reaches every module that reads it
    fr = BC.frame([[60.0, 80.0, 100.0, 120.0]] * 3, np.zeros((3, 4)))
    e = BC.Mock([fr], 3, 4)
    before = (MA.MultiAgentRSUBlockageEnv._desired_dbm(e, 0, 1), CF.desired(e, 0, 1, 30.0),
              float(LS.mean_sinr_db(e, fr)[0][0, 1]), float(AG.mean_dbm(e, 0)[1]), float(PH.rsrp_frame(e, fr)[0, 1]))
    RE.set_beam_gain(20.0)
    after = (MA.MultiAgentRSUBlockageEnv._desired_dbm(e, 0, 1), CF.desired(e, 0, 1, 30.0),
             float(LS.mean_sinr_db(e, fr)[0][0, 1]), float(AG.mean_dbm(e, 0)[1]), float(PH.rsrp_frame(e, fr)[0, 1]))
    RE.set_beam_gain(None)
    back = MA.MultiAgentRSUBlockageEnv._desired_dbm(e, 0, 1)
    rep['U3_gain_switch'] = bool(all(abs((a - b) - 5.0) < 1e-9 for a, b in zip(after, before))
                                 and back == before[0] and MA.G_SERV == 15.0)

    # U4: the A2 pattern; U5: the A3 misalignment loss and its pointing clock
    rep['U4_pattern'] = bool(RE.pattern_gain_db(0.0) == 15.0 and abs(RE.pattern_gain_db(15.0) - 12.0) < 1e-12
                             and RE.pattern_gain_db(34.0) == 0.0 and RE.pattern_gain_db(120.0) == 0.0)
    rep['U5_loss'] = bool(RE.misalign_loss_db(0.0) == 0.0 and abs(RE.misalign_loss_db(30.0) - 12.0) < 1e-12
                          and RE.misalign_loss_db(90.0) == 30.0)
    f = _FakeEnv()
    f._az = np.zeros((20, 1, 1))
    for t in range(20):
        f._az[t, 0, 0] = np.radians(2.0 * t)                       # the vehicle moves 2 degrees per slot
    f._beam_serv = np.array([-1])
    f._beam_t0 = np.array([0])
    f._beam_last = np.array([-10])
    got = [RE.RealismEnv._misalign_db(f, 0, 0, t) for t in range(13)]
    want = [RE.misalign_loss_db(2.0 * (t - max(0, t - t % 5))) for t in range(13)]
    rep['U5_pointing_every_5_slots'] = bool(np.allclose(got, want, rtol=0, atol=1e-9))
    f._beam_last = np.array([5])
    rep['U5_repointed_after_gap'] = bool(RE.RealismEnv._misalign_db(f, 0, 0, 13) == 0.0)

    # U6: fading
    fr6 = [dict(active=np.ones(400, dtype=bool), ag=[dict(depth=np.array([0.0, 5.0])) for _ in range(400)])
           for _ in range(50)]
    g1 = RE.fading_gains(fr6, 400, 2, 7)
    g2 = RE.fading_gains(fr6, 400, 2, 7)
    g3 = RE.fading_gains(fr6, 400, 2, 8)
    rep['U6_unit_mean'] = bool(abs(g1[:, :, 0].mean() - 1.0) < 0.02 and abs(g1[:, :, 1].mean() - 1.0) < 0.02)
    rep['U6_rayleigh_spread_wider'] = bool(g1[:, :, 1].var() > 0.9 and g1[:, :, 0].var() < 0.3)
    rep['U6_common_random_numbers'] = bool(np.array_equal(g1, g2) and not np.array_equal(g1, g3))

    # U7: the break-even rule
    rep['U7_break_even'] = bool(break_even([('a', [0.5, 1.0]), ('b', [-0.1, 0.8]), ('c', [-1.0, -0.2])]) == 'b'
                                and break_even([('a', [0.5, 1.0])]) is None)

    # U8: every world and setting builds its switches; unknown switch names are refused
    ok = True
    for w in WORLDS:
        cfg = config(w)
        cfg.pop('GAIN', None)
        ok = ok and all(hasattr(RE.RealismEnv, k) for k in cfg)
    for c in COORD:
        cfg = config(None, c)
        cfg.pop('HL', None)
        ok = ok and all(hasattr(RE.RealismEnv, k) or k == 'JITTER' for k in cfg)
    rep['U8_switch_names'] = bool(ok and len(WORLDS) == 15 and len(COORD) == 9 and len(ARMS) == 10
                                  and len(world_cells()) == 150 and len(coord_cells()) == 13)

    flat = [v for k, v in rep.items() if isinstance(v, bool)]
    rep['ok'] = bool(all(flat))
    rep['code'] = code_md5()
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(path, 'w'), indent=1)
    status('[unit] ' + ('PASS' if rep['ok'] else 'FAIL') + ' ' +
           json.dumps({k: v for k, v in rep.items() if k not in ('_versions', 'code')}))


# ---------------------------------------------------------------- prep: the B1-B3 frames, built once
def prep_mode():
    path = os.path.join(OUT, 'PREP.json')
    assert not os.path.exists(path), 'prep already ran'
    u = json.load(open(os.path.join(OUT, 'UNIT.json')))
    assert u['ok'] and u['code'] == code_md5(), 'unit tests did not pass with this code'
    env, _ = make_env(None, None, 'A2')
    rec = {'_versions': versions(), 'files': {}}
    for blk in B_BLOCKS:
        t0 = time.time()
        for sid in env.scene_ids:
            assert not os.path.exists(RE.cache_path(blk, sid)), 'frames already built: archive geom_cache_realism first'
            RE.rebuilt_frames(env, sid, blk)
            p = RE.cache_path(blk, sid)
            rec['files'][os.path.relpath(p, HERE).replace(os.sep, '/')] = md5(p)
        status(f'[prep] {blk}: {len(env.scene_ids)} scenes ({time.time() - t0:.0f} s)')
    stats = {}
    base = []
    for blk in [None] + B_BLOCKS:
        dep = []
        for sid in env.scene_ids:
            obj = MA.MultiAgentRSUBlockageEnv._precompute(env, sid) if blk is None else RE.rebuilt_frames(env, sid, blk)
            for fr in obj['frames']:
                for a in fr['ag']:
                    if a is not None:
                        dep.extend(list(a['depth']))
        dep = np.asarray(dep)
        stats[str(blk)] = dict(link_slots=int(dep.size), blocked_share=float((dep > 0).mean()),
                               mean_loss_when_blocked=float(dep[dep > 0].mean()))
    rec['blockage_stats'] = stats
    rec['code'] = code_md5()
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[prep] done: {stats}')


# ---------------------------------------------------------------- gates (real scenes)
def gate_mode():
    path = os.path.join(OUT, 'GATE.json')
    assert not os.path.exists(path), 'gates already ran: archive GATE.json first'
    for f in ('UNIT.json', 'PREP.json'):
        rec = json.load(open(os.path.join(OUT, f)))
        assert rec.get('ok', True) and rec['code'] == code_md5(), f'{f} was made with other code'
    fzm = MHO.check_freeze()
    assert abs(float(fzm['beta']) - BETA) < 1e-15
    seeds = seeds_test()
    rep = {'_versions': versions()}
    # G0: every switch off reproduces the stored episodes (first 40)
    g0 = {}
    for arm, A in ARMS:
        rows = run_job(None, None, arm, A, seeds[:N_GATE])
        st = stored_v0(arm, A)
        new = AC.roundtrip(rows)
        cols = [c for c in st.columns if c not in ('world', 'coord', 'arm') and not c.startswith('p_')]
        old = st.set_index('seed').loc[list(new['seed'])].reset_index()
        new['scene'] = new['scene'].astype(str)
        ok = all(c in new.columns for c in cols) and bool(LC6.frames_equal_typed(
            new[cols].reset_index(drop=True), old[cols].reset_index(drop=True)))
        g0[f'{arm}_{A}'] = bool(ok)
        status(f'[G0] {arm} {A} = stored: {ok}')
    rep['G0'] = g0
    # G1: the frame builder with the paper's blockage rule rebuilds every stored slot
    env, _ = make_env(None, None, 'A2')
    t0 = time.time()
    n = mism = 0
    first = None
    for sid in env.scene_ids:
        obj = MA.MultiAgentRSUBlockageEnv._precompute(env, sid)
        nb = RE.build_frames(env, sid, None)
        for k, (a, b) in enumerate(zip(nb['frames'], obj['frames'])):
            n += 1
            if not BC.frame_same(a, b):
                mism += 1
                first = first or f'{sid} slot {k}'
    rep['G1'] = dict(slots=n, mismatches=mism, first=first, seconds=round(time.time() - t0, 1),
                     ok=bool(mism == 0 and n == 6000))
    status(f'[G1] {rep["G1"]}')
    # G2: the mechanics on real scenes (LSRA under A2, first 10 episodes; DEBUG records)
    g2 = {}
    for w in ('H1', 'H2a', 'H2b'):
        env, _ = make_env(w, None, 'A2')
        env.DEBUG = True
        rows = run_real(env, policy('LSRA', 'A2'), seeds[:10])
        ev = [e for e in env.real_events if e[0] == 'reest']
        room_ok = all(e[5][e[4]] < CAP for e in ev)
        g2[w] = dict(reest=len(ev), too_late=int(sum(r['h2_too_late'] for r in rows)),
                     target=int(sum(r['h2_target'] for r in rows)), h1_unserved=int(sum(r['h1_unserved'] for r in rows)),
                     reest_station_had_room=bool(room_ok),
                     flight_mismatch=int(sum(r['flight_mismatch'] for r in rows)))
        g2[w]['ok'] = bool(room_ok and g2[w]['flight_mismatch'] == 0 and
                           (g2[w]['h1_unserved'] > 0 if w == 'H1' else len(ev) > 0))
    for c in ('C1a', 'C1b', 'C2b', 'C3c'):
        env, _ = make_env(None, c, 'A2')
        rows = run_real(env, policy('LSRA', 'A2'), seeds[:20])
        d = AC.roundtrip(rows)
        e = dict(starts=int(d.starts.sum()), dropped=int(d.dropped.sum()), c3_failed=int(d.c3_failed.sum()),
                 jit_min=int(d.jit_min[d.jit_min >= 0].min()) if (d.jit_min >= 0).any() else None,
                 jit_max=int(d.jit_max.max()), flight_mismatch=int(d.flight_mismatch.sum()))
        if c == 'C2b':
            frac = e['dropped'] / max(e['dropped'] + e['starts'], 1)
            e['drop_share'] = frac
            e['ok'] = bool(abs(frac - 0.20) < 0.06)
        elif c == 'C3c':
            frac = e['c3_failed'] / max(e['starts'], 1)
            e['fail_share'] = frac
            e['ok'] = bool(0.04 < frac < 0.16 and e['flight_mismatch'] == 0)
        elif c == 'C1a':
            e['ok'] = bool(e['jit_min'] is not None and e['jit_min'] >= HL - 5 and e['jit_max'] <= HL + 5
                           and e['flight_mismatch'] == 0)
        else:
            e['ok'] = bool(e['jit_min'] is not None and e['jit_min'] >= HL and e['jit_max'] <= HL + 5
                           and e['flight_mismatch'] == 0)
        g2[c] = e
    rep['G2'] = g2
    rep['G2_ok'] = bool(all(v['ok'] for v in g2.values()))
    status(f'[G2] {g2}')
    # G3: every cell runs (first episode); the accounting adds up
    g3 = {}
    for w, c, arm, A in world_cells() + coord_cells():
        try:
            r = AC.roundtrip(run_job(w, c, arm, A, seeds[:1]))
            ok = bool(len(r) == 1)
            ok = ok and bool((r.starts == r.requests - r.refused - r.dropped).all())
            if c != 'C2a' and c != 'C2b':
                ok = ok and int(r.dec_over.sum()) == 0
            if A == 'A1':
                ok = ok and int(r.loop_violations.sum()) == 0
        except Exception as ex:
            ok = f'error: {type(ex).__name__}: {ex}'
        g3[cell_name(w, c, arm, A)] = ok
    rep['G3'] = dict(cells=g3, ok=bool(len(g3) == 163 and all(v is True for v in g3.values())))
    status(f'[G3] ok={rep["G3"]["ok"]}')
    # G4: timing (LSRA under A2, 10 episodes per world and setting)
    tm = {}
    for w in list(WORLDS) + list(COORD):
        t0 = time.time()
        if w in WORLDS:
            run_job(w, None, 'LSRA', 'A2', seeds[:10])
        else:
            run_job(None, w, 'LSRA', 'A2', seeds[:10])
        tm[w] = round((time.time() - t0) / 10.0, 3)
    total = sum(tm[w] for w in WORLDS) * 10 * N_EP + sum(tm[c] for c, _ in COORD_ARMS) * N_EP
    rep['G4'] = dict(s_per_episode=tm, projected_cpu_hours=round(total / 3600.0, 2),
                     projected_wall_hours_6_workers=round(total / 3600.0 / 6.0, 2))
    status(f'[G4] {rep["G4"]}')
    rep['ok'] = bool(all(g0.values()) and rep['G1']['ok'] and rep['G2_ok'] and rep['G3']['ok'])
    rep['code'] = code_md5()
    json.dump(rep, open(path, 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- cells and analysis
def load_cell(w, c, arm, A):
    if (w in (None, 'V0')) and not c:
        return stored_v0(arm, A)
    if c == 'C4c':
        return stored_lsra_hl15()
    d = pd.read_csv(cell_path(None if w == 'V0' else w, c, arm, A))
    d['scene'] = d['scene'].astype(str)
    return d.sort_values('seed').reset_index(drop=True)


def paired(x, y):
    assert len(x) == N_EP and len(y) == N_EP, 'a cell does not have 480 episodes'
    assert list(x.seed) == list(y.seed) and list(x.scene) == list(y.scene), 'pairing by seed and scene failed'
    return (x.worst.values - y.worst.values) * 100.0, y.scene.values


def analyse_mode(dry=False):
    fz = {'written': 'dry run on relabelled stored cells'} if dry else check_freeze()
    res_path = os.path.join(OUT, 'REAL_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    status('[analyse] start')
    R = {'_freeze': fz['written'], '_versions': versions()}
    worlds = ['V0'] + list(WORLDS)
    cells = {(w, arm, A): load_cell(w, None, arm, A) for w in worlds for arm, A in ARMS}
    ccells = {(c, arm): load_cell(None, c, arm, 'A2') for c, arm in COORD_ARMS}
    ccells[('C4c', 'LSRA')] = load_cell(None, 'C4c', 'LSRA', 'A2')
    problems = []
    ref = cells[('V0', 'LSRA', 'A2')]
    for key, d in list(cells.items()) + list(ccells.items()):
        if not (len(d) == N_EP and list(d.seed) == list(ref.seed) and list(d.scene) == list(ref.scene)):
            problems.append(f'{key}: does not pair')
            continue
        if key[0] == 'V0' or key[0] == 'C4c':
            continue
        if 'dropped' in d.columns:
            if not bool((d.starts == d.requests - d.refused - d.dropped).all()):
                problems.append(f'{key}: starts differ from requests minus refused minus dropped')
        if key[0] not in ('C2a', 'C2b') and int(d.dec_over.sum()) != 0:
            problems.append(f'{key}: created landing excess')
        if len(key) == 3 and key[2] == 'A1' and int(d.loop_violations.sum()) != 0:
            problems.append(f'{key}: loop violations')
    R['problems'] = problems
    if problems:
        json.dump(R, open(res_path, 'w'), indent=1)
        status('[analyse] STOPPED: ' + '; '.join(problems[:5]))
        return
    sb = [BOOT_SEED0]

    def boot(d, sc, q=(2.5, 97.5)):
        ci = RM.scene_boot(d, sc, sb[0], q)
        sb[0] += 1
        return ci

    q9917 = (100.0 * 0.05 / 6 / 2, 100.0 - 100.0 * 0.05 / 6 / 2)
    q975 = (100.0 * 0.05 / 2 / 2, 100.0 - 100.0 * 0.05 / 2 / 2)
    W = {}
    for w in worlds:
        l2 = cells[(w, 'LSRA', 'A2')]
        rw1 = {}
        for b in RW1_BASE:
            d, sc = paired(cells[(w, b, 'A2')], l2)
            ci = boot(d, sc, q9917)
            rw1[b] = dict(mean=float(d.mean()), ci_9917=ci, beats=holds_above(ci))
        d, sc = paired(cells[(w, 'LSRA', 'A1')], l2)
        ci2 = boot(d, sc)
        rw3 = {}
        for b in ('R2', 'A3T'):
            d3, sc3 = paired(cells[(w, b, 'A1')], l2)
            ci3 = boot(d3, sc3, q975)
            rw3[b] = dict(mean=float(d3.mean()), ci_975=ci3, beats=holds_above(ci3))
        levels = {}
        for arm, A in ARMS:
            x = cells[(w, arm, A)]
            v = x.worst.values * 100.0
            e = dict(worst=float(v.mean()), ci_95=boot(v, x.scene.values),
                     ho_per_vehicle_min=float(x.starts.sum() / x.p_act.sum() * 600.0))
            if 'pingpong' in x.columns:
                e['quick_returns_per_vehicle_min'] = float(x.pingpong.sum() / x.p_act.sum() * 600.0)
            if A != 'A0':
                e['refused_share'] = float(x.refused.sum() / max(x.requests.sum(), 1))
            for k in ('h1_unserved', 'h2_too_late', 'h2_target', 'reest'):
                if k in x.columns:
                    e[k + '_per_episode'] = float(x[k].mean())
            wa = np.maximum(x.worst_act.values, 1.0)
            e['causes_worst'] = {k: float((x[f'w_{k}'].values / wa).mean() * 100.0) for k in RM.CAUSES}
            levels[f'{arm}_{A}'] = e
        W[w] = dict(RW1=dict(per_baseline=rw1, held=bool(all(v['beats'] for v in rw1.values())),
                             not_beaten=[b for b, v in rw1.items() if not v['beats']]),
                    RW2=dict(mean=float(d.mean()), ci_95=ci2, held=holds_above(ci2)),
                    RW3=dict(per_baseline=rw3, held=bool(all(v['beats'] for v in rw3.values()))),
                    levels=levels)
    R['worlds'] = W
    l1v0, r2v0 = cells[('V0', 'LSRA', 'A1')], cells[('V0', 'R2', 'A1')]
    C = {}
    allc = list(COORD) + ['C4c']
    for c in allc:
        lc = ccells[(c, 'LSRA')]
        e = {'LSRA_worst': float(lc.worst.mean() * 100.0)}
        for name, refc in (('vs_LSRA_A1', l1v0), ('vs_R2_A1', r2v0), ('vs_LSRA_A2_ideal', cells[('V0', 'LSRA', 'A2')])):
            d, sc = paired(refc, lc)
            e[name] = dict(mean=float(d.mean()), ci_95=boot(d, sc))
        if 'dropped' in lc.columns:
            for k in ('dropped', 'c3_failed', 'reest'):
                e[k + '_per_episode'] = float(lc[k].mean())
        C[c] = e
    for c, b in (('C2b', 'R3'), ('C2b', 'A3T'), ('C3c', 'R3'), ('C3c', 'A3T')):
        x = ccells[(c, b)]
        d, sc = paired(x, cells[('V0', b, 'A2')])
        d2, sc2 = paired(x, ccells[(c, 'LSRA')])
        C[c][f'{b}_cost_of_imperfection'] = dict(mean=float(d.mean()), ci_95=boot(d, sc))
        C[c][f'{b}_minus_LSRA_same_setting'] = dict(mean=float(d2.mean()), ci_95=boot(d2, sc2))
    R['coordination'] = C
    be = {}
    for seq, settings in SEQUENCES.items():
        be[seq] = dict(vs_LSRA_A1=break_even([(s, C[s]['vs_LSRA_A1']['ci_95']) for s in settings]),
                       vs_R2_A1=break_even([(s, C[s]['vs_R2_A1']['ci_95']) for s in settings]))
    R['break_even'] = be
    R['robustness'] = {k: [w for w in worlds if W[w][k]['held']] for k in ('RW1', 'RW2', 'RW3')}
    R['last_boot_seed'] = sb[0] - 1
    g = dict(
        RW1_held_in_V0_S1_S2_B1_B2_B3_A1a_A1b_H1_H2a_F1=all(W[w]['RW1']['held'] for w in
                                                             ('V0', 'S1', 'S2', 'B1', 'B2', 'B3', 'A1a', 'A1b', 'H1', 'H2a', 'F1')),
        RW2_RW3_held_in_every_world_but_ALL=all(W[w]['RW2']['held'] and W[w]['RW3']['held'] for w in worlds if w != 'ALL'),
        C2b_C3c_keep_LSRA_A2_ahead_of_LSRA_A1_and_R2_A1=all(holds_above(C[c][k]['ci_95']) for c in ('C2b', 'C3c')
                                                         for k in ('vs_LSRA_A1', 'vs_R2_A1')),
        C4_no_break_even_within_half_second=be['C4']['vs_LSRA_A1'] is None and be['C4']['vs_R2_A1'] is None)
    R['guesses'] = {k: bool(v) for k, v in g.items()}
    R['ok'] = True
    json.dump(R, open(res_path, 'w'), indent=1)
    write_txt(R, os.path.join(OUT, 'REAL_RESULT.txt'))
    status(f'[analyse] done; robustness {R["robustness"]}')


def write_txt(R, path):
    def ci(v):
        return f'[{v[0]:+.2f}, {v[1]:+.2f}]'
    L = ['REALISM RUN FOR LSRA (60 held-out DAIR scenes, 480 episodes per cell, 4 places, 1.0 s; J = worst-vehicle RLF %;',
         'differences: the other arm minus LSRA under coordinated admission, pp, paired by episode; scene bootstraps)', '']
    for w, v in R['worlds'].items():
        L.append(f"{w}: LSRA A2 {v['levels']['LSRA_A2']['worst']:.2f} %  |  RW1 {'HELD' if v['RW1']['held'] else 'NOT HELD ' + str(v['RW1']['not_beaten'])}"
                 f"  |  RW2 {v['RW2']['mean']:+.2f} {ci(v['RW2']['ci_95'])}  |  RW3 {'HELD' if v['RW3']['held'] else 'NOT HELD'}")
        L.append('    RW1: ' + '  '.join(f"{b} {x['mean']:+.2f} {ci(x['ci_9917'])}" for b, x in v['RW1']['per_baseline'].items()))
        L.append('    RW3: ' + '  '.join(f"{b} {x['mean']:+.2f} {ci(x['ci_975'])}" for b, x in v['RW3']['per_baseline'].items()))
    L.append('')
    L.append('imperfect coordination (LSRA under A2; reference minus LSRA, pp, 95 %):')
    for c, v in R['coordination'].items():
        L.append(f"  {c}: LSRA {v['LSRA_worst']:.2f} %  vs LSRA A1 {v['vs_LSRA_A1']['mean']:+.2f} {ci(v['vs_LSRA_A1']['ci_95'])}"
                 f"  vs R2 A1 {v['vs_R2_A1']['mean']:+.2f} {ci(v['vs_R2_A1']['ci_95'])}"
                 f"  vs ideal {v['vs_LSRA_A2_ideal']['mean']:+.2f} {ci(v['vs_LSRA_A2_ideal']['ci_95'])}")
    L.append(f"break-even: {R['break_even']}")
    L.append(f"robustness (worlds where each conclusion holds): {R['robustness']}")
    L.append(f"guesses recorded before the run: {R['guesses']}")
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- dry run (stored V0 cells relabelled)
def dry_mode():
    global OUT
    rec_path = os.path.join(OUT_REAL, 'DRY.json')
    assert not os.path.exists(rec_path), 'dry run already made: archive DRY.json and models/_dry_rc first'
    assert not os.path.exists(OUT_DRY), 'models/_dry_rc exists: archive it first'
    MHO.check_freeze()
    OUT = OUT_DRY
    try:
        for w, c, arm, A in world_cells() + coord_cells():
            x = stored_v0(arm, A).copy()
            for k in ('requests', 'refused', 'loop_violations', 'dropped', 'c3_failed', 'reest', 'h1_unserved',
                      'h2_too_late', 'h2_target'):
                if k not in x.columns:
                    x[k] = 0
            x.to_csv(cell_path(w, c, arm, A), index=False)
        analyse_mode(dry=True)
        res = json.load(open(os.path.join(OUT_DRY, 'REAL_RESULT.json')))
    finally:
        OUT = OUT_REAL
    AR = json.load(open(os.path.join(HERE, 'models', 'admission_check', 'ADMIT_RESULT.json')))['secondary']
    A3R = json.load(open(os.path.join(HERE, 'models', 'a3_check', 'A3_RESULT.json')))['secondary']['table']
    BR = json.load(open(os.path.join(HERE, 'models', 'baselines_check', 'BASE_RESULT.json')))
    SC = json.load(open(os.path.join(HERE, 'models', 'stored_checks', 'STORED_CHECKS_RESULT.json')))['check_A']['cells']
    want_rw1 = {'R1i': AR['table']['A2_hl10_R1i']['minus_METHOD_same_admission']['mean'],
                'R2': AR['table']['A2_hl10_R2']['minus_METHOD_same_admission']['mean'],
                'R3': AR['table']['A2_hl10_R3']['minus_METHOD_same_admission']['mean'],
                'A3S5': A3R['A2_C4_hl10_A3S5']['minus_METHOD_same_admission']['mean'],
                'A3T': BR['HN1_A2_C4']['per_arm']['A3T']['mean'],
                'PHO': BR['HN1_A2_C4']['per_arm']['PHO']['mean']}
    want_rw2 = AR['contrasts']['hl10_METHOD_A1_minus_A2']['mean']
    want_rw3 = {'R2': SC['C4_R2_hl10']['T']['mean'], 'A3T': BR['split']['C4_A3T']['T']['mean']}
    mism, cnt = [], [0]

    def cmp(name, u, v, tol=1e-9):
        cnt[0] += 1
        if not abs(float(u) - float(v)) <= tol:
            mism.append(f'{name}: {u} vs {v}')

    ok_run = not res.get('problems')
    if ok_run:
        for w, v in res['worlds'].items():
            for b in RW1_BASE:
                cmp(f'{w}_RW1_{b}', v['RW1']['per_baseline'][b]['mean'], want_rw1[b])
            cmp(f'{w}_RW2', v['RW2']['mean'], want_rw2)
            for b in ('R2', 'A3T'):
                cmp(f'{w}_RW3_{b}', v['RW3']['per_baseline'][b]['mean'], want_rw3[b])
        for c in COORD:
            v = res['coordination'][c]
            cmp(f'{c}_vs_LSRA_A1', v['vs_LSRA_A1']['mean'], want_rw2)
            cmp(f'{c}_vs_R2_A1', v['vs_R2_A1']['mean'], want_rw3['R2'])
            cmp(f'{c}_vs_ideal', v['vs_LSRA_A2_ideal']['mean'], 0.0)
    ok = bool(ok_run and not mism and cnt[0] >= 150)
    json.dump(dict(ok=ok, compared=cnt[0], mismatches=mism, problems=res.get('problems'),
                   result_md5=md5(os.path.join(OUT_DRY, 'REAL_RESULT.json')),
                   note='stored V0 cells relabelled as every world and setting; no realism episode used',
                   code=code_md5()), open(rec_path, 'w'), indent=1)
    status(f'[dry] ok={ok} compared={cnt[0]} mismatches={len(mism)}')


# ---------------------------------------------------------------- freeze and the run
def freeze_files():
    roots = [os.path.normcase(os.path.abspath(r)) for r in (HERE, RM.HERE, CV_DIR)]
    files = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        fa = os.path.abspath(f)
        if not os.path.isfile(fa):
            continue
        if any(os.path.normcase(fa).startswith(r + os.sep) for r in roots):
            files[os.path.relpath(fa, HERE).replace(os.sep, '/')] = fa
    data = [MHO.POOL_H, os.path.join(MHO.OUT, 'SEEDS_heldout60.csv'), os.path.join(MHO.OUT, 'FREEZE.json')]
    data += stored_files()
    data += sorted(glob.glob(os.path.join(RE.CACHE_DIR, '*', '*.pkl')))
    data += [os.path.join(HERE, PREREG)] + [os.path.join(OUT_REAL, f) for f in ('UNIT.json', 'PREP.json', 'GATE.json', 'DRY.json')]
    for f in data:
        fa = os.path.abspath(f)
        files[os.path.relpath(fa, HERE).replace(os.sep, '/')] = fa
    return files


def freeze_mode():
    path = os.path.join(OUT_REAL, 'REAL_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    now = code_md5()
    for f in ('UNIT.json', 'PREP.json', 'GATE.json', 'DRY.json'):
        rec = json.load(open(os.path.join(OUT_REAL, f)))
        assert rec.get('ok', True), f'{f} did not pass: no freeze'
        assert rec['code'] == now, f'{f} was made with other code: run the checks again'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen plan is not in the folder'
    prep = json.load(open(os.path.join(OUT_REAL, 'PREP.json')))
    assert all(md5(os.path.join(HERE, k)) == v for k, v in prep['files'].items()), 'a built frame file changed'
    assert len(prep['files']) == 180
    MHO.check_freeze()
    tuned()
    files = freeze_files()
    rec = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), versions=versions(),
               md5={k: md5(v) for k, v in sorted(files.items())})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written: {len(files)} files, md5 {md5(path)}')


def check_freeze():
    path = os.path.join(OUT_REAL, 'REAL_FREEZE.json')
    assert os.path.exists(path), 'not frozen'
    fz = json.load(open(path))
    bad = [k for k, v in fz['md5'].items() if md5(os.path.join(HERE, k)) != v]
    assert not bad, 'changed after the freeze: ' + ', '.join(bad[:10])
    extra = [k for k in freeze_files() if k not in fz['md5']]
    assert not extra, 'read but not frozen: ' + ', '.join(extra[:10])
    MHO.check_freeze()
    return fz


def jobs():
    out = []
    for w, c, arm, A in world_cells() + coord_cells():
        out.append(dict(key=cell_name(w, c, arm, A), out=cell_path(w, c, arm, A),
                        run=lambda w=w, c=c, arm=arm, A=A: run_job(w, c, arm, A, seeds_test())))
    return out


def run_jobs(js, worker):
    for j in js:
        if os.path.exists(j['out']) or not BC.claim(j['out'] + '.claim'):
            continue
        t0 = time.time()
        try:
            rows = j['run']()
        except Exception as ex:
            status(f'STOP [run] worker {worker}: {j["key"]} failed: {type(ex).__name__}: {ex}')
            raise
        assert len(rows) == N_EP
        write_atomic(rows, j['out'])
        status(f'[run] worker {worker}: {j["key"]} ({len(rows)} episodes, {time.time() - t0:.0f} s)')


def wait_all(paths, worker):
    last_n, last_change = -1, time.time()
    while True:
        n = sum(os.path.exists(p) for p in paths)
        if n == len(paths):
            return
        if n != last_n:
            last_n, last_change = n, time.time()
        elif time.time() - last_change > NO_PROGRESS_STOP_S:
            status(f'STOP [run] worker {worker}: no new output for {NO_PROGRESS_STOP_S} s ({n} of {len(paths)})')
            raise SystemExit(2)
        time.sleep(POLL_S)


def pipeline_mode(worker):
    check_freeze()
    tuned()
    status(f'[pipeline] worker {worker} starts (pid {os.getpid()})')
    js = jobs()
    run_jobs(js, worker)
    wait_all([j['out'] for j in js], worker)
    if BC.claim(os.path.join(OUT, 'analyse.claim')):
        status(f'[analyse] worker {worker} runs it')
        analyse_mode()
    status(f'[pipeline] worker {worker} done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['unit', 'prep', 'gate', 'dry', 'freeze', 'pipeline'])
    ap.add_argument('--worker', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT_REAL, exist_ok=True)
    {'unit': unit_mode, 'prep': prep_mode, 'gate': gate_mode, 'dry': dry_mode, 'freeze': freeze_mode,
     'pipeline': lambda: pipeline_mode(a.worker)}[a.mode]()


if __name__ == '__main__':
    main()
