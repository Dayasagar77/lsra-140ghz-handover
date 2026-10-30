#!/usr/bin/env python3
"""baselines_check.py -- stronger baselines for LSRA (28 Oct 2026).

Plan: PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation (the FROZEN copy in this
folder is checked by md5 before every registered step). Daya approved building and testing at about 10:24 AM
IST on 28 Oct 2026; the freeze and the run need his OK after the gates.

New arms: A3T, A3TNF and A3LB (a3_general.A3Gen; stage 1 tunes them on the 45 design scenes), PHO, PHCV and PHOR
(ph_rule.PHRule, Na et al.'s rule) and LSRAM (lsra_m.LSRAM). A3O, the per-scene bound, is the measured A3 grid
run on the held-out scenes. LSRA and the earlier arms are not run again: their stored episodes are paired with
the new ones by seed and scene.

Modes, in this order, each launched by its own .cmd through WMI on the laptop (Windows Python 3.11.9):
  unit                  hand-made tests, no scenes                                         -> UNIT.json
  gate                  identities, every-cell runs and timing on real scenes; no new arm
                        is compared with LSRA                                              -> GATE.json
  dry                   the registered analysis on stored arms relabelled as the new arms
                        (into models/_dry_bc); every number that can be checked must equal
                        the earlier checks' numbers                                        -> DRY.json
  freeze                md5 of every file the run reads, once unit, gate and dry passed  -> BASE_FREEZE.json
  pipeline --worker k   the run: stage 1 (the A3 grid, then kappa), the frozen choice (TUNED.json), stage 2 (the
                        test cells and the A3O grid), then the analysis, once. Workers take jobs one at a time
                        through claim files, so no job runs twice; a worker with nothing left to take waits for
                        the others, and stops if no output appears for an hour.
Outputs: models/baselines_check/.
"""
import os, sys, io, json, time, argparse, hashlib, platform

HERE = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'LEARNED_FORECAST_PAPER', 'cv_check'))
sys.path.insert(0, HERE)
sys.path.append(CV_DIR)                    # cv_forecast.py of the learned-forecast study (frozen 27 Oct), read only
import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import load_check_c6 as LC6
import a3_check as A3C
import paper2ma_env as MA
import paper2ma_env_admit as AD
import a3_rule as A3
import lsra as LS
import cv_forecast as CVF
import a3_general as AG
import ph_rule as PH
import lsra_m as LM

OUT_REAL = os.path.join(HERE, 'models', 'baselines_check')
OUT_DRY = os.path.join(HERE, 'models', '_dry_bc')
OUT = OUT_REAL
PREREG = 'PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation_FROZEN_20261028.md'
ADM = {'A0': 'none', 'A1': 'per_request', 'A2': 'coordinated'}
CAPS = [4, 5, 6]
HL = 10
BETA = 0.05
TUNE_CELLS = [(cap, A) for cap in CAPS for A in ('A1', 'A2')]
ORACLE_CELLS = [(4, 'A2'), (5, 'A2')]
N_EP = 480
N_DESIGN = 360
N_GATE = 40
MTS = 10                                   # TR 36.839 minimum time of stay: 1 s = 10 slots
A3_ARMS = ['A3T', 'A3TNF', 'A3LB']
PH_ARMS = ['PHO', 'PHCV', 'PHOR']
FAMILY = A3_ARMS + PH_ARMS
BOOT_SEED0 = 12000
POLL_S = 30
NO_PROGRESS_STOP_S = 3600
CV_STORED = os.path.join(CV_DIR, 'results', 'CV_EPISODES_C{cap}_hl10.csv')
# dry run: the stored arm that stands in for each new arm ('a3' = the stored A3 episodes, 'base' = LSRA's files)
DRY_SRC = {'A3T': ('a3', 'A3S5'), 'A3TNF': ('base', 'R1'), 'A3LB': ('base', 'R2'), 'PHO': ('base', 'R3'),
           'PHCV': ('base', 'R0'), 'PHOR': ('base', 'R1i'), 'LSRAM': ('base', 'ABL_nolanding')}
REF_RESULT = {4: os.path.join('models', 'admission_check', 'ADMIT_RESULT.json'),
              5: os.path.join('models', 'load_check', 'LOAD_RESULT.json'),
              6: os.path.join('models', 'load_check_c6', 'LOAD6_RESULT.json')}
A3_RESULT = os.path.join('models', 'a3_check', 'A3_RESULT.json')
SC_RESULT = os.path.join('models', 'stored_checks', 'STORED_CHECKS_RESULT.json')


# ---------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def code_md5():
    """md5 of this script and the three new modules, written into every check record and compared at the freeze."""
    return {f: md5(os.path.join(HERE, f)) for f in ('baselines_check.py', 'a3_general.py', 'ph_rule.py', 'lsra_m.py')}


def status(msg):
    os.makedirs(OUT, exist_ok=True)
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__)


def write_atomic(rows, path):
    tmp = path + '.partial'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def claim(path):
    """Create the claim file atomically; False if another worker created it first."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, 'w') as f:
        f.write(f'pid {os.getpid()} at {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
    return True


_POOLS = {}
_SEEDS = {}
_ENVS = {}


def pool(kind):
    if kind not in _POOLS:
        _POOLS[kind] = pd.read_csv(RM.POOL if kind == 'design' else MHO.POOL_H)
    return _POOLS[kind]


def seeds_design():
    if 'design' not in _SEEDS:
        _SEEDS['design'] = {sp: list(pd.read_csv(RM.seed_path(sp)).seed.astype(int)) for sp in RM.SPLITS}
    return _SEEDS['design']


def seeds_test():
    if 'test' not in _SEEDS:
        _SEEDS['test'] = AC.seeds_all()
    return _SEEDS['test']


def make_env_design(split, admit, cap):
    """The design-scene environment of stage 1: load_check_c6.make_env_cap with the design pool and its split, as
    cv_check.make_env of the learned-forecast study (gate G1c checks it against that study's stored episodes)."""
    env = AD.AdmissionEnv(pool('design'), n_agents=16, n_subbands=3, gnb_cap=cap, seed=0, split=split,
                          compute_anticipation=True, ho_latency=HL)
    env.ATTACH = True
    env.ADMIT = admit
    return env


def env_for(kind, cap, A, split=None):
    """Environments are reused across jobs of the same cell (every episode starts with a reset). At most two are
    kept (the two design splits of one cell), to keep each worker's memory small."""
    key = (kind, cap, A, split)
    if key not in _ENVS:
        if len(_ENVS) >= 2:
            _ENVS.clear()
        _ENVS[key] = (make_env_design(split, ADM[A], cap) if kind == 'design'
                      else LC6.make_env_cap(pool('heldout'), HL, ADM[A], cap))
    return _ENVS[key]


class PingPong:
    """Wraps a policy and counts, per episode, the quick returns: a handover that takes a vehicle back to the station
    it came from, started less than MTS slots (1 s, the minimum time of stay of TR 36.839) after it landed where it
    is. With the 1.0 s handover delay a return cannot land within 1 s of the landing it undoes, so the time of stay
    is measured from the landing to the start of the return. Read only: the policy's decision is passed through."""

    def __init__(self, fn, mts=MTS):
        self.fn, self.mts = fn, int(mts)
        self.counts = []

    def __call__(self, env, obs, sinr, rng, ag, temp):
        t, M = env.t, env.M
        act = np.asarray(env.frames[t]['active'], dtype=bool).copy()
        serv = np.asarray(env.serv, dtype=int).copy()
        timer = np.asarray(env._ho_timer, dtype=int).copy()
        target = np.asarray(env._ho_target, dtype=int).copy()
        if t == 0:
            self.counts.append(0)
            self.came_from = np.full(M, -1, dtype=int)
            self.landed_at = np.full(M, -10 ** 9, dtype=int)
        else:
            for i in range(M):
                if not (act[i] and self.prev_act[i]):          # absent, or (re)appeared in this slot
                    self.came_from[i] = -1
                    self.landed_at[i] = -10 ** 9
                    continue
                if serv[i] != self.prev_serv[i]:              # landed between the previous slot and this one
                    self.came_from[i] = self.prev_serv[i]
                    self.landed_at[i] = t
                if self.prev_timer[i] == 0 and timer[i] > 0:  # a handover started in the previous slot
                    if target[i] == self.came_from[i] and (t - 1) - self.landed_at[i] < self.mts:
                        self.counts[-1] += 1
        self.prev_act, self.prev_serv, self.prev_timer = act, serv, timer
        return self.fn(env, obs, sinr, rng, ag, temp)


def run(env, fn, seeds):
    """admission_check.run_cell (method_harness.run_eval_m) with the quick-return count added to every row."""
    pp = PingPong(fn)
    rows = AC.run_cell(env, pp, seeds)
    assert len(pp.counts) == len(rows) == len(seeds), 'one quick-return count per episode'
    for r, c in zip(rows, pp.counts):
        r['pingpong'] = int(c)
    return rows


def make_arm(name, params=None):
    """A fresh policy object (the rules hold per-episode state)."""
    if name in ('A3T', 'A3TNF', 'A3LB', 'A3O'):
        return AG.A3Gen(**params)
    if name == 'PHO':
        return PH.PHRule('oracle', room=False)
    if name == 'PHCV':
        return PH.PHRule('cv', room=False)
    if name == 'PHOR':
        return PH.PHRule('oracle', room=True)
    if name == 'LSRAM':
        return LM.LSRAM(k=4, source='measured', sigma_rule='fixed', sigma=8.2, beta=BETA)
    if name == 'LSRA':
        return LS.make_lsra('now', landing=True, minmax=False, beta=BETA)
    raise ValueError(name)


def test_cells():
    cells = []
    for cap in CAPS:
        for A in ('A1', 'A2'):
            cells += [(arm, cap, A) for arm in A3_ARMS]
        for A in ('A0', 'A1', 'A2'):
            cells += [(arm, cap, A) for arm in PH_ARMS]
        cells += [('LSRAM', cap, A) for A in ('A1', 'A2')]
    return cells


def tune_path(cap, A, p, sub='tune'):
    d = os.path.join(OUT, sub, f'C{cap}_{A}')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, AG.setting_id(p) + '.csv')


def test_path(arm, cap, A):
    d = os.path.join(OUT, 'test')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{arm}_C{cap}_{A}.csv')


def pick_a3(table):
    """table: (mean J %, handovers, params) per setting. Lowest J; ties: fewer handovers, then n, Off, K."""
    return sorted(table, key=lambda r: (r[0], r[1], r[2]['n_ttt'], r[2]['off'], r[2]['k']))[0]


def pick_kappa(table):
    """table: (mean J %, params) per kappa. Lowest J; ties: the smaller kappa."""
    return sorted(table, key=lambda r: (r[0], r[1]['kappa']))[0]


def a3o_pick(per):
    """per: (setting id, episodes sorted by seed) for every setting, all with the same seeds and scenes. For every
    scene, the setting with the lowest mean J over its episodes (ties: the earlier in grid order). Returns the chosen
    episodes (seed, scene, worst; in seed order) and the setting chosen per scene."""
    base = per[0][1]
    scenes = base.scene.astype(str).values
    for _, d in per:
        assert list(d.seed) == list(base.seed) and list(d.scene.astype(str)) == list(scenes), 'grid files do not pair'
    W = np.stack([d.worst.values.astype(float) for _, d in per])
    worst = np.empty(len(base))
    pick = {}
    for sc in sorted(set(scenes)):
        m = scenes == sc
        k = int(np.argmin(W[:, m].mean(axis=1)))            # the first minimum is the earliest in grid order
        pick[sc] = per[k][0]
        worst[m] = W[k, m]
    out = pd.DataFrame(dict(seed=base.seed.values, scene=scenes, worst=worst))
    return out, pick


def frame_same(mine, fr):
    """cv_check.frame_same: one slot's geometry, exactly."""
    if not np.array_equal(mine['active'], fr['active']) or not np.array_equal(mine['bg'], fr['bg']):
        return False
    for a, b in zip(mine['ag'], fr['ag']):
        if (a is None) != (b is None):
            return False
        if a is not None and not (np.array_equal(a['dist'], b['dist']) and np.array_equal(a['depth'], b['depth'])):
            return False
    return len(mine['ag']) == len(fr['ag'])


# ---------------------------------------------------------------- unit tests (no scenes)
class Mock:
    """What the new rules read: frames (ag dist/depth, active, bg), _z, serv, _ho_timer, _ho_target, pt, sb, t, M,
    n_gnb, n_sb, gnb_cap, ho_lat, scene_id."""

    def __init__(self, frames, M, G, z=None, cap=4, hl=2):
        self.frames, self.M, self.n_gnb, self.n_sb, self.gnb_cap, self.ho_lat = frames, M, G, 3, cap, hl
        self._z = np.zeros((len(frames), M, G)) if z is None else z
        self.serv = np.zeros(M, dtype=int)
        self._ho_timer = np.zeros(M, dtype=int)
        self._ho_target = np.zeros(M, dtype=int)
        self.pt = np.full(M, float(MA.PT_MAX))
        self.sb = np.array([i % 3 for i in range(M)], dtype=int)
        self.t = 0
        self.scene_id = 'mock'


def frame(dist, depth, active=None, bg=None):
    dist, depth = np.asarray(dist, dtype=float), np.asarray(depth, dtype=float)
    M, G = dist.shape
    active = np.ones(M, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    ag = [dict(dist=dist[i].copy(), depth=depth[i].copy()) if active[i] else None for i in range(M)]
    return dict(ag=ag, active=active, bg=np.zeros(G) if bg is None else np.asarray(bg, dtype=float))


class _PPEnv:
    pass


def _pp_episode(pp, states, active=None):
    """Drive the quick-return counter through one episode of one vehicle: states[t] = (serv, timer, target)."""
    T = len(states)
    act = [True] * T if active is None else active
    e = _PPEnv()
    e.M = 1
    e.frames = [dict(active=np.array([act[t]])) for t in range(T)]
    for t, (s, tm, tg) in enumerate(states):
        e.t = t
        e.serv, e._ho_timer, e._ho_target = np.array([s]), np.array([tm]), np.array([tg])
        pp(e, None, None, None, None, 0.0)


def unit_mode():
    path = os.path.join(OUT, 'UNIT.json')
    assert not os.path.exists(path), 'unit tests already ran: archive UNIT.json first'
    rep = {'_versions': versions()}
    rng = np.random.default_rng(20261028)

    # U1: A3Gen with a TR 36.839 set and kappa = 0 equals A3Rule, slot by slot, on random episodes in which the
    #     serving station changes and vehicles go in flight, (dis)appear and are blocked
    ok = True
    for set_id in sorted(A3.SETS):
        for _ in range(3):
            T, M, G = 40, 4, 4
            frs = [frame(rng.uniform(20, 150, (M, G)), (rng.random((M, G)) < 0.3) * rng.uniform(5, 40, (M, G)),
                         active=rng.random(M) < 0.9) for _ in range(T)]
            z = rng.normal(size=(T, M, G))
            e1, e2 = Mock(frs, M, G, z), Mock(frs, M, G, z)
            r1, r2 = A3.A3Rule(set_id), AG.A3Gen(**AG.set_params(set_id))
            for t in range(T):
                serv = rng.integers(0, G, M) if rng.random() < 0.2 else e1.serv
                timer = np.where(rng.random(M) < 0.15, rng.integers(1, 4, M), 0)
                for e in (e1, e2):
                    e.t, e.serv, e._ho_timer = t, np.array(serv), np.array(timer)
                a1, g1 = r1(e1, None, None, None, None, 0.0)
                a2, g2 = r2(e2, None, None, None, None, 0.0)
                ok = ok and bool(np.array_equal(a1, a2) and np.array_equal(g1, g2))
    rep['U1_A3Gen_equals_A3Rule'] = bool(ok)

    # U2: the cell individual offset (formula, clip, kappa = 0) and its effect on the target
    e = Mock([frame(np.full((6, 4), 50.0), np.zeros((6, 4)))], 6, 4, cap=4)
    e.serv = np.array([0, 0, 0, 0, 1, 2])
    N = np.array([4.0, 1.0, 1.0, 0.0])
    fr = e.frames[0]
    O8 = AG.A3Gen(1, 3.0, 0, source='mean', kappa=8.0)._cio(e, fr)
    O16 = AG.A3Gen(1, 3.0, 0, source='mean', kappa=16.0)._cio(e, fr)
    O0 = AG.A3Gen(1, 3.0, 0, source='mean', kappa=0.0)._cio(e, fr)
    rep['U2_cio_formula'] = bool(np.allclose(O8, 2.0 * (N[:, None] - N[None, :]), rtol=0, atol=1e-12))
    rep['U2_cio_clip'] = bool(np.allclose(O16, np.clip(4.0 * (N[:, None] - N[None, :]), -10.0, 10.0), rtol=0, atol=1e-12)
                              and O16.max() == 10.0 and O16.min() == -10.0)
    rep['U2_cio_zero_without_kappa'] = bool(np.array_equal(O0, np.zeros((4, 4))))
    a, g = AG.A3Gen(1, 3.0, 0, source='mean', kappa=8.0)(e, None, None, None, None, 0.0)
    rep['U2_cio_moves_to_emptiest'] = bool(list(a[:, 0]) == [1, 1, 1, 1, 0, 0] and list(g) == [3, 3, 3, 3, -1, -1])

    # U3: Na et al.'s decision, branch by branch
    tau = PH.TAU_DBM
    R1 = np.array([[-80.0] * 5, [-70.0] * 5, [-75.0] * 5, [-60.0, -60.0, -60.0, -100.0, -60.0]])
    R1b = np.array([[-80.0] * 5, [-70.0] * 5, [-79.0, -60.0, -60.0, -60.0, -60.0], [-60.0, -60.0, -60.0, -100.0, -60.0]])
    R2 = np.array([[-93.0, -93.0, -95.0, -95.0, -95.0], [-91.0] * 5, [-85.0] * 5, [-70.0, -70.0, -70.0, -100.0, -100.0]])
    R3 = np.array([[-80.0, -80.0, -95.0, -95.0, -95.0], [-91.0, -91.0, -95.0, -95.0, -95.0],
                   [-85.0, -95.0, -95.0, -95.0, -95.0], [-70.0, -70.0, -70.0, -100.0, -100.0]])
    R4 = R3.copy()
    R4[3, 0] = -85.0
    R5 = R3.copy()
    R5[1] = [-75.0, -75.0, -75.0, -95.0, -95.0]      # prefix tie with station 3, which has the higher average
    R6 = R3.copy()
    R6[1] = [-72.0, -72.0, -72.0, -95.0, -95.0]      # prefix tie with station 3; station 1 has the higher average
    D = PH.decide
    u3 = {
        'case1_highest_average': D(R1, 0, tau, 3.0, 0.0, dwell_ok=True) == 1,
        'case1_dwell_not_met': D(R1, 0, tau, 3.0, 0.0, dwell_ok=False) == -1,
        'case1_no_station_beats_by_delta_avg': D(R1, 0, tau, 12.0, 0.0, dwell_ok=True) == -1,
        'case1_delta_init_filters': (D(R1b, 0, tau, 3.0, 0.0, dwell_ok=True) == 2
                                     and D(R1b, 0, tau, 3.0, 5.0, dwell_ok=True) == 1),
        'case2_reliable_highest_average': D(R2, 0, tau, 3.0, 0.0, dwell_ok=False) == 2,
        'case2_no_dwell_condition': D(R2, 0, tau, 3.0, 0.0, dwell_ok=True) == 2,
        'case2_allowed_subset': D(R2, 0, tau, 3.0, 0.0, dwell_ok=False, allowed=[1, 3]) == 1,
        'case2_delta_init_blocks': D(R2, 0, tau, 3.0, 10.0, dwell_ok=False) == -1,
        'case2_fallback_longest_above': D(R3, 0, tau, 3.0, 0.0, dwell_ok=False) == 3,
        'case2_fallback_start_not_better': D(R4, 0, tau, 3.0, 0.0, dwell_ok=False) == -1,
        'case2_fallback_tie_to_higher_average_a': D(R5, 0, tau, 3.0, 0.0, dwell_ok=False) == 3,
        'case2_fallback_tie_to_higher_average_b': D(R6, 0, tau, 3.0, 0.0, dwell_ok=False) == 1,
        'no_candidate': D(R2, 0, tau, 3.0, 0.0, dwell_ok=False, allowed=[0]) == -1,
    }
    rep['U3_na_rule'] = {k: bool(v) for k, v in u3.items()}

    # U4: PHRule on a hand-made episode: the dwell time, then the room count within a slot (one place per station;
    #     both vehicles want station 1 in slot 2; with the room check the second one is held back)
    def drive_ph(room):
        frs = [frame([[10.0, 100.0, 100.0, 100.0]] * 2, np.zeros((2, 4))) for _ in range(4)]
        frs += [frame([[200.0, 10.0, 100.0, 100.0]] * 2, np.zeros((2, 4))) for _ in range(8)]
        e = Mock(frs, 2, 4, cap=1, hl=1)
        e.serv = np.array([0, 2])
        pol = PH.PHRule('oracle', room=room, t_win=2, t_min=2)
        got = []
        for t in range(3):
            e.t = t
            a, g = pol(e, None, None, None, None, 0.0)
            got.append(([int(v) for v in a[:, 0]], [int(v) for v in g]))
        return got
    rep['U4_dwell_then_both_move'] = bool(drive_ph(False) == [([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([1, 1], [1, 1])])
    rep['U4_room_counted_within_slot'] = bool(drive_ph(True) == [([0, 0], [-1, -1]), ([0, 0], [-1, -1]), ([1, 0], [1, -1])])

    # U5: LSRA-M
    M, G, T = 5, 4, 3
    frs = [frame(rng.uniform(20, 300, (M, G)), (rng.random((M, G)) < 0.3) * rng.uniform(5, 40, (M, G)),
                 bg=np.full(G, 1e-9)) for _ in range(T)]
    e = Mock(frs, M, G, z=rng.normal(size=(T, M, G)), cap=4)
    gm = LM.LSRAM(k=0, source='mean', sigma_rule='depth', beta=BETA)
    worst = 0.0
    for t in range(T):
        e.t = t
        gm(e, None, None, None, None, 0.0)
        worst = max(worst, float(np.abs(gm.last_risk - LS.window_risk(e, 'now')).max()))
    rep['U5a_gate_mode_risk_equals_lsra'] = bool(worst < 1e-12)
    rep['U5a_max_risk_diff'] = worst
    rep['U5b_filter_coefficient_K4'] = bool(abs(LM.LSRAM(k=4).a - (1.0 - 0.5 ** 0.5)) < 1e-15)
    M = 4
    frs = [frame(rng.uniform(20, 300, (M, G)), (rng.random((M, G)) < 0.3) * rng.uniform(5, 40, (M, G)),
                 active=[a_, True, True, True], bg=np.full(G, 1e-9)) for a_ in (True, False, True)]
    e = Mock(frs, M, G, z=rng.normal(size=(3, M, G)), cap=4)
    lm = LM.LSRAM(k=4)
    ms = []
    for t in range(3):
        e.t = t
        ms.append({i: A3.measured_dbm(e, i) for i in range(M) if frs[t]['active'][i]})
        lm(e, None, None, None, None, 0.0)
    al = lm.a
    want1 = (1 - al) * ((1 - al) * ms[0][1] + al * ms[1][1]) + al * ms[2][1]
    rep['U5c_filter_and_restart'] = bool(np.allclose(lm.F[0], ms[2][0], rtol=0, atol=1e-12)
                                         and np.allclose(lm.F[1], want1, rtol=0, atol=1e-12))
    F = lm.F
    fr = frs[2]
    want = np.zeros((M, G))
    for i in range(M):
        for g in range(G):
            I = float(fr['bg'][g])
            for j in range(M):
                if j != i and fr['active'][j] and e.sb[j] == e.sb[i]:
                    I += 10 ** ((F[j, g] - MA.G_SERV + MA.G_SIDE) / 10.0)
            sinr = 10 * np.log10(10 ** (F[i, g] / 10.0) / (MA.N0_MW + I))
            want[i, g] = ndtr((MA.SINR_OUT_DB - sinr) / 8.2)
    rep['U5d_risk_rebuilt_by_hand'] = bool(np.allclose(lm.last_risk, want, rtol=0, atol=1e-12))
    rep['U5d_largest_risk'] = float(want.max())
    pol = LS.make_lsra('now', landing=True, minmax=False, beta=BETA)
    same = True
    for _ in range(50):
        M, G = 8, 4
        frs = [frame(rng.uniform(20, 300, (M, G)), (rng.random((M, G)) < 0.3) * rng.uniform(5, 40, (M, G)),
                     active=rng.random(M) < 0.85, bg=np.full(G, 1e-9))]
        e = Mock(frs, M, G, cap=2)
        e.serv = rng.integers(0, G, M)
        load = np.zeros(G, dtype=int)
        for i in range(M):
            if frs[0]['active'][i] and rng.random() < 0.3:
                g = int(rng.integers(0, G))
                if load[g] < e.gnb_cap:
                    e._ho_timer[i] = int(rng.integers(1, 4))
                    e._ho_target[i] = g
                    load[g] += 1
        a1, t1 = pol(e, None, None, None, None, 0.0)
        a2, t2 = LM.lsra_decide(e, LS.window_risk(e, 'now'), BETA)
        same = same and bool(np.array_equal(a1, a2) and np.array_equal(t1, t2))
    rep['U5e_decision_equals_lsra'] = bool(same)

    # U6: the quick-return counter: returns started 0 and 9 slots after the landing count; 10 and 12 do not; a vehicle
    #     that disappears and reappears starts afresh
    pp = PingPong(lambda *args: None)
    ep1 = ([(0, 0, 0), (0, 3, 1), (0, 2, 1), (0, 1, 1), (1, 0, 1), (1, 3, 0), (1, 2, 0), (1, 1, 0)] + [(0, 0, 0)] * 13
           + [(0, 3, 1), (0, 2, 1), (0, 1, 1)] + [(1, 0, 1)] * 6)
    ep2 = ([(0, 0, 0), (0, 3, 1), (0, 2, 1), (0, 1, 1)] + [(1, 0, 1)] * 10 + [(1, 3, 0), (1, 2, 0), (1, 1, 0)]
           + [(0, 0, 0)] * 11 + [(0, 3, 1), (0, 2, 1), (0, 1, 1)] + [(1, 0, 1)] * 9)
    ep3 = [(0, 0, 0), (0, 3, 1), (0, 2, 1), (0, 1, 1), (1, 0, 1), (1, 0, 1), (1, 0, 1), (1, 3, 0), (1, 2, 0),
           (1, 1, 0), (0, 0, 0), (0, 0, 0)]
    _pp_episode(pp, ep1)
    _pp_episode(pp, ep2)
    _pp_episode(pp, ep3, active=[True] * 5 + [False] + [True] * 6)
    rep['U6_quick_returns'] = bool(pp.counts == [1, 1, 0])
    rep['U6_counts'] = list(pp.counts)

    # U7: the stage-1 choices and their tie rules
    P = lambda n, o, k: dict(n_ttt=n, off=o, k=k, source='measured', kappa=0.0)
    t1 = [(5.0, 100.0, P(3, 2.0, 1)), (4.0, 300.0, P(1, 0.0, 0)), (4.0, 200.0, P(7, 6.0, 4)),
          (4.0, 200.0, P(2, 1.0, 4)), (4.0, 200.0, P(2, 1.0, 1)), (4.0, 200.0, P(2, 3.0, 0))]
    t2 = [(3.0, dict(kappa=0.0)), (2.5, dict(kappa=8.0)), (2.5, dict(kappa=4.0)), (2.7, dict(kappa=2.0))]
    rep['U7_pick_a3'] = bool(pick_a3(t1)[2] == P(2, 1.0, 1))
    rep['U7_pick_kappa'] = bool(pick_kappa(t2)[1]['kappa'] == 4.0)

    # U8: the per-scene choice of A3O
    def ep(w):
        return pd.DataFrame(dict(seed=[1, 2, 3, 4], scene=['a', 'a', 'b', 'b'], worst=w))
    out, pick = a3o_pick([('s0', ep([0.25, 0.75, 0.5, 0.5])), ('s1', ep([0.25, 0.25, 1.0, 1.0])),
                          ('s2', ep([0.5, 0.5, 0.5, 0.5]))])
    rep['U8_a3o_pick'] = bool(pick == {'a': 's1', 'b': 's0'} and list(out.worst) == [0.25, 0.25, 0.5, 0.5])

    # U9: the grid and the TR 36.839 settings
    gm_, gn_ = AG.grid('measured'), AG.grid('mean')
    rep['U9_grid_sizes'] = bool(len(gm_) == 120 and len(gn_) == 80
                                and len({AG.setting_id(p) for p in gm_ + gn_}) == 200)
    rep['U9_tr36839_slots'] = bool([AG.set_params(k)['n_ttt'] for k in sorted(A3.SETS)] == [5, 2, 2, 1, 1])

    flat = []
    for k, v in rep.items():
        if isinstance(v, bool):
            flat.append(v)
        elif isinstance(v, dict) and k.startswith('U'):
            flat += list(v.values())
    rep['ok'] = bool(all(flat))
    rep['code'] = code_md5()
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(path, 'w'), indent=1)
    status('[unit] ' + ('PASS' if rep['ok'] else 'FAIL') + ' ' +
           json.dumps({k: v for k, v in rep.items() if k not in ('_versions', 'code')}))


# ---------------------------------------------------------------- gates (real scenes; no new arm compared with LSRA)
def gate_mode():
    path = os.path.join(OUT, 'GATE.json')
    assert not os.path.exists(path), 'gates already ran: archive GATE.json first'
    u = json.load(open(os.path.join(OUT, 'UNIT.json')))
    assert u['ok'] and u['code'] == code_md5(), 'unit tests did not pass with this code'
    fzm = MHO.check_freeze()
    assert abs(float(fzm['beta']) - BETA) < 1e-15, 'beta differs from the frozen held-out test'
    seeds = seeds_test()
    sd = seeds_design()
    rep = {'_versions': versions()}
    env = env_for('heldout', 4, 'A2')
    g1 = {}
    # G1a: A3Gen with the TR 36.839 parameters reproduces the stored A3S1..A3S5 episodes (4 places, A2, 1.0 s)
    stored = pd.read_csv(A3C.cell_path('A2', 4, HL))
    for set_id in sorted(A3.SETS):
        t0 = time.time()
        rows = run(env, AG.A3Gen(**AG.set_params(set_id)), seeds[:N_GATE])
        for x in rows:
            x.update(hl=HL, arm=f'A3S{set_id}', adm='A2', cap=4)
        g1[f'A3S{set_id}'] = bool(A3C.same_as_stored(rows, stored, f'A3S{set_id}', HL))
        status(f'[G1a] A3Gen with Set {set_id} = stored A3S{set_id}: {g1[f"A3S{set_id}"]} ({time.time() - t0:.0f} s)')
    # G1b: LSRA through this runner reproduces its stored episodes (held-out scenes)
    stored_l = pd.read_csv(A3C.stored_path('A2', 4, HL))
    rows = run(env, make_arm('LSRA'), seeds[:N_GATE])
    for x in rows:
        x.update(hl=HL, arm='METHOD', adm='A2')
    g1['LSRA_heldout'] = bool(A3C.same_as_stored(rows, stored_l, 'METHOD', HL))
    status(f'[G1b] LSRA = stored (held-out): {g1["LSRA_heldout"]}')
    # G1c: the stage-1 design environment reproduces the learned-forecast study's stored LSRA episodes
    for cap in (4, 6):
        cvs = pd.read_csv(CV_STORED.format(cap=cap))
        cols = list(cvs.columns)
        ok = True
        for split in RM.SPLITS:
            r = run(make_env_design(split, ADM['A2'], cap), make_arm('LSRA'), sd[split][:20])
            for x in r:
                x.update(split=split, hl=HL, arm='HELD', adm='A2', cap=cap)
            new = AC.roundtrip(r)
            old = cvs[(cvs['arm'] == 'HELD') & (cvs['split'] == split)].set_index('seed').loc[list(new['seed'])]
            old = old.reset_index()
            ok = ok and all(c in new.columns for c in cols) and bool(LC6.frames_equal_typed(
                new[cols].reset_index(drop=True), old[cols].reset_index(drop=True)))
        g1[f'LSRA_design_C{cap}'] = bool(ok)
        status(f'[G1c] design environment, {cap} places: LSRA = stored = {ok}')
    # G1d: the straight-line forecast's geometry code rebuilds every stored slot of the 60 held-out scenes exactly
    t0 = time.time()
    n = mism = 0
    first = None
    for sid in env.scene_ids:
        obj = env._precompute(sid)
        g = CVF.SceneGeom(env, sid, obj['agents'])
        if len(g.rows) != len(obj['frames']):
            mism += 1
            first = first or f'{sid}: slot count'
            continue
        for k, frm in enumerate(obj['frames']):
            n += 1
            if not frame_same(g.recorded(k), frm):
                mism += 1
                first = first or f'{sid} slot {k}'
    g1['CV_geometry_heldout'] = dict(scenes=len(env.scene_ids), slots=n, mismatches=mism, first_mismatch=first,
                                     seconds=round(time.time() - t0, 1),
                                     ok=bool(mism == 0 and n == 6000 and len(env.scene_ids) == 60))
    status(f'[G1d] {g1["CV_geometry_heldout"]}')
    rep['G1'] = g1
    rep['G1_ok'] = bool(all((v['ok'] if isinstance(v, dict) else v) for v in g1.values()))
    # G2: LSRA-M on noise-free inputs with sigma by blockage = LSRA: the risk in every slot, the decision in every
    #     slot, and the episodes
    st = dict(worst=0.0, slots=0, differ=0)
    gmode = LM.LSRAM(k=0, source='mean', sigma_rule='depth', beta=BETA)
    ref = make_arm('LSRA')

    def checked(env_, obs, sinr, rng, ag, temp):
        a1, t1 = gmode(env_, obs, sinr, rng, ag, temp)
        st['worst'] = max(st['worst'], float(np.abs(gmode.last_risk - LS.window_risk(env_, 'now')).max()))
        a2, t2 = ref(env_, obs, sinr, rng, ag, temp)
        st['slots'] += 1
        st['differ'] += int(not (np.array_equal(a1, a2) and np.array_equal(t1, t2)))
        return a1, t1

    rows = run(env, checked, seeds[:N_GATE])
    for x in rows:
        x.update(hl=HL, arm='METHOD', adm='A2')
    eq = bool(A3C.same_as_stored(rows, stored_l, 'METHOD', HL))
    rep['G2'] = dict(max_risk_diff=st['worst'], slots=st['slots'], slots_with_other_decision=st['differ'],
                     episodes_equal_stored_LSRA=eq, ok=bool(st['worst'] < 1e-12 and st['differ'] == 0 and eq))
    status(f'[G2] {rep["G2"]}')
    # G3: every new arm runs in every test cell and every stage-1 cell (first seed); the accounting adds up
    stub = dict(n_ttt=2, off=2.0, k=1, source='measured', kappa=0.0)
    params = {'A3T': stub, 'A3TNF': dict(stub, source='mean', k=4), 'A3LB': dict(stub, kappa=4.0)}
    g3 = {}
    for arm, cap, A in test_cells():
        try:
            r = AC.roundtrip(run(env_for('heldout', cap, A), make_arm(arm, params.get(arm)), seeds[:1]))
            ok = bool(len(r) == 1 and 'pingpong' in r.columns)
            if A != 'A0':
                ok = ok and int(r.dec_over.sum()) == 0 and bool((r.starts == r.requests - r.refused).all())
            if A == 'A1':
                ok = ok and int(r.loop_violations.sum()) == 0
        except Exception as ex:                             # recorded, never hidden
            ok = f'error: {type(ex).__name__}: {ex}'
        g3[f'{arm}_C{cap}_{A}'] = ok
    for cap, A in TUNE_CELLS:
        for src in ('measured', 'mean'):
            try:
                ok = True
                for split in RM.SPLITS:
                    r = AC.roundtrip(run(env_for('design', cap, A, split), AG.A3Gen(**dict(stub, source=src)),
                                         sd[split][:1]))
                    ok = ok and len(r) == 1 and int(r.dec_over.sum()) == 0 and bool(
                        (r.starts == r.requests - r.refused).all())
                    if A == 'A1':
                        ok = ok and int(r.loop_violations.sum()) == 0
                ok = bool(ok)
            except Exception as ex:
                ok = f'error: {type(ex).__name__}: {ex}'
            g3[f'design_C{cap}_{A}_{src}'] = ok
    rep['G3'] = dict(cells=g3, ok=bool(len(g3) == 63 and all(v is True for v in g3.values())))
    status(f'[G3] ok={rep["G3"]["ok"]}')
    # G4: timing, to set the number of workers
    tm = {}
    t0 = time.time()
    for split in RM.SPLITS:
        run(env_for('design', 4, 'A2', split), AG.A3Gen(**stub), sd[split][:20])
    tm['design_A3_s_per_episode'] = (time.time() - t0) / 40.0
    for arm in ['A3T', 'A3TNF', 'A3LB', 'PHO', 'PHOR', 'LSRAM']:
        t0 = time.time()
        run(env_for('heldout', 4, 'A2'), make_arm(arm, params.get(arm)), seeds[:20])
        tm[f'{arm}_s_per_episode'] = (time.time() - t0) / 20.0
    e4 = env_for('heldout', 4, 'A2')
    t0 = time.time()
    done = []
    for s in seeds:
        e4.reset(seed=s)
        if e4.scene_id in done:
            continue
        for t in range(len(e4.frames) - 1):
            e4.t = t
            PH.cv_window(e4)
        done.append(e4.scene_id)
        if len(done) == 2:
            break
    tm['PHCV_forecast_s_per_scene'] = (time.time() - t0) / 2.0
    run(e4, make_arm('PHCV'), seeds[:20])                    # fills the forecast cache for these scenes
    t0 = time.time()
    run(e4, make_arm('PHCV'), seeds[:20])
    tm['PHCV_s_per_episode_forecast_cached'] = (time.time() - t0) / 20.0
    s_ep = {a: tm[f'{a}_s_per_episode'] for a in ['A3T', 'A3TNF', 'A3LB', 'PHO', 'PHOR', 'LSRAM']}
    s_ep['PHCV'] = tm['PHCV_s_per_episode_forecast_cached']
    stage1 = (200 * 6 + 4 * 6) * N_DESIGN * tm['design_A3_s_per_episode']
    test = sum(N_EP * s_ep[arm] for arm, cap, A in test_cells()) + 9 * 60 * tm['PHCV_forecast_s_per_scene']
    oracle = 240 * N_EP * tm['A3T_s_per_episode']
    tm['projected_cpu_hours'] = dict(stage1=round(stage1 / 3600, 2), test=round(test / 3600, 2),
                                     oracle=round(oracle / 3600, 2))
    tm['projected_wall_hours_10_workers_if_no_slowdown'] = round((stage1 + test + oracle) / 3600 / 10, 2)
    rep['G4'] = tm
    status(f'[G4] {tm}')
    rep['ok'] = bool(rep['G1_ok'] and rep['G2']['ok'] and rep['G3']['ok'])
    rep['code'] = code_md5()
    json.dump(rep, open(path, 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- dry run of the analysis (stored arms relabelled)
def dry_mode():
    global OUT
    rec_path = os.path.join(OUT_REAL, 'DRY.json')
    assert not os.path.exists(rec_path), 'dry run already made: archive DRY.json and models/_dry_bc first'
    assert not os.path.exists(OUT_DRY), 'models/_dry_bc exists: archive it first'
    MHO.check_freeze()
    OUT = OUT_DRY
    try:
        for arm, cap, A in test_cells():
            kind, src = DRY_SRC[arm]
            s = pd.read_csv(A3C.cell_path(A, cap, HL) if kind == 'a3' else A3C.stored_path(A, cap, HL))
            x = s[(s['arm'] == src) & (s['hl'] == HL)].copy()
            assert len(x) == N_EP, f'{src} has {len(x)} episodes'
            for c in ('requests', 'refused', 'loop_violations'):
                if c not in x.columns:
                    x[c] = 0
            x['arm'], x['adm'], x['cap'], x['pingpong'] = arm, A, cap, 0
            x.sort_values('seed').to_csv(test_path(arm, cap, A), index=False)
        for cap, A in ORACLE_CELLS:
            s = pd.read_csv(A3C.cell_path(A, cap, HL))
            for j, p in enumerate(AG.grid('measured')):
                x = s[(s['arm'] == f'A3S{j % 5 + 1}') & (s['hl'] == HL)].copy()
                assert len(x) == N_EP
                x.sort_values('seed').to_csv(tune_path(cap, A, p, 'oracle'), index=False)
        stub_m, stub_n = AG.grid('measured')[0], AG.grid('mean')[0]
        tuned = {f'C{cap}_{A}': dict(A3T=stub_m, A3TNF=stub_n, A3LB=dict(stub_m)) for cap, A in TUNE_CELLS}
        tpath = os.path.join(OUT, 'TUNED.json')
        json.dump(tuned, open(tpath, 'w'), indent=1)
        json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), md5=md5(tpath), note='dry run stand-in'),
                  open(os.path.join(OUT, 'TUNED_FREEZE.json'), 'w'), indent=1)
        analyse_mode(dry=True)
        res = json.load(open(os.path.join(OUT_DRY, 'BASE_RESULT.json')))
        lsra_w = {(A, cap): float(lsra_cell(A, cap).worst.mean() * 100.0) for cap, A in ORACLE_CELLS}
    finally:
        OUT = OUT_REAL
    refs = {cap: json.load(open(os.path.join(HERE, f)))['secondary']['table'] for cap, f in REF_RESULT.items()}
    a3ref = json.load(open(os.path.join(HERE, A3_RESULT)))['secondary']['table']
    sc = json.load(open(os.path.join(HERE, SC_RESULT)))['check_A']['cells']

    def ref(arm, cap, A):
        kind, src = DRY_SRC[arm]
        return a3ref[f'{A}_C{cap}_hl{HL}_{src}'] if kind == 'a3' else refs[cap][f'{A}_hl{HL}_{src}']

    mism = []
    cnt = [0]

    def cmp(name, u, v, tol=1e-9):
        cnt[0] += 1
        if not abs(float(u) - float(v)) <= tol:
            mism.append(f'{name}: {u} vs {v}')

    ok_run = not res.get('problems')
    if ok_run:
        Mx = res['mechanisms']
        for arm, cap, A in test_cells():
            r, m = ref(arm, cap, A), Mx[f'{arm}_C{cap}_{A}']
            key = f'{arm}_C{cap}_{A}'
            cmp(key + '_worst', m['worst'], r['worst'])
            cmp(key + '_ho', m['ho_per_vehicle_min'] / 600.0, r['ho_per_car_step'])
            for c in RM.CAUSES:
                cmp(f'{key}_cause_{c}', m['causes_worst'][c], r['causes_worst'][c])
            if A != 'A0':
                if 'requests_per_car_step' in r:
                    cmp(key + '_requests', m['requests_per_vehicle_min'] / 600.0, r['requests_per_car_step'])
                if 'refused_share' in r:
                    cmp(key + '_refused_share', m['refused_share'], r['refused_share'])
        for fam, A in (('HN1_A2_C4', 'A2'), ('HN2_A1_C4', 'A1')):
            for arm in FAMILY:
                cmp(f'{fam}_{arm}', res[fam]['per_arm'][arm]['mean'], ref(arm, 4, A)['minus_METHOD_same_admission']['mean'])
        for cap in (5, 6):
            for A in ('A1', 'A2'):
                for arm in FAMILY:
                    cmp(f'sec_C{cap}_{A}_{arm}', res['secondary_other_loads'][f'C{cap}_{A}'][arm]['mean'],
                        ref(arm, cap, A)['minus_METHOD_same_admission']['mean'])
        for cap in CAPS:
            for A in ('A1', 'A2'):
                L, r = res['LSRAM'][f'C{cap}_{A}'], ref('LSRAM', cap, A)
                cmp(f'LSRAM_C{cap}_{A}_worst', L['LSRAM_worst'], r['worst'])
                cmp(f'LSRAM_C{cap}_{A}_minus', L['LSRAM_minus_LSRA'], r['minus_METHOD_same_admission']['mean'])
        for cap in CAPS:
            for arm in FAMILY:
                src = DRY_SRC[arm][1]
                for q in ('T', 'P1', 'Ab', 'I'):
                    cmp(f'split_C{cap}_{arm}_{q}', res['split'][f'C{cap}_{arm}'][q]['mean'], sc[f'C{cap}_{src}_hl{HL}'][q]['mean'])
        for cap, A in ORACLE_CELLS:
            s = pd.read_csv(A3C.cell_path(A, cap, HL))
            s = s[s['hl'] == HL]
            indep = float(s.groupby(['scene', 'arm']).worst.mean().unstack().min(axis=1).mean() * 100.0)
            cmp(f'A3O_C{cap}_{A}_worst', res['A3O_bound'][f'C{cap}_{A}']['A3O_worst'], indep)
            cmp(f'A3O_C{cap}_{A}_minus', res['A3O_bound'][f'C{cap}_{A}']['minus_LSRA'], indep - lsra_w[(A, cap)])
    ok = bool(ok_run and not mism and cnt[0] >= 500)
    json.dump(dict(ok=ok, compared=cnt[0], mismatches=mism, problems=res.get('problems'),
                   result_md5=md5(os.path.join(OUT_DRY, 'BASE_RESULT.json')),
                   note='stored arms relabelled: ' + json.dumps(DRY_SRC) + '; A3O stand-in: setting j = stored A3S(j mod 5 + 1)',
                   code=code_md5()), open(rec_path, 'w'), indent=1)
    status(f'[dry] ok={ok} compared={cnt[0]} mismatches={len(mism)} problems={res.get("problems")}')


# ---------------------------------------------------------------- freeze
def freeze_files():
    """Every file the run reads, keyed by its path relative to this folder: the code actually imported (this script,
    Paper 2's modules, cv_forecast), the pools, seed lists, stored episodes and results, the frozen plan, the check
    records."""
    roots = [os.path.normcase(os.path.abspath(r)) for r in (HERE, RM.HERE, CV_DIR)]
    files = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        fa = os.path.abspath(f)
        if not os.path.isfile(fa):                         # e.g. torch's stand-in modules '_classes.py', '_ops.py'
            continue
        if any(os.path.normcase(fa).startswith(r + os.sep) for r in roots):
            files[os.path.relpath(fa, HERE).replace(os.sep, '/')] = fa
    data = [RM.POOL, MHO.POOL_H, os.path.join(MHO.OUT, 'SEEDS_heldout60.csv'), os.path.join(MHO.OUT, 'FREEZE.json')]
    data += [RM.seed_path(sp) for sp in RM.SPLITS]
    data += [A3C.stored_path(A, cap, HL) for A in ADM for cap in CAPS]
    data += [A3C.cell_path(A, cap, HL) for A in ADM for cap in CAPS]
    data += [CV_STORED.format(cap=c) for c in (4, 6)]
    data += [os.path.join(HERE, f) for f in list(REF_RESULT.values()) + [A3_RESULT, SC_RESULT]]
    data += [os.path.join(HERE, PREREG)] + [os.path.join(OUT_REAL, f) for f in ('UNIT.json', 'GATE.json', 'DRY.json')]
    for f in data:
        fa = os.path.abspath(f)
        files[os.path.relpath(fa, HERE).replace(os.sep, '/')] = fa
    return files


def freeze_mode():
    path = os.path.join(OUT_REAL, 'BASE_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    now = code_md5()
    for f in ('UNIT.json', 'GATE.json', 'DRY.json'):
        rec = json.load(open(os.path.join(OUT_REAL, f)))
        assert rec['ok'], f'{f} did not pass: no freeze'
        assert rec['code'] == now, f'{f} was made with other code: run the checks again'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen plan is not in the folder'
    MHO.check_freeze()
    files = freeze_files()
    rec = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), versions=versions(), beta=BETA, hl=HL,
               grid=dict(n_ttt=AG.GRID_NTTT, off=AG.GRID_OFF, k_measured=AG.GRID_K_MEASURED, k_mean=AG.GRID_K_MEAN,
                         kappa=AG.GRID_KAPPA),
               md5={k: md5(v) for k, v in sorted(files.items())})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written: {len(files)} files, md5 {md5(path)}')


def check_freeze():
    path = os.path.join(OUT_REAL, 'BASE_FREEZE.json')
    assert os.path.exists(path), 'not frozen'
    fz = json.load(open(path))
    bad = [k for k, v in fz['md5'].items() if md5(os.path.join(HERE, k)) != v]
    assert not bad, 'changed after the freeze: ' + ', '.join(bad)
    extra = [k for k in freeze_files() if k not in fz['md5']]
    assert not extra, 'read but not frozen: ' + ', '.join(extra)
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run: jobs
def run_design(cap, A, p):
    sd = seeds_design()
    rows = []
    for split in RM.SPLITS:
        r = run(env_for('design', cap, A, split), AG.A3Gen(**p), sd[split])
        for x in r:
            x.update(split=split, hl=HL, cap=cap, adm=A, **{f'p_{k}': v for k, v in p.items()})
        rows += r
    assert len(rows) == N_DESIGN, f'{len(rows)} design episodes'
    return rows


def run_heldout(arm, cap, A, p):
    rows = run(env_for('heldout', cap, A), make_arm(arm, p), seeds_test())
    for x in rows:
        x.update(hl=HL, arm=arm, adm=A, cap=cap)
        if p is not None:
            x.update(**{f'p_{k}': v for k, v in p.items()})
    assert len(rows) == N_EP, f'{len(rows)} test episodes'
    return rows


def tune_jobs():
    jobs = []
    for cap, A in TUNE_CELLS:
        for src in ('measured', 'mean'):
            for p in AG.grid(src):
                jobs.append(dict(key=f'tune C{cap} {A} {AG.setting_id(p)}', out=tune_path(cap, A, p),
                                 run=lambda cap=cap, A=A, p=p: run_design(cap, A, p)))
    return jobs


def lb_jobs(tuned_t):
    jobs = []
    for cap, A in TUNE_CELLS:
        base = tuned_t[f'C{cap}_{A}']['A3T']['params']
        for kappa in AG.GRID_KAPPA[1:]:
            p = dict(base, kappa=float(kappa))
            jobs.append(dict(key=f'tune_lb C{cap} {A} {AG.setting_id(p)}', out=tune_path(cap, A, p, 'tune_lb'),
                             run=lambda cap=cap, A=A, p=p: run_design(cap, A, p)))
    return jobs


def stage2_jobs(tuned):
    jobs = []
    for arm, cap, A in test_cells():
        p = tuned[f'C{cap}_{A}'][arm] if arm in A3_ARMS else None
        jobs.append(dict(key=f'test {arm} C{cap} {A}', out=test_path(arm, cap, A),
                         run=lambda arm=arm, cap=cap, A=A, p=p: run_heldout(arm, cap, A, p)))
    for cap, A in ORACLE_CELLS:
        for p in AG.grid('measured'):
            jobs.append(dict(key=f'A3O C{cap} {A} {AG.setting_id(p)}', out=tune_path(cap, A, p, 'oracle'),
                             run=lambda cap=cap, A=A, p=p: run_heldout('A3O', cap, A, p)))
    return jobs


def summarise(path, p):
    d = pd.read_csv(path)
    assert len(d) == N_DESIGN, f'{path}: {len(d)} rows'
    for k, v in p.items():
        assert (d[f'p_{k}'].astype(str) == str(v)).all(), f'{path}: column p_{k} is not {v}'
    return float(d.worst.mean() * 100.0), float(d.starts.sum())


def select_t_mode():
    out = os.path.join(OUT, 'TUNED_T.json')
    assert not os.path.exists(out), 'already selected'
    res = {}
    for cap, A in TUNE_CELLS:
        cell = {}
        for src, arm in (('measured', 'A3T'), ('mean', 'A3TNF')):
            table = [summarise(tune_path(cap, A, p), p) + (p,) for p in AG.grid(src)]
            best = pick_a3(table)
            top = sorted(table, key=lambda r: (r[0], r[1], r[2]['n_ttt'], r[2]['off'], r[2]['k']))[:5]
            cell[arm] = dict(params=best[2], design_worst=best[0], design_handovers=best[1],
                             top5=[dict(setting=AG.setting_id(r[2]), worst=r[0], handovers=r[1]) for r in top])
        res[f'C{cap}_{A}'] = cell
    json.dump(res, open(out, 'w'), indent=1)
    status('[select_t] ' + json.dumps({k: {a: (AG.setting_id(v[a]['params']), round(v[a]['design_worst'], 3))
                                           for a in v} for k, v in res.items()}))


def select_lb_mode():
    out = os.path.join(OUT, 'TUNED.json')
    fzp = os.path.join(OUT, 'TUNED_FREEZE.json')
    assert not os.path.exists(out) and not os.path.exists(fzp), 'already selected'
    tt = json.load(open(os.path.join(OUT, 'TUNED_T.json')))
    res = {}
    for cap, A in TUNE_CELLS:
        key = f'C{cap}_{A}'
        base = dict(tt[key]['A3T']['params'])
        table = [(tt[key]['A3T']['design_worst'], base)]
        for kappa in AG.GRID_KAPPA[1:]:
            p = dict(base, kappa=float(kappa))
            table.append((summarise(tune_path(cap, A, p, 'tune_lb'), p)[0], p))
        best = pick_kappa(table)
        res[key] = dict(A3T=tt[key]['A3T']['params'], A3TNF=tt[key]['A3TNF']['params'], A3LB=best[1],
                        design_worst=dict(A3T=tt[key]['A3T']['design_worst'], A3TNF=tt[key]['A3TNF']['design_worst'],
                                          A3LB=best[0]),
                        kappa_table=[dict(kappa=p['kappa'], worst=w) for w, p in table])
    json.dump(res, open(out, 'w'), indent=1)
    json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M:%S'), md5=md5(out)), open(fzp, 'w'), indent=1)
    status(f'[select_lb] TUNED.json frozen, md5 {md5(out)}: ' + json.dumps(
        {k: {a: AG.setting_id(v[a]) for a in A3_ARMS} for k, v in res.items()}))


def tuned_record():
    path = os.path.join(OUT, 'TUNED.json')
    fz = json.load(open(os.path.join(OUT, 'TUNED_FREEZE.json')))
    assert md5(path) == fz['md5'], 'TUNED.json changed after it was frozen'
    return json.load(open(path))


def run_jobs(jobs, worker, stage):
    for j in jobs:
        if os.path.exists(j['out']) or not claim(j['out'] + '.claim'):
            continue
        t0 = time.time()
        try:
            rows = j['run']()
        except Exception as ex:
            status(f'STOP [{stage}] worker {worker}: {j["key"]} failed: {type(ex).__name__}: {ex}')
            raise
        write_atomic(rows, j['out'])
        status(f'[{stage}] worker {worker}: {j["key"]} ({len(rows)} episodes, {time.time() - t0:.0f} s)')


def wait_all(paths, stage, worker):
    last_n, last_change = -1, time.time()
    while True:
        n = sum(os.path.exists(p) for p in paths)
        if n == len(paths):
            return
        if n != last_n:
            last_n, last_change = n, time.time()
        elif time.time() - last_change > NO_PROGRESS_STOP_S:
            status(f'STOP [{stage}] worker {worker}: no new output for {NO_PROGRESS_STOP_S} s ({n} of {len(paths)})')
            raise SystemExit(2)
        time.sleep(POLL_S)


def once(name, fn, out_path, worker, wait=True):
    """The first worker to claim the step runs it; the others wait for its output (or leave, if wait is False)."""
    if os.path.exists(out_path):
        return
    if claim(os.path.join(OUT, name + '.claim')):
        status(f'[{name}] worker {worker} runs it')
        fn()
        assert os.path.exists(out_path), f'{name} did not write {out_path}'
    elif wait:
        wait_all([out_path], name, worker)


def pipeline_mode(worker):
    check_freeze()
    assert abs(float(MHO.check_freeze()['beta']) - BETA) < 1e-15
    status(f'[pipeline] worker {worker} starts (pid {os.getpid()})')
    tj = tune_jobs()
    run_jobs(tj, worker, 'tune')
    wait_all([j['out'] for j in tj], 'tune', worker)
    once('select_t', select_t_mode, os.path.join(OUT, 'TUNED_T.json'), worker)
    lj = lb_jobs(json.load(open(os.path.join(OUT, 'TUNED_T.json'))))
    run_jobs(lj, worker, 'tune_lb')
    wait_all([j['out'] for j in lj], 'tune_lb', worker)
    once('select_lb', select_lb_mode, os.path.join(OUT, 'TUNED_FREEZE.json'), worker)
    sj = stage2_jobs(tuned_record())
    run_jobs(sj, worker, 'test')
    wait_all([j['out'] for j in sj], 'test', worker)
    once('analyse', analyse_mode, os.path.join(OUT, 'BASE_RESULT.json'), worker, wait=False)
    status(f'[pipeline] worker {worker} done')


# ---------------------------------------------------------------- analysis (once)
def lsra_cell(A, cap):
    s = pd.read_csv(A3C.stored_path(A, cap, HL))
    s = s[(s['arm'] == 'METHOD') & (s['hl'] == HL)].copy()
    s['scene'] = s['scene'].astype(str)
    return s.sort_values('seed').reset_index(drop=True)


def new_cell(arm, cap, A):
    d = pd.read_csv(test_path(arm, cap, A))
    d['scene'] = d['scene'].astype(str)
    return d.sort_values('seed').reset_index(drop=True)


def paired(x, y):
    assert len(x) == N_EP and len(y) == N_EP, 'a cell does not have 480 episodes'
    assert list(x.seed) == list(y.seed) and list(x.scene) == list(y.scene), 'pairing by seed and scene failed'
    return (x.worst.values - y.worst.values) * 100.0, y.scene.values


def oracle_cell(cap, A):
    per = []
    for p in AG.grid('measured'):
        d = pd.read_csv(tune_path(cap, A, p, 'oracle'))
        d['scene'] = d['scene'].astype(str)
        assert len(d) == N_EP, f'A3O {AG.setting_id(p)}: {len(d)} rows'
        per.append((AG.setting_id(p), d.sort_values('seed').reset_index(drop=True)))
    return a3o_pick(per)


def analyse_mode(dry=False):
    fz = {'written': 'dry run on relabelled stored arms'} if dry else check_freeze()
    res_path = os.path.join(OUT, 'BASE_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    tuned = tuned_record()
    status('[analyse] start')
    R = {'_freeze': fz['written'], '_versions': versions(), 'tuned': tuned}
    cells = {c: new_cell(*c) for c in test_cells()}
    lsra = {(A, cap): lsra_cell(A, cap) for A in ('A1', 'A2') for cap in CAPS}
    problems = []
    for (arm, cap, A), d in cells.items():
        y = lsra[('A2' if A == 'A0' else A, cap)]
        if not (len(d) == N_EP and list(d.seed) == list(y.seed) and list(d.scene) == list(y.scene)):
            problems.append(f'{arm} C{cap} {A}: does not pair with LSRA by seed and scene')
            continue
        if A != 'A0':
            if int(d.dec_over.sum()) != 0:
                problems.append(f'{arm} C{cap} {A}: created landing excess')
            if not bool((d.starts == d.requests - d.refused).all()):
                problems.append(f'{arm} C{cap} {A}: starts differ from requests minus refused')
        if A == 'A1' and int(d.loop_violations.sum()) != 0:
            problems.append(f'{arm} C{cap} {A}: refused requests that are not closed loops')
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

    qb = (100.0 * 0.05 / 6 / 2, 100.0 - 100.0 * 0.05 / 6 / 2)        # 99.17 %, Bonferroni over six

    def family(A):
        per = {}
        for arm in FAMILY:
            d, sc = paired(cells[(arm, 4, A)], lsra[(A, 4)])
            ci = boot(d, sc, qb)
            per[arm] = dict(mean=float(d.mean()), ci_9917=ci, beats=bool(ci[0] > 0.0))
        return dict(per_arm=per, held=bool(all(v['beats'] for v in per.values())),
                    not_shown=[a for a, v in per.items() if not v['beats']])

    R['HN1_A2_C4'] = family('A2')
    R['HN2_A1_C4'] = family('A1')
    S = {}
    for cap in (5, 6):
        for A in ('A1', 'A2'):
            S[f'C{cap}_{A}'] = {}
            for arm in FAMILY:
                d, sc = paired(cells[(arm, cap, A)], lsra[(A, cap)])
                S[f'C{cap}_{A}'][arm] = dict(mean=float(d.mean()), ci_95=boot(d, sc))
    R['secondary_other_loads'] = S
    O = {}
    for cap, A in ORACLE_CELLS:
        a3o, pick = oracle_cell(cap, A)
        d, sc = paired(a3o, lsra[(A, cap)])
        O[f'C{cap}_{A}'] = dict(A3O_worst=float(a3o.worst.mean() * 100.0),
                                LSRA_worst=float(lsra[(A, cap)].worst.mean() * 100.0),
                                minus_LSRA=float(d.mean()), ci_95=boot(d, sc),
                                settings_used=len(set(pick.values())), setting_per_scene=pick)
    R['A3O_bound'] = O
    LMr = {}
    for cap in CAPS:
        for A in ('A1', 'A2'):
            lm = cells[('LSRAM', cap, A)]
            d, sc = paired(lm, lsra[(A, cap)])
            best = min(FAMILY, key=lambda a: float(cells[(a, cap, A)].worst.mean()))
            d2, sc2 = paired(cells[(best, cap, A)], lm)
            LMr[f'C{cap}_{A}'] = dict(LSRAM_worst=float(lm.worst.mean() * 100.0), LSRAM_minus_LSRA=float(d.mean()),
                                      ci_95=boot(d, sc), best_new_baseline=best, best_minus_LSRAM=float(d2.mean()),
                                      ci_95_best=boot(d2, sc2))
    R['LSRAM'] = LMr
    T = {}
    for cap in CAPS:
        l1, l2 = lsra[('A1', cap)], lsra[('A2', cap)]
        for arm in FAMILY:
            b1, b2 = cells[(arm, cap, 'A1')], cells[(arm, cap, 'A2')]
            for x in (b1, b2, l1):
                assert list(x.seed) == list(l2.seed) and list(x.scene) == list(l2.scene), 'split: pairing failed'
            jb1, jb2, jl1, jl2 = (x.worst.values * 100.0 for x in (b1, b2, l1, l2))
            sc = l2.scene.values
            q = dict(T=jb1 - jl2, P1=jb1 - jl1, Ab=jb1 - jb2)
            q['I'] = q['T'] - q['P1'] - q['Ab']
            e = {k: dict(mean=float(v.mean()), ci_95=boot(v, sc)) for k, v in q.items()}
            P2, Al = float((jb2 - jl2).mean()), float((jl1 - jl2).mean())
            e['levels'] = dict(B_A1=float(jb1.mean()), B_A2=float(jb2.mean()), L_A1=float(jl1.mean()), L_A2=float(jl2.mean()))
            e['P2'], e['Al'] = P2, Al
            e['share_planner'] = (e['P1']['mean'] + P2) / 2.0
            e['share_admission'] = (e['Ab']['mean'] + Al) / 2.0
            if arm in A3_ARMS:
                e['same_setting_under_A1_and_A2'] = bool(tuned[f'C{cap}_A1'][arm] == tuned[f'C{cap}_A2'][arm])
            T[f'C{cap}_{arm}'] = e
    R['split'] = T
    Mx = {}
    for (arm, cap, A), d in cells.items():
        wa = np.maximum(d.worst_act.values, 1.0)
        w = d.worst.values * 100.0
        e = dict(worst=float(w.mean()), ci_95=boot(w, d.scene.values),
                 ho_per_vehicle_min=float(d.starts.sum() / d.p_act.sum() * 600.0),
                 pingpong_per_vehicle_min=float(d.pingpong.sum() / d.p_act.sum() * 600.0),
                 pingpong_share_of_handovers=float(d.pingpong.sum() / max(d.starts.sum(), 1)),
                 causes_worst={k: float((d[f'w_{k}'].values / wa).mean() * 100.0) for k in RM.CAUSES})
        if A != 'A0':
            e['requests_per_vehicle_min'] = float(d.requests.sum() / d.p_act.sum() * 600.0)
            e['refused_share'] = float(d.refused.sum() / max(d.requests.sum(), 1))
        Mx[f'{arm}_C{cap}_{A}'] = e
    R['mechanisms'] = Mx
    R['last_boot_seed'] = sb[0] - 1
    g = dict(
        HN1_held=R['HN1_A2_C4']['held'],
        HN2_not_held=not R['HN2_A1_C4']['held'],
        A3T_A2_C4_above_6pct=Mx['A3T_C4_A2']['worst'] > 6.0,
        A3O_C4_above_LSRA=O['C4_A2']['minus_LSRA'] > 0.0,
        A3O_C5_within_1pp_of_LSRA=abs(O['C5_A2']['minus_LSRA']) <= 1.0,
        LSRAM_worse_than_LSRA_by_0p5_to_2pp_C4_A2=0.5 <= LMr['C4_A2']['LSRAM_minus_LSRA'] <= 2.0,
        LSRAM_beats_A3T_C4_A2=LMr['C4_A2']['LSRAM_worst'] < Mx['A3T_C4_A2']['worst'],
        PHO_fewer_handovers_than_every_A3_arm_C4_A2=Mx['PHO_C4_A2']['ho_per_vehicle_min'] < min(
            Mx[f'{a}_C4_A2']['ho_per_vehicle_min'] for a in A3_ARMS),
        PHO_most_requests_refused_C4_A2=Mx['PHO_C4_A2']['refused_share'] > 0.5)
    R['guesses'] = {k: bool(v) for k, v in g.items()}
    h1 = R['HN1_A2_C4']
    R['decision'] = dict(
        HN1=("held: with coordinated admission at full load, LSRA beats the tuned A3, the load-balancing A3 and "
             "Na et al.'s predictive handover, even with a perfect forecast" if h1['held'] else
             'not held against ' + ', '.join(h1['not_shown']) + ': named, and the claim is narrowed to the others'),
        HN2='reported as it falls, with the planner/admission split: ' + (
            'held' if R['HN2_A1_C4']['held'] else 'not held against ' + ', '.join(R['HN2_A1_C4']['not_shown'])))
    R['ok'] = True
    json.dump(R, open(res_path, 'w'), indent=1)
    write_txt(R, os.path.join(OUT, 'BASE_RESULT.txt'))
    status(f'[analyse] done; HN1 held={R["HN1_A2_C4"]["held"]}, HN2 held={R["HN2_A1_C4"]["held"]}')


def write_txt(R, path):
    def ci(v):
        return f'[{v[0]:+.2f}, {v[1]:+.2f}]'
    L = ["NEW BASELINES FOR LSRA: tuned A3, load-balancing A3, Na et al.'s predictive handover, LSRA-M",
         '(60 held-out DAIR scenes, 480 episodes per cell, 1.0 s; J = worst-vehicle RLF in %; differences are the',
         'baseline minus LSRA in percentage points, paired by episode; intervals are scene bootstraps)', '',
         'settings chosen on the design scenes (TUNED.json):']
    for k, v in R['tuned'].items():
        L.append(f'  {k}: ' + '  '.join(f'{a} {AG.setting_id(v[a])}' for a in A3_ARMS))
    for key, title in (('HN1_A2_C4', 'HN1, coordinated admission (A2), 4 places, 99.17 %'),
                       ('HN2_A1_C4', 'HN2, per-request admission (A1), 4 places, 99.17 %')):
        L.append('')
        L.append(title + ':')
        for a, v in R[key]['per_arm'].items():
            L.append(f"  {a}: {v['mean']:+.2f} {ci(v['ci_9917'])} -> {'BEATS' if v['beats'] else 'not shown'}")
        L.append('  ' + ('HELD' if R[key]['held'] else 'NOT HELD'))
    L.append('')
    L.append('the same at 5 and 6 places (95 %):')
    for k, v in R['secondary_other_loads'].items():
        L.append(f'  {k}: ' + '  '.join(f"{a} {w['mean']:+.2f} {ci(w['ci_95'])}" for a, w in v.items()))
    L.append('')
    L.append('A3O, the best measured A3 setting per test scene in hindsight (a bound, A2):')
    for k, v in R['A3O_bound'].items():
        L.append(f"  {k}: A3O {v['A3O_worst']:.2f} %, LSRA {v['LSRA_worst']:.2f} %, A3O minus LSRA "
                 f"{v['minus_LSRA']:+.2f} {ci(v['ci_95'])}; {v['settings_used']} settings used")
    L.append('')
    L.append('LSRA-M, LSRA fed only measurement reports:')
    for k, v in R['LSRAM'].items():
        L.append(f"  {k}: {v['LSRAM_worst']:.2f} %; minus LSRA {v['LSRAM_minus_LSRA']:+.2f} {ci(v['ci_95'])}; "
                 f"best new baseline {v['best_new_baseline']} minus LSRA-M {v['best_minus_LSRAM']:+.2f} {ci(v['ci_95_best'])}")
    L.append('')
    L.append('today (baseline, A1) against the proposal (LSRA, A2): T = P1 (planner alone) + Ab (admission alone) + I (both):')
    for k, v in R['split'].items():
        L.append(f"  {k}: T {v['T']['mean']:+.2f} {ci(v['T']['ci_95'])}  P1 {v['P1']['mean']:+.2f} {ci(v['P1']['ci_95'])}  "
                 f"Ab {v['Ab']['mean']:+.2f} {ci(v['Ab']['ci_95'])}  I {v['I']['mean']:+.2f} {ci(v['I']['ci_95'])}")
    L.append('')
    L.append('levels and mechanisms: J % [95 %] | handovers per vehicle-minute | quick returns per vehicle-minute | '
             'refused share | causes of the worst vehicle (%)')
    for k, v in R['mechanisms'].items():
        L.append(f"  {k}: {v['worst']:.2f} [{v['ci_95'][0]:.2f}, {v['ci_95'][1]:.2f}] | {v['ho_per_vehicle_min']:.3f} | "
                 f"{v['pingpong_per_vehicle_min']:.4f} | {v.get('refused_share', float('nan')):.3f} | "
                 + ' '.join(f'{c} {x:.1f}' for c, x in v['causes_worst'].items()))
    L.append('')
    L.append(f"guesses recorded before the run: {R['guesses']}")
    L.append(f"decision: {R['decision']}")
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['unit', 'gate', 'dry', 'freeze', 'pipeline'])
    ap.add_argument('--worker', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT_REAL, exist_ok=True)
    if a.mode == 'unit':
        unit_mode()
    elif a.mode == 'gate':
        gate_mode()
    elif a.mode == 'dry':
        dry_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    else:
        pipeline_mode(a.worker)


if __name__ == '__main__':
    main()
