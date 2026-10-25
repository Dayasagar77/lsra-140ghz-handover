#!/usr/bin/env python3
"""The 3GPP A3 baseline: the five TR 36.839 parameter sets against LSRA (DRAFT, 25 Oct 2026; nothing run).

Pre-registration: PREREG_MA_A3Baseline_DRAFT_v1_20261025.md (needs Daya's OK).

Same 60 held-out scenes, seed list, environment class (paper2ma_env_admit.AdmissionEnv, built exactly as in
load_check_c6.make_env_cap), harness (admission_check.run_cell -> method_harness.run_eval_m) and admission
models as the earlier checks. New: five arms A3S1..A3S5 (a3_rule.A3Rule), run under A0, A1 and A2 at 4, 5 and
6 places and at 0.2, 0.5, 1.0 and 1.5 s (36 cells). LSRA is not run again: each A3 episode is paired, by seed,
with LSRA's stored episode of the same cell, and gate G0 first shows that this runner reproduces stored
episodes exactly.

Before Daya's OK (no A3 episode on the test scenes):
  --mode unit     U1-U9: the A3 rule on hand-made cases (no scenes)
  --mode dry      the registered analysis on stored episodes of R0, R1, R1i, R2 and R3 relabelled as
                  A3S1..A3S5 (into models/_dry_a3); it must reproduce the earlier checks' numbers to 1e-9
  --mode gate0    G0: stored episodes reproduced exactly (R1 and METHOD, first 40 seeds, 1.0 s, every
                  admission model at every number of places, every stored column)
After Daya's OK, before the freeze:
  --mode gate     G1: the A3 measurement equals the environment's own received power; G2: the A3 arms are
                  deterministic and the request accounting adds up; G3: every A3 arm runs in every cell
                  (first seed only; only pass or fail is recorded)
  --mode freeze   writes A3_FREEZE.json once unit, dry, gate0 and gate passed
The run:
  --mode test --adm A --cap C --hl H   the five A3 arms in one cell; each cell runs once
  --mode analyse                       the registered analysis, once
"""
import os, sys, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import load_check_c6 as LC6
import paper2ma_env as MA
import a3_rule as A3

HERE = RM.HERE
OUT_REAL = os.path.join(HERE, 'models', 'a3_check')
OUT_DRY = os.path.join(HERE, 'models', '_dry_a3')
OUT = OUT_REAL
ADM = {'A0': 'none', 'A1': 'per_request', 'A2': 'coordinated'}
CAPS = [4, 5, 6]
HLS = MHO.HLS
A3_ARMS = [f'A3S{k}' for k in sorted(A3.SETS)]
DRY_BASE = ['R0', 'R1', 'R1i', 'R2', 'R3']          # stored arms relabelled as A3S1..A3S5 in the dry run
OLD_RESULT = {4: 'models/admission_check/ADMIT_RESULT.json', 5: 'models/load_check/LOAD_RESULT.json',
              6: 'models/load_check_c6/LOAD6_RESULT.json'}
N_GATE = 40
N_G1 = 5
N_G2 = 10
N_EP = 480
PREREG = 'PREREG_MA_A3Baseline_FROZEN_20261025.md'


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def code_md5():
    """md5 of the two new files, written into every check record and compared at the freeze."""
    return {f: md5(os.path.join(HERE, f)) for f in ('a3_rule.py', 'a3_check.py')}


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def stored_path(A, cap, hl):
    """LSRA's stored episodes for one cell (the frozen runs of the earlier checks)."""
    if cap == 4:
        if A == 'A0':
            return os.path.join(MHO.OUT, f'HELDOUT_EPISODES_hl{hl}.csv')
        return os.path.join(AC.OUT, f'ADMIT_EPISODES_{A}_hl{hl}.csv')
    if cap == 5:
        return os.path.join(HERE, 'models', 'load_check', f'LOAD_EPISODES_C5_{A}_hl{hl}.csv')
    if cap == 6:
        return os.path.join(HERE, 'models', 'load_check_c6', f'LOAD_EPISODES_C6_{A}_hl{hl}.csv')
    raise ValueError(cap)


def read_files():
    return [os.path.relpath(stored_path(A, c, h), HERE).replace(os.sep, '/')
            for A in ADM for c in CAPS for h in HLS]


def cell_path(A, cap, hl):
    return os.path.join(OUT, f'A3_EPISODES_C{cap}_{A}_hl{hl}.csv')


def run_arms(env, seeds, A, cap, hl, out_path=None, quiet=False):
    rows = []
    for name, fn in A3.arms():
        t0 = time.time()
        r = AC.run_cell(env, fn, seeds)
        for x in r:
            x.update(hl=hl, arm=name, adm=A, cap=cap)
        rows += r
        if out_path:
            pd.DataFrame(rows).to_csv(out_path + '.partial', index=False)
        if not quiet:
            status(f'cell C{cap} {A} hl={hl} {name}: {len(r)} episodes ({time.time() - t0:.0f} s)')
    return rows


# ---------------------------------------------------------------- unit tests (no scenes)
class MockEnv:
    """Just what a3_rule reads: frames (ag dist/depth, active), _z, serv, _ho_timer, pt, t, M, n_gnb."""

    def __init__(self, z, depth=None, dist=100.0, active=None):
        z = np.asarray(z, dtype=float)
        self.T, self.M, self.n_gnb = z.shape
        self._z = z
        dep = np.zeros_like(z) if depth is None else np.asarray(depth, dtype=float)
        act = np.ones((self.T, self.M), dtype=bool) if active is None else np.asarray(active, dtype=bool)
        self.frames = []
        for t in range(self.T):
            ag = [dict(dist=np.full(self.n_gnb, dist), depth=dep[t, i]) if act[t, i] else None
                  for i in range(self.M)]
            self.frames.append(dict(ag=ag, active=act[t].copy()))
        self.serv = np.zeros(self.M, dtype=int)
        self._ho_timer = np.zeros(self.M, dtype=int)
        self.pt = np.full(self.M, float(MA.PT_MAX))
        self.t = 0


def z_for(diff_db):
    """Shadowing draws (clear links) that make station g stronger than station 0 by diff_db[g] dB."""
    return -np.asarray(diff_db, dtype=float) / MA.CH.SIG_LOS


def steady(T, diff_db):
    z = np.zeros((T, 1, len(diff_db)))
    z[:, 0, :] = z_for(diff_db)
    return z


def drive(rule, env, steps):
    """Call the rule at t = 0 .. steps-1; returns the (request, target) of vehicle 0 per slot."""
    out = []
    for t in range(steps):
        env.t = t
        a, tgt = rule(env, None, None, None, None, 0.0)
        out.append((int(a[0, 0]), int(tgt[0])))
    return out


def first_request(s):
    r = [t for t, (q, _) in enumerate(s) if q == 1]
    return r[0] if r else None


def unit_mode():
    MHO.check_freeze()
    R = {}
    a4 = 1.0 - 0.5 ** 0.5                                  # K = 4: a = 1/2, per 0.1 s slot 1 - (1/2)^(1/2)
    # U1: the slot conversions
    R['U1_ttt_slots_5_2_2_1_1'] = [A3.ttt_slots(A3.SETS[k][0]) for k in sorted(A3.SETS)] == [5, 2, 2, 1, 1]
    R['U1_ttt_slots_exact_multiples'] = [A3.ttt_slots(x) for x in (0.0, 100.0, 200.0, 99.0)] == [1, 2, 3, 1]
    al = {k: A3.alpha_slot(A3.SETS[k][2]) for k in sorted(A3.SETS)}
    R['U1_alpha'] = bool(abs(al[1] - a4) < 1e-12 and abs(al[2] - a4) < 1e-12 and abs(al[5] - 1.0) < 1e-12
                         and abs(al[3] - (1.0 - (1.0 - 2 ** -0.25) ** 0.5)) < 1e-12 and al[3] == al[4])
    # U2: a neighbour 6 dB stronger from the start; Set 1 (TTT 5 slots, offset 3) requests at slot 4, to it
    s = drive(A3.A3Rule(1), MockEnv(steady(12, [0, 6, -10, -10])), 12)
    R['U2_set1_requests_at_slot4'] = (s[:4] == [(0, -1)] * 4 and s[4] == (1, 1))
    # U3: a one-slot dip at slot 3
    z = steady(12, [0, 6, -10, -10]); z[3, 0, :] = z_for([0, 1, -10, -10])
    s = drive(A3.A3Rule(2), MockEnv(z), 12)                # Set 2 (K 4): filtered 6 - 5 a4 = 4.54 dB > 3
    R['U3a_set2_filter_rides_out_a_dip'] = (s[0] == (0, -1) and all(x == (1, 1) for x in s[1:]))
    z = steady(12, [0, 6, -10, -10]); z[3, 0, :] = z_for([0, -6, -10, -10])
    s = drive(A3.A3Rule(5), MockEnv(z), 12)                # Set 5: no filter, TTT 1, offset -1
    R['U3b_set5_follows_each_slot'] = (s[2] == (1, 1) and s[3] == (0, -1) and s[4] == (1, 1))
    s = drive(A3.A3Rule(1), MockEnv(z), 12)                # Set 1: filtered 6 - 12 a4 = 2.49 dB < 3 breaks the count
    R['U3c_set1_dip_restarts_count_request_at_slot8'] = (first_request(s) == 8)
    # U4: Set 5 (offset -1): a neighbour 0.5 dB weaker triggers at once, 1.5 dB weaker never
    R['U4a_set5_weaker_by_0p5_triggers'] = (drive(A3.A3Rule(5), MockEnv(steady(4, [0, -0.5, -10, -10])), 4)[0]
                                            == (1, 1))
    R['U4b_set5_weaker_by_1p5_never'] = all(q == 0 for q, _ in
                                            drive(A3.A3Rule(5), MockEnv(steady(4, [0, -1.5, -10, -10])), 4))
    # U5: two neighbours triggered: the strongest is chosen
    R['U5_strongest_triggered'] = drive(A3.A3Rule(4), MockEnv(steady(6, [0, 5, 8, -10])), 6)[0] == (1, 2)
    # U6: in flight -> no request; after landing on the new station nothing is better by the offset
    env = MockEnv(steady(10, [0, 6, -10, -10])); rule = A3.A3Rule(4)
    out = []
    for t in range(10):
        env.t = t
        if t == 1:
            env._ho_timer[0] = 3                            # as the environment after a request at slot 0, L = 3
        if 1 < t <= 4:
            env._ho_timer[0] -= 1
            if env._ho_timer[0] == 0:
                env.serv[0] = 1
        a, tgt = rule(env, None, None, None, None, 0.0)
        out.append((int(a[0, 0]), int(tgt[0])))
    R['U6_no_request_in_flight_then_none_from_new_station'] = (out[0] == (1, 1)
                                                               and all(o == (0, -1) for o in out[1:]))
    # U7: a 5 dB step at slot 2 through the K = 4 filter: 1.46, 2.50, 3.23 dB at slots 2, 3, 4, so the
    # condition (> 3 dB) first holds at slot 4 and Set 1 requests on its fifth consecutive slot, at slot 8
    z = steady(12, [0, 0, -10, -10]); z[2:, 0, :] = z_for([0, 5, -10, -10])
    s = drive(A3.A3Rule(1), MockEnv(z), 12)
    k = next(n for n in range(1, 20) if 5.0 * (1.0 - (1.0 - a4) ** n) > 3.0)
    R['U7_filter_step_then_ttt_request_at_slot8'] = (k == 3 and first_request(s) == 8)
    # U8: a vehicle that leaves (slot 3) and reappears restarts its filter and count: Set 1 requests at slot 8
    act = np.ones((14, 1), dtype=bool); act[3, 0] = False
    s = drive(A3.A3Rule(1), MockEnv(steady(14, [0, 6, -10, -10]), active=act), 14)
    R['U8_reappearance_restarts'] = (s[3] == (0, -1) and first_request(s) == 8)
    # U9: slots out of order are refused; t = 0 starts a new episode (state cleared)
    rule = A3.A3Rule(2); env = MockEnv(steady(6, [0, 6, -10, -10]))
    drive(rule, env, 3)
    try:
        env.t = 4
        rule(env, None, None, None, None, 0.0)
        R['U9a_out_of_order_refused'] = False
    except RuntimeError:
        R['U9a_out_of_order_refused'] = True
    s = drive(rule, MockEnv(steady(6, [0, 6, -10, -10])), 6)
    R['U9b_new_episode_at_t0'] = (s[0] == (0, -1) and s[1] == (1, 1))
    R['ok'] = bool(all(v for k2, v in R.items() if k2 != 'ok'))
    R['code'] = code_md5()
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'UNIT.json'), 'w'), indent=1)
    status(f'[unit] {R}')


# ---------------------------------------------------------------- dry run of the analysis (stored arms only)
def dry_mode():
    """The registered analysis on stored episodes of R0, R1, R1i, R2 and R3 relabelled as A3S1..A3S5.
    No A3 episode is used. Every level, handover rate, cause share, refused share and difference from LSRA must
    equal the earlier checks' results, and the two family means must equal HR2's and HR1's."""
    global OUT
    MHO.check_freeze()
    stored_md5 = {f: md5(os.path.join(HERE, f)) for f in read_files()}
    OUT = OUT_DRY
    os.makedirs(OUT, exist_ok=True)
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                p = cell_path(A, cap, hl)
                assert not os.path.exists(p), 'dry run already made: archive models/_dry_a3 first'
                s = pd.read_csv(stored_path(A, cap, hl))
                parts = []
                for k, b in zip(A3_ARMS, DRY_BASE):
                    x = s[(s['arm'] == b) & (s['hl'] == hl)].copy()
                    assert len(x) == N_EP, f'{b} has {len(x)} episodes in {p}'
                    x['arm'] = k
                    for c, v in (('requests', 0), ('refused', 0), ('loop_violations', 0), ('adm', A), ('cap', cap)):
                        if c not in x.columns:
                            x[c] = v
                    parts.append(x)
                pd.concat(parts, ignore_index=True).to_csv(p, index=False)
    analyse(dry=True)
    res = json.load(open(os.path.join(OUT, 'A3_RESULT.json')))
    OUT = OUT_REAL
    mism, n = [], 0
    if 'secondary' in res:
        T = res['secondary']['table']
        for cap, f in OLD_RESULT.items():
            OT = json.load(open(os.path.join(HERE, f)))['secondary']['table']
            for A in ADM:
                for hl in HLS:
                    for k, b in list(zip(A3_ARMS, DRY_BASE)) + [('METHOD', 'METHOD')]:
                        new, old = T[f'{A}_C{cap}_hl{hl}_{k}'], OT[f'{A}_hl{hl}_{b}']
                        pairs = [('worst', new['worst'], old['worst']),
                                 ('ho', new['ho_per_car_step'], old['ho_per_car_step'])]
                        pairs += [(f'cause_{c}', new['causes_worst'][c], old['causes_worst'][c]) for c in RM.CAUSES]
                        if k != 'METHOD':
                            pairs.append(('minus', new['minus_METHOD_same_admission']['mean'],
                                          old['minus_METHOD_same_admission']['mean']))
                        for key in ('requests_per_car_step', 'refused_share'):
                            if key in new and key in old:
                                pairs.append((key, new[key], old[key]))
                        for name, u, v in pairs:
                            n += 1
                            if not abs(float(u) - float(v)) <= 1e-9:
                                mism.append(f'{A}_C{cap}_hl{hl}_{k}_{name}')
        AR = json.load(open(os.path.join(AC.OUT, 'ADMIT_RESULT.json')))
        for key_new, key_old in (('HA1_A2', 'HR2_A2'), ('HA2_A1', 'HR1_A1')):
            for k, b in zip(A3_ARMS[1:], DRY_BASE[1:]):
                n += 1
                if not abs(res[key_new]['per_set'][k]['mean'] - AR[key_old]['per_baseline'][b]['mean']) <= 1e-9:
                    mism.append(f'{key_new}_{k}')
    ok = bool('secondary' in res and res['checks']['ok'] and n >= 1500 and not mism)
    json.dump(dict(ok=ok, compared=n, mismatches=mism, checks=res['checks'],
                   result_md5=md5(os.path.join(OUT_DRY, 'A3_RESULT.json')),
                   note='stored R0, R1, R1i, R2, R3 relabelled as A3S1..A3S5; no A3 episode used',
                   code=code_md5(), stored_md5=stored_md5),
              open(os.path.join(OUT_REAL, 'DRY.json'), 'w'), indent=1)
    status(f'[dry] ok={ok} compared={n} mismatches={len(mism)} checks={res["checks"]}')


# ---------------------------------------------------------------- gates (scenes)
def same_as_stored(rows, stored, name, hl):
    """Every column of the stored file, exactly, for the seeds just run."""
    new = AC.roundtrip(rows)
    cols = list(stored.columns)
    if any(c not in new.columns for c in cols):
        return False
    old = stored[(stored['arm'] == name) & (stored['hl'] == hl)]
    old = old.set_index('seed').loc[list(new['seed'])].reset_index()
    return bool(LC6.frames_equal_typed(new[cols].reset_index(drop=True), old[cols].reset_index(drop=True)))


def gate0_mode():
    """G0: this runner reproduces the stored episodes exactly, for every admission model and number of places."""
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    g0 = {}
    for A in ADM:
        for cap in CAPS:
            stored = pd.read_csv(stored_path(A, cap, 10))
            env = LC6.make_env_cap(pool, 10, ADM[A], cap)
            for name, fn in MHO.arms(fzm['beta']):
                if name not in ('R1', 'METHOD'):
                    continue
                t0 = time.time()
                rows = AC.run_cell(env, fn, seeds[:N_GATE])
                for x in rows:
                    x.update(hl=10, arm=name, adm=A, cap=cap)
                key = f'C{cap}_{A}_{name}'
                g0[key] = same_as_stored(rows, stored, name, 10)
                status(f'[gate0] {key}: identical to the stored run = {g0[key]} ({time.time() - t0:.0f} s)')
    rep = dict(_versions=AC.versions(), n_seeds=N_GATE, hl=10, G0_reproduces_stored=g0,
               ok=bool(len(g0) == 18 and all(g0.values())), code=code_md5())
    json.dump(rep, open(os.path.join(OUT, 'GATE0.json'), 'w'), indent=1)
    status(f'[gate0] ok={rep["ok"]}')


def gate_mode():
    MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    rep = {'_versions': AC.versions()}
    # G1: the A3 measurement equals the environment's own received power, every slot, vehicle and station
    env = LC6.make_env_cap(pool, 10, ADM['A2'], 4)
    rule = A3.A3Rule(2)
    st = dict(worst=0.0, n=0)

    def checked(env_, obs, sinr, rng, ag, temp):
        fr = env_.frames[env_.t]
        for i in range(env_.M):
            if not fr['active'][i]:
                continue
            m = A3.measured_dbm(env_, i)
            for g in range(env_.n_gnb):
                ref = env_._desired_dbm(i, g) - env_._z[env_.t, i, g] * MA._sigma(fr['ag'][i]['depth'][g])
                st['worst'] = max(st['worst'], abs(float(m[g]) - float(ref)))
                st['n'] += 1
        return rule(env_, obs, sinr, rng, ag, temp)

    AC.run_cell(env, checked, seeds[:N_G1])
    rep['G1_values_compared'] = st['n']
    rep['G1_max_abs_diff_db'] = st['worst']
    rep['G1_ok'] = bool(st['n'] > 0 and st['worst'] < 1e-9)
    status(f'[gate G1] {st["n"]} values; largest difference from the environment received power {st["worst"]:.3g} dB')
    # G2: deterministic, and the request accounting adds up (A1, 4 places, 1.0 s, 10 seeds, twice)
    env = LC6.make_env_cap(pool, 10, ADM['A1'], 4)
    r1 = AC.roundtrip(run_arms(env, seeds[:N_G2], 'A1', 4, 10, quiet=True))
    r2 = AC.roundtrip(run_arms(env, seeds[:N_G2], 'A1', 4, 10, quiet=True))
    rep['G2_deterministic'] = bool(LC6.frames_equal_typed(r1, r2))
    rep['G2_starts_eq_requests_minus_refused'] = bool((r1.starts == r1.requests - r1.refused).all())
    rep['G2_created_excess'] = int(r1.dec_over.sum())
    rep['G2_loop_violations'] = int(r1.loop_violations.sum())
    rep['G2_every_arm_requests'] = bool(all(int(r1[r1.arm == a].requests.sum()) > 0 for a in A3_ARMS))
    rep['G2_ok'] = bool(rep['G2_deterministic'] and rep['G2_starts_eq_requests_minus_refused']
                        and rep['G2_created_excess'] == 0 and rep['G2_loop_violations'] == 0
                        and rep['G2_every_arm_requests'])
    status(f'[gate G2] ok={rep["G2_ok"]}')
    # G3: every A3 arm runs in every cell (first seed); only pass or fail is recorded
    g3 = {}
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                env = LC6.make_env_cap(pool, hl, ADM[A], cap)
                try:
                    r = AC.roundtrip(run_arms(env, seeds[:1], A, cap, hl, quiet=True))
                    ok = len(r) == len(A3_ARMS)
                    if A != 'A0':
                        ok = ok and int(r.dec_over.sum()) == 0 and bool((r.starts == r.requests - r.refused).all())
                    if A == 'A1':
                        ok = ok and int(r.loop_violations.sum()) == 0
                except Exception as e:                       # recorded, never hidden
                    ok = f'error: {type(e).__name__}: {e}'
                g3[f'C{cap}_{A}_hl{hl}'] = ok
    rep['G3_every_cell_runs'] = g3
    rep['G3_ok'] = bool(len(g3) == 36 and all(v is True for v in g3.values()))
    status(f'[gate G3] ok={rep["G3_ok"]}')
    rep['ok'] = bool(rep['G1_ok'] and rep['G2_ok'] and rep['G3_ok'])
    rep['code'] = code_md5()
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_files():
    return (['a3_rule.py', 'a3_check.py', 'load_check_c6.py', 'admission_check.py', 'paper2ma_env_admit.py',
             'paper2ma_env_attach.py', 'paper2ma_env.py', 'paper2_env.py', 'channel_model_nyu.py',
             'method_harness.py', 'method_heldout.py', 'remeasure_ma.py', 'lsra.py', PREREG,
             'models/a3_check/UNIT.json', 'models/a3_check/DRY.json', 'models/a3_check/GATE0.json',
             'models/a3_check/GATE.json', 'models/method_heldout/FREEZE.json',
             'models/admission_check/ADMIT_FREEZE.json', 'models/load_check/LOAD_FREEZE.json',
             'models/load_check_c6/LOAD6_FREEZE.json'] + read_files())


def freeze_mode():
    now = code_md5()
    for f in ('UNIT.json', 'DRY.json', 'GATE0.json', 'GATE.json'):
        rec = json.load(open(os.path.join(OUT, f)))
        assert rec['ok'], f'{f} did not pass: no freeze'
        assert rec['code'] == now, f'{f} was made with other code: run the checks again'
    dry = json.load(open(os.path.join(OUT, 'DRY.json')))
    assert dry['stored_md5'] == {f: md5(os.path.join(HERE, f)) for f in read_files()}, 'a stored file changed'
    path = os.path.join(OUT, 'A3_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'],
               sets={k: dict(ttt_ms=v[0], a3_offset_db=v[1], k=v[2], ttt_slots=A3.ttt_slots(v[0]),
                             a_slot=A3.alpha_slot(v[2])) for k, v in A3.SETS.items()},
               md5={f: md5(os.path.join(HERE, f)) for f in freeze_files()})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_a3_freeze():
    fz = json.load(open(os.path.join(OUT_REAL, 'A3_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run
def test_mode(A, cap, hl):
    check_a3_freeze()
    out = cell_path(A, cap, hl)
    assert not os.path.exists(out), 'this cell already ran: each cell runs once'
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    env = LC6.make_env_cap(pool, hl, ADM[A], cap)
    status(f'test C{cap} {A} hl={hl} start')
    rows = run_arms(env, seeds, A, cap, hl, out_path=out)
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test C{cap} {A} hl={hl} DONE')


# ---------------------------------------------------------------- analysis (once)
def analyse(dry=False):
    fz = {'note': 'dry run on relabelled stored arms'} if dry else check_a3_freeze()
    res_path = os.path.join(OUT, 'A3_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    parts = []
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                d = pd.read_csv(cell_path(A, cap, hl))
                d['scene'] = d['scene'].astype(str)
                parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    lsra = {}
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                s = pd.read_csv(stored_path(A, cap, hl))
                s = s[(s['arm'] == 'METHOD') & (s['hl'] == hl)].copy()
                s['scene'] = s['scene'].astype(str)
                lsra[(A, cap, hl)] = s.sort_values('seed')

    def cell(A, cap, hl, arm):
        if arm == 'METHOD':
            return lsra[(A, cap, hl)]
        return df[(df.adm == A) & (df.cap == cap) & (df.hl == hl) & (df.arm == arm)].sort_values('seed')

    def paired(x, y):
        assert len(x) == N_EP and list(x.seed) == list(y.seed), 'pairing by seed failed'
        assert list(x.scene) == list(y.scene), 'scenes differ between paired episodes'
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    def diff(A, cap, hl, arm):
        return paired(cell(A, cap, hl, arm), cell(A, cap, hl, 'METHOD'))

    R = {'_freeze': fz, '_versions': AC.versions(), 'n_episodes_per_cell': N_EP}
    adm = df[df['adm'].isin(['A1', 'A2'])]
    pairing = {}
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                for arm in A3_ARMS:
                    x, y = cell(A, cap, hl, arm), cell(A, cap, hl, 'METHOD')
                    pairing[f'{A}_C{cap}_hl{hl}_{arm}'] = bool(len(x) == N_EP and list(x.seed) == list(y.seed)
                                                               and list(x.scene) == list(y.scene))
    checks = dict(M1_created_excess_under_A1_A2=int(adm.dec_over.sum()),
                  M3_loop_violations_A1=int(df[df['adm'] == 'A1'].loop_violations.sum()),
                  M4_rows_where_starts_differ=int((adm.starts != adm.requests - adm.refused).sum()),
                  M5_all_cells_paired=bool(len(pairing) == 180 and all(pairing.values())))
    checks['ok'] = bool(checks['M1_created_excess_under_A1_A2'] == 0 and checks['M3_loop_violations_A1'] == 0
                        and checks['M4_rows_where_starts_differ'] == 0 and checks['M5_all_cells_paired'])
    R['checks'] = checks
    if not checks['M5_all_cells_paired']:
        R['pairing_failures'] = [k for k, v in pairing.items() if not v]
        json.dump(R, open(res_path, 'w'), indent=1)
        status(f'analyse STOPPED: pairing failed in {len(R["pairing_failures"])} cells')
        return

    def family(A, seed0):
        per = {}
        for k, arm in enumerate(A3_ARMS):
            d, sc = diff(A, 4, 10, arm)
            ci = RM.scene_boot(d, sc, seed0 + k, (0.5, 99.5))
            per[arm] = dict(mean=float(d.mean()), scene_99=ci, scene_95=RM.scene_boot(d, sc, seed0 + 50 + k),
                            beats=bool(ci[0] > 0.0))
        return dict(per_set=per, held=bool(all(v['beats'] for v in per.values())))

    R['HA1_A2'] = family('A2', 7000)
    R['HA2_A1'] = family('A1', 7100)

    sb = 7300
    S = {}
    tp = {}
    for arm in A3_ARMS:
        d, sc = paired(cell('A1', 4, 10, arm), cell('A2', 4, 10, 'METHOD'))
        tp[arm] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, sc, sb)); sb += 1
    S['today_A3_under_A1_minus_LSRA_under_A2_4places_1s'] = tp
    T = {}
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                for arm in A3_ARMS + ['METHOD']:
                    c = cell(A, cap, hl, arm)
                    w = c.worst.values * 100.0
                    wa = np.maximum(c.worst_act.values, 1.0)
                    e = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                             ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                             dec_over=int(c.dec_over.sum()),
                             causes_worst={kk: float((c[f'w_{kk}'].values / wa).mean() * 100) for kk in RM.CAUSES})
                    sb += 1
                    if A != 'A0':                            # requests are counted only under A1 and A2
                        e['requests_per_car_step'] = float(c.requests.sum() / c.p_act.sum())
                        e['refused_share'] = float(c.refused.sum() / max(c.requests.sum(), 1))
                    if arm != 'METHOD':
                        d, s2 = diff(A, cap, hl, arm)
                        e['minus_METHOD_same_admission'] = dict(mean=float(d.mean()),
                                                                scene_95=RM.scene_boot(d, s2, sb)); sb += 1
                    T[f'{A}_C{cap}_hl{hl}_{arm}'] = e
    S['table'] = T
    best = {}
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                vals = {arm: T[f'{A}_C{cap}_hl{hl}_{arm}']['worst'] for arm in A3_ARMS}
                b = min(vals, key=vals.get)
                best[f'{A}_C{cap}_hl{hl}'] = dict(best_set=b, worst=vals[b],
                                                  METHOD=T[f'{A}_C{cap}_hl{hl}_METHOD']['worst'])
    S['best_A3_set_per_cell_descriptive'] = best
    R['secondary'] = S

    h1, h2 = R['HA1_A2']['held'], R['HA2_A1']['held']
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif h1 and h2:
        outcome = ('LSRA beats every TR 36.839 A3 set under coordinated and under per-request admission '
                   '(4 places, 1.0 s)')
    elif h1:
        outcome = ('LSRA beats every A3 set under coordinated admission; under per-request admission it is not '
                   'shown to beat every set')
    elif h2:
        outcome = ('LSRA beats every A3 set under per-request admission, but under coordinated admission it is '
                   'not shown to beat every set')
    else:
        outcome = 'LSRA is not shown to beat every A3 set under either admission model'
    R['decision'] = dict(outcome=outcome, HA1_held=h1, HA2_held=h2)
    hps = {arm: T[f'A0_C4_hl10_{arm}']['ho_per_car_step'] for arm in A3_ARMS}
    R['guesses'] = dict(
        HA1_held=h1, HA2_not_held=not h2,
        best_A3_under_A2_4places_1s_between_6_and_10=bool(
            6.0 <= min(T[f'A2_C4_hl10_{a}']['worst'] for a in A3_ARMS) <= 10.0),
        best_A3_under_A1_4places_1s_between_8_5_and_11=bool(
            8.5 <= min(T[f'A1_C4_hl10_{a}']['worst'] for a in A3_ARMS) <= 11.0),
        set5_most_and_set1_fewest_handovers_A0_4places_1s=bool(
            max(hps, key=hps.get) == 'A3S5' and min(hps, key=hps.get) == 'A3S1'),
        LSRA_lead_over_best_A3_under_A2_smaller_at_6_than_at_4_places=bool(
            T['A2_C6_hl10_METHOD']['worst'] - best['A2_C6_hl10']['worst'] >
            T['A2_C4_hl10_METHOD']['worst'] - best['A2_C4_hl10']['worst']))
    json.dump(R, open(res_path, 'w'), indent=1)

    L = ['THE 3GPP A3 BASELINE: the five TR 36.839 sets against LSRA',
         f'({N_EP} episodes per cell; ' + ('DRY RUN on relabelled stored arms, a code check only' if dry else
                                            '60 held-out DAIR scenes, the frozen seeds') + ')', '',
         f"must-hold checks: {checks}", '']
    for key, title in [('HA1_A2', 'HA1, coordinated admission A2'), ('HA2_A1', 'HA2, per-request admission A1')]:
        L.append(f'{title} (4 places, 1.0 s; A3 set minus LSRA, worst-vehicle RLF pp; 99 % scene interval):')
        for arm, v in R[key]['per_set'].items():
            L.append(f"  {arm}: {v['mean']:+.2f} [{v['scene_99'][0]:+.2f}, {v['scene_99'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    L.append('')
    L.append("today's practice: A3 under A1 minus LSRA under A2 (4 places, 1.0 s; 95 % scene intervals):")
    for arm, v in tp.items():
        L.append(f"  {arm}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append('')
    for A in ADM:
        for cap in CAPS:
            L.append(f'{A}, {cap} places: arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS)
                     + ' | handovers/vehicle-step @1.0 s | refused share @1.0 s')
            for arm in A3_ARMS + ['METHOD']:
                e10 = T[f'{A}_C{cap}_hl10_{arm}']
                L.append(f'  {arm} | ' + ' | '.join(
                    f"{T[f'{A}_C{cap}_hl{h}_{arm}']['worst']:.2f} [{T[f'{A}_C{cap}_hl{h}_{arm}']['scene_95'][0]:.2f}, "
                    f"{T[f'{A}_C{cap}_hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
                    + f" | {e10['ho_per_car_step']:.4f} | {e10.get('refused_share', float('nan')):.3f}")
            L.append('')
    L.append('every A3 set minus LSRA, same admission (95 % scene intervals):')
    for A in ADM:
        for cap in CAPS:
            for hl in HLS:
                L.append(f'  {A} {cap} places {hl * 0.1:.1f} s: ' + '  '.join(
                    f"{arm} {T[f'{A}_C{cap}_hl{hl}_{arm}']['minus_METHOD_same_admission']['mean']:+.2f} "
                    f"[{T[f'{A}_C{cap}_hl{hl}_{arm}']['minus_METHOD_same_admission']['scene_95'][0]:+.2f}, "
                    f"{T[f'{A}_C{cap}_hl{hl}_{arm}']['minus_METHOD_same_admission']['scene_95'][1]:+.2f}]"
                    for arm in A3_ARMS))
    L.append(''); L.append('best A3 set per cell (descriptive; chosen after the run, so it favours A3):')
    for k2, v in best.items():
        L.append(f"  {k2}: {v['best_set']} {v['worst']:.2f}  (LSRA {v['METHOD']:.2f})")
    L.append(''); L.append('causes of the worst vehicle at 1.0 s, 4 places:')
    for A in ADM:
        for arm in A3_ARMS + ['METHOD']:
            cw = T[f'{A}_C4_hl10_{arm}']['causes_worst']
            L.append(f'  {A} {arm}: ' + '  '.join(f'{kk} {cw[kk]:.2f}' for kk in RM.CAUSES))
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'A3_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['unit', 'dry', 'gate0', 'gate', 'freeze', 'test', 'analyse'], required=True)
    ap.add_argument('--adm', choices=list(ADM))
    ap.add_argument('--cap', type=int, choices=CAPS)
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(OUT_REAL, exist_ok=True)
    if a.mode == 'unit':
        unit_mode()
    elif a.mode == 'dry':
        dry_mode()
    elif a.mode == 'gate0':
        gate0_mode()
    elif a.mode == 'gate':
        gate_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    elif a.mode == 'test':
        assert a.hl in HLS and a.adm in ADM and a.cap in CAPS
        test_mode(a.adm, a.cap, a.hl)
    else:
        analyse()
