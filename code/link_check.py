#!/usr/bin/env python3
"""Link-budget check: the main comparison at a 23 dBm transmit power and with the 3GPP -8 dB out-of-sync
threshold (DRAFT, 22 Oct 2026). Pre-registration: PREREG_MA_LinkBudget_P23_T8_DRAFT_v1_20261022.md
(needs Daya's OK before any variant runs).

Same 60 held-out scenes, seed list, frozen arms (method_heldout.arms, beta 0.05), harness and environment
class (paper2ma_env_admit.AdmissionEnv) as the admission check, at 4 places under coordinated admission (A2),
the setting of the registered test HR2. ONE change per variant, made by setting module constants of
paper2ma_env before the environment is built:
  P23    every vehicle (controlled and background) transmits at 23 dBm instead of 30 dBm
  T8     the out-of-sync threshold is -8 dB instead of -13.8 dB (N310 = 3 unchanged)
  P23T8  both
Every policy, the harness and the method's forecast read these constants from paper2ma_env at run time
(MA.PT_MAX, MA.SINR_OUT_DB, MA.INTERF_PT_DBM). U1d checks this on the source of every module the runner
imports; U1e checks that no episode reads Paper 1's copies (paper2_env). The background interference is
precomputed per scene and cached on disk under a key that does not contain the power, so the 23 dBm variants
use their own cache folder (geom_cache_ma_p23), filled once by --mode warm, in one process, before any run,
and checked there scene by scene against the default cache. A test refuses to start on an incomplete cache
or on a cache whose files changed after the warm-up. Nothing is deleted: a folder to be rebuilt is renamed.
  --mode unit                 U1: link-patch checks (no new results)
  --mode warm                 fills geom_cache_ma_p23 for the 60 held-out scenes (23 dBm background) and checks it
  --mode gate                 G0: default link, FRESH cache folder: reproduces the admission check's A2 runs
                              at 4 places exactly (first 40 seeds, 1.0 s, all 10 arms, every column)
  --mode freeze               writes LINK_FREEZE.json once U1, warm and G0 passed with this very file (on Daya's OK)
  --mode test --var V --hl H  all 10 frozen arms under A2 at 4 places, one variant, one delay; each cell once
  --mode analyse              the registered analysis, once
"""
import os, sys, ast, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import paper2ma_env as MA
import paper2ma_env_admit as AD
import paper2_env as PE
import lsra as LS

HERE = RM.HERE
SELF = os.path.abspath(__file__)
OUT = os.path.join(HERE, 'models', 'link_check')
CACHE_P23 = os.path.join(HERE, 'geom_cache_ma_p23')
CACHE_GATE = os.path.join(HERE, 'geom_cache_ma_gate')
CACHE_UNIT = os.path.join(HERE, 'geom_cache_ma_unit')
HLS = MHO.HLS
CAP = 4
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']
N_GATE = 40
N_UNIT = 5
N_POISON = 4
PREREG = 'PREREG_MA_LinkBudget_P23_T8_FROZEN_20261022.md'
VARIANTS = {'P23': dict(pt=23.0, thr=None), 'T8': dict(pt=None, thr=-8.0), 'P23T8': dict(pt=23.0, thr=-8.0)}
NO_REFUSAL_ARMS = ['METHOD', 'METHOD_oracle', 'ABL_worstcarfirst', 'R1i']   # plan on the landing state
DEFAULT_FILES = [f'models/admission_check/ADMIT_EPISODES_A2_hl{h}.csv' for h in HLS]
NAMES = ('SINR_OUT_DB', 'PT_MAX', 'PT_MIN', 'INTERF_PT_DBM')

# the default link, read once at import (asserted, so a silent change of the simulator is caught)
D0 = dict(PT_MAX=float(MA.PT_MAX), PT_MIN=float(MA.PT_MIN), INTERF_PT_DBM=float(MA.INTERF_PT_DBM),
          SINR_OUT_DB=float(MA.SINR_OUT_DB))
assert D0['PT_MAX'] == 30.0 and D0['PT_MIN'] == 23.0 and D0['INTERF_PT_DBM'] == 30.0, D0
# the default threshold is -92 dBm minus the noise floor: -13.82 dB, written -13.8 dB in the paper
assert D0['SINR_OUT_DB'] == float(PE.SINR_OUT_DB) == float(PE.RLF_THRESH_DBM - PE.CM.N0_DBM), D0
assert round(D0['SINR_OUT_DB'], 1) == -13.8, D0


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fresh_dir(d):
    """An empty folder at d. An existing one is renamed aside (never deleted)."""
    if os.path.exists(d):
        os.rename(d, f'{d}_superseded_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(d)


def set_link(pt=None, thr=None):
    """Sets the link constants of paper2ma_env; None restores the default. PT_MIN keeps the default 7 dB
    distance to PT_MAX: it is used only to scale the observation and the reward and as the lower clip of the
    power (every arm asks for full power), so it does not enter any decision of the arms compared here."""
    MA.PT_MAX = D0['PT_MAX'] if pt is None else float(pt)
    MA.PT_MIN = D0['PT_MIN'] if pt is None else float(pt) - (D0['PT_MAX'] - D0['PT_MIN'])
    MA.INTERF_PT_DBM = D0['INTERF_PT_DBM'] if pt is None else float(pt)
    MA.SINR_OUT_DB = D0['SINR_OUT_DB'] if thr is None else float(thr)
    return dict(PT_MAX=MA.PT_MAX, PT_MIN=MA.PT_MIN, INTERF_PT_DBM=MA.INTERF_PT_DBM, SINR_OUT_DB=MA.SINR_OUT_DB)


def make_env(pool, hl, cache_dir=None):
    """As admission_check.make_env3(pool, hl, 'coordinated'), with an optional cache folder."""
    env = AD.AdmissionEnv(pool, n_agents=16, n_subbands=3, gnb_cap=CAP, seed=0, split=None,
                          compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = 'coordinated'
    if cache_dir is not None:
        env.CACHE_DIR = cache_dir
    return env


def cell_path(var, hl):
    return os.path.join(OUT, f'LINK_EPISODES_{var}_hl{hl}.csv')


def same_value(a, b):
    """Exact equality of two geometry entries (NaN equal to NaN for floating arrays)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def frames_equal_typed(a, b):
    """Exact comparison: numeric columns as floats (NaN equal to NaN), all others as text."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True):
                return False
        elif not all(str(u) == str(v) for u, v in zip(a[c].tolist(), b[c].tolist())):
            return False
    return True


def cache_file(cache_dir, env, sid):
    """The cache file of one scene (same key and name as paper2ma_env._precompute)."""
    key = (sid, env.M, env.n_sb, env.k_bg, tuple(env.gnb_modes), env.want_ant)
    return os.path.join(cache_dir, f'{sid}_{hashlib.md5(repr(key).encode()).hexdigest()[:10]}.pkl')


def cache_complete(cache_dir, env):
    return all(os.path.exists(cache_file(cache_dir, env, sid)) for sid in env.scene_ids)


def cache_digest(cache_dir, env):
    """One md5 over the scene files of this environment (name and content, in scene order)."""
    h = hashlib.md5()
    for sid in env.scene_ids:
        f = cache_file(cache_dir, env, sid)
        h.update(os.path.basename(f).encode()); h.update(md5(f).encode())
    return h.hexdigest()


def compare_scene(A, Bx):
    """Default-link scene object A against a 23 dBm one Bx: (geometry identical, max relative error of the
    background ratio against 10^(-0.7), number of positive background entries compared)."""
    same = (A['agents'] == Bx['agents'] and len(A['frames']) == len(Bx['frames']))
    err, n = 0.0, 0
    for fa, fb in zip(A['frames'], Bx['frames']):
        same &= bool(np.array_equal(fa['active'], fb['active'])) and len(fa['ag']) == len(fb['ag'])
        for x, y in zip(fa['ag'], fb['ag']):
            if (x is None) != (y is None):
                same = False
            elif x is not None:
                same &= (set(x) == set(y)) and all(same_value(x[k], y[k]) for k in x)
        bg0, bg1 = np.asarray(fa['bg'], float), np.asarray(fb['bg'], float)
        pos = bg0 > 0
        n += int(pos.sum())
        if pos.any():
            err = max(err, float(np.max(np.abs(bg1[pos] / bg0[pos] / 10 ** (-0.7) - 1.0))))
        same &= bool(np.all(bg1[~pos] == 0))
    return bool(same), err, n


def scan_by_value():
    """U1d. No module this runner imports (and none they import, among the local files) copies the four link
    constants by value: no 'from <local module> import' of them (or *), no default argument or decorator that
    reads them, and no import-time statement (module or class body) that reads them, except their own
    definitions: in paper2_env.py (Paper 1's module, where they come from) and the assignments of the four
    names themselves in paper2ma_env.py."""
    local = {f[:-3] for f in os.listdir(HERE) if f.endswith('.py')}
    me = ast.parse(open(SELF, encoding='utf-8').read())
    todo = [a.name.split('.')[0] for n in me.body if isinstance(n, ast.Import) for a in n.names]
    seen, trees = set(), {}
    while todo:
        m = todo.pop()
        if m in seen or m not in local or m == 'link_check':
            continue
        seen.add(m)
        t = ast.parse(open(os.path.join(HERE, m + '.py'), encoding='utf-8').read())
        trees[m] = t
        for n in ast.walk(t):
            if isinstance(n, ast.Import):
                todo += [a.name.split('.')[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                todo.append(n.module.split('.')[0])

    def refs(node):
        return [x for x in ast.walk(node)
                if (isinstance(x, ast.Name) and x.id in NAMES) or (isinstance(x, ast.Attribute) and x.attr in NAMES)]

    def targets(s):
        out = []
        for t in s.targets:
            out += list(t.elts) if isinstance(t, ast.Tuple) else [t]
        return out

    hits = []

    def body(m, stmts, where):
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in list(s.args.defaults) + [d for d in s.args.kw_defaults if d is not None] + s.decorator_list:
                    if refs(d):
                        hits.append(f'default argument or decorator: {m}.py:{s.lineno} {where}{s.name}')
            elif isinstance(s, ast.ClassDef):
                body(m, s.body, where + s.name + '.')
            elif refs(s):
                own = m == 'paper2_env' or (
                    m == 'paper2ma_env' and where == '' and isinstance(s, ast.Assign)
                    and all(isinstance(e, ast.Name) for e in targets(s))
                    and any(e.id in NAMES for e in targets(s)))
                if not own:
                    hits.append(f'import-time use: {m}.py:{s.lineno} {where or "<module>"} {ast.unparse(s)[:120]}')

    for m, t in sorted(trees.items()):
        for n in ast.walk(t):
            if (isinstance(n, ast.ImportFrom) and n.module and n.module.split('.')[0] in local
                    and any(a.name in NAMES or a.name == '*' for a in n.names)):
                hits.append(f'from-import: {m}.py:{n.lineno} from {n.module}')
        body(m, t.body, '')
    return sorted(seen), hits


def gate_compare(env, fzm, seeds, label):
    """Runs every arm on the seeds at 1.0 s and compares every column with the admission check's A2 run."""
    old = pd.read_csv(os.path.join(HERE, 'models', 'admission_check', 'ADMIT_EPISODES_A2_hl10.csv'))
    per = {}
    for name, fn in MHO.arms(fzm['beta']):
        t0 = time.time()
        rows = AC.run_cell(env, fn, seeds)
        for x in rows:
            x.update(hl=10, arm=name, adm='A2')
        new = AC.roundtrip(rows)
        o = old[old['arm'] == name].set_index('seed').loc[list(new['seed'])].reset_index()[list(new.columns)]
        per[name] = bool(frames_equal_typed(new.reset_index(drop=True), o.reset_index(drop=True)))
        status(f'[{label}] {name}: identical to the admission check A2 run = {per[name]} ({time.time() - t0:.0f} s)')
    return per


# ---------------------------------------------------------------- U1: link-patch checks (no new results)
def unit_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    for sub in ('default', 'p23'):
        fresh_dir(os.path.join(CACHE_UNIT, sub))
    rep = {'_versions': AC.versions(), 'runner_md5': md5(SELF), 'D0': D0}
    # (a) geometry identical and background scaled by exactly 10^(-0.7) when the background power is 23 dBm
    set_link()
    e0 = make_env(pool, 10, os.path.join(CACHE_UNIT, 'default'))
    scenes = e0.scene_ids[:N_UNIT]                     # the scene ids exactly as the environment holds them
    f0 = {sid: e0._precompute(sid) for sid in scenes}
    set_link(pt=23.0)
    e1 = make_env(pool, 10, os.path.join(CACHE_UNIT, 'p23'))
    f1 = {sid: e1._precompute(sid) for sid in scenes}
    geom_same, ratio_err, n_bg = True, 0.0, 0
    for sid in scenes:
        g, e, k = compare_scene(f0[sid], f1[sid])
        geom_same &= g; ratio_err = max(ratio_err, e); n_bg += k
    rep['scenes'] = [str(s) for s in scenes]
    rep['U1a_geometry_identical'] = bool(geom_same)
    rep['U1a_bg_ratio_max_rel_error'] = ratio_err
    rep['U1a_bg_entries_compared'] = n_bg
    # (b) after reset, every controlled vehicle transmits at 23 dBm; desired power falls by exactly 7 dB
    e1.reset(seed=AC.seeds_all()[0])                   # still at 23 dBm
    rep['U1b_all_pt_23'] = bool(np.all(np.asarray(e1.pt) == 23.0))
    set_link()
    e0b = make_env(pool, 10, os.path.join(CACHE_UNIT, 'default'))
    e0b.reset(seed=AC.seeds_all()[0])
    act = [i for i in range(e1.M) if e1.frames[0]['active'][i]]
    dd = [e0b._desired_dbm(i, int(e0b.serv[i])) - e1._desired_dbm(i, int(e0b.serv[i])) for i in act]
    rep['U1b_desired_drop_db'] = [float(min(dd)), float(max(dd))] if dd else None
    rep['U1b_same_scene'] = bool(e0b.scene_id == e1.scene_id)
    # (c) the method's forecast reads the threshold: risk at -8 dB >= risk at -13.8 dB, strictly somewhere
    fr = e0b.frames[0]
    r0 = LS.risk_frame(e0b, fr)
    set_link(thr=-8.0)
    r1 = LS.risk_frame(e0b, fr)
    set_link()
    rep['U1c_risk_never_lower'] = bool(np.all(r1 >= r0 - 1e-15))
    rep['U1c_risk_higher_somewhere'] = bool(np.any(r1 > r0 + 1e-12))
    # (d) no module copies the constants by value (source scan of the import closure)
    closure, hits = scan_by_value()
    rep['U1d_modules_scanned'] = closure
    rep['U1d_by_value_copies'] = hits
    # (e) no episode reads Paper 1's copies (paper2_env): with those four set to NaN, the first seeds at
    #     1.0 s under the default link reproduce the admission check's stored A2 episodes, every arm
    saved = {k: getattr(PE, k) for k in NAMES}
    try:
        for k in NAMES:
            setattr(PE, k, float('nan'))
        set_link()
        per = gate_compare(make_env(pool, 10), fzm, AC.seeds_all()[:N_POISON], 'unit e')
    finally:
        for k, v in saved.items():
            setattr(PE, k, v)
    rep['U1e_paper1_constants_NaN_same_episodes'] = per
    rep['ok'] = bool(rep['U1a_geometry_identical'] and ratio_err < 1e-12 and n_bg > 0 and rep['U1b_all_pt_23']
                     and rep['U1b_same_scene'] and dd and abs(min(dd) - 7.0) < 1e-9 and abs(max(dd) - 7.0) < 1e-9
                     and rep['U1c_risk_never_lower'] and rep['U1c_risk_higher_somewhere']
                     and not hits and len(closure) > 5 and all(per.values()) and len(per) == 10)
    json.dump(rep, open(os.path.join(OUT, 'UNIT.json'), 'w'), indent=1)
    status(f'[unit] ok={rep["ok"]} (geometry identical {geom_same}, bg ratio error {ratio_err:.2e} over {n_bg} '
           f'entries, desired drop {rep["U1b_desired_drop_db"]}, by-value copies {len(hits)} in {len(closure)} '
           f'modules, NaN test {sum(per.values())}/{len(per)})')


# ---------------------------------------------------------------- warm: the 23 dBm background cache
def warm_mode():
    MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    fresh_dir(CACHE_P23)
    set_link(pt=23.0)
    env = make_env(pool, 10, CACHE_P23)
    t0 = time.time()
    p23 = {sid: env._precompute(sid) for sid in env.scene_ids}
    secs = round(time.time() - t0, 1)
    set_link()
    complete = cache_complete(CACHE_P23, env)
    # the 23 dBm cache against the default cache, scene by scene
    e0 = make_env(pool, 10)
    default_complete = cache_complete(e0.CACHE_DIR, e0)
    geom, err, n = True, 0.0, 0
    for sid in env.scene_ids:
        g, e, k = compare_scene(e0._precompute(sid), p23[sid])
        geom &= g; err = max(err, e); n += k
    rep = dict(_versions=AC.versions(), runner_md5=md5(SELF), scenes=len(env.scene_ids), seconds=secs,
               complete=bool(complete), default_cache_complete=bool(default_complete),
               geometry_identical_to_default_cache=bool(geom), bg_ratio_max_rel_error=err, bg_entries_compared=n,
               digest_p23=cache_digest(CACHE_P23, env), digest_default=cache_digest(e0.CACHE_DIR, e0))
    rep['ok'] = bool(complete and default_complete and geom and err < 1e-12 and n > 0)
    json.dump(rep, open(os.path.join(OUT, 'WARM.json'), 'w'), indent=1)
    status(f'[warm] ok={rep["ok"]}: {len(env.scene_ids)} scenes in {secs} s; geometry identical to the default '
           f'cache {geom}; bg ratio error {err:.2e} over {n} entries')


# ---------------------------------------------------------------- G0: default link, fresh cache = A2 run
def gate_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    fresh_dir(CACHE_GATE)
    set_link()
    per = gate_compare(make_env(pool, 10, CACHE_GATE), fzm, AC.seeds_all()[:N_GATE], 'gate')
    rep = dict(_versions=AC.versions(), runner_md5=md5(SELF), n_seeds=N_GATE, hl=10, per_arm=per,
               ok=bool(all(per.values()) and len(per) == 10))
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_mode():
    path = os.path.join(OUT, 'LINK_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    recs = {f: json.load(open(os.path.join(OUT, f))) for f in ('UNIT.json', 'WARM.json', 'GATE.json')}
    for f, r in recs.items():
        assert r.get('ok') is True, f'{f} did not pass: no freeze'
        assert r.get('runner_md5') == md5(SELF), f'{f} was written by another version of this runner: no freeze'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    env = make_env(pool, 10)
    dig = dict(p23=cache_digest(CACHE_P23, env), default=cache_digest(env.CACHE_DIR, env))
    assert dig['p23'] == recs['WARM.json']['digest_p23'], 'the 23 dBm cache changed after the warm-up'
    assert dig['default'] == recs['WARM.json']['digest_default'], 'the default cache changed after the warm-up'
    files = (['link_check.py', 'admission_check.py', 'paper2ma_env_admit.py', 'paper2ma_env_attach.py',
              'paper2ma_env.py', 'paper2_env.py', 'lsra.py', 'method_heldout.py', 'method_harness.py',
              'remeasure_ma.py', 'assign_value.py', 'capacity_rule.py', 'paper2ma_confirm.py',
              'paper2ma_antic_test.py', PREREG,
              'models/link_check/UNIT.json', 'models/link_check/WARM.json', 'models/link_check/GATE.json',
              'models/method_heldout/FREEZE.json', 'models/admission_check/ADMIT_FREEZE.json'] + DEFAULT_FILES)
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'], cap=CAP,
               variants=VARIANTS, cache_digests=dig, md5={f: md5(os.path.join(HERE, f)) for f in files})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_link_freeze():
    fz = json.load(open(os.path.join(OUT, 'LINK_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run
def test_mode(var, hl):
    fz = check_link_freeze()
    out = cell_path(var, hl)
    assert not os.path.exists(out), 'this cell already ran: each cell runs once'
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    v = VARIANTS[var]
    link = set_link(**v)
    cache = CACHE_P23 if v['pt'] is not None else None
    env = make_env(pool, hl, cache)
    assert cache_complete(env.CACHE_DIR, env), f'incomplete cache: {env.CACHE_DIR}'
    want = fz['cache_digests']['p23' if cache is not None else 'default']
    assert cache_digest(env.CACHE_DIR, env) == want, f'the cache changed after the freeze: {env.CACHE_DIR}'
    rows = []
    status(f'test {var} hl={hl} start (beta {fz["beta"]}; link {link}; cache {env.CACHE_DIR})')
    for name, fn in MHO.arms(fz['beta']):
        t0 = time.time()
        r = AC.run_cell(env, fn, seeds)
        for x in r:
            x.update(hl=hl, arm=name, adm='A2', var=var)
        rows += r
        pd.DataFrame(rows).to_csv(out + '.partial', index=False)
        status(f'test {var} hl={hl} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test {var} hl={hl} DONE')


# ---------------------------------------------------------------- analysis (once)
def analyse():
    fz = check_link_freeze()
    res_path = os.path.join(OUT, 'LINK_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    names = [a for a, _ in MHO.arms(fz['beta'])]
    parts = []
    for var in VARIANTS:
        for h in HLS:
            d = pd.read_csv(cell_path(var, h))
            d['scene'] = d['scene'].astype(str)
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    dp = []
    for h in HLS:
        d = pd.read_csv(os.path.join(HERE, 'models', 'admission_check', f'ADMIT_EPISODES_A2_hl{h}.csv'))
        d['scene'] = d['scene'].astype(str)
        d['var'] = 'DEFAULT'
        dp.append(d)
    dfd = pd.concat(dp, ignore_index=True)

    def cell(var, hl, arm, d=None):
        d = df if d is None else d
        return d[(d['var'] == var) & (d['hl'] == hl) & (d['arm'] == arm)].sort_values('seed')

    def diff(va, a, vb, b, hl, da=None, db=None):
        x, y = cell(va, hl, a, da), cell(vb, hl, b, db)
        assert list(x.seed) == list(y.seed) and len(x) == 480
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    R = {'_freeze': fz, '_versions': AC.versions()}
    m1 = int(df.dec_over.sum())
    m2 = {f'{var}_hl{h}_{arm}': int(cell(var, h, arm).refused.sum())
          for var in VARIANTS for h in HLS for arm in NO_REFUSAL_ARMS}
    m4 = int((df.starts != df.requests - df.refused).sum())
    checks = dict(M1_created_excess_total=m1, M2_refusals_of_landing_state_planners=m2,
                  M4_rows_where_starts_differ=m4)
    checks['ok'] = bool(m1 == 0 and all(v == 0 for v in m2.values()) and m4 == 0)
    R['checks'] = checks

    def family(var, seed0):
        per = {}
        for k, b in enumerate(PRIMARY_BASELINES):
            d, sc = diff(var, b, var, 'METHOD', 10)
            ci = RM.scene_boot(d, sc, seed0 + k, (0.625, 99.375))
            per[b] = dict(mean=float(d.mean()), scene_9875=ci, scene_95=RM.scene_boot(d, sc, seed0 + 50 + k),
                          beats=bool(ci[0] > 0.0))
        return dict(per_baseline=per, held=bool(all(v['beats'] for v in per.values())))

    R['HB1_P23'] = family('P23', 7000)
    R['HB2_T8'] = family('T8', 7100)
    R['HB3_P23T8'] = family('P23T8', 7200)

    T = {}; sb = 7300
    for var in VARIANTS:
        for hl in HLS:
            for arm in names:
                c = cell(var, hl, arm); w = c.worst.values * 100.0; wa = np.maximum(c.worst_act.values, 1.0)
                e = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                         ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                         refused_share=float(c.refused.sum() / max(c.requests.sum(), 1)),
                         causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
                sb += 1
                if arm != 'METHOD':
                    d, s2 = diff(var, arm, var, 'METHOD', hl)
                    e['minus_METHOD'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
                T[f'{var}_hl{hl}_{arm}'] = e
    V = {}
    for var in VARIANTS:
        for arm in names:
            d, s2 = diff(var, arm, 'DEFAULT', arm, 10, None, dfd)
            V[f'{var}_minus_default_1s_{arm}'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
    R['secondary'] = dict(table=T, contrasts=V)
    held = {k: R[k]['held'] for k in ('HB1_P23', 'HB2_T8', 'HB3_P23T8')}
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif all(held.values()):
        outcome = 'the main comparison holds at 23 dBm, at the 3GPP -8 dB threshold, and with both'
    else:
        outcome = ('not held under ' + ', '.join(k for k, v in held.items() if not v)
                   + ': the paper ties the main result to its link budget and reports where it does not hold')
    R['decision'] = dict(outcome=outcome, **held)
    R['guesses'] = dict(
        METHOD_P23_1s_between_4_and_10=bool(4.0 <= T['P23_hl10_METHOD']['worst'] <= 10.0),
        METHOD_T8_1s_between_4_and_10=bool(4.0 <= T['T8_hl10_METHOD']['worst'] <= 10.0),
        METHOD_P23T8_1s_between_8_and_25=bool(8.0 <= T['P23T8_hl10_METHOD']['worst'] <= 25.0),
        HB1_held=held['HB1_P23'], HB2_held=held['HB2_T8'], HB3_held=held['HB3_P23T8'])
    json.dump(R, open(res_path, 'w'), indent=1)

    L = ['LINK-BUDGET CHECK: 4 places, coordinated admission (A2), the 60 held-out scenes, 480 episodes per cell',
         f'variants: {VARIANTS}', '', f'must-hold checks: {checks}', '']
    for key, title in [('HB1_P23', 'HB1, 23 dBm'), ('HB2_T8', 'HB2, -8 dB threshold'), ('HB3_P23T8', 'HB3, both')]:
        L.append(f'{title} (1.0 s; baseline minus METHOD, worst-vehicle RLF pp; 98.75 % scene interval):')
        for b, v in R[key]['per_baseline'].items():
            L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    L.append('')
    for var in VARIANTS:
        L.append(f'{var}: arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS) + ' | handovers/vehicle-step @1.0 s')
        for arm in names:
            L.append(f'  {arm} | ' + ' | '.join(
                f"{T[f'{var}_hl{h}_{arm}']['worst']:.2f} [{T[f'{var}_hl{h}_{arm}']['scene_95'][0]:.2f}, "
                f"{T[f'{var}_hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
                + f" | {T[f'{var}_hl10_{arm}']['ho_per_car_step']:.4f}")
        L.append('')
    L.append('every arm minus METHOD, 1.0 s (95 % scene intervals):')
    for var in VARIANTS:
        L.append(f'  {var}: ' + '  '.join(
            f"{arm} {T[f'{var}_hl10_{arm}']['minus_METHOD']['mean']:+.2f} "
            f"[{T[f'{var}_hl10_{arm}']['minus_METHOD']['scene_95'][0]:+.2f}, "
            f"{T[f'{var}_hl10_{arm}']['minus_METHOD']['scene_95'][1]:+.2f}]" for arm in names if arm != 'METHOD'))
    L.append(''); L.append('causes of the worst vehicle at 1.0 s:')
    for var in VARIANTS:
        for arm in names:
            cw = T[f'{var}_hl10_{arm}']['causes_worst']
            L.append(f'  {var} {arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append('variant minus default link, same arm and seeds, 1.0 s (95 % scene intervals):')
    for k, v in V.items():
        L.append(f"  {k}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'LINK_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['unit', 'warm', 'gate', 'freeze', 'test', 'analyse'], required=True)
    ap.add_argument('--var', choices=list(VARIANTS))
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'unit':
        unit_mode()
    elif a.mode == 'warm':
        warm_mode()
    elif a.mode == 'gate':
        gate_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    elif a.mode == 'test':
        assert a.hl in HLS and a.var in VARIANTS
        test_mode(a.var, a.hl)
    else:
        analyse()
