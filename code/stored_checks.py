#!/usr/bin/env python3
"""stored_checks.py -- two checks on stored results (DRAFT, 28 Oct 2026; not run on real outcomes yet).

Plan: PREREG_LSRA_StoredChecks_PlannerAdmissionSplit_BlockageDurations (DRAFT v1, 28 Oct 2026; needs Daya's OK).

Check A (guide's correction 2). For each comparison, split the gap between today's practice (a baseline B with
per-request admission A1) and the proposal (LSRA with coordinated admission A2). Per episode, paired by seed
and scene, with J the worst-vehicle RLF in percent and L = LSRA:
    T  = J(B,A1) - J(L,A2)     P1 = J(B,A1) - J(L,A1)     Ab = J(B,A1) - J(B,A2)     I = T - P1 - Ab
    P2 = J(B,A2) - J(L,A2)     Al = J(L,A1) - J(L,A2)
    share_planner = (P1 + P2) / 2      share_admission = (Ab + Al) / 2
Only the stored per-episode files of the frozen earlier runs are read; nothing is simulated.

Check B (guide's correction 1). From the environment's own geometry cache of the 60 held-out scenes: how long
link blockages last, how many stations are clear, and how close each vehicle's best link sits to the outage
threshold (geometry only; no planner).

Modes:  unit      synthetic tests (no real outcome is read)
        freeze    md5 record of the code, the frozen plan and every input file
        analyse   the one real run (refuses if a result exists or a frozen file changed)
"""
import os, sys, json, math, time, pickle, hashlib, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'models', 'stored_checks')
PLAN = 'PREREG_LSRA_StoredChecks_PlannerAdmissionSplit_BlockageDurations_FROZEN_20261028.md'
CAPS = [4, 5, 6]
HLS = [2, 5, 10, 15]
HL_MAIN = 10
RULES = ['R0', 'R1', 'R1i', 'R2', 'R3']
A3S = ['A3S1', 'A3S2', 'A3S3', 'A3S4', 'A3S5']
BASELINES = RULES + A3S
LSRA = 'METHOD'
N_EP = 480
TOL = 1e-9
QUANT = ['T', 'P1', 'Ab', 'I', 'P2', 'Al', 'share_planner', 'share_admission']
SEED0 = 9101
STATIONS = ('SW', 'SE', 'NW', 'NE')
N_AGENTS, N_SB, K_BG, WANT_ANT = 16, 3, 4, True
BLOCK_DB = 1.0                      # the paper's "blocked" definition (blockage3d.BLOCK_THR_DB)
SLOT_S = 0.1
SEEDS_CSV = os.path.join(HERE, 'models', 'method_heldout', 'SEEDS_heldout60.csv')
STATS_CSV = os.path.join(HERE, 'HELDOUT60_SCENE_STATS.csv')
CACHE_DIR = os.path.join(HERE, 'geom_cache_ma')
RESULT_JSON = {4: os.path.join(HERE, 'models', 'admission_check', 'ADMIT_RESULT.json'),
               5: os.path.join(HERE, 'models', 'load_check', 'LOAD_RESULT.json'),
               6: os.path.join(HERE, 'models', 'load_check_c6', 'LOAD6_RESULT.json'),
               'A3': os.path.join(HERE, 'models', 'a3_check', 'A3_RESULT.json')}


# ----------------------------------------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def status(msg):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S') + '  ' + msg + '\n')
    print(msg, flush=True)


def ep_path(A, cap, hl, arm, here=None):
    """Stored per-episode file holding arm under admission A, cap places, delay hl (frozen earlier runs)."""
    here = here or HERE
    if arm in A3S:
        return os.path.join(here, 'models', 'a3_check', f'A3_EPISODES_C{cap}_{A}_hl{hl}.csv')
    if cap == 4:
        if A == 'A0':
            return os.path.join(here, 'models', 'method_heldout', f'HELDOUT_EPISODES_hl{hl}.csv')
        return os.path.join(here, 'models', 'admission_check', f'ADMIT_EPISODES_{A}_hl{hl}.csv')
    sub = {5: 'load_check', 6: 'load_check_c6'}[cap]
    return os.path.join(here, 'models', sub, f'LOAD_EPISODES_C{cap}_{A}_hl{hl}.csv')


def input_files():
    """Every stored file the analysis reads (for the freeze record)."""
    files = set()
    for cap in CAPS:
        for hl in HLS:
            for A in ('A0', 'A1', 'A2'):
                files.add(ep_path(A, cap, hl, LSRA))
                files.add(ep_path(A, cap, hl, 'A3S1'))
    files.update(RESULT_JSON.values())
    files.update([SEEDS_CSV, STATS_CSV])
    return sorted(files)


class Store:
    """Reads stored episode files once and returns one cell (arm, admission, places, delay), sorted by seed."""

    def __init__(self, here=None):
        self.here = here
        self.files = {}

    def cell(self, A, cap, hl, arm):
        p = ep_path(A, cap, hl, arm, self.here)
        if p not in self.files:
            d = pd.read_csv(p)
            d['scene'] = d['scene'].astype(str)
            self.files[p] = d
        d = self.files[p]
        c = d[(d['arm'] == arm) & (d['hl'] == hl)]
        if 'cap' in c.columns:
            c = c[c['cap'] == cap]
        if 'adm' in c.columns:
            c = c[c['adm'] == A]
        return c.sort_values('seed').reset_index(drop=True)


def stored_mean(results, A, cap, hl, arm):
    """The level reported by the frozen earlier analyses (secondary.table[...]['worst'], in percent)."""
    if arm in A3S:
        return results['A3']['secondary']['table'][f'{A}_C{cap}_hl{hl}_{arm}']['worst']
    return results[cap]['secondary']['table'][f'{A}_hl{hl}_{arm}']['worst']


def paired(cells, n_ep=N_EP):
    """cells: four data frames (B,A1), (B,A2), (L,A1), (L,A2). Same seeds and scenes, n_ep each, or raise."""
    ref = cells[0]
    for c in cells:
        if len(c) != n_ep:
            raise AssertionError(f'cell has {len(c)} episodes, not {n_ep}')
        if list(c.seed) != list(ref.seed) or list(c.scene) != list(ref.scene):
            raise AssertionError('pairing by seed and scene failed')
    return [c.worst.values * 100.0 for c in cells], ref.scene.values


def split(jb1, jb2, jl1, jl2):
    T = jb1 - jl2
    P1 = jb1 - jl1
    Ab = jb1 - jb2
    I = T - P1 - Ab
    P2 = jb2 - jl2
    Al = jl1 - jl2
    return dict(T=T, P1=P1, Ab=Ab, I=I, P2=P2, Al=Al,
                share_planner=(P1 + P2) / 2.0, share_admission=(Ab + Al) / 2.0)


def identities_ok(q):
    """T = P1 + Ab + I = P1 + Al = Ab + P2 = share_planner + share_admission, per episode."""
    errs = [np.abs(q['T'] - (q['P1'] + q['Ab'] + q['I'])),
            np.abs(q['T'] - (q['P1'] + q['Al'])),
            np.abs(q['T'] - (q['Ab'] + q['P2'])),
            np.abs(q['T'] - (q['share_planner'] + q['share_admission'])),
            np.abs(q['I'] - (q['P2'] - q['P1']))]
    return float(max(float(e.max()) for e in errs))


# ------------------------------------------------------------------------------------------------ check A
def check_a(store, results, boot, caps=None, baselines=None, hls=None, n_ep=N_EP, hl_main=HL_MAIN):
    """boot(d, scenes, seed) -> [lo, hi]; returns (R, problems). caps, baselines, hls default to the plan's."""
    caps = CAPS if caps is None else caps
    baselines = BASELINES if baselines is None else baselines
    hls = HLS if hls is None else hls
    problems = []
    R = {'definitions': 'T = J(B,A1) - J(L,A2); P1 = J(B,A1) - J(L,A1); Ab = J(B,A1) - J(B,A2); I = T - P1 - Ab; '
                        'P2 = J(B,A2) - J(L,A2); Al = J(L,A1) - J(L,A2); share_planner = (P1+P2)/2; '
                        'share_admission = (Ab+Al)/2; J = worst-vehicle RLF in %, per episode, paired by seed and scene',
         'cells': {}}
    seed = SEED0
    m2 = {}
    max_ident = 0.0
    for cap in caps:
        for B in baselines:
            for hl in hls:
                cells = [store.cell('A1', cap, hl, B), store.cell('A2', cap, hl, B),
                         store.cell('A1', cap, hl, LSRA), store.cell('A2', cap, hl, LSRA)]
                try:
                    (jb1, jb2, jl1, jl2), sc = paired(cells, n_ep)
                except AssertionError as e:
                    problems.append(f'M1 C{cap} {B} hl{hl}: {e}')
                    continue
                for (A, arm), x in zip([('A1', B), ('A2', B), ('A1', LSRA), ('A2', LSRA)], [jb1, jb2, jl1, jl2]):
                    key = f'{A}_C{cap}_hl{hl}_{arm}'
                    if key not in m2:
                        dev = abs(float(x.mean()) - float(stored_mean(results, A, cap, hl, arm)))
                        m2[key] = dev
                        if dev > TOL:
                            problems.append(f'M2 {key}: mean differs from the stored value by {dev:.3g}')
                q = split(jb1, jb2, jl1, jl2)
                max_ident = max(max_ident, identities_ok(q))
                R['cells'][f'C{cap}_{B}_hl{hl}'] = dict(
                    levels=dict(B_A1=float(jb1.mean()), B_A2=float(jb2.mean()),
                                L_A1=float(jl1.mean()), L_A2=float(jl2.mean())))
    if max_ident > TOL:
        problems.append(f'M3 identities fail by {max_ident:.3g}')
    R['M2_max_deviation_from_stored'] = float(max(m2.values())) if m2 else None
    R['M3_max_identity_error'] = max_ident
    if problems:
        return R, problems
    # intervals, in the fixed order: load, baseline, quantity, delay
    for cap in caps:
        for B in baselines:
            qs = {}
            for hl in hls:
                cells = [store.cell('A1', cap, hl, B), store.cell('A2', cap, hl, B),
                         store.cell('A1', cap, hl, LSRA), store.cell('A2', cap, hl, LSRA)]
                (jb1, jb2, jl1, jl2), sc = paired(cells, n_ep)
                qs[hl] = (split(jb1, jb2, jl1, jl2), sc)
            for name in QUANT:
                for hl in hls:
                    q, sc = qs[hl]
                    d = q[name]
                    ci = boot(d, sc, seed)
                    seed += 1
                    R['cells'][f'C{cap}_{B}_hl{hl}'][name] = dict(mean=float(d.mean()), scene_95=ci,
                                                                   above_zero=bool(ci[0] > 0.0),
                                                                   below_zero=bool(ci[1] < 0.0))
    R['last_seed_used'] = seed - 1
    # descriptive: best A3 set per load by its A2 level at 1.0 s (for the paper's compact table)
    best = {}
    for cap in caps:
        lv = {s: R['cells'][f'C{cap}_{s}_hl{hl_main}']['levels']['B_A2'] for s in A3S if s in baselines}
        if lv:
            best[f'C{cap}'] = min(lv, key=lv.get)
    R['best_A3_set_by_A2_level_1s'] = best
    return R, problems


def check_m4(store):
    """LSRA's runs under A0 and A2 are identical (Corollary 1), at every load and delay."""
    out = {}
    for cap in CAPS:
        for hl in HLS:
            a0, a2 = store.cell('A0', cap, hl, LSRA), store.cell('A2', cap, hl, LSRA)
            out[f'C{cap}_hl{hl}'] = bool(len(a0) == N_EP and list(a0.seed) == list(a2.seed)
                                         and list(a0.scene) == list(a2.scene)
                                         and np.array_equal(a0.worst.values, a2.worst.values))
    return out


# ------------------------------------------------------------------------------------------------ check B
def cache_file(sid, cache_dir=None):
    """The cache file paper2ma_env._precompute uses for this scene and configuration (same key, same hash)."""
    key = (str(sid), N_AGENTS, N_SB, K_BG, STATIONS, WANT_ANT)
    h = hashlib.md5(repr(key).encode()).hexdigest()[:10]
    return os.path.join(cache_dir or CACHE_DIR, f'{sid}_{h}.pkl')


def link_events(present, blocked):
    """Maximal runs of consecutive frames that are present and blocked. Returns [(length, censored)].
    An event is censored if it touches the first or last frame, or an absence."""
    F = len(present)
    ev = []
    t = 0
    while t < F:
        if present[t] and blocked[t]:
            s = t
            while t + 1 < F and present[t + 1] and blocked[t + 1]:
                t += 1
            e = t
            cens = (s == 0) or (e == F - 1) or (not present[s - 1]) or (not present[e + 1])
            ev.append((e - s + 1, bool(cens)))
        t += 1
    return ev


def scene_arrays(obj):
    fr = obj['frames']
    F, M, G = len(fr), N_AGENTS, len(STATIONS)
    present = np.array([np.asarray(f['active'], dtype=bool) for f in fr])
    depth = np.zeros((F, M, G))
    dist = np.ones((F, M, G))
    for t, f in enumerate(fr):
        for i, a in enumerate(f['ag']):
            if (a is not None) != bool(present[t, i]):
                raise AssertionError('active flag and geometry disagree')
            if a is not None:
                depth[t, i] = a['depth']
                dist[t, i] = a['dist']
    bg = np.array([np.asarray(f['bg'], dtype=float) for f in fr])
    return present, depth, dist, bg


def margins(present, depth, dist, bg, MA):
    """Best station by mean SNR (background interference, no co-channel interference, no shadowing)."""
    PL = MA.CH.ci_pathloss_db(np.maximum(dist, 1.0), True)
    des = MA.PT_MAX + MA.G_SERV - PL - depth
    snr = des - 10.0 * np.log10(MA.N0_MW + bg)[:, None, :]
    g = np.argmax(snr, axis=2)
    best = np.take_along_axis(snr, g[..., None], axis=2)[..., 0]
    dbest = np.take_along_axis(depth, g[..., None], axis=2)[..., 0]
    m = best - MA.SINR_OUT_DB
    sig = np.where(dbest > 0, MA.CH.SIG_NLOS, MA.CH.SIG_LOS)
    return m[present], sig[present]


def check_b(scenes, MA, ndtr, cache_dir=None, stats_csv=None):
    problems = []
    ev_all = []
    n_links = n_link_slots = n_blocked = 0
    blocked_per_slot = 0
    clear_counts = np.zeros(5, dtype=np.int64)
    ms, ss = [], []
    per_scene = []
    depths = []
    for sid in scenes:
        cf = cache_file(sid, cache_dir)
        if not os.path.exists(cf):
            problems.append(f'cache missing for scene {sid}')
            continue
        obj = pickle.load(open(cf, 'rb'))
        present, depth, dist, bg = scene_arrays(obj)
        F = present.shape[0]
        blocked = present[..., None] & (depth >= BLOCK_DB)
        for i in range(N_AGENTS):
            if not present[:, i].any():
                continue
            for g in range(len(STATIONS)):
                n_links += 1
                n_link_slots += int(present[:, i].sum())
                n_blocked += int(blocked[:, i, g].sum())
                ev_all += link_events(present[:, i], blocked[:, i, g])
        blocked_per_slot += int(blocked.sum())
        clear = (depth < BLOCK_DB).sum(axis=2)
        clear_counts += np.bincount(clear[present], minlength=5)[:5]
        depths.append(depth[blocked])
        m, s = margins(present, depth, dist, bg, MA)
        ms.append(m)
        ss.append(s)
        act = present.sum(axis=1)
        per_scene.append(dict(scene=str(sid), frames=int(F), mean_active=float(act.mean()),
                              full16=float((act == N_AGENTS).mean())))
    if problems:
        return {}, problems
    if n_blocked != blocked_per_slot:
        problems.append('blocked share differs between the per-link and per-slot counts')
    ps = pd.DataFrame(per_scene)
    if stats_csv is not None:
        st = pd.read_csv(stats_csv)
        st['scene'] = st['scene'].astype(str)
        mg = ps.merge(st, on='scene', suffixes=('', '_stored'))
        if len(mg) != len(ps) or (mg.frames != mg.frames_stored).any() \
                or (np.abs(mg.mean_active - mg.mean_active_stored) > 1e-6).any() \
                or (np.abs(mg.full16 - mg.full16_stored) > 1e-6).any():
            problems.append('scene statistics differ from HELDOUT60_SCENE_STATS.csv')
    L = np.array([e[0] for e in ev_all], dtype=float) * SLOT_S
    C = np.array([e[1] for e in ev_all], dtype=bool)

    def dist_stats(x):
        if len(x) == 0:
            return None
        return dict(n=int(len(x)), median_s=float(np.median(x)), mean_s=float(x.mean()),
                    p90_s=float(np.percentile(x, 90)), max_s=float(x.max()),
                    share_ge_0p5s=float((x >= 0.5 - 1e-9).mean()), share_ge_1s=float((x >= 1.0 - 1e-9).mean()))

    dep = np.concatenate(depths) if depths else np.zeros(0)
    m = np.concatenate(ms)
    s = np.concatenate(ss)
    R = dict(
        scenes=int(len(per_scene)), links=int(n_links), present_link_slots=int(n_link_slots),
        blocked_share=float(n_blocked / n_link_slots),
        events_all=dist_stats(L), events_uncensored=dist_stats(L[~C]), censored_share=float(C.mean()),
        loss_on_blocked_slots_db=dict(median=float(np.median(dep)), p90=float(np.percentile(dep, 90))),
        clear_stations_share={str(k): float(clear_counts[k] / clear_counts.sum()) for k in range(5)},
        vehicles_present_mean=float(ps.mean_active.mean()), all16_share=float(ps.full16.mean()),
        best_link_margin=dict(n_vehicle_slots=int(len(m)),
                              below_threshold=float((m < 0).mean()),
                              within_1_sigma=float(((m >= 0) & (m < s)).mean()),
                              within_1_to_2_sigma=float(((m >= s) & (m < 2 * s)).mean()),
                              beyond_2_sigma=float((m >= 2 * s).mean()),
                              mean_outage_prob_from_shadowing=float(ndtr(-m / s).mean())))
    return R, problems


# ------------------------------------------------------------------------------------------------ modes
def unit_mode():
    from scipy.special import ndtr
    rep = {}
    # UA1: the split on hand-made numbers
    q = split(np.array([9.0]), np.array([8.0]), np.array([8.5]), np.array([3.5]))
    rep['UA1_split'] = bool(abs(q['T'][0] - 5.5) < 1e-12 and abs(q['P1'][0] - 0.5) < 1e-12
                            and abs(q['Ab'][0] - 1.0) < 1e-12 and abs(q['I'][0] - 4.0) < 1e-12
                            and abs(q['P2'][0] - 4.5) < 1e-12 and abs(q['Al'][0] - 5.0) < 1e-12
                            and abs(q['share_planner'][0] - 2.5) < 1e-12 and abs(q['share_admission'][0] - 3.0) < 1e-12
                            and identities_ok(q) < 1e-12)
    # UA2: pairing failures are caught
    a = pd.DataFrame(dict(seed=np.arange(N_EP), scene=['s%d' % (k % 60) for k in range(N_EP)], worst=0.0))
    b = a.copy(); b.loc[0, 'scene'] = 'other'
    c = a.iloc[:-1].copy()
    caught = 0
    for bad in (b, c):
        try:
            paired([a, a, a, bad])
        except AssertionError:
            caught += 1
    rep['UA2_pairing_failures_caught'] = bool(caught == 2)
    # UA3: Store filters arm, delay, admission and places on a synthetic mixed file
    tmp = os.path.join(OUT, '_unit_tmp')
    os.makedirs(os.path.join(tmp, 'models', 'a3_check'), exist_ok=True)
    rows = []
    for arm in ('A3S1', 'A3S2'):
        for hl in (10, 15):
            for cap in (4, 5):
                for A in ('A1', 'A2'):
                    for k in range(3):
                        rows.append(dict(seed=100 + k, scene=k, worst=(hl + cap + (A == 'A2')) / 1000.0 + k * 1e-4,
                                         arm=arm, hl=hl, cap=cap, adm=A))
    pd.DataFrame(rows).sample(frac=1.0, random_state=1).to_csv(
        os.path.join(tmp, 'models', 'a3_check', 'A3_EPISODES_C4_A2_hl10.csv'), index=False)
    st = Store(here=tmp)
    cc = st.cell('A2', 4, 10, 'A3S1')
    rep['UA3_store_filters'] = bool(len(cc) == 3 and list(cc.seed) == [100, 101, 102]
                                    and np.allclose(cc.worst.values, [(10 + 4 + 1) / 1000.0 + k * 1e-4 for k in range(3)])
                                    and list(cc.scene) == ['0', '1', '2'])
    # UA4: check_a end to end on a small synthetic store (6 episodes): values, M2 flag, M1 flag
    os.makedirs(os.path.join(tmp, 'models', 'admission_check'), exist_ok=True)
    lev = {('A1', 'R1'): 0.09, ('A2', 'R1'): 0.08, ('A1', LSRA): 0.085, ('A2', LSRA): 0.035}

    def write_syn(bad_scene=False):
        for A in ('A1', 'A2'):
            rows = []
            for arm in ('R1', LSRA):
                for k in range(6):
                    sc = k % 3
                    if bad_scene and A == 'A2' and arm == LSRA and k == 0:
                        sc = 9
                    rows.append(dict(seed=500 + k, scene=sc, worst=lev[(A, arm)], arm=arm, hl=10, adm=A))
            pd.DataFrame(rows).to_csv(os.path.join(tmp, 'models', 'admission_check',
                                                   f'ADMIT_EPISODES_{A}_hl10.csv'), index=False)

    def fake_results(shift=0.0):
        t = {f'{A}_hl10_{arm}': {'worst': v * 100.0 + (shift if (A, arm) == ('A1', 'R1') else 0.0)}
             for (A, arm), v in lev.items()}
        return {4: {'secondary': {'table': t}}}

    def run_syn(results):
        return check_a(Store(here=tmp), results, lambda d, s, seed: [float(d.min()), float(d.max())],
                       caps=[4], baselines=['R1'], hls=[10], n_ep=6)

    write_syn()
    Rg, pg = run_syn(fake_results())
    cg = Rg['cells'].get('C4_R1_hl10', {})
    good = (not pg and abs(cg['T']['mean'] - 5.5) < 1e-9 and abs(cg['P1']['mean'] - 0.5) < 1e-9
            and abs(cg['Ab']['mean'] - 1.0) < 1e-9 and abs(cg['I']['mean'] - 4.0) < 1e-9
            and cg['T']['above_zero'] and Rg['last_seed_used'] == SEED0 + len(QUANT) - 1)
    _, pm = run_syn(fake_results(shift=1e-6))
    write_syn(bad_scene=True)
    _, pp_ = run_syn(fake_results())
    rep['UA4_check_a_values'] = bool(good)
    rep['UA4_m2_flag'] = bool(len(pm) == 1 and pm[0].startswith('M2'))
    rep['UA4_m1_flag'] = bool(len(pp_) == 1 and pp_[0].startswith('M1'))
    # UB1: blockage events and censoring
    P = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    Bk = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
    ev = link_events(P, Bk)
    rep['UB1_events'] = bool(ev == [(1, True), (2, True), (1, True), (3, True)])
    P2_ = np.ones(8, dtype=bool)
    B2_ = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype=bool)
    rep['UB1b_uncensored'] = bool(link_events(P2_, B2_) == [(2, False), (1, False)])
    # UB2: margins on a hand-made frame
    class _CH:
        SIG_LOS, SIG_NLOS = 2.7, 8.2

        @staticmethod
        def ci_pathloss_db(d, los):
            return 75.0 + 20.0 * np.log10(d)

    class _MA:
        CH = _CH
        PT_MAX, G_SERV, N0_MW, SINR_OUT_DB = 30.0, 15.0, 1e-8, -13.8

    pres = np.ones((1, 1), dtype=bool)
    dep = np.array([[[0.0, 20.0]]])
    dis = np.array([[[10.0, 10.0]]])
    bgr = np.array([[0.0, 0.0]])
    m, s = margins(pres, dep, dis, bgr, _MA)
    snr0 = 30.0 + 15.0 - 95.0 - 10.0 * math.log10(1e-8)
    rep['UB2_margin'] = bool(abs(m[0] - (snr0 + 13.8)) < 1e-9 and s[0] == 2.7)
    # UB3: the cache hash matches paper2ma_env's own expression (same tuple, same repr)
    key = ('10915', 16, 3, 4, tuple(['SW', 'SE', 'NW', 'NE']), True)
    rep['UB3_hash_expr'] = bool(cache_file('10915', '/x').endswith(
        '10915_' + hashlib.md5(repr(key).encode()).hexdigest()[:10] + '.pkl'))
    # UB4: the cache file of every held-out scene exists (file names only; nothing is read)
    scenes = sorted(set(pd.read_csv(SEEDS_CSV)['scene'].astype(str)))
    rep['UB4_all_60_caches_exist'] = bool(len(scenes) == 60 and all(os.path.exists(cache_file(s)) for s in scenes))
    rep['ok'] = bool(all(v for k, v in rep.items() if k != 'ok'))
    json.dump(rep, open(os.path.join(OUT, 'UNIT.json'), 'w'), indent=1)
    status('unit ' + ('PASS' if rep['ok'] else 'FAIL') + ' ' + json.dumps(rep))


def freeze_mode():
    fz_path = os.path.join(OUT, 'STORED_CHECKS_FREEZE.json')
    assert not os.path.exists(fz_path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PLAN)), 'frozen plan not found'
    unit = json.load(open(os.path.join(OUT, 'UNIT.json')))
    assert unit.get('ok'), 'unit tests did not pass'
    scenes = sorted(set(pd.read_csv(SEEDS_CSV)['scene'].astype(str)))
    files = input_files() + [cache_file(s) for s in scenes]
    rec = dict(written=time.strftime('%Y-%m-%d %H:%M:%S'),
               code={'stored_checks.py': md5(os.path.abspath(__file__)), PLAN: md5(os.path.join(HERE, PLAN)),
                     'models/stored_checks/UNIT.json': md5(os.path.join(OUT, 'UNIT.json'))},
               inputs={os.path.relpath(p, HERE): md5(p) for p in files})
    json.dump(rec, open(fz_path, 'w'), indent=1)
    status(f'freeze written: {len(rec["inputs"])} input files')


def analyse_mode():
    import remeasure_ma as RM
    import paper2ma_env as MA
    from scipy.special import ndtr
    res_path = os.path.join(OUT, 'STORED_CHECKS_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    assert RM.BOOT == 10000, 'bootstrap size changed'
    fz = json.load(open(os.path.join(OUT, 'STORED_CHECKS_FREEZE.json')))
    changed = [k for k, v in fz['code'].items() if md5(os.path.join(HERE, k)) != v]
    changed += [k for k, v in fz['inputs'].items() if md5(os.path.join(HERE, k)) != v]
    if changed:
        status('analyse STOPPED: frozen files changed: ' + ', '.join(changed[:10]))
        return
    status('analyse started')
    results = {k: json.load(open(p)) for k, p in RESULT_JSON.items()}
    store = Store()
    R = {'_freeze_written': fz['written'],
         '_versions': dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)}
    R['M4_LSRA_A0_equals_A2'] = check_m4(store)
    A, pa = check_a(store, results, RM.scene_boot)
    R['check_A'] = A
    scenes = sorted(set(pd.read_csv(SEEDS_CSV)['scene'].astype(str)))
    Bres, pb = check_b(scenes, MA, ndtr, stats_csv=STATS_CSV)
    R['check_B'] = Bres
    problems = pa + pb
    if not all(R['M4_LSRA_A0_equals_A2'].values()):
        problems.append('M4 LSRA A0 and A2 differ in some cell')
    R['problems'] = problems
    R['ok'] = bool(not problems)
    json.dump(R, open(res_path, 'w'), indent=1)
    write_text(R)
    status('analyse done; ok=' + str(R['ok']))


def write_text(R):
    L = ['STORED CHECKS RESULT', 'ok: %s' % R['ok']]
    if R['problems']:
        L += ['PROBLEMS:'] + R['problems']
    A = R.get('check_A', {})
    if A.get('cells') and not R['problems']:
        L.append('')
        L.append('Check A at 1.0 s (pp; 95 % scene intervals): T = P1 + Ab + I')
        for cap in CAPS:
            for B in BASELINES:
                c = A['cells'][f'C{cap}_{B}_hl{HL_MAIN}']
                lv = c['levels']
                row = f'C{cap} {B:5s} levels B/A1 {lv["B_A1"]:6.2f} B/A2 {lv["B_A2"]:6.2f} L/A1 {lv["L_A1"]:6.2f} L/A2 {lv["L_A2"]:6.2f} |'
                for n in ('T', 'P1', 'Ab', 'I'):
                    q = c[n]
                    row += f' {n} {q["mean"]:+6.2f} [{q["scene_95"][0]:+.2f}, {q["scene_95"][1]:+.2f}]'
                L.append(row)
        L.append('best A3 set by A2 level at 1.0 s: ' + json.dumps(A.get('best_A3_set_by_A2_level_1s')))
    Bc = R.get('check_B', {})
    if Bc:
        L.append('')
        L.append('Check B: ' + json.dumps(Bc))
    open(os.path.join(OUT, 'STORED_CHECKS_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['unit', 'freeze', 'analyse'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {'unit': unit_mode, 'freeze': freeze_mode, 'analyse': analyse_mode}[a.mode]()


if __name__ == '__main__':
    main()
