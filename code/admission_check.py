#!/usr/bin/env python3
"""Robustness check of the frozen held-out test: admission at handover preparation (DRAFT, 20 Oct 2026).

Pre-registration: PREREG_MA_AdmissionAtPreparation_Robustness_DRAFT_v1_20261020.md (needs Daya's OK).
Nothing here has been run.
  --mode unit                 G1: the admission rules on hand-made cases and 500 random small cases
                              checked against brute force (no scenes, no policies)
  --mode gate                 G0: with admission off, the new class reproduces the frozen held-out
                              episodes exactly (first 40 seeds, 1.0 s, all 10 arms, every column)
  --mode freeze               writes ADMIT_FREEZE.json once both gates passed (on Daya's OK)
  --mode test --adm A --hl H  all 10 frozen arms under admission model A (A1 per request, A2
                              coordinated) at one delay; refuses without the freeze, if any frozen file
                              changed, or if this cell already ran
  --mode analyse              the registered analysis, once
The method, the baselines, the harness, the scenes and the seeds are the frozen ones of the held-out
test (models/method_heldout/FREEZE.json is checked before every step). Only the environment class
differs (paper2ma_env_admit.AdmissionEnv).
"""
import os, sys, io, json, time, argparse, hashlib, platform, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import remeasure_ma as RM
import method_harness as MH
import method_heldout as MHO
import paper2ma_env_admit as AD

HERE = RM.HERE
OUT = os.path.join(HERE, 'models', 'admission_check')
HLS = MHO.HLS
ADM = {'A1': 'per_request', 'A2': 'coordinated'}
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']
SAME_UNDER_A2 = ['R1i', 'METHOD', 'METHOD_oracle', 'ABL_worstcarfirst']   # never created excess (frozen run)
N_GATE = 40
PREREG = 'PREREG_MA_AdmissionAtPreparation_Robustness_FROZEN_20261020.md'


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__)


def make_env3(pool, hl, admit):
    """Same construction as remeasure_ma2.make_env2(pool, None, hl, attach=True), new class."""
    env = AD.AdmissionEnv(pool, n_agents=16, n_subbands=3, gnb_cap=4, seed=0, split=None,
                          compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = admit
    return env


def frozen(hl):
    df = pd.read_csv(os.path.join(MHO.OUT, f'HELDOUT_EPISODES_hl{hl}.csv'))
    df['scene'] = df['scene'].astype(str)
    return df


def frozen_cols(fz):
    return [c for c in fz.columns if c not in ('hl', 'arm')]


def seeds_all():
    return list(pd.read_csv(os.path.join(MHO.OUT, 'SEEDS_heldout60.csv')).seed.astype(int))


def run_cell(env, fn, seeds):
    env.adm_log = []
    _, _, _, rows = MH.run_eval_m(env, fn, seeds)
    assert len(env.adm_log) == len(seeds) == len(rows), 'one admission record per episode'
    for r, a in zip(rows, env.adm_log):
        r.update(requests=a['requests'], refused=a['refused'], loop_violations=a['loop_violations'])
    return rows


def equal_frames(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        x, y = a[c].values, b[c].values
        if a[c].dtype == object or b[c].dtype == object:
            if not all(str(u) == str(v) for u, v in zip(x, y)):
                return False
        elif not np.array_equal(x.astype(float), y.astype(float)):
            return False
    return True


def same_as_frozen(new_df, fz, arm, hl):
    """new_df: rows as read back from CSV text (same pandas as the frozen run)."""
    cols = frozen_cols(fz)
    a = new_df[cols].reset_index(drop=True).copy()
    a['scene'] = a['scene'].astype(str)
    old = fz[(fz['arm'] == arm) & (fz['hl'] == hl)].set_index('seed').loc[list(a['seed'])].reset_index()
    return equal_frames(a, old[cols].reset_index(drop=True))


def roundtrip(rows):
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


# ---------------------------------------------------------------- G1: unit tests (no scenes)
def unit_mode():
    rep = {'_versions': versions()}
    cases = []

    def case(name, G, cap, nbar, reqs, order, a1, a2, loops=None):
        g1 = AD.admit_per_request(reqs, np.array(nbar), cap, order)
        g2 = AD.admit_coordinated(reqs, np.array(nbar), cap, order)
        ok = (g1 == a1 and g2 == a2)
        rec = dict(name=name, A1=g1, A2=g2, A1_expected=a1, A2_expected=a2)
        if loops is not None:
            ref = [k for k in range(len(reqs)) if k not in set(g1)]
            rec['A1_refused_are_loops'] = AD.refused_is_closed_loops(reqs, ref, G)
            ok = ok and rec['A1_refused_are_loops'] == loops
        rec['ok'] = bool(ok)
        cases.append(rec)

    # (car, from, to); order = strongest first
    case('swap between two full stations', 2, 1, [1, 1], [(0, 0, 1), (1, 1, 0)], [0, 1], [], [0, 1], loops=True)
    case('chain into a free place, listed in the worst order', 3, 1, [1, 1, 0],
         [(0, 0, 1), (1, 1, 2)], [0, 1], [0, 1], [0, 1])
    case('two requests for one free place (stronger wins)', 3, 1, [1, 1, 0],
         [(0, 0, 2), (1, 1, 2)], [1, 0], [1], [1])
    case('into a station over capacity after an appearance', 3, 1, [2, 1, 0], [(5, 1, 0)], [0], [], [])
    case('closed loop plus a chain', 4, 1, [1, 1, 1, 0],
         [(0, 0, 1), (1, 1, 0), (2, 2, 3)], [0, 1, 2], [2], [0, 1, 2], loops=True)
    case('coordinated keeps the largest feasible part', 3, 1, [1, 1, 1],
         [(0, 0, 1), (1, 1, 0), (2, 2, 0)], [0, 1, 2], [], [0, 1])

    rng = np.random.default_rng(20261020)
    mism = dict(A2_not_largest=0, A2_tiebreak=0, A2_excess=0, A1_excess=0, A1_more_than_A2=0,
                A1_refused_not_loops=0)
    n_loop_cases = 0
    N = 500
    for _ in range(N):
        G = int(rng.integers(2, 5)); cap = int(rng.integers(1, 4))
        nbar = rng.integers(0, cap + 2, G)                       # sometimes over capacity
        reqs = []; car = 0
        for g in range(G):
            for _ in range(int(rng.integers(0, min(int(nbar[g]), 2) + 1))):
                reqs.append((car, g, int(rng.choice([x for x in range(G) if x != g])))); car += 1
        order = [int(k) for k in rng.permutation(len(reqs))]
        a1 = AD.admit_per_request(reqs, nbar, cap, order)
        a2 = AD.admit_coordinated(reqs, nbar, cap, order)
        rank = {k: r for r, k in enumerate(order)}
        n = len(reqs)

        def wsum(sub):
            return sum(1.0 - rank[k] / n for k in sub) if n else 0.0

        best_n, best_w = -1, -1.0
        for m in range(n + 1):
            for sub in itertools.combinations(range(n), m):
                if AD.creates_excess(nbar, reqs, list(sub), cap):
                    continue
                w = wsum(sub)
                if m > best_n or (m == best_n and w > best_w + 1e-12):
                    best_n, best_w = m, w
        mism['A2_not_largest'] += int(len(a2) != best_n)
        mism['A2_tiebreak'] += int(len(a2) == best_n and abs(wsum(a2) - best_w) > 1e-9)
        mism['A2_excess'] += int(AD.creates_excess(nbar, reqs, a2, cap))
        mism['A1_excess'] += int(AD.creates_excess(nbar, reqs, a1, cap))
        mism['A1_more_than_A2'] += int(len(a1) > len(a2))
        if (nbar <= cap).all() and not AD.creates_excess(nbar, reqs, list(range(n)), cap):
            n_loop_cases += 1
            ref = [k for k in range(n) if k not in set(a1)]
            mism['A1_refused_not_loops'] += int(not AD.refused_is_closed_loops(reqs, ref, G))
    rep['cases'] = cases
    rep['random'] = dict(n=N, cases_where_loop_property_applies=n_loop_cases, mismatches=mism)
    rep['ok'] = bool(all(c['ok'] for c in cases) and all(v == 0 for v in mism.values()))
    json.dump(rep, open(os.path.join(OUT, 'UNIT.json'), 'w'), indent=1)
    status(f'[unit] ok={rep["ok"]} cases={[c["ok"] for c in cases]} random={mism}')


# ---------------------------------------------------------------- G0: admission off = frozen run
def gate_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = seeds_all()[:N_GATE]
    fz = frozen(10)
    env = make_env3(pool, 10, 'none')
    per = {}
    for name, fn in MHO.arms(fzm['beta']):
        t0 = time.time()
        rows = run_cell(env, fn, seeds)
        per[name] = bool(same_as_frozen(roundtrip(rows), fz, name, 10))
        status(f'[gate] {name}: identical to the frozen run = {per[name]} ({time.time() - t0:.0f} s)')
    rep = dict(_versions=versions(), n_seeds=N_GATE, hl=10, per_arm=per, ok=bool(all(per.values())))
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_mode():
    u = json.load(open(os.path.join(OUT, 'UNIT.json')))
    g = json.load(open(os.path.join(OUT, 'GATE.json')))
    assert u['ok'] and g['ok'], 'a gate did not pass: no freeze'
    path = os.path.join(OUT, 'ADMIT_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    files = (['paper2ma_env_admit.py', 'admission_check.py', PREREG,
              'models/admission_check/UNIT.json', 'models/admission_check/GATE.json',
              'models/method_heldout/FREEZE.json']
             + [f'models/method_heldout/HELDOUT_EPISODES_hl{h}.csv' for h in HLS])
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'],
               md5={f: md5(os.path.join(HERE, f)) for f in files})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_admit_freeze():
    fz = json.load(open(os.path.join(OUT, 'ADMIT_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run
def test_mode(A, hl):
    fz = check_admit_freeze()
    out = os.path.join(OUT, f'ADMIT_EPISODES_{A}_hl{hl}.csv')
    assert not os.path.exists(out), 'this cell already ran: each cell runs once'
    pool = pd.read_csv(MHO.POOL_H)
    seeds = seeds_all()
    env = make_env3(pool, hl, ADM[A])
    rows = []
    status(f'test {A} hl={hl} start (beta {fz["beta"]})')
    for name, fn in MHO.arms(fz['beta']):
        t0 = time.time()
        r = run_cell(env, fn, seeds)
        for x in r:
            x.update(hl=hl, arm=name, adm=A)
        rows += r
        pd.DataFrame(rows).to_csv(out + '.partial', index=False)
        status(f'test {A} hl={hl} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test {A} hl={hl} DONE')


# ---------------------------------------------------------------- analysis (once)
def analyse():
    fz = check_admit_freeze()
    res_path = os.path.join(OUT, 'ADMIT_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    names = [a for a, _ in MHO.arms(fz['beta'])]
    parts = []
    for h in HLS:
        f0 = frozen(h); f0['adm'] = 'A0'; parts.append(f0)
    new = {}
    for A in ADM:
        for h in HLS:
            d = pd.read_csv(os.path.join(OUT, f'ADMIT_EPISODES_{A}_hl{h}.csv'))
            d['scene'] = d['scene'].astype(str)
            new[(A, h)] = d
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)

    def cell(A, hl, arm):
        return df[(df['adm'] == A) & (df['hl'] == hl) & (df['arm'] == arm)].sort_values('seed')

    def diff(A, a, B, b, hl):
        x, y = cell(A, hl, a), cell(B, hl, b)
        assert list(x.seed) == list(y.seed) and len(x) == 480
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    R = {'_freeze': fz, '_versions': versions()}
    # must-hold checks
    newdf = pd.concat(list(new.values()), ignore_index=True)
    m1 = int(newdf.dec_over.sum())
    m2 = {}
    for h in HLS:
        fzh = frozen(h)
        for arm in SAME_UNDER_A2:
            d = new[('A2', h)]
            d = d[d['arm'] == arm].sort_values('seed')
            m2[f'hl{h}_{arm}'] = bool(same_as_frozen(d, fzh, arm, h) and int(d.refused.sum()) == 0)
    m3 = int(newdf[newdf['adm'] == 'A1'].loop_violations.sum())
    m4 = int((newdf.starts != newdf.requests - newdf.refused).sum())
    checks = dict(M1_created_excess_total=m1, M2_same_as_frozen_under_A2=m2, M3_loop_violations=m3,
                  M4_rows_where_starts_differ=m4)
    checks['ok'] = bool(m1 == 0 and all(m2.values()) and m3 == 0 and m4 == 0)
    R['checks'] = checks

    def family(A, seed0):
        per = {}
        for k, b in enumerate(PRIMARY_BASELINES):
            d, sc = diff(A, b, A, 'METHOD', 10)
            ci = RM.scene_boot(d, sc, seed0 + k, (0.625, 99.375))
            per[b] = dict(mean=float(d.mean()), scene_9875=ci, scene_95=RM.scene_boot(d, sc, seed0 + 50 + k),
                          beats=bool(ci[0] > 0.0))
        return dict(per_baseline=per, held=bool(all(v['beats'] for v in per.values())))

    R['HR1_A1'] = family('A1', 3000)
    R['HR2_A2'] = family('A2', 3100)
    d3, sc3 = diff('A2', 'ABL_nolanding', 'A2', 'METHOD', 10)
    ci3 = RM.scene_boot(d3, sc3, 3200)
    R['HR3_A2_nolanding_minus_METHOD'] = dict(mean=float(d3.mean()), scene_95=ci3, held=bool(ci3[0] > 0.0))

    # secondary
    S = {}; sb = 3300; T = {}
    for A in ['A0', 'A1', 'A2']:
        for hl in HLS:
            for arm in names:
                c = cell(A, hl, arm); w = c.worst.values * 100.0; wa = np.maximum(c.worst_act.values, 1.0)
                e = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                         ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                         causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
                sb += 1
                if A != 'A0':
                    e['requests_per_car_step'] = float(c.requests.sum() / c.p_act.sum())
                    e['refused_per_car_step'] = float(c.refused.sum() / c.p_act.sum())
                    e['refused_share'] = float(c.refused.sum() / max(c.requests.sum(), 1))
                if arm != 'METHOD':
                    d, s2 = diff(A, arm, A, 'METHOD', hl)
                    e['minus_METHOD_same_admission'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb))
                    sb += 1
                T[f'{A}_hl{hl}_{arm}'] = e
    S['table'] = T
    V = {}
    for hl in HLS:
        d, s2 = diff('A1', 'METHOD', 'A2', 'METHOD', hl)
        V[f'hl{hl}_METHOD_A1_minus_A2'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
        for A in ['A1', 'A2']:
            d, s2 = diff(A, 'R3', 'A0', 'R3', hl)
            V[f'hl{hl}_R3_{A}_minus_A0'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
    for A in ['A1', 'A2']:
        r10, sc = diff(A, 'R3', A, 'METHOD', 10); r2, _ = diff(A, 'R3', A, 'METHOD', 2)
        V[f'{A}_R3_cost_of_delay_minus_METHOD_cost'] = dict(mean=float((r10 - r2).mean()),
                                                           scene_95=RM.scene_boot(r10 - r2, sc, sb)); sb += 1
    S['contrasts'] = V
    R['secondary'] = S

    # decisions fixed in advance
    h1, h2, h3 = R['HR1_A1']['held'], R['HR2_A2']['held'], R['HR3_A2_nolanding_minus_METHOD']['held']
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif h1 and h2:
        outcome = 'survives admission at preparation, with or without coordination'
    elif h2:
        outcome = 'the gain needs coordinated admission: state the condition; standards check before claiming a RIC can do it'
    elif not R['HR2_A2']['per_baseline']['R3']['beats']:
        outcome = 'R3 with coordinated admission is not beaten: withdraw the claim against R3 in its current form; pause the write-up and discuss'
    else:
        outcome = 'HR2 not held against a simple baseline: report as is and discuss'
    R['decision'] = dict(outcome=outcome, HR1_held=h1, HR2_held=h2, HR3_held=h3,
                         may_claim_landing_state_helps_beyond_admission=bool(checks['ok'] and h3))
    R['guesses'] = dict(
        METHOD_A1_1s_between_3_6_and_5_0=bool(3.6 <= T['A1_hl10_METHOD']['worst'] <= 5.0),
        R3_A1_1s_between_4_and_7=bool(4.0 <= T['A1_hl10_R3']['worst'] <= 7.0),
        R3_A2_1s_between_3_5_and_6=bool(3.5 <= T['A2_hl10_R3']['worst'] <= 6.0),
        nolanding_A2_1s_between_4_and_6=bool(4.0 <= T['A2_hl10_ABL_nolanding']['worst'] <= 6.0),
        HR1_held=h1, HR2_held=h2, HR3_held=h3)
    json.dump(R, open(res_path, 'w'), indent=1)

    L = ['ADMISSION AT HANDOVER PREPARATION: robustness check of the frozen held-out test',
         '(60 held-out DAIR scenes, 480 episodes per cell, the frozen arms and seeds)', '',
         f"must-hold checks: {checks}", '']
    for key, title in [('HR1_A1', 'HR1, A1 per-request admission'), ('HR2_A2', 'HR2, A2 coordinated admission')]:
        L.append(f'{title} (1.0 s; baseline minus METHOD, worst-car RLF pp; 98.75 % scene interval):')
        for b, v in R[key]['per_baseline'].items():
            L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    v3 = R['HR3_A2_nolanding_minus_METHOD']
    L.append(f"HR3 (A2, 1.0 s, ABL_nolanding minus METHOD, 95 %): {v3['mean']:+.2f} "
             f"[{v3['scene_95'][0]:+.2f}, {v3['scene_95'][1]:+.2f}] -> {'HELD' if v3['held'] else 'NOT HELD'}")
    L.append('')
    for A in ['A0', 'A1', 'A2']:
        L.append(f'{A}: arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS)
                 + ' | handovers/car-step @1.0 s | refused/car-step @1.0 s')
        for arm in names:
            e10 = T[f'{A}_hl10_{arm}']
            L.append(f'  {arm} | ' + ' | '.join(
                f"{T[f'{A}_hl{h}_{arm}']['worst']:.2f} [{T[f'{A}_hl{h}_{arm}']['scene_95'][0]:.2f}, "
                f"{T[f'{A}_hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
                + f" | {e10['ho_per_car_step']:.4f} | {e10.get('refused_per_car_step', float('nan')):.4f}")
        L.append('')
    L.append('causes of the worst car at 1.0 s:')
    for A in ['A0', 'A1', 'A2']:
        for arm in names:
            cw = T[f'{A}_hl10_{arm}']['causes_worst']
            L.append(f'  {A} {arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append('contrasts (95 % scene intervals):')
    for k, v in V.items():
        L.append(f"  {k}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'ADMIT_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['unit', 'gate', 'freeze', 'test', 'analyse'], required=True)
    ap.add_argument('--adm', choices=list(ADM))
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'unit':
        unit_mode()
    elif a.mode == 'gate':
        gate_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    elif a.mode == 'test':
        assert a.hl in HLS and a.adm in ADM
        test_mode(a.adm, a.hl)
    else:
        analyse()
