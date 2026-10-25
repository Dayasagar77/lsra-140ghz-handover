#!/usr/bin/env python3
"""Load check, third level: the three admission models at 6 places per station (DRAFT, 22 Oct 2026).

Pre-registration: PREREG_MA_LoadCheck_LighterLoad_C6_DRAFT_v1_20261022.md (needs Daya's OK).
Nothing here has been run on 6 places.

Same 60 held-out scenes, seed list, frozen arms (method_heldout.arms, beta 0.05), harness
(method_harness.run_eval_m) and environment class (paper2ma_env_admit.AdmissionEnv) as the held-out test,
the admission check (4 places) and the first load check (5 places, load_check.py, unchanged). ONE change:
every station has 6 places, so 16 vehicles always leave at least 8 of the 24 places free (load 0.67).
  --mode gate                 G0a: this runner at 4 places with admission off reproduces the frozen held-out
                              episodes exactly (first 40 seeds, 1.0 s, all 10 arms);
                              G0b: at 5 places under A1 it reproduces the first load check's episodes exactly
                              (first 40 seeds, 1.0 s, all 10 arms)
  --mode freeze               writes LOAD6_FREEZE.json once both gates passed (on Daya's OK)
  --mode test --adm A --hl H  all 10 frozen arms at 6 places under A0 (admission at landing only),
                              A1 (per request) or A2 (coordinated), at one delay; each cell runs once
  --mode analyse              the registered analysis, once
"""
import os, sys, json, time, argparse, hashlib, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import paper2ma_env_admit as AD

HERE = RM.HERE
OUT = os.path.join(HERE, 'models', 'load_check_c6')
C5DIR = os.path.join(HERE, 'models', 'load_check')
HLS = MHO.HLS
CAP = 6
ADM = {'A0': 'none', 'A1': 'per_request', 'A2': 'coordinated'}
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']
SIMPLE_RULES = ['R1', 'R1i', 'R2']
N_GATE = 40
PREREG = 'PREREG_MA_LoadCheck_LighterLoad_C6_FROZEN_20261022.md'
EXTRA = ('hl', 'arm', 'adm', 'cap', 'requests', 'refused', 'loop_violations')
# finished files read for the paired comparisons with 5 and 4 places (md5 recorded at the freeze)
READ_FILES = (['models/load_check/LOAD_FREEZE.json']
              + [f'models/load_check/LOAD_EPISODES_C5_{A}_hl10.csv' for A in ADM]
              + ['models/method_heldout/HELDOUT_EPISODES_hl10.csv',
                 'models/admission_check/ADMIT_EPISODES_A1_hl10.csv',
                 'models/admission_check/ADMIT_EPISODES_A2_hl10.csv'])


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def make_env_cap(pool, hl, admit, cap):
    """As load_check.make_env_cap: remeasure_ma2.make_env2(pool, None, hl, attach=True) with the number of
    places per station as a parameter and the admission option of paper2ma_env_admit."""
    env = AD.AdmissionEnv(pool, n_agents=16, n_subbands=3, gnb_cap=cap, seed=0, split=None,
                          compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = admit
    return env


def cell_path(A, hl):
    return os.path.join(OUT, f'LOAD_EPISODES_C{CAP}_{A}_hl{hl}.csv')


def frames_equal_typed(a, b):
    """Exact comparison, column by column: numeric columns as floats (NaN equal to NaN), all others as text.
    Added 22 Oct after the first gate attempt: admission_check.equal_frames treats only the object dtype as
    text, and under pandas 3 the text columns 'arm' and 'adm' are read with the string dtype, so G0b
    stopped with a ValueError before comparing anything. Used only by G0b."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True):
                return False
        elif not all(str(u) == str(v) for u, v in zip(a[c].tolist(), b[c].tolist())):
            return False
    return True


# ---------------------------------------------------------------- G0a and G0b
def gate_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()[:N_GATE]
    # G0a: 4 places, admission off = the frozen held-out run
    fz = AC.frozen(10)
    env4 = make_env_cap(pool, 10, 'none', 4)
    per_a = {}
    for name, fn in MHO.arms(fzm['beta']):
        t0 = time.time()
        rows = AC.run_cell(env4, fn, seeds)
        per_a[name] = bool(AC.same_as_frozen(AC.roundtrip(rows), fz, name, 10))
        status(f'[gate a] {name}: identical to the frozen run = {per_a[name]} ({time.time() - t0:.0f} s)')
    # G0b: 5 places under A1 = the first load check (every column, the same 40 seeds)
    c5 = pd.read_csv(os.path.join(C5DIR, 'LOAD_EPISODES_C5_A1_hl10.csv'))
    env5 = make_env_cap(pool, 10, ADM['A1'], 5)
    per_b = {}
    for name, fn in MHO.arms(fzm['beta']):
        t0 = time.time()
        rows = AC.run_cell(env5, fn, seeds)
        for x in rows:
            x.update(hl=10, arm=name, adm='A1', cap=5)
        new = AC.roundtrip(rows)
        old = c5[c5['arm'] == name].set_index('seed').loc[list(new['seed'])].reset_index()
        old = old[list(new.columns)]
        per_b[name] = bool(frames_equal_typed(new.reset_index(drop=True), old.reset_index(drop=True)))
        status(f'[gate b] {name}: identical to the 5-place A1 run = {per_b[name]} ({time.time() - t0:.0f} s)')
    rep = dict(_versions=AC.versions(), n_seeds=N_GATE, hl=10, G0a_cap4_admission_off=per_a,
               G0b_cap5_A1=per_b, ok=bool(all(per_a.values()) and all(per_b.values())))
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_mode():
    g = json.load(open(os.path.join(OUT, 'GATE.json')))
    assert g['ok'], 'the gate did not pass: no freeze'
    path = os.path.join(OUT, 'LOAD6_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    files = (['load_check_c6.py', 'admission_check.py', 'paper2ma_env_admit.py', PREREG,
              'models/load_check_c6/GATE.json', 'models/method_heldout/FREEZE.json',
              'models/admission_check/ADMIT_FREEZE.json'] + READ_FILES)
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'], cap=CAP,
               md5={f: md5(os.path.join(HERE, f)) for f in files})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_load6_freeze():
    fz = json.load(open(os.path.join(OUT, 'LOAD6_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run
def test_mode(A, hl):
    fz = check_load6_freeze()
    out = cell_path(A, hl)
    assert not os.path.exists(out), 'this cell already ran: each cell runs once'
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    env = make_env_cap(pool, hl, ADM[A], CAP)
    rows = []
    status(f'test C{CAP} {A} hl={hl} start (beta {fz["beta"]})')
    for name, fn in MHO.arms(fz['beta']):
        t0 = time.time()
        r = AC.run_cell(env, fn, seeds)
        for x in r:
            x.update(hl=hl, arm=name, adm=A, cap=CAP)
        rows += r
        pd.DataFrame(rows).to_csv(out + '.partial', index=False)
        status(f'test C{CAP} {A} hl={hl} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test C{CAP} {A} hl={hl} DONE')


# ---------------------------------------------------------------- analysis (once)
def analyse():
    fz = check_load6_freeze()
    res_path = os.path.join(OUT, 'LOAD6_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    names = [a for a, _ in MHO.arms(fz['beta'])]
    parts = []
    for A in ADM:
        for h in HLS:
            d = pd.read_csv(cell_path(A, h))
            d['scene'] = d['scene'].astype(str)
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # 5 places (first load check) and 4 places (frozen A0, admission check A1/A2), 1.0 s, same seeds
    p5 = []
    for A in ADM:
        d = pd.read_csv(os.path.join(C5DIR, f'LOAD_EPISODES_C5_{A}_hl10.csv'))
        d['scene'] = d['scene'].astype(str)
        p5.append(d)
    df5 = pd.concat(p5, ignore_index=True)
    f0 = AC.frozen(10)
    f0['adm'] = 'A0'
    p4 = [f0]
    for A in ('A1', 'A2'):
        d = pd.read_csv(os.path.join(AC.OUT, f'ADMIT_EPISODES_{A}_hl10.csv'))
        d['scene'] = d['scene'].astype(str)
        p4.append(d)
    df4 = pd.concat(p4, ignore_index=True)
    LOADS = {4: df4, 5: df5, CAP: df}

    def cell(A, hl, arm, d=None):
        d = df if d is None else d
        return d[(d['adm'] == A) & (d['hl'] == hl) & (d['arm'] == arm)].sort_values('seed')

    def diff(A, a, B, b, hl, d=None):
        x, y = cell(A, hl, a, d), cell(B, hl, b, d)
        assert list(x.seed) == list(y.seed) and len(x) == 480
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    R = {'_freeze': fz, '_versions': AC.versions()}
    # must-hold checks
    adm = df[df['adm'].isin(['A1', 'A2'])]
    m1 = int(adm.dec_over.sum())
    core = [c for c in df.columns if c not in EXTRA]
    m2 = {}
    for h in HLS:
        for arm in names:
            a0 = cell('A0', h, arm)
            if int(a0.dec_over.sum()) != 0:
                continue                                  # this arm created excess at A0: no identity expected
            a2 = cell('A2', h, arm)
            same = AC.equal_frames(a2[core].reset_index(drop=True), a0[core].reset_index(drop=True))
            m2[f'hl{h}_{arm}'] = bool(same and int(a2.refused.sum()) == 0)
    m3 = int(df[df['adm'] == 'A1'].loop_violations.sum())
    m4 = int((adm.starts != adm.requests - adm.refused).sum())
    checks = dict(M1_created_excess_total=m1, M2_same_as_A0_under_A2=m2, M3_loop_violations=m3,
                  M4_rows_where_starts_differ=m4)
    checks['ok'] = bool(m1 == 0 and len(m2) > 0 and all(m2.values()) and m3 == 0 and m4 == 0)
    R['checks'] = checks

    def family(A, seed0):
        per = {}
        for k, b in enumerate(PRIMARY_BASELINES):
            d, sc = diff(A, b, A, 'METHOD', 10)
            ci = RM.scene_boot(d, sc, seed0 + k, (0.625, 99.375))
            per[b] = dict(mean=float(d.mean()), scene_9875=ci, scene_95=RM.scene_boot(d, sc, seed0 + 50 + k),
                          beats=bool(ci[0] > 0.0))
        return dict(per_baseline=per, held=bool(all(v['beats'] for v in per.values())))

    R['HL3_A1'] = family('A1', 6000)
    R['HL4_A2'] = family('A2', 6100)

    S = {}; sb = 6300; T = {}
    for A in ADM:
        for hl in HLS:
            for arm in names:
                c = cell(A, hl, arm); w = c.worst.values * 100.0; wa = np.maximum(c.worst_act.values, 1.0)
                e = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                         ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                         requests_per_car_step=float(c.requests.sum() / c.p_act.sum()),
                         refused_per_car_step=float(c.refused.sum() / c.p_act.sum()),
                         refused_share=float(c.refused.sum() / max(c.requests.sum(), 1)),
                         causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
                sb += 1
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
        for A in ('A1', 'A2'):
            d, s2 = diff(A, 'R3', 'A0', 'R3', hl)
            V[f'hl{hl}_R3_{A}_minus_A0'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
    # 6 places minus 5 places and minus 4 places, same seeds, 1.0 s
    for A in ADM:
        for arm in ('METHOD', 'R1', 'R1i', 'R2', 'R3'):
            x = cell(A, 10, arm)
            for other in (5, 4):
                y = cell(A, 10, arm, LOADS[other])
                assert list(x.seed) == list(y.seed) and len(x) == 480
                d = (x.worst.values - y.worst.values) * 100.0
                V[f'{A}_{arm}_{CAP}places_minus_{other}places_1s'] = dict(
                    mean=float(d.mean()), scene_95=RM.scene_boot(d, y.scene.values, sb)); sb += 1
    # the load curve at 1.0 s (descriptive): METHOD's lead over each simple rule and over R3, at 4, 5 and 6 places
    LC = {}
    for A in ADM:
        for load in (4, 5, CAP):
            dd = LOADS[load]
            for b in SIMPLE_RULES + ['R3']:
                d, s2 = diff(A, b, A, 'METHOD', 10, dd)
                LC[f'{A}_{load}places_{b}_minus_METHOD'] = dict(mean=float(d.mean()),
                                                                scene_95=RM.scene_boot(d, s2, sb)); sb += 1
    V['load_curve'] = LC
    S['contrasts'] = V
    R['secondary'] = S

    h3, h4 = R['HL3_A1']['held'], R['HL4_A2']['held']
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif h3 and h4:
        outcome = 'at load 0.67 the gain holds under both admission models: per-request admission no longer blocks it at this load'
    elif h4:
        outcome = 'at load 0.67 the gain still needs coordinated admission'
    else:
        outcome = ('HL4 not held: at load 0.67 the method is not shown to beat all four baselines even with '
                   'coordinated admission; the claim is limited to full and near-full load')
    R['decision'] = dict(outcome=outcome, HL3_held=h3, HL4_held=h4)
    R['guesses'] = dict(
        METHOD_A2_1s_between_1p5_and_3=bool(1.5 <= T['A2_hl10_METHOD']['worst'] <= 3.0),
        R1i_A2_1s_between_3_and_5=bool(3.0 <= T['A2_hl10_R1i']['worst'] <= 5.0),
        METHOD_A1_minus_A2_1s_below_0p5pp=bool(V['hl10_METHOD_A1_minus_A2']['mean'] < 0.5),
        HL3_held=h3, HL4_held=h4)
    json.dump(R, open(res_path, 'w'), indent=1)

    L = [f'LOAD CHECK, THIRD LEVEL: {CAP} places per station (16 vehicles, {4 * CAP} places), the three admission models',
         '(60 held-out DAIR scenes, 480 episodes per cell, the frozen arms and seeds)', '',
         f"must-hold checks: {checks}", '']
    for key, title in [('HL3_A1', 'HL3, A1 per-request admission'), ('HL4_A2', 'HL4, A2 coordinated admission')]:
        L.append(f'{title} (1.0 s; baseline minus METHOD, worst-vehicle RLF pp; 98.75 % scene interval):')
        for b, v in R[key]['per_baseline'].items():
            L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    L.append('')
    for A in ADM:
        L.append(f'{A}: arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS)
                 + ' | handovers/vehicle-step @1.0 s | refused share @1.0 s')
        for arm in names:
            e10 = T[f'{A}_hl10_{arm}']
            L.append(f'  {arm} | ' + ' | '.join(
                f"{T[f'{A}_hl{h}_{arm}']['worst']:.2f} [{T[f'{A}_hl{h}_{arm}']['scene_95'][0]:.2f}, "
                f"{T[f'{A}_hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
                + f" | {e10['ho_per_car_step']:.4f} | {e10['refused_share']:.3f}")
        L.append('')
    L.append('every arm minus METHOD, same admission (95 % scene intervals):')
    for A in ADM:
        for hl in HLS:
            L.append(f'  {A} {hl * 0.1:.1f} s: ' + '  '.join(
                f"{arm} {T[f'{A}_hl{hl}_{arm}']['minus_METHOD_same_admission']['mean']:+.2f} "
                f"[{T[f'{A}_hl{hl}_{arm}']['minus_METHOD_same_admission']['scene_95'][0]:+.2f}, "
                f"{T[f'{A}_hl{hl}_{arm}']['minus_METHOD_same_admission']['scene_95'][1]:+.2f}]"
                for arm in names if arm != 'METHOD'))
    L.append(''); L.append('causes of the worst vehicle at 1.0 s:')
    for A in ADM:
        for arm in names:
            cw = T[f'{A}_hl10_{arm}']['causes_worst']
            L.append(f'  {A} {arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append('contrasts (95 % scene intervals):')
    for k, v in V.items():
        if k == 'load_curve':
            continue
        L.append(f"  {k}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append(''); L.append('load curve at 1.0 s, baseline minus METHOD (95 % scene intervals; descriptive):')
    for k, v in LC.items():
        L.append(f"  {k}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'LOAD6_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['gate', 'freeze', 'test', 'analyse'], required=True)
    ap.add_argument('--adm', choices=list(ADM))
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'gate':
        gate_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    elif a.mode == 'test':
        assert a.hl in HLS and a.adm in ADM
        test_mode(a.adm, a.hl)
    else:
        analyse()
