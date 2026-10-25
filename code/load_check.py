#!/usr/bin/env python3
"""Load check: the three admission models when the stations have spare places (DRAFT, 20 Oct 2026).

Pre-registration: PREREG_MA_LoadCheck_SparePlaces_C5_DRAFT_v1_20261020.md (needs Daya's OK).
Nothing here has been run.

Same 60 held-out scenes, seed list, frozen arms (method_heldout.arms, beta 0.05), harness
(method_harness.run_eval_m) and environment class (paper2ma_env_admit.AdmissionEnv) as the held-out
test and the admission check. ONE change: every station has 5 places instead of 4, so 16 cars can
never fill all 20 places at once.
  --mode gate                 G0: this runner at 4 places with admission off reproduces the frozen
                              held-out episodes exactly (first 40 seeds, 1.0 s, all 10 arms)
  --mode freeze               writes LOAD_FREEZE.json once the gate passed (on Daya's OK)
  --mode test --adm A --hl H  all 10 frozen arms at 5 places under A0 (admission at landing only),
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
OUT = os.path.join(HERE, 'models', 'load_check')
HLS = MHO.HLS
CAP = 5
ADM = {'A0': 'none', 'A1': 'per_request', 'A2': 'coordinated'}
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']
N_GATE = 40
PREREG = 'PREREG_MA_LoadCheck_SparePlaces_C5_FROZEN_20261020.md'
EXTRA = ('hl', 'arm', 'adm', 'cap', 'requests', 'refused', 'loop_violations')


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def make_env_cap(pool, hl, admit, cap):
    """As remeasure_ma2.make_env2(pool, None, hl, attach=True), with the number of places per station
    as a parameter and the admission option of paper2ma_env_admit."""
    env = AD.AdmissionEnv(pool, n_agents=16, n_subbands=3, gnb_cap=cap, seed=0, split=None,
                          compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = admit
    return env


def cell_path(A, hl):
    return os.path.join(OUT, f'LOAD_EPISODES_C{CAP}_{A}_hl{hl}.csv')


# ---------------------------------------------------------------- G0: runner at 4 places = frozen run
def gate_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()[:N_GATE]
    fz = AC.frozen(10)
    env = make_env_cap(pool, 10, 'none', 4)
    per = {}
    for name, fn in MHO.arms(fzm['beta']):
        t0 = time.time()
        rows = AC.run_cell(env, fn, seeds)
        per[name] = bool(AC.same_as_frozen(AC.roundtrip(rows), fz, name, 10))
        status(f'[gate] {name}: identical to the frozen run = {per[name]} ({time.time() - t0:.0f} s)')
    rep = dict(_versions=AC.versions(), n_seeds=N_GATE, hl=10, cap=4, per_arm=per, ok=bool(all(per.values())))
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_mode():
    g = json.load(open(os.path.join(OUT, 'GATE.json')))
    assert g['ok'], 'the gate did not pass: no freeze'
    path = os.path.join(OUT, 'LOAD_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    files = ['load_check.py', 'admission_check.py', 'paper2ma_env_admit.py', PREREG,
             'models/load_check/GATE.json', 'models/method_heldout/FREEZE.json',
             'models/admission_check/ADMIT_FREEZE.json']
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'], cap=CAP,
               md5={f: md5(os.path.join(HERE, f)) for f in files})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_load_freeze():
    fz = json.load(open(os.path.join(OUT, 'LOAD_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


# ---------------------------------------------------------------- the run
def test_mode(A, hl):
    fz = check_load_freeze()
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
    fz = check_load_freeze()
    res_path = os.path.join(OUT, 'LOAD_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    names = [a for a, _ in MHO.arms(fz['beta'])]
    parts = []
    for A in ADM:
        for h in HLS:
            d = pd.read_csv(cell_path(A, h))
            d['scene'] = d['scene'].astype(str)
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    # the 4-place runs, for paired comparisons (same seeds): frozen A0 and the admission check A1/A2
    p4 = []
    for h in HLS:
        f0 = AC.frozen(h); f0['adm'] = 'A0'; p4.append(f0)
        for A in ('A1', 'A2'):
            d = pd.read_csv(os.path.join(AC.OUT, f'ADMIT_EPISODES_{A}_hl{h}.csv'))
            d['scene'] = d['scene'].astype(str); p4.append(d)
    df4 = pd.concat(p4, ignore_index=True)

    def cell(A, hl, arm, d=None):
        d = df if d is None else d
        return d[(d['adm'] == A) & (d['hl'] == hl) & (d['arm'] == arm)].sort_values('seed')

    def diff(A, a, B, b, hl):
        x, y = cell(A, hl, a), cell(B, hl, b)
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

    R['HL1_A1'] = family('A1', 4000)
    R['HL2_A2'] = family('A2', 4100)

    S = {}; sb = 4300; T = {}
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
    # 5 places minus 4 places, same seeds, 1.0 s
    for A in ADM:
        for arm in ('METHOD', 'R1', 'R1i', 'R2', 'R3'):
            x, y = cell(A, 10, arm), cell(A, 10, arm, df4)
            assert list(x.seed) == list(y.seed)
            d = (x.worst.values - y.worst.values) * 100.0
            V[f'{A}_{arm}_5places_minus_4places_1s'] = dict(mean=float(d.mean()),
                                                           scene_95=RM.scene_boot(d, y.scene.values, sb)); sb += 1
    S['contrasts'] = V
    R['secondary'] = S

    h1, h2 = R['HL1_A1']['held'], R['HL2_A2']['held']
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif h1 and h2:
        outcome = 'with spare places the gain survives per-request admission: coordinated admission is needed only near full load'
    elif h2:
        outcome = 'even with spare places the gain needs coordinated admission'
    else:
        outcome = 'HL2 not held: unexpected after the 4-place result; report as is and discuss'
    R['decision'] = dict(outcome=outcome, HL1_held=h1, HL2_held=h2)
    R['guesses'] = dict(
        METHOD_A1_1s_between_2_and_4=bool(2.0 <= T['A1_hl10_METHOD']['worst'] <= 4.0),
        METHOD_A1_minus_A2_1s_below_1pp=bool(V['hl10_METHOD_A1_minus_A2']['mean'] < 1.0),
        R1_A1_1s_between_4_and_7=bool(4.0 <= T['A1_hl10_R1']['worst'] <= 7.0),
        R3_A1_1s_between_4_and_7=bool(4.0 <= T['A1_hl10_R3']['worst'] <= 7.0),
        HL1_held=h1, HL2_held=h2)
    json.dump(R, open(res_path, 'w'), indent=1)

    L = [f'LOAD CHECK: {CAP} places per station (16 cars, {4 * CAP} places), the three admission models',
         '(60 held-out DAIR scenes, 480 episodes per cell, the frozen arms and seeds)', '',
         f"must-hold checks: {checks}", '']
    for key, title in [('HL1_A1', 'HL1, A1 per-request admission'), ('HL2_A2', 'HL2, A2 coordinated admission')]:
        L.append(f'{title} (1.0 s; baseline minus METHOD, worst-car RLF pp; 98.75 % scene interval):')
        for b, v in R[key]['per_baseline'].items():
            L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    L.append('')
    for A in ADM:
        L.append(f'{A}: arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS)
                 + ' | handovers/car-step @1.0 s | refused share @1.0 s')
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
    L.append(''); L.append('causes of the worst car at 1.0 s:')
    for A in ADM:
        for arm in names:
            cw = T[f'{A}_hl10_{arm}']['causes_worst']
            L.append(f'  {A} {arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append('contrasts (95 % scene intervals):')
    for k, v in V.items():
        L.append(f"  {k}: {v['mean']:+.2f} [{v['scene_95'][0]:+.2f}, {v['scene_95'][1]:+.2f}]")
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'LOAD_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
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
