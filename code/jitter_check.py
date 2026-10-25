#!/usr/bin/env python3
"""Delay-difference check: handover delays that differ from car to car (DRAFT, 20 Oct 2026; not run).

Pre-registration: PREREG_MA_DelayDifference_Jitter_DRAFT_v1_20261020.md (needs Daya's OK).
Question: coordinated admission (A2) and LSRA's swaps were tested with every handover taking exactly
L slots, so the cars of a loop land together. Does LSRA's gain survive when each handover takes
L + d slots, d uniform in {-J, ..., +J}, drawn per car and slot (common random numbers)?
Same 60 held-out scenes, 480 frozen seeds, frozen arms (method_heldout.arms, beta from FREEZE.json),
harness (method_harness.run_eval_m) and admission model A2 as before; only the delays differ
(paper2ma_env_jitter.JitterEnv). 1.0 s nominal delay (L = 10 slots), 4 and 5 places per station.
  --mode unit                 U1-U4 on the jitter class
  --mode gate                 G0: J = 0 reproduces the frozen A2 runs at 4 and 5 places (first 40 seeds,
                              all 10 arms, every column)
  --mode freeze               writes JITTER_FREEZE.json once unit and gate passed (on Daya's OK)
  --mode test --cap C --j J   all 10 frozen arms under A2 at C places with jitter J; each cell runs once
  --mode analyse              the registered analysis, once
"""
import os, sys, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import paper2ma_env_jitter as JE

HERE = RM.HERE
OUT = os.path.join(HERE, 'models', 'jitter_check')
HL = 10
CAPS = (4, 5)
JS = (1, 2)
PRIMARY_J = 2
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']
N_GATE = 40
N_UNIT = 5
PREREG = 'PREREG_MA_DelayDifference_Jitter_FROZEN_20261020.md'
FROZEN_A2 = {4: 'models/admission_check/ADMIT_EPISODES_A2_hl10.csv',
             5: 'models/load_check/LOAD_EPISODES_C5_A2_hl10.csv'}
JCOLS = ['starts_j', 'early', 'late', 'min_delay', 'max_delay', 'excess_car_steps', 'flight_mismatch']
EXTRA = ('hl', 'arm', 'adm', 'cap', 'jit', 'requests', 'refused', 'loop_violations') + tuple(JCOLS)


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def make_env_j(pool, cap, J, debug=False):
    """As load_check.make_env_cap(pool, 10, 'coordinated', cap), with per-car delays."""
    env = JE.JitterEnv(pool, n_agents=16, n_subbands=3, gnb_cap=cap, seed=0, split=None,
                       compute_anticipation=True, ho_latency=HL)
    env.ATTACH = True
    env.ADMIT = 'coordinated'
    env.JITTER = int(J)
    env.DEBUG = bool(debug)
    return env


def run_cell_j(env, fn, seeds):
    env.jit_log = []
    rows = AC.run_cell(env, fn, seeds)
    assert len(env.jit_log) == len(seeds) == len(rows), 'one jitter record per episode'
    for r, j in zip(rows, env.jit_log):
        r.update(jit=int(env.JITTER), starts_j=j['starts'], early=j['early'], late=j['late'],
                 min_delay=j['min_delay'], max_delay=j['max_delay'], excess_car_steps=j['excess_car_steps'],
                 flight_mismatch=j['flight_mismatch'])
    return rows


def frozen_a2(cap):
    df = pd.read_csv(os.path.join(HERE, FROZEN_A2[cap]))
    df['scene'] = df['scene'].astype(str)
    return df


def core_cols(df):
    return [c for c in df.columns if c not in EXTRA]


def same_as_frozen_a2(new_df, fz, arm):
    cols = core_cols(fz) + ['requests', 'refused']
    a = new_df[cols].reset_index(drop=True).copy()
    a['scene'] = a['scene'].astype(str)
    old = fz[fz['arm'] == arm].set_index('seed').loc[list(a['seed'])].reset_index()
    return AC.equal_frames(a, old[cols].reset_index(drop=True))


# ---------------------------------------------------------------- U1-U4: the jitter class itself
def unit_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()[:N_UNIT]
    arms = dict(MHO.arms(fzm['beta']))
    rep = {'_versions': AC.versions()}
    # U1: delays lie in [L-J, L+J] and every landing comes exactly `delay` active slots after its start
    u1 = {}
    for J in JS:
        for name in ('METHOD', 'R3'):
            env = make_env_j(pool, 4, J, debug=True)
            env.events = []
            rows = run_cell_j(env, arms[name], seeds)
            st = [e for e in env.events if e[0] == 'start']
            ld = [e for e in env.events if e[0] == 'land']
            dl = [e[3] for e in st]
            u1[f'J{J}_{name}'] = dict(
                starts=len(st), landings=len(ld),
                delays_in_range=bool(all(HL - J <= x <= HL + J for x in dl)),
                some_early=bool(any(x < HL for x in dl)), some_late=bool(any(x > HL for x in dl)),
                landings_flown_equals_delay=bool(all(e[3] == e[4] for e in ld)),
                flight_mismatch=int(sum(r['flight_mismatch'] for r in rows)))
    rep['U1'] = u1
    ok1 = all(v['delays_in_range'] and v['some_early'] and v['some_late'] and v['landings_flown_equals_delay']
              and v['flight_mismatch'] == 0 and v['landings'] > 0 for v in u1.values())
    # U2: the delay table depends on the seed only (common random numbers)
    env = make_env_j(pool, 4, 2)
    env.reset(seed=seeds[0]); d1 = env._D.copy()
    env.reset(seed=seeds[0]); d2 = env._D.copy()
    env.reset(seed=seeds[1]); d3 = env._D.copy()
    rep['U2'] = dict(same_seed_same_table=bool(np.array_equal(d1, d2)),
                     other_seed_other_table=bool(not np.array_equal(d1[:min(len(d1), len(d3))],
                                                                    d3[:min(len(d1), len(d3))])))
    ok2 = rep['U2']['same_seed_same_table'] and rep['U2']['other_seed_other_table']
    # U3: d is uniform on {-J..J} (pooled over the 480 seeds)
    u3 = {}
    allseeds = AC.seeds_all()
    for J in JS:
        envj = make_env_j(pool, 4, J)
        cnt = np.zeros(2 * J + 1)
        for s in allseeds:
            envj.reset(seed=s)
            v, c = np.unique(envj._D, return_counts=True)
            for a, b in zip(v, c):
                cnt[int(a) + J] += b
        fr = cnt / cnt.sum()
        u3[f'J{J}'] = dict(freq=[float(x) for x in fr],
                           within_0_01=bool(np.all(np.abs(fr - 1.0 / (2 * J + 1)) < 0.01)))
    rep['U3'] = u3
    ok3 = all(v['within_0_01'] for v in u3.values())
    # U4: J = 0 changes no delay
    env0 = make_env_j(pool, 4, 0, debug=True)
    env0.events = []
    rows0 = run_cell_j(env0, arms['METHOD'], seeds)
    d0 = [e[3] for e in env0.events if e[0] == 'start']
    rep['U4'] = dict(starts=len(d0), all_equal_L=bool(all(x == HL for x in d0)),
                     flight_mismatch=int(sum(r['flight_mismatch'] for r in rows0)))
    ok4 = rep['U4']['all_equal_L'] and rep['U4']['flight_mismatch'] == 0 and len(d0) > 0
    rep['ok'] = bool(ok1 and ok2 and ok3 and ok4)
    json.dump(rep, open(os.path.join(OUT, 'UNIT.json'), 'w'), indent=1)
    status(f'[unit] U1 {ok1} U2 {ok2} U3 {ok3} U4 {ok4} -> ok={rep["ok"]}')


# ---------------------------------------------------------------- G0: J = 0 equals the frozen A2 runs
def gate_mode():
    fzm = MHO.check_freeze()
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()[:N_GATE]
    per = {}
    lsra_excess = {}
    for cap in CAPS:
        fz = frozen_a2(cap)
        env = make_env_j(pool, cap, 0)
        for name, fn in MHO.arms(fzm['beta']):
            t0 = time.time()
            rows = run_cell_j(env, fn, seeds)
            per[f'C{cap}_{name}'] = bool(same_as_frozen_a2(AC.roundtrip(rows), fz, name))
            if name == 'METHOD':
                lsra_excess[f'C{cap}'] = int(sum(r['excess_car_steps'] for r in rows))
            status(f'[gate] C{cap} {name}: identical to the frozen A2 run = {per[f"C{cap}_{name}"]} '
                   f'({time.time() - t0:.0f} s)')
    rep = dict(_versions=AC.versions(), n_seeds=N_GATE, hl=HL, per_arm=per,
               lsra_excess_car_steps_at_J0=lsra_excess,          # recorded only (cars that appear can
               ok=bool(all(per.values())))                          # meet a station full at landing)
    json.dump(rep, open(os.path.join(OUT, 'GATE.json'), 'w'), indent=1)
    status(f'[gate] ok={rep["ok"]}')


# ---------------------------------------------------------------- freeze
def freeze_mode():
    u = json.load(open(os.path.join(OUT, 'UNIT.json')))
    g = json.load(open(os.path.join(OUT, 'GATE.json')))
    assert u['ok'] and g['ok'], 'unit tests or gate did not pass: no freeze'
    path = os.path.join(OUT, 'JITTER_FREEZE.json')
    assert not os.path.exists(path), 'already frozen'
    assert os.path.exists(os.path.join(HERE, PREREG)), 'the frozen pre-registration is not in the folder'
    fzm = MHO.check_freeze()
    files = ['jitter_check.py', 'paper2ma_env_jitter.py', 'paper2ma_env_admit.py', 'admission_check.py', PREREG,
             'models/jitter_check/UNIT.json', 'models/jitter_check/GATE.json',
             'models/method_heldout/FREEZE.json', 'models/admission_check/ADMIT_FREEZE.json',
             'models/load_check/LOAD_FREEZE.json', FROZEN_A2[4], FROZEN_A2[5]]
    rec = dict(written_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), beta=fzm['beta'],
               md5={f: md5(os.path.join(HERE, f)) for f in files})
    json.dump(rec, open(path, 'w'), indent=1)
    status(f'[freeze] written, md5 {md5(path)}')


def check_jitter_freeze():
    fz = json.load(open(os.path.join(OUT, 'JITTER_FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    MHO.check_freeze()
    return fz


def cell_path(cap, J):
    return os.path.join(OUT, f'JITTER_EPISODES_C{cap}_J{J}.csv')


# ---------------------------------------------------------------- the run
def test_mode(cap, J):
    fz = check_jitter_freeze()
    out = cell_path(cap, J)
    assert not os.path.exists(out), 'this cell already ran: each cell runs once'
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    env = make_env_j(pool, cap, J)
    rows = []
    status(f'test C{cap} J{J} start (beta {fz["beta"]})')
    for name, fn in MHO.arms(fz['beta']):
        t0 = time.time()
        r = run_cell_j(env, fn, seeds)
        for x in r:
            x.update(hl=HL, arm=name, adm='A2', cap=cap)
        rows += r
        pd.DataFrame(rows).to_csv(out + '.partial', index=False)
        status(f'test C{cap} J{J} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test C{cap} J{J} DONE')


# ---------------------------------------------------------------- analysis (once)
def analyse():
    fz = check_jitter_freeze()
    res_path = os.path.join(OUT, 'JITTER_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    names = [a for a, _ in MHO.arms(fz['beta'])]
    D = {}
    for cap in CAPS:
        f0 = frozen_a2(cap)
        D[(cap, 0)] = f0
        for J in JS:
            d = pd.read_csv(cell_path(cap, J))
            d['scene'] = d['scene'].astype(str)
            D[(cap, J)] = d

    def cell(cap, J, arm):
        d = D[(cap, J)]
        return d[d['arm'] == arm].sort_values('seed')

    def diff(cap, Ja, a, Jb, b):
        x, y = cell(cap, Ja, a), cell(cap, Jb, b)
        assert list(x.seed) == list(y.seed) and len(x) == 480
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    R = {'_freeze': fz, '_versions': AC.versions()}
    jit = pd.concat([D[(c, J)] for c in CAPS for J in JS], ignore_index=True)
    m1 = int(jit.dec_over.sum())
    m4 = int((jit.starts != jit.requests - jit.refused).sum())
    m5 = int(jit.flight_mismatch.sum())
    m6 = 0
    for J in JS:
        s = jit[(jit.jit == J) & (jit.starts_j > 0)]
        m6 += int(((s.min_delay < HL - J) | (s.max_delay > HL + J)).sum())
    m7 = int((jit.starts != jit.starts_j).sum())
    checks = dict(M1_created_excess_total=m1, M4_rows_where_starts_differ=m4, M5_flight_mismatch=m5,
                  M6_delays_out_of_range=m6, M7_rows_where_jitter_starts_differ=m7)
    checks['ok'] = bool(m1 == 0 and m4 == 0 and m5 == 0 and m6 == 0 and m7 == 0)
    R['checks'] = checks

    def family(cap, J, seed0):
        per = {}
        for k, b in enumerate(PRIMARY_BASELINES):
            d, sc = diff(cap, J, b, J, 'METHOD')
            ci = RM.scene_boot(d, sc, seed0 + k, (0.625, 99.375))
            per[b] = dict(mean=float(d.mean()), scene_9875=ci, scene_95=RM.scene_boot(d, sc, seed0 + 50 + k),
                          beats=bool(ci[0] > 0.0))
        return dict(per_baseline=per, held=bool(all(v['beats'] for v in per.values())))

    R['HJ1_C4_J2'] = family(4, 2, 6000)
    R['HJ2_C5_J2'] = family(5, 2, 6100)
    S = {'J1_families': {'C4_J1': family(4, 1, 6200), 'C5_J1': family(5, 1, 6300)}}
    sb = 6500; T = {}; V = {}
    for cap in CAPS:
        for J in (0,) + JS:
            for arm in names:
                c = cell(cap, J, arm); w = c.worst.values * 100.0; wa = np.maximum(c.worst_act.values, 1.0)
                e = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                         ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                         refused_share=float(c.refused.sum() / max(c.requests.sum(), 1)),
                         causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
                sb += 1
                if J > 0:
                    e['excess_car_steps_per_car_step'] = float(c.excess_car_steps.sum() / c.p_act.sum())
                    e['early_share'] = float(c.early.sum() / max(c.starts_j.sum(), 1))
                    e['late_share'] = float(c.late.sum() / max(c.starts_j.sum(), 1))
                    d, s2 = diff(cap, J, arm, 0, arm)
                    e['minus_J0_same_arm'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
                if arm != 'METHOD':
                    d, s2 = diff(cap, J, arm, J, 'METHOD')
                    e['minus_METHOD_same_J'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb)); sb += 1
                T[f'C{cap}_J{J}_{arm}'] = e
    S['table'] = T
    R['secondary'] = S
    h1, h2 = R['HJ1_C4_J2']['held'], R['HJ2_C5_J2']['held']
    if not checks['ok']:
        outcome = 'VOID: a must-hold check failed; find the cause, report, new OK before any re-run'
    elif h1 and h2:
        outcome = 'the gain survives delay differences of up to +-0.2 s at both loads'
    elif h1:
        outcome = 'at full load the gain survives; with spare places the narrow margin does not survive'
    else:
        outcome = 'coordinated admission must also make the cars of a loop land together; state it and report the cost'
    R['decision'] = dict(outcome=outcome, HJ1_held=h1, HJ2_held=h2)
    R['guesses'] = dict(
        METHOD_C4_J2_minus_J0_below_0_5=bool(T['C4_J2_METHOD']['minus_J0_same_arm']['mean'] < 0.5),
        METHOD_C4_J1_minus_J0_below_0_2=bool(T['C4_J1_METHOD']['minus_J0_same_arm']['mean'] < 0.2),
        HJ1_held=h1, HJ2_not_held=(not h2))
    json.dump(R, open(res_path, 'w'), indent=1)

    L = ['DELAY-DIFFERENCE CHECK: each handover takes 10 + d slots, d uniform in {-J..J}; coordinated admission (A2)',
         '(60 held-out DAIR scenes, 480 episodes per cell, frozen arms and seeds; J = 0 is the frozen A2 run)', '',
         f'must-hold checks: {checks}', '']
    for key, title in [('HJ1_C4_J2', 'HJ1, 4 places, J = 2'), ('HJ2_C5_J2', 'HJ2, 5 places, J = 2')]:
        L.append(f'{title} (baseline minus METHOD, worst-car RLF pp; 98.75 % scene interval):')
        for b, v in R[key]['per_baseline'].items():
            L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> "
                     f"{'BEATS' if v['beats'] else 'not shown'}")
        L.append(f"  {'HELD' if R[key]['held'] else 'NOT HELD'}")
    L.append('')
    for key, fam in S['J1_families'].items():
        L.append(f'{key} (secondary): ' + '  '.join(
            f"{b} {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}]"
            for b, v in fam['per_baseline'].items()) + f"  -> {'held' if fam['held'] else 'not held'}")
    L.append('')
    for cap in CAPS:
        L.append(f'{cap} places: arm | J0 | J1 | J2 (worst-car RLF %, 95 % scene interval) | J2 minus J0 | '
                 f'excess car-steps per car-step at J2')
        for arm in names:
            cells = [T[f'C{cap}_J{J}_{arm}'] for J in (0,) + JS]
            m = cells[2]['minus_J0_same_arm']
            L.append(f'  {arm} | ' + ' | '.join(f"{c['worst']:.2f} [{c['scene_95'][0]:.2f}, {c['scene_95'][1]:.2f}]"
                                                 for c in cells)
                     + f" | {m['mean']:+.2f} [{m['scene_95'][0]:+.2f}, {m['scene_95'][1]:+.2f}]"
                     + f" | {cells[2]['excess_car_steps_per_car_step']:.5f}")
        L.append('')
    L.append('causes of the worst car (J = 2):')
    for cap in CAPS:
        for arm in names:
            cw = T[f'C{cap}_J2_{arm}']['causes_worst']
            L.append(f'  C{cap} {arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'JITTER_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['unit', 'gate', 'freeze', 'test', 'analyse'], required=True)
    ap.add_argument('--cap', type=int)
    ap.add_argument('--j', type=int)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'unit':
        unit_mode()
    elif a.mode == 'gate':
        gate_mode()
    elif a.mode == 'freeze':
        freeze_mode()
    elif a.mode == 'test':
        assert a.cap in CAPS and a.j in JS
        test_mode(a.cap, a.j)
    else:
        analyse()
