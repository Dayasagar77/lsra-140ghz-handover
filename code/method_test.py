#!/usr/bin/env python3
"""Test of the landing-state rotation method (pre-registration PREREG_LSRA_MethodTest_Dev45_Heldout60).

Stage 1, development check on the 45 design scenes (Daya's OK, 20 Oct):
  --mode dev --hl H      run every stage-1 arm at one delay on the 20 Oct seed lists (both splits)
  --mode dev_analyse     merge the four delays, choose beta by the fixed rule, write DEV_RESULT.*
Stage 2 (held-out; needs a separate OK and a frozen pre-registration) is not implemented here yet.

Environment: paper2ma_env_attach.CapacityAttachEnv (capacity-aware attachment), 16 cars, capacity 4.
Harness: method_harness.run_eval_m. No held-out scene is touched by this script.
"""
import os, sys, json, time, argparse, hashlib, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import paper2ma_confirm as CF
import assign_value as AV
import capacity_rule as CR
import paper2ma_antic_test as AT
import remeasure_ma as RM
import remeasure_ma2 as RM2
import lsra as LS
import method_harness as MH

HERE = RM.HERE
DEV = os.path.join(HERE, 'models', 'method_dev')
SEEDS = os.path.join(HERE, 'models', 'remeasure_ma')
HLS = [2, 5, 10, 15]
BETAS = [0.001, 0.01, 0.05]


def dev_arms():
    arms = [('R0', CF.pol_loadaware), ('R1', CR.pol_loadaware_cap), ('R1i', LS.pol_r1i),
            ('R2', AV.pol_hung_restricted), ('R3', CF.pol_hungarian), ('R3F', AT.pol_hung_oracle)]
    for b in BETAS:
        arms.append((f'LSRA_b{b}', LS.make_lsra('now', beta=b)))
    arms += [('LSRA_oracle_b0.001', LS.make_lsra('oracle', beta=0.001)),
             ('LSRA_nolanding_b0.001', LS.make_lsra('now', landing=False, beta=0.001)),
             ('LSRA_nominmax_b0.001', LS.make_lsra('now', minmax=False, beta=0.001))]
    return arms


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(DEV, 'DEV_STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def dev(hl):
    os.makedirs(DEV, exist_ok=True)
    out = os.path.join(DEV, f'DEV_EPISODES_hl{hl}.csv')
    assert not os.path.exists(out), 'this delay already ran'
    pool = pd.read_csv(RM.POOL)
    seeds = {sp: list(pd.read_csv(os.path.join(SEEDS, f'SEEDS_{sp}.csv')).seed.astype(int)) for sp in RM.SPLITS}
    rows = []
    status(f'dev hl={hl} start; lsra.py {md5(os.path.join(HERE, "lsra.py"))}; harness {md5(os.path.join(HERE, "method_harness.py"))}')
    for split in RM.SPLITS:
        env = RM2.make_env2(pool, split, hl, attach=True)
        for name, fn in dev_arms():
            t0 = time.time()
            _, _, _, r = MH.run_eval_m(env, fn, seeds[split])
            for x in r:
                x.update(split=split, hl=hl, arm=name)
            rows += r
            pd.DataFrame(rows).to_csv(out + '.partial', index=False)
            status(f'dev hl={hl} {split} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'dev hl={hl} DONE')


def dev_analyse():
    df = pd.concat([pd.read_csv(os.path.join(DEV, f'DEV_EPISODES_hl{h}.csv')) for h in HLS], ignore_index=True)
    df['scene'] = df['scene'].astype(str)
    R = {'_versions': dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                           pandas=pd.__version__),
         '_md5': {f: md5(os.path.join(HERE, f)) for f in ['lsra.py', 'method_harness.py', 'method_test.py',
                                                          'paper2ma_env_attach.py', 'paper2ma_env.py']}}
    arms = [a for a, _ in dev_arms()]

    def cell(hl, arm):
        return df[(df['hl'] == hl) & (df['arm'] == arm)].sort_values(['split', 'seed'])

    # beta rule: largest beta within 0.5 pp of the best at 1.0 s (all 45 scenes)
    w10 = {b: float(cell(10, f'LSRA_b{b}').worst.mean() * 100) for b in BETAS}
    best = min(w10.values())
    chosen = max(b for b in BETAS if w10[b] <= best + 0.5)
    R['beta_rule'] = dict(worst_at_1s=w10, best=best, chosen=chosen)
    lsra = f'LSRA_b{chosen}'
    sb = 100
    T = {}
    for hl in HLS:
        for arm in arms:
            c = cell(hl, arm); w = c.worst.values * 100.0
            wa = np.maximum(c.worst_act.values, 1.0)
            T[f'hl{hl}_{arm}'] = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                                      mean_rlf=float(c['mean'].mean() * 100), ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                                      dec_over_per_1000_steps=float(c.dec_over.sum() / c.p_act.sum() * 1000),
                                      land_over_total=int(c.land_over.sum()),
                                      causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
            sb += 1
    R['table'] = T
    C = {}
    for hl in HLS:
        b = cell(hl, lsra)
        for arm in arms:
            if arm == lsra:
                continue
            a = cell(hl, arm)
            assert list(a.seed) == list(b.seed) and list(a.split) == list(b.split)
            d = (a.worst.values - b.worst.values) * 100.0
            C[f'hl{hl}_{arm}_minus_LSRA'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, b.scene.values, sb))
            sb += 1
    R['contrasts'] = C
    json.dump(R, open(os.path.join(DEV, 'DEV_RESULT.json'), 'w'), indent=1)
    L = [f"STAGE 1 (45 design scenes, 360 episodes per cell; design data, no claims). beta rule: {R['beta_rule']}", '']
    L.append('arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS) + ' | handovers/car-step @1.0 s | created landing excess per 1000 car-steps @1.0 s')
    for arm in arms:
        L.append(f'{arm} | ' + ' | '.join(
            f"{T[f'hl{h}_{arm}']['worst']:.2f} [{T[f'hl{h}_{arm}']['scene_95'][0]:.2f}, {T[f'hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
            + f" | {T[f'hl10_{arm}']['ho_per_car_step']:.4f} | {T[f'hl10_{arm}']['dec_over_per_1000_steps']:.2f}")
    L.append('')
    L.append(f'contrasts vs {lsra} (arm minus LSRA, worst-car RLF pp, 95% scene interval):')
    for hl in HLS:
        L.append(f'{hl * 0.1:.1f} s: ' + '  '.join(
            f"{arm} {C[f'hl{hl}_{arm}_minus_LSRA']['mean']:+.2f} [{C[f'hl{hl}_{arm}_minus_LSRA']['scene_95'][0]:+.2f}, {C[f'hl{hl}_{arm}_minus_LSRA']['scene_95'][1]:+.2f}]"
            for arm in arms if arm != lsra))
    L.append('')
    L.append('causes of the worst car at 1.0 s (A attach, L landed-full, F full-other, W waiting, B outage):')
    for arm in arms:
        cw = T[f'hl10_{arm}']['causes_worst']
        L.append(f'{arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    open(os.path.join(DEV, 'DEV_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('dev_analyse DONE')


# ------------------------------------------------------------------------------------------------
# Stage-1 follow-up (20 Oct, after stage 1; Daya asked for the best-quality option): the method
# without the worst-car-first stage ('ms' = minimum total risk + beta per move). The beta rule of
# stage 1 is applied to this form; its forecast reference and no-landing ablation are run in the
# same form at all three betas. Design scenes only.
# ------------------------------------------------------------------------------------------------
def dev2_arms():
    arms = [(f'LSRAms_b{b}', LS.make_lsra('now', minmax=False, beta=b)) for b in [0.01, 0.05]]
    for b in BETAS:
        arms.append((f'LSRAms_oracle_b{b}', LS.make_lsra('oracle', minmax=False, beta=b)))
        arms.append((f'LSRAms_nolanding_b{b}', LS.make_lsra('now', landing=False, minmax=False, beta=b)))
    return arms


def dev2(hl):
    out = os.path.join(DEV, f'DEV2_EPISODES_hl{hl}.csv')
    assert not os.path.exists(out), 'this delay already ran'
    pool = pd.read_csv(RM.POOL)
    seeds = {sp: list(pd.read_csv(os.path.join(SEEDS, f'SEEDS_{sp}.csv')).seed.astype(int)) for sp in RM.SPLITS}
    rows = []
    status(f'dev2 hl={hl} start; lsra.py {md5(os.path.join(HERE, "lsra.py"))}')
    for split in RM.SPLITS:
        env = RM2.make_env2(pool, split, hl, attach=True)
        for name, fn in dev2_arms():
            t0 = time.time()
            _, _, _, r = MH.run_eval_m(env, fn, seeds[split])
            for x in r:
                x.update(split=split, hl=hl, arm=name)
            rows += r
            pd.DataFrame(rows).to_csv(out + '.partial', index=False)
            status(f'dev2 hl={hl} {split} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'dev2 hl={hl} DONE')


def dev2_analyse():
    d1 = pd.concat([pd.read_csv(os.path.join(DEV, f'DEV_EPISODES_hl{h}.csv')) for h in HLS], ignore_index=True)
    d2 = pd.concat([pd.read_csv(os.path.join(DEV, f'DEV2_EPISODES_hl{h}.csv')) for h in HLS], ignore_index=True)
    d1['arm'] = d1['arm'].replace({'LSRA_nominmax_b0.001': 'LSRAms_b0.001'})
    df = pd.concat([d1, d2], ignore_index=True)
    df['scene'] = df['scene'].astype(str)

    def cell(hl, arm):
        return df[(df['hl'] == hl) & (df['arm'] == arm)].sort_values(['split', 'seed'])

    w10 = {b: float(cell(10, f'LSRAms_b{b}').worst.mean() * 100) for b in BETAS}
    best = min(w10.values())
    chosen = max(b for b in BETAS if w10[b] <= best + 0.5)
    R = {'beta_rule_ms': dict(worst_at_1s=w10, best=best, chosen=chosen),
         '_md5': {f: md5(os.path.join(HERE, f)) for f in ['lsra.py', 'method_harness.py', 'method_test.py']}}
    method = f'LSRAms_b{chosen}'
    arms = ['R0', 'R1', 'R1i', 'R2', 'R3', 'R3F', method, f'LSRAms_oracle_b{chosen}',
            f'LSRAms_nolanding_b{chosen}', 'LSRA_b0.05'] + [f'LSRAms_b{b}' for b in BETAS if b != chosen]
    sb = 500; T = {}; C = {}
    for hl in HLS:
        m = cell(hl, method)
        for arm in arms:
            c = cell(hl, arm); w = c.worst.values * 100.0
            wa = np.maximum(c.worst_act.values, 1.0)
            T[f'hl{hl}_{arm}'] = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                                      ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                                      dec_over_per_1000_steps=float(c.dec_over.sum() / c.p_act.sum() * 1000),
                                      causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
            sb += 1
            if arm != method:
                assert list(c.seed) == list(m.seed) and list(c.split) == list(m.split)
                d = (c.worst.values - m.worst.values) * 100.0
                C[f'hl{hl}_{arm}_minus_method'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, m.scene.values, sb))
                sb += 1
    R['method'] = method; R['table'] = T; R['contrasts'] = C
    json.dump(R, open(os.path.join(DEV, 'DEV2_RESULT.json'), 'w'), indent=1)
    L = [f"STAGE-1 FOLLOW-UP (45 design scenes; design data, no claims). beta rule (min-total-risk form): {R['beta_rule_ms']}",
         f'method = {method}', '',
         'arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS) + ' | handovers/car-step @1.0 s | created landing excess /1000 car-steps @1.0 s']
    for arm in arms:
        L.append(f'{arm} | ' + ' | '.join(
            f"{T[f'hl{h}_{arm}']['worst']:.2f} [{T[f'hl{h}_{arm}']['scene_95'][0]:.2f}, {T[f'hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
            + f" | {T[f'hl10_{arm}']['ho_per_car_step']:.4f} | {T[f'hl10_{arm}']['dec_over_per_1000_steps']:.2f}")
    L.append(''); L.append(f'contrasts (arm minus {method}, pp, 95% scene interval):')
    for hl in HLS:
        L.append(f'{hl * 0.1:.1f} s: ' + '  '.join(
            f"{arm} {C[f'hl{hl}_{arm}_minus_method']['mean']:+.2f} [{C[f'hl{hl}_{arm}_minus_method']['scene_95'][0]:+.2f}, {C[f'hl{hl}_{arm}_minus_method']['scene_95'][1]:+.2f}]"
            for arm in arms if arm != method))
    L.append(''); L.append('causes of the worst car at 1.0 s:')
    for arm in arms:
        cw = T[f'hl10_{arm}']['causes_worst']
        L.append(f'{arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    open(os.path.join(DEV, 'DEV2_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('dev2_analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dev', 'dev_analyse', 'dev2', 'dev2_analyse'], required=True)
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(DEV, exist_ok=True)
    if a.mode == 'dev':
        assert a.hl in HLS
        dev(a.hl)
    elif a.mode == 'dev_analyse':
        dev_analyse()
    elif a.mode == 'dev2':
        assert a.hl in HLS
        dev2(a.hl)
    else:
        dev2_analyse()
