#!/usr/bin/env python3
"""Stage 2: frozen test of the landing-state method on the 60 held-out DAIR scenes.

Pre-registration: PREREG_LSRA_MethodTest_Heldout60_FROZEN_20261020.md (needs Daya's OK).
  --mode seeds          balanced seed list (8 per scene, first seeds of 930,000,000 + j that pick each
                        scene), checked by env.reset; no policy is run
  --mode test --hl H    run every stage-2 arm at one delay (refuses if FREEZE.json is missing, any
                        frozen file changed, or this delay already ran)
  --mode analyse        registered analysis, once
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
OUT = os.path.join(HERE, 'models', 'method_heldout')
POOL_H = os.path.join(HERE, 'dair_geometry_pool_heldout60_20261020.csv')
POOL_H_MD5 = '3a68d055876a733ee21de37700385a75'
SEED_BASE = 930_000_000
PER_SCENE = 8
HLS = [2, 5, 10, 15]
PRIMARY_BASELINES = ['R1', 'R1i', 'R2', 'R3']


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def arms(beta):
    return [('R0', CF.pol_loadaware), ('R1', CR.pol_loadaware_cap), ('R1i', LS.pol_r1i),
            ('R2', AV.pol_hung_restricted), ('R3', CF.pol_hungarian), ('R3F', AT.pol_hung_oracle),
            ('METHOD', LS.make_lsra('now', minmax=False, beta=beta)),
            ('METHOD_oracle', LS.make_lsra('oracle', minmax=False, beta=beta)),
            ('ABL_nolanding', LS.make_lsra('now', landing=False, minmax=False, beta=beta)),
            ('ABL_worstcarfirst', LS.make_lsra('now', minmax=True, beta=0.05))]


def seeds_mode():
    os.makedirs(OUT, exist_ok=True)
    assert md5(POOL_H) == POOL_H_MD5, 'held-out pool changed'
    path = os.path.join(OUT, 'SEEDS_heldout60.csv')
    assert not os.path.exists(path), 'seed list already exists'
    pool = pd.read_csv(POOL_H)
    env = RM2.make_env2(pool, None, 2, attach=True)
    ids = list(env.scene_ids); n = len(ids)
    cnt = {sid: 0 for sid in ids}; lst = []; j = 0
    while min(cnt.values()) < PER_SCENE:
        s = SEED_BASE + j
        sid = ids[np.random.default_rng(s).integers(n)]
        if cnt[sid] < PER_SCENE:
            cnt[sid] += 1; lst.append((s, sid))
        j += 1
        assert j < 500000
    mism = 0
    for s, sid in lst:
        _, inf = env.reset(seed=s)
        mism += int(str(inf['scene']) != str(sid))
    df = pd.DataFrame(lst, columns=['seed', 'scene']); df.to_csv(path, index=False)
    per = df.groupby('scene').size()
    rep = dict(n_scenes=int(per.size), episodes=len(df), all_eight=bool((per == PER_SCENE).all()),
               reset_mismatch=mism, max_j=int(max(s for s, _ in lst) - SEED_BASE), md5=md5(path))
    json.dump(rep, open(os.path.join(OUT, 'SEEDS_heldout60_check.json'), 'w'), indent=1)
    status(f'[seeds] {rep}')


def check_freeze():
    fz = json.load(open(os.path.join(OUT, 'FREEZE.json')))
    for f, m in fz['md5'].items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed after the freeze'
    return fz


def test_mode(hl):
    fz = check_freeze()
    out = os.path.join(OUT, f'HELDOUT_EPISODES_hl{hl}.csv')
    assert not os.path.exists(out), 'this delay already ran: the test runs once'
    pool = pd.read_csv(POOL_H)
    seeds = list(pd.read_csv(os.path.join(OUT, 'SEEDS_heldout60.csv')).seed.astype(int))
    env = RM2.make_env2(pool, None, hl, attach=True)
    rows = []
    status(f'test hl={hl} start (beta {fz["beta"]})')
    for name, fn in arms(fz['beta']):
        t0 = time.time()
        _, _, _, r = MH.run_eval_m(env, fn, seeds)
        for x in r:
            x.update(hl=hl, arm=name)
        rows += r
        pd.DataFrame(rows).to_csv(out + '.partial', index=False)
        status(f'test hl={hl} {name} done ({len(r)} episodes, {time.time() - t0:.0f} s)')
    pd.DataFrame(rows).to_csv(out, index=False)
    status(f'test hl={hl} DONE')


def analyse():
    fz = check_freeze()
    res_path = os.path.join(OUT, 'HELDOUT_RESULT.json')
    assert not os.path.exists(res_path), 'analysis already ran'
    df = pd.concat([pd.read_csv(os.path.join(OUT, f'HELDOUT_EPISODES_hl{h}.csv')) for h in HLS], ignore_index=True)
    df['scene'] = df['scene'].astype(str)
    names = [a for a, _ in arms(fz['beta'])]

    def cell(hl, arm):
        return df[(df['hl'] == hl) & (df['arm'] == arm)].sort_values('seed')

    def diff(hl, a, b):
        x, y = cell(hl, a), cell(hl, b)
        assert list(x.seed) == list(y.seed)
        return (x.worst.values - y.worst.values) * 100.0, y.scene.values

    R = {'_freeze': fz, '_versions': dict(python=platform.python_version(), numpy=np.__version__,
                                          scipy=scipy.__version__, pandas=pd.__version__)}
    # primary HM1 (1.0 s): each baseline minus METHOD, 98.75 % scene interval above 0
    hm1 = {}
    for k, b in enumerate(PRIMARY_BASELINES):
        d, sc = diff(10, b, 'METHOD')
        ci = RM.scene_boot(d, sc, 1000 + k, (0.625, 99.375))
        hm1[b] = dict(mean=float(d.mean()), scene_9875=ci, scene_95=RM.scene_boot(d, sc, 1000 + k),
                      beats=bool(ci[0] > 0.0))
    R['HM1'] = dict(per_baseline=hm1, held=bool(all(v['beats'] for v in hm1.values())))
    # mechanism check
    m = df[df['arm'] == 'METHOD']
    R['mechanism'] = dict(dec_over_total=int(m.dec_over.sum()), land_over_total=int(m.land_over.sum()),
                          ok=bool(m.dec_over.sum() == 0 and m.land_over.sum() == 0))
    # secondary
    S = {}
    a10, sc = diff(10, 'R3', 'METHOD'); a2, _ = diff(2, 'R3', 'METHOD')
    S['HM3_R3_cost_minus_METHOD_cost'] = dict(mean=float((a10 - a2).mean()), scene_95=RM.scene_boot(a10 - a2, sc, 1100))
    d02, sc = diff(2, 'METHOD', 'R3')
    S['METHOD_minus_R3_at_0.2s'] = dict(mean=float(d02.mean()), scene_95=RM.scene_boot(d02, sc, 1101))
    sb = 1200; T = {}
    for hl in HLS:
        for arm in names:
            c = cell(hl, arm); w = c.worst.values * 100.0; wa = np.maximum(c.worst_act.values, 1.0)
            T[f'hl{hl}_{arm}'] = dict(worst=float(w.mean()), scene_95=RM.scene_boot(w, c.scene.values, sb),
                                      mean_rlf=float(c['mean'].mean() * 100), ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                                      dec_over_per_1000_steps=float(c.dec_over.sum() / c.p_act.sum() * 1000),
                                      causes_worst={k: float((c[f'w_{k}'].values / wa).mean() * 100) for k in RM.CAUSES})
            sb += 1
            if arm != 'METHOD':
                d, s2 = diff(hl, arm, 'METHOD')
                T[f'hl{hl}_{arm}']['minus_METHOD'] = dict(mean=float(d.mean()), scene_95=RM.scene_boot(d, s2, sb))
                sb += 1
    S['table'] = T
    R['secondary'] = S
    # decisions fixed in advance
    nl = T['hl10_ABL_nolanding']['minus_METHOD']
    R['decision'] = dict(HM1_held=R['HM1']['held'],
                         landing_state_not_what_helps=bool(abs(nl['mean']) < 1.0 and nl['scene_95'][0] <= 0 <= nl['scene_95'][1]))
    R['guesses'] = dict(method_1s_between_3_and_5=bool(3.0 <= T['hl10_METHOD']['worst'] <= 5.0),
                        R3_1s_between_9_and_11=bool(9.0 <= T['hl10_R3']['worst'] <= 11.0),
                        beats_all_four=R['HM1']['held'],
                        nolanding_close_to_R3=bool(abs(T['hl10_ABL_nolanding']['worst'] - T['hl10_R3']['worst']) < 1.0),
                        method_within_0_5_of_R3_at_0_2=bool(abs(d02.mean()) <= 0.5),
                        method_hands_over_more_than_R1_R2=bool(T['hl10_METHOD']['ho_per_car_step'] >
                                                               max(T['hl10_R1']['ho_per_car_step'], T['hl10_R2']['ho_per_car_step'])))
    json.dump(R, open(res_path, 'w'), indent=1)
    L = ['HELD-OUT TEST (60 new DAIR val scenes, 480 episodes per cell, capacity-aware attachment)',
         f"method = landing-state assignment, total risk + beta per move, beta = {fz['beta']}", '',
         'HM1 (1.0 s; baseline minus METHOD, worst-car RLF pp; 98.75 % scene interval):']
    for b, v in hm1.items():
        L.append(f"  {b}: {v['mean']:+.2f} [{v['scene_9875'][0]:+.2f}, {v['scene_9875'][1]:+.2f}] -> {'BEATS' if v['beats'] else 'not shown'}")
    L.append(f"HM1 {'HELD' if R['HM1']['held'] else 'NOT HELD'}; mechanism check {R['mechanism']}")
    L.append(f"HM3 (R3 cost of delay minus METHOD cost, 1.0 - 0.2 s): {S['HM3_R3_cost_minus_METHOD_cost']}")
    L.append(f"METHOD minus R3 at 0.2 s: {S['METHOD_minus_R3_at_0.2s']}"); L.append('')
    L.append('arm | ' + ' | '.join(f'{h * 0.1:.1f} s' for h in HLS) + ' | handovers/car-step @1.0 s | created excess /1000 @1.0 s')
    for arm in names:
        L.append(f'{arm} | ' + ' | '.join(
            f"{T[f'hl{h}_{arm}']['worst']:.2f} [{T[f'hl{h}_{arm}']['scene_95'][0]:.2f}, {T[f'hl{h}_{arm}']['scene_95'][1]:.2f}]" for h in HLS)
            + f" | {T[f'hl10_{arm}']['ho_per_car_step']:.4f} | {T[f'hl10_{arm}']['dec_over_per_1000_steps']:.2f}")
    L.append(''); L.append('causes of the worst car at 1.0 s:')
    for arm in names:
        cw = T[f'hl10_{arm}']['causes_worst']; L.append(f'{arm}: ' + '  '.join(f'{k} {cw[k]:.2f}' for k in RM.CAUSES))
    L.append(''); L.append(f"decision: {R['decision']}"); L.append(f"guesses: {R['guesses']}")
    open(os.path.join(OUT, 'HELDOUT_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('analyse DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['seeds', 'test', 'analyse'], required=True)
    ap.add_argument('--hl', type=int)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'seeds':
        seeds_mode()
    elif a.mode == 'test':
        assert a.hl in HLS
        test_mode(a.hl)
    else:
        analyse()
