#!/usr/bin/env python3
"""Re-measurement with capacity-aware attachment (rule D0 of PREREG_MA_REMEASURE_FROZEN_20261020.md).

Pre-registered in PREREG_MA_REMEASURE2_Attach_FROZEN_20261020.md. No training. The five policies, the
harness (run_eval_log), the analysis (analyse, write_txt) and the seed lists are reused unchanged
from remeasure_ma.py; only the environment class differs (paper2ma_env_attach.CapacityAttachEnv).

  --mode tests : T0 md5s; T1 fix switched off == base environment (bit-identical per episode against
                 CF.run_eval on the base environment, and the stored G1/G2 values exactly), old seeds,
                 test split, hl = 10, all five policies; T2 fix switched on: the class's own checks
                 (no station without room, start within capacity) on the old seeds, and the fix acts
                 (some attachments differ from the base rule). Prints pass/fail and counts only.
  --mode run   : requires passed tests and unchanged files; runs every cell once on the 20 Oct seed
                 lists; registered analysis; paired comparison with the 20 Oct episodes (S5).
"""
import os, sys, json, time, argparse, hashlib, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import paper2_controller as C
import paper2ma_confirm as CF
import assign_value as AV
import remeasure_ma as RM
import paper2ma_env_attach as PA

HERE = RM.HERE
OUT = os.path.join(HERE, 'models', 'remeasure_ma2')
OLD = os.path.join(HERE, 'models', 'remeasure_ma')
MD5_REGISTERED = {
    'paper2ma_env.py': '42c9dba38c90141e3c96b8678d14857b',
    'paper2ma_confirm.py': 'a900171af7343dbf249a547cb84aa1ca',
    'capacity_rule.py': 'b3c4f289b940b59ca94abc0be3061bc1',
    'assign_value.py': '1e7dd45241670c2b762732adca263e6d',
    'paper2ma_antic_test.py': '2672762d2437af9591a84a59771d3859',
    'dair_geometry_pool.csv': '3de813a13293b9810b21fe7683a71c26',
    'remeasure_ma.py': 'c90eb0ad3c038b306f5e1a18e73e58b1',
    os.path.join('models', 'remeasure_ma', 'SEEDS_train.csv'): 'f34f6f3655f8daf7da8efc6244e416ac',
    os.path.join('models', 'remeasure_ma', 'SEEDS_test.csv'): '774f9748ea56e53ac7b59e5fb04f4308',
    os.path.join('models', 'remeasure_ma', 'REMEASURE_EPISODES.csv'): '9563f0611ce346a1bc13432d4a46d00c',
}


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def make_env2(pool, split, hl, attach=True):
    env = PA.CapacityAttachEnv(pool, n_agents=16, n_subbands=3, gnb_cap=4, seed=0, split=split,
                               compute_anticipation=True, ho_latency=hl)
    env.ATTACH = attach
    return env


def tests():
    os.makedirs(OUT, exist_ok=True)
    pool = pd.read_csv(RM.POOL)
    t = {'_versions': dict(python=platform.python_version(), numpy=np.__version__,
                           scipy=scipy.__version__, pandas=pd.__version__)}
    got = {f: md5(os.path.join(HERE, f)) for f in MD5_REGISTERED}
    t['T0'] = dict(ok=bool(all(got[f] == MD5_REGISTERED[f] for f in got)),
                   script=md5(os.path.abspath(__file__)), attach=md5(os.path.join(HERE, 'paper2ma_env_attach.py')))
    status(f'[T0] registered md5s match: {t["T0"]["ok"]}')
    seeds_old = [C.EVAL_SEED_BASE + k for k in range(120)]
    base = AV.make_env(pool, 'test', 10)
    off = make_env2(pool, 'test', 10, attach=False)
    t1 = {}
    for name, fn in RM.POLICIES:
        e1, f1, h1 = CF.run_eval(base, fn, seeds_old)
        e2, f2, h2, _ = RM.run_eval_log(off, fn, seeds_old)
        eq = bool(np.array_equal(e1, e2) and np.array_equal(f1, f2) and np.array_equal(h1, h2))
        worst = float(e2.mean()) * 100.0
        tgt = RM.G1_TARGET.get(name, RM.G2_TARGET.get(name))
        t1[name] = dict(equivalent=eq, identical=bool(worst == tgt))
        status(f'[T1] fix off, {name}: equivalent to base={eq} stored value exact={worst == tgt}')
    t['T1'] = t1
    t['T1_ok'] = bool(all(v['equivalent'] and v['identical'] for v in t1.values()))
    on = make_env2(pool, 'test', 10, attach=True)
    try:
        _, _, _, _ = RM.run_eval_log(on, CF.pol_hungarian, seeds_old)
        t2 = dict(ran=True, attachments=int(on.n_attach), changed=int(on.n_changed), error=None)
    except Exception as ex:                                   # the class raises on any violation
        t2 = dict(ran=False, attachments=int(on.n_attach), changed=int(on.n_changed), error=repr(ex))
    t2['ok'] = bool(t2['ran'] and t2['changed'] > 0)
    t['T2'] = t2
    status(f'[T2] fix on: {t2}')
    t['_pass'] = bool(t['T0']['ok'] and t['T1_ok'] and t2['ok'])
    json.dump(t, open(os.path.join(OUT, 'TESTS.json'), 'w'), indent=1)
    status(f'TESTS {"PASS" if t["_pass"] else "FAIL"}')
    return t['_pass']


def paired_fix_effect(new, old):
    """S5: fixed minus old attachment, per policy, delay and split; paired by seed (same episodes)."""
    S5 = {}; sb = 900
    for split in RM.SPLITS:
        for hl in RM.HLS:
            for pol, _ in RM.POLICIES:
                a = new[(new['split'] == split) & (new['hl'] == hl) & (new['policy'] == pol)].sort_values('seed')
                b = old[(old['split'] == split) & (old['hl'] == hl) & (old['policy'] == pol)].sort_values('seed')
                assert list(a.seed) == list(b.seed)
                d = (a.worst.values - b.worst.values) * 100.0
                S5[f'{split}_hl{hl}_{pol}'] = dict(fixed=float(a.worst.mean() * 100.0), old=float(b.worst.mean() * 100.0),
                                                   diff=float(d.mean()), scene_95=RM.scene_boot(d, a.scene.values, sb))
                sb += 1
    return S5


def run():
    t = json.load(open(os.path.join(OUT, 'TESTS.json')))
    assert t['_pass'], 'tests did not pass'
    assert md5(os.path.abspath(__file__)) == t['T0']['script'], 'script changed after the tests'
    assert md5(os.path.join(HERE, 'paper2ma_env_attach.py')) == t['T0']['attach'], 'attach file changed'
    for f, m in MD5_REGISTERED.items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed'
    pool = pd.read_csv(RM.POOL)
    seeds = {sp: list(pd.read_csv(os.path.join(OLD, f'SEEDS_{sp}.csv')).seed.astype(int)) for sp in RM.SPLITS}
    csv = os.path.join(OUT, 'REMEASURE2_EPISODES.csv')
    assert not os.path.exists(csv), 'this run already exists: it runs once'
    allrows = []; counters = {}
    status('RUN START')
    for split in RM.SPLITS:
        for hl in RM.HLS:
            for name, fn in RM.POLICIES:
                env = make_env2(pool, split, hl, attach=True)       # fresh env per cell: clean counters
                t0 = time.time()
                w, m, o, rows = RM.run_eval_log(env, fn, seeds[split])
                for r in rows:
                    r.update(split=split, hl=hl, policy=name)
                allrows += rows
                counters[f'{split}_hl{hl}_{name}'] = dict(attachments=int(env.n_attach), changed=int(env.n_changed))
                pd.DataFrame(allrows).to_csv(csv + '.partial', index=False)
                status(f'cell {split} hl={hl} {name} done ({len(rows)} episodes, {time.time() - t0:.0f} s)')
    df = pd.DataFrame(allrows)
    df.to_csv(csv, index=False)
    status('RUN DONE; analysis')
    R = RM.analyse(df)
    R['guess_check_of_20oct_run_rule'] = R.pop('guess_check', None)   # the 20 Oct guess; kept out of the text
    old = pd.read_csv(os.path.join(OLD, 'REMEASURE_EPISODES.csv'))
    R['S5'] = paired_fix_effect(df, old)
    R['attach_counters'] = counters
    sh = R['causes']['train_R3']['shares']
    R['guess2'] = dict(r3_delay_cost_below_8=bool(R['causes']['train_R3']['delta_total'] < 8.0),
                       L_share_at_least_half=(None if sh['L'] is None else bool(sh['L'] >= 0.5)),
                       A_share_below_0_10=(None if sh['A'] is None else bool(sh['A'] < 0.10)),
                       H1_not_held_below_2pp=bool((not R['H1']['held']) and R['H1']['mean'] < 2.0))
    R['_tests'] = t
    R['_episodes_csv_md5'] = md5(csv)
    json.dump(R, open(os.path.join(OUT, 'REMEASURE2_RESULT.json'), 'w'), indent=1)
    txt = os.path.join(OUT, 'REMEASURE2_RESULT.txt')
    RM.write_txt(R, txt)
    L = ['', '--- S5: effect of the attachment fix (fixed minus old), worst-car RLF %, same episodes, 95% scene interval ---']
    for split in RM.SPLITS:
        for hl in RM.HLS:
            L.append(f'{split} {hl * 0.1:.1f} s: ' + '  '.join(
                f"{p} {R['S5'][f'{split}_hl{hl}_{p}']['old']:.2f}->{R['S5'][f'{split}_hl{hl}_{p}']['fixed']:.2f} "
                f"({R['S5'][f'{split}_hl{hl}_{p}']['diff']:+.2f} [{R['S5'][f'{split}_hl{hl}_{p}']['scene_95'][0]:+.2f}, "
                f"{R['S5'][f'{split}_hl{hl}_{p}']['scene_95'][1]:+.2f}])" for p, _ in RM.POLICIES))
    L.append('')
    L.append(f"Recorded guesses (re-run): {R['guess2']}")
    L.append('Attachments decided by the fix (and how many differ from the old rule), R3 cells: ' + '; '.join(
        f"{k}: {v['attachments']} ({v['changed']})" for k, v in counters.items() if k.endswith('_R3')))
    with open(txt, 'a', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    status('ANALYSIS DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['tests', 'run'], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'tests':
        sys.exit(0 if tests() else 2)
    run()
