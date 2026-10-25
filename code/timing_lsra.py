#!/usr/bin/env python3
"""Decision time per step of LSRA and the deployable baselines (20 Oct 2026).

Protocol: PROTOCOL_Timing_DecisionTimePerStep_20261020.md (written before the run).
One decision = one call of a policy at one 0.1 s step; it returns the handover decisions of all cars.
Set-up: the frozen held-out test (60 unseen scenes, the 480 frozen seeds), 1.0 s handover delay,
4 places per station, admission at landing only (A0). The environment is built as in the admission
check (AdmissionEnv with admission off), which reproduced the frozen episodes exactly in gate G0.
numpy/scipy are held to one thread (environment variables set before numpy is imported).
Arms: METHOD (LSRA), R1, R1i, R2, R3 from method_heldout.arms, beta from the frozen FREEZE.json.
The timer only reads a clock. Integrity check: each arm's 480 episodes must equal the frozen
held-out episodes column for column; an arm that differs is marked VOID in the result.
For METHOD the two parts are timed as well: the risk forecast (lsra.window_risk) and the assignment
(lsra.bottleneck_then_mincost), with the number of free cars and of places on offer.
Each arm first runs N_WARM episodes untimed (warm-up), then all 480 episodes timed.
Outputs (models/timing/): TIMING_RAW_<arm>.csv (one row per decision), TIMING_RESULT.json,
TIMING_RESULT.txt, STATUS.txt. Runs once: refuses if TIMING_RESULT.json exists.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, time, hashlib, platform, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import remeasure_ma as RM
import method_heldout as MHO
import admission_check as AC
import lsra as LS

HERE = RM.HERE
OUT = os.path.join(HERE, 'models', 'timing')
HL = 10
TIMED = ['METHOD', 'R1', 'R1i', 'R2', 'R3']
N_WARM = 3
FROZEN_MD5 = {'admission_check.py': '1b16d561d8a1e14a715cc5b0dc5e8e95',
              'paper2ma_env_admit.py': '5a644059a82ba06a73ad2f1f47968c90',
              'models/method_heldout/FREEZE.json': '76f010f740673ef3fa66f4cf8914b0a5'}
PROTOCOL = 'PROTOCOL_Timing_DecisionTimePerStep_20261020.md'
COLS = ['seed', 't', 'n_active', 'n_inflight', 'policy_ns', 'risk_ns', 'assign_ns', 'n_free', 'n_slots']


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


REC = {'on': False, 'seed': -1, 'rows': [], 'part': {}}
_window_risk, _assign = LS.window_risk, LS.bottleneck_then_mincost


def window_risk_timed(env, forecast):
    t0 = time.perf_counter_ns()
    r = _window_risk(env, forecast)
    REC['part']['risk'] = time.perf_counter_ns() - t0
    return r


def assign_timed(R, slot_g, serv, beta, floor, minmax=True):
    t0 = time.perf_counter_ns()
    out = _assign(R, slot_g, serv, beta, floor, minmax=minmax)
    REC['part'].update(assign=time.perf_counter_ns() - t0, nF=int(R.shape[0]), slots=int(len(slot_g)))
    return out


def timed(fn):
    def f(env, obs, sinr, rng, ag, temp):
        if not REC['on']:
            return fn(env, obs, sinr, rng, ag, temp)
        act = np.asarray(env.frames[env.t]['active'], dtype=bool)
        n_act = int(act.sum()); n_fl = int(((np.asarray(env._ho_timer) > 0) & act).sum())
        REC['part'] = {}
        t0 = time.perf_counter_ns()
        out = fn(env, obs, sinr, rng, ag, temp)
        dt = time.perf_counter_ns() - t0
        p = REC['part']
        REC['rows'].append((REC['seed'], int(env.t), n_act, n_fl, dt, p.get('risk', -1), p.get('assign', -1),
                            p.get('nF', -1), p.get('slots', -1)))
        return out
    return f


def stats_ms(ns):
    x = np.asarray(ns, dtype=float) / 1e6
    q = np.percentile(x, [50, 90, 99, 99.9])
    return dict(n=int(x.size), mean=float(x.mean()), median=float(q[0]), p90=float(q[1]), p99=float(q[2]),
                p999=float(q[3]), max=float(x.max()), share_over_10ms=float((x > 10.0).mean()),
                share_over_100ms=float((x > 100.0).mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    res_path = os.path.join(OUT, 'TIMING_RESULT.json')
    assert not os.path.exists(res_path), 'the timing already ran: it runs once'
    assert os.path.exists(os.path.join(HERE, PROTOCOL)), 'the protocol is not in the folder'
    for f, m in FROZEN_MD5.items():
        assert md5(os.path.join(HERE, f)) == m, f'{f} changed'
    fzm = MHO.check_freeze()
    LS.window_risk, LS.bottleneck_then_mincost = window_risk_timed, assign_timed
    pool = pd.read_csv(MHO.POOL_H)
    seeds = AC.seeds_all()
    fz = AC.frozen(HL)
    env = AC.make_env3(pool, HL, 'none')
    _reset = env.reset

    def reset(*a, **k):
        REC['seed'] = int(k.get('seed', -1))
        return _reset(*a, **k)
    env.reset = reset
    arms = dict(MHO.arms(fzm['beta']))
    res = dict(_protocol_md5=md5(os.path.join(HERE, PROTOCOL)), _script_md5=md5(os.path.abspath(__file__)),
               _versions=AC.versions(),
               _machine=dict(platform=platform.platform(), processor=platform.processor(),
                             cpu_count=os.cpu_count(), python_build=platform.python_build(),
                             threads_env={v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                                          'OPENBLAS_NUM_THREADS')},
                             gc_enabled=gc.isenabled(),
                             perf_counter=str(time.get_clock_info('perf_counter'))),
               setup=dict(hl=HL, delay_s=HL * 0.1, places=4, admission='A0 (none)', episodes=len(seeds),
                          warmup_episodes=N_WARM, beta=fzm['beta']),
               arms={})
    status(f'timing start: arms {TIMED}, {len(seeds)} episodes each, warm-up {N_WARM}')
    for name in TIMED:
        fn = timed(arms[name])
        REC['on'] = False
        AC.run_cell(env, fn, seeds[:N_WARM])
        REC['on'] = True; REC['rows'] = []
        t0 = time.time()
        rows = AC.run_cell(env, fn, seeds)
        wall = time.time() - t0
        REC['on'] = False
        same = bool(AC.same_as_frozen(AC.roundtrip(rows), fz, name, HL))
        raw = pd.DataFrame(REC['rows'], columns=COLS)
        raw.to_csv(os.path.join(OUT, f'TIMING_RAW_{name}.csv'), index=False)
        a = dict(identical_to_frozen=same, wall_s=wall, decisions=int(len(raw)),
                 policy=stats_ms(raw.policy_ns))
        bins = {'1-8': (1, 8), '9-12': (9, 12), '13-16': (13, 16)}
        a['policy_by_active_cars'] = {}
        for k, (lo, hi) in bins.items():
            sel = raw[(raw.n_active >= lo) & (raw.n_active <= hi)]
            if len(sel):
                s = stats_ms(sel.policy_ns)
                a['policy_by_active_cars'][k] = {q: s[q] for q in ('n', 'median', 'p99', 'max')}
        if name == 'METHOD':
            a['risk_forecast'] = stats_ms(raw.risk_ns[raw.risk_ns >= 0])
            a['assignment'] = stats_ms(raw.assign_ns[raw.assign_ns >= 0])
            a['free_cars'] = dict(mean=float(raw.n_free.mean()), max=int(raw.n_free.max()))
            a['places_on_offer'] = dict(mean=float(raw.n_slots.mean()), max=int(raw.n_slots.max()))
            a['missing_part_timings'] = int(((raw.risk_ns < 0) | (raw.assign_ns < 0)).sum())
        res['arms'][name] = a
        p = a['policy']
        status(f'{name}: identical to frozen = {same}; {a["decisions"]} decisions; median {p["median"]:.3f} ms, '
               f'p99 {p["p99"]:.3f} ms, max {p["max"]:.3f} ms; wall {wall:.0f} s')
    res['all_identical_to_frozen'] = bool(all(v['identical_to_frozen'] for v in res['arms'].values()))
    res['void_arms'] = [k for k, v in res['arms'].items() if not v['identical_to_frozen']]
    json.dump(res, open(res_path, 'w'), indent=1)
    L = ['Decision time per step (one call of the policy for all cars), held-out test, 1.0 s, 4 places, A0',
         f'machine: {res["_machine"]["platform"]} | {res["_machine"]["processor"]} | {os.cpu_count()} logical CPUs',
         f'versions: {res["_versions"]}; one thread for numpy/scipy; gc enabled = {gc.isenabled()}',
         f'all arms identical to the frozen episodes: {res["all_identical_to_frozen"]} (void: {res["void_arms"]})',
         '', f'{"arm":8s} {"decisions":>9s} {"mean":>7s} {"median":>7s} {"p90":>7s} {"p99":>7s} {"p99.9":>7s} '
             f'{"max":>8s} {">10ms":>7s}  (ms)']
    for name in TIMED:
        p = res['arms'][name]['policy']
        L.append(f'{name:8s} {p["n"]:9d} {p["mean"]:7.3f} {p["median"]:7.3f} {p["p90"]:7.3f} {p["p99"]:7.3f} '
                 f'{p["p999"]:7.3f} {p["max"]:8.3f} {100 * p["share_over_10ms"]:6.3f}%')
    m = res['arms']['METHOD']
    L += ['', 'METHOD parts (ms): risk forecast median {:.3f} p99 {:.3f}; assignment median {:.3f} p99 {:.3f}'.format(
        m['risk_forecast']['median'], m['risk_forecast']['p99'], m['assignment']['median'], m['assignment']['p99']),
        f'METHOD free cars per decision: mean {m["free_cars"]["mean"]:.2f}, max {m["free_cars"]["max"]}; '
        f'places on offer: mean {m["places_on_offer"]["mean"]:.2f}, max {m["places_on_offer"]["max"]}',
        '', 'policy time by number of active cars (median / p99 / max, ms):']
    for name in TIMED:
        L.append(f'{name:8s} ' + '  '.join(f'{k}: {v["median"]:.3f} / {v["p99"]:.3f} / {v["max"]:.3f} (n {v["n"]})'
                                           for k, v in res['arms'][name]['policy_by_active_cars'].items()))
    open(os.path.join(OUT, 'TIMING_RESULT.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    status('timing done: ' + ('all arms valid' if res['all_identical_to_frozen'] else f'VOID arms {res["void_arms"]}'))


if __name__ == '__main__':
    main()
