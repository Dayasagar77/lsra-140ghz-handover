#!/usr/bin/env python3
"""Multi-car re-measurement at a 1 s handover delay on unseen scenes, with a failure-cause breakdown.

Pre-registered in PREREG_MA_Remeasure_1sDelay_UnseenScenes_FailureCauses (DRAFT v1, 20 Oct 2026;
frozen copy PREREG_MA_REMEASURE_FROZEN_20261020.md). No training. No change to the environment or
to any policy: the five policies are imported unchanged, and the harness is CF.run_eval's loop plus
read-only logging.

  --mode gates : seed lists + T1, module md5s, G0 (equivalence), G1 (stage-16 identity),
                 G2 (stage-12 identity). Prints pass/fail only (no failure-cause numbers).
  --mode run   : requires a passed gate file and unchanged seed lists; runs every fresh cell once,
                 then the registered analysis. T2 is asserted on every episode.
"""
import os, sys, json, time, argparse, hashlib, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, scipy
import paper2ma_env as MA, paper2_controller as C
import paper2ma_confirm as CF
import assign_value as AV
import capacity_rule as CR
import paper2ma_antic_test as AT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'models', 'remeasure_ma')
POOL = os.path.join(HERE, 'dair_geometry_pool.csv')
POLICIES = [('R0', CF.pol_loadaware), ('R1', CR.pol_loadaware_cap), ('R2', AV.pol_hung_restricted),
            ('R3', CF.pol_hungarian), ('R3F', AT.pol_hung_oracle)]
HLS = [2, 5, 10, 15]
SPLITS = ['train', 'test']           # train = primary (unseen for the contrasts); test = secondary
SEED_BASE = 910_000_000
PER_SCENE = 8
BOOT = 10000
CAUSES = ['A', 'L', 'F', 'W', 'B']
G1_TARGET = {'R0': 45.19631161795504, 'R1': 28.320655007379074, 'R2': 19.042027922175812,
             'R3': 18.313201958530993}                       # CAPACITY_RULE.json, test_hl10
G2_TARGET = {'R3F': 17.813597764365028}                     # ANTIC_MAP_M16.json, M16_hl10
MD5_REGISTERED = {'paper2ma_env.py': '42c9dba3', 'paper2ma_confirm.py': 'a900171a',
                  'capacity_rule.py': 'b3c4f289', 'assign_value.py': '1e7dd452',
                  'paper2ma_antic_test.py': '2672762d', 'dair_geometry_pool.csv': '3de813a1'}


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def status(msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(os.path.join(OUT, 'STATUS.txt'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


# --------------------------------------------------------------------------------------------
# Harness: CF.run_eval's loop, unchanged, plus read-only logging.
# --------------------------------------------------------------------------------------------
def run_eval_log(env, policy_fn, seeds):
    pw, pm, po, rows = [], [], [], []
    rng = np.random.default_rng(20260726)
    cap = int(env.gnb_cap)
    for s in seeds:
        obs, info0 = env.reset(seed=s); M = env.M
        env.sb = np.array([i % env.n_sb for i in range(M)], dtype=int)
        sinr = np.array([env._sinr_db(i) for i in range(M)])
        rlf = np.zeros(M); act = np.zeros(M); ok = np.zeros(M); done = False
        # ---- logging state (only reads env) ----
        G = env.n_gnb
        cause = np.zeros((M, len(CAUSES)))
        prev_active = env.frames[0]['active'].copy()
        prev_serv = env.serv.copy()
        last_land = np.full(G, -10 ** 9); last_app = np.full(G, -10 ** 9)
        self_app = np.full(M, -10 ** 9)
        n_start = 0; n_blind = 0; inflight_sum = 0.0; n_frames = 0
        while not done:
            act_pre = env.frames[env.t]['active'].copy()
            serv_pre = env.serv.copy(); tim_pre = env._ho_timer.copy(); tgt_pre = env._ho_target.copy()
            a, tgt = policy_fn(env, obs, sinr, rng, None, 0.0)
            obs, rew, term, done, info = env.step(a, ho_targets=tgt)
            for i in range(M):
                if info['active'][i]:
                    sinr[i] = obs[i, 0] * 30.0
                    rlf[i] += info['rlf'][i]; ok[i] += info['link_ok'][i]; act[i] += 1
            # ---- logging (reads only) ----
            tau = env.t
            act_now = info['active']; serv_now = info['serv']
            started = np.where(info['ho'])[0]
            if len(started):
                land = np.zeros(G)
                for j in range(M):
                    if act_pre[j]:
                        land[tgt_pre[j] if tim_pre[j] > 0 else serv_pre[j]] += 1
                for j in started:                                   # car-index order
                    g_from = int(serv_pre[j]); g_to = int(env._ho_target[j])
                    n_start += 1
                    if land[g_to] >= cap:
                        n_blind += 1
                    land[g_to] += 1; land[g_from] -= 1
            for j in range(M):
                if act_now[j] and not prev_active[j]:
                    last_app[serv_now[j]] = tau; self_app[j] = tau
                if act_now[j] and serv_now[j] != prev_serv[j]:
                    last_land[serv_now[j]] = tau
            for i in range(M):
                if not (act_now[i] and info['rlf'][i]):
                    continue
                tau0 = tau - int(env._consec[i]) + 1                # first step of this failure run
                g = int(serv_now[i])
                if not info['admitted'][i]:
                    if self_app[i] >= tau0 or last_app[g] >= tau0 or tau0 <= 1:
                        k = 0                                       # A: attachment
                    elif last_land[g] >= tau0:
                        k = 1                                       # L: landed on a full station
                    else:
                        k = 2                                       # F: full, other
                else:
                    k = 3 if env._ho_timer[i] > 0 else 4            # W: waiting / B: outage
                cause[i, k] += 1
            inflight_sum += float(np.sum((env._ho_timer > 0) & act_now)); n_frames += 1
            prev_active = act_now.copy(); prev_serv = serv_now.copy()
        assert np.array_equal(cause.sum(axis=1), rlf), 'T2 failed: causes do not add up'
        aa = np.maximum(act, 1)
        w = float((rlf / aa).max()); m = float(rlf.sum() / aa.sum()); o = float(ok.sum() / aa.sum())
        pw.append(w); pm.append(m); po.append(o)
        wi = int(np.argmax(rlf / aa))
        row = dict(seed=int(s), scene=str(info0['scene']), worst=w, mean=m, ok=o, worst_car=wi,
                   worst_act=float(act[wi]), worst_rlf=float(rlf[wi]), p_act=float(act.sum()),
                   p_rlf=float(rlf.sum()), starts=n_start, blind=n_blind,
                   inflight=inflight_sum / max(n_frames, 1))
        for k, c in enumerate(CAUSES):
            row[f'w_{c}'] = float(cause[wi, k]); row[f'p_{c}'] = float(cause[:, k].sum())
        rows.append(row)
    return np.array(pw), np.array(pm), np.array(po), rows


# --------------------------------------------------------------------------------------------
# Seed lists (balanced: the first PER_SCENE seeds of SEED_BASE + j that pick each scene)
# --------------------------------------------------------------------------------------------
def seed_list(pool, split):
    env = AV.make_env(pool, split, 2)
    ids = list(env.scene_ids); n = len(ids)
    cnt = {sid: 0 for sid in ids}; out = []; j = 0
    while min(cnt.values()) < PER_SCENE:
        s = SEED_BASE + j
        sid = ids[np.random.default_rng(s).integers(n)]
        if cnt[sid] < PER_SCENE:
            cnt[sid] += 1; out.append((s, sid))
        j += 1
        assert j < 200000, 'seed search did not finish'
    return out, env


def seed_path(split):
    return os.path.join(OUT, f'SEEDS_{split}.csv')


def gates():
    os.makedirs(OUT, exist_ok=True)
    pool = pd.read_csv(POOL)
    g = {'_versions': dict(python=platform.python_version(), numpy=np.__version__,
                           scipy=scipy.__version__, pandas=pd.__version__)}
    # module md5s
    got = {f: md5(os.path.join(HERE, f))[:8] for f in MD5_REGISTERED}
    g['md5'] = dict(got=got, ok=bool(all(got[f] == MD5_REGISTERED[f] for f in got)))
    g['md5']['script'] = md5(os.path.abspath(__file__))
    status(f'[md5] modules match registration: {g["md5"]["ok"]}')
    # seed lists + T1
    t1 = {}
    for split in SPLITS:
        lst, env = seed_list(pool, split)
        df = pd.DataFrame(lst, columns=['seed', 'scene'])
        df.to_csv(seed_path(split), index=False)
        per = df.groupby('scene').size()
        old = set(range(C.EVAL_SEED_BASE, C.EVAL_SEED_BASE + 120))
        mism = 0
        for s, sid in lst:
            _, inf = env.reset(seed=s)
            mism += int(str(inf['scene']) != str(sid))
        t1[split] = dict(n_episodes=len(lst), n_scenes=int(per.size), all_eight=bool((per == PER_SCENE).all()),
                         overlap_old=int(sum(1 for s, _ in lst if s in old)), reset_mismatch=mism,
                         max_j=int(max(s for s, _ in lst) - SEED_BASE), md5=md5(seed_path(split)))
        status(f'[T1] {split}: {t1[split]}')
    g['T1'] = t1
    g['T1_ok'] = bool(all(v['all_eight'] and v['overlap_old'] == 0 and v['reset_mismatch'] == 0
                          for v in t1.values()))
    # G0 / G1 / G2 on the old seeds (test split, hl = 10)
    seeds_old = [C.EVAL_SEED_BASE + k for k in range(120)]
    env = AV.make_env(pool, 'test', 10)
    gg = {}
    for name, fn in POLICIES:
        t0 = time.time()
        e1, f1, h1 = CF.run_eval(env, fn, seeds_old)
        e2, f2, h2, _ = run_eval_log(env, fn, seeds_old)
        eq = bool(np.array_equal(e1, e2) and np.array_equal(f1, f2) and np.array_equal(h1, h2))
        worst = float(e2.mean()) * 100.0
        tgt = G1_TARGET.get(name, G2_TARGET.get(name))
        gg[name] = dict(equivalent=eq, worst=worst, target=tgt, identical=bool(worst == tgt),
                        delta=abs(worst - tgt), wall_s=round(time.time() - t0, 1))
        status(f'[G0/G1/G2] {name}: equivalent={eq} identical={worst == tgt} ({worst!r} vs {tgt!r})')
    g['G'] = gg
    g['G0_ok'] = bool(all(v['equivalent'] for v in gg.values()))
    g['G1_ok'] = bool(all(gg[k]['identical'] for k in G1_TARGET))
    g['G2_ok'] = bool(all(gg[k]['identical'] for k in G2_TARGET))
    g['_pass'] = bool(g['md5']['ok'] and g['T1_ok'] and g['G0_ok'] and g['G1_ok'] and g['G2_ok'])
    json.dump(g, open(os.path.join(OUT, 'GATES.json'), 'w'), indent=1)
    status(f'GATES {"PASS" if g["_pass"] else "FAIL"}')
    return g['_pass']


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------
def scene_boot(d, scenes, seed, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    uniq = sorted(set(scenes)); idx = [np.where(scenes == u)[0] for u in uniq]; S = len(uniq)
    bs = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(0, S, S)
        bs[b] = d[np.concatenate([idx[p] for p in pick])].mean()
    return [float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))]


def ep_boot(d, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def analyse(df):
    R = {'_note': 'per-episode worst-car RLF in %; scene bootstrap primary, episode bootstrap for comparison'}
    sb = 10                                                    # bootstrap seed counter for secondary items

    def cell(split, hl, pol):
        c = df[(df['split'] == split) & (df['hl'] == hl) & (df['policy'] == pol)].sort_values('seed')
        return c

    def diff(split, hl, a, b):
        ca, cb = cell(split, hl, a), cell(split, hl, b)
        assert list(ca.seed) == list(cb.seed)
        return (ca.worst.values - cb.worst.values) * 100.0, ca.scene.values

    # ---- primary (train, hl = 10) ----
    d1, sc = diff('train', 10, 'R1', 'R2')
    i975 = scene_boot(d1, sc, 0, (1.25, 98.75))
    H1 = dict(mean=float(d1.mean()), scene_975=i975, scene_95=scene_boot(d1, sc, 0), episode_95=ep_boot(d1, 0),
              held=bool(i975[0] > 0.0))
    d2, sc = diff('train', 10, 'R3', 'R3F')
    j975 = scene_boot(d2, sc, 1, (1.25, 98.75))
    if j975[1] < 3.0:
        v2 = 'held'
    elif j975[0] > 0.0 and float(d2.mean()) >= 3.0:
        v2 = 'foresight matters'
    else:
        v2 = 'inconclusive'
    H2 = dict(mean=float(d2.mean()), scene_975=j975, scene_95=scene_boot(d2, sc, 1), episode_95=ep_boot(d2, 1),
              verdict=v2)
    R['H1'] = H1; R['H2'] = H2

    # ---- secondary ----
    S1 = {}
    for split in SPLITS:
        a10, sc = diff(split, 10, 'R1', 'R2'); a2, _ = diff(split, 2, 'R1', 'R2')
        d = a10 - a2
        S1[split] = dict(mean=float(d.mean()), scene_95=scene_boot(d, sc, 2), episode_95=ep_boot(d, 2))
    R['S1'] = S1
    S2 = {}
    for split in SPLITS:
        for hl in HLS:
            key = f'{split}_hl{hl}'; S2[key] = {}
            for pol, _ in POLICIES:
                c = cell(split, hl, pol); w = c.worst.values * 100.0
                S2[key][pol] = dict(worst=float(w.mean()), scene_95=scene_boot(w, c.scene.values, sb),
                                    episode_95=ep_boot(w, sb), mean_rlf=float(c['mean'].mean() * 100.0),
                                    link_ok=float(c.ok.mean() * 100.0),
                                    ho_per_car_step=float(c.starts.sum() / c.p_act.sum()),
                                    inflight=float(c.inflight.mean()))
                sb += 1
            for a, b in [('R0', 'R1'), ('R1', 'R2'), ('R2', 'R3'), ('R3', 'R3F')]:
                d, sc = diff(split, hl, a, b)
                S2[key][f'{a}-{b}'] = dict(mean=float(d.mean()), scene_95=scene_boot(d, sc, sb),
                                           episode_95=ep_boot(d, sb))
                sb += 1
    R['S2'] = S2
    S3 = {}
    for split in SPLITS:
        c10, c2 = cell(split, 10, 'R3'), cell(split, 2, 'R3')
        d = (c10.worst.values - c2.worst.values) * 100.0
        S3[split] = dict(mean=float(d.mean()), scene_95=scene_boot(d, c10.scene.values, 3))
    R['S3'] = S3
    S4 = {}
    for split in SPLITS:
        for hl in HLS:
            for pol, _ in POLICIES:
                c = cell(split, hl, pol)
                S4[f'{split}_hl{hl}_{pol}'] = dict(starts=int(c.starts.sum()), blind=int(c.blind.sum()),
                                                   blind_share=float(c.blind.sum() / max(c.starts.sum(), 1)))
    R['S4'] = S4

    # ---- cause breakdown (section 8) ----
    CB = {}
    for split in SPLITS:
        for pol, _ in POLICIES:
            rates = {}
            for hl in HLS:
                c = cell(split, hl, pol)
                wa = np.maximum(c.worst_act.values, 1.0)
                rates[hl] = dict(worst={k: float((c[f'w_{k}'].values / wa).mean() * 100.0) for k in CAUSES},
                                 pooled={k: float(c[f'p_{k}'].sum() / c.p_act.sum() * 100.0) for k in CAUSES})
            dlt = {k: rates[10]['worst'][k] - rates[2]['worst'][k] for k in CAUSES}
            tot = sum(dlt.values())
            shares = {k: (dlt[k] / tot if tot > 0 else None) for k in CAUSES}
            CB[f'{split}_{pol}'] = dict(rates=rates, delta_1s_minus_02s=dlt, delta_total=tot, shares=shares)
    R['causes'] = CB
    sh = CB['train_R3']['shares']
    if sh['A'] is None:
        dec = dict(D0=None, lever='D4 (no cost of delay to split)')
    else:
        lever = 'D4'
        if sh['L'] >= 0.5:
            lever = 'D1'
        elif sh['W'] >= 0.5:
            lever = 'D2'
        elif sh['B'] >= 0.5:
            lever = 'D3'
        dec = dict(D0=bool(sh['A'] >= 0.25), lever=lever)
    R['decision'] = dec
    if sh['A'] is not None:
        by_share = max(CAUSES, key=lambda k: sh[k])
        by_rate = max(CAUSES, key=lambda k: CB['train_R3']['rates'][10]['worst'][k])
        R['guess_check'] = dict(largest_by_delay_share=by_share, largest_by_rate_at_1s=by_rate,
                                L_largest_by_share=bool(by_share == 'L'), L_largest_by_rate=bool(by_rate == 'L'),
                                A_share_at_least_0_10=bool(sh['A'] >= 0.10))
    return R


def write_txt(R, path):
    L = []
    h1, h2 = R['H1'], R['H2']
    L.append('PRIMARY (30 train-split scenes, fresh seeds, 1.0 s delay; scene bootstrap, 97.5 % intervals)')
    L.append(f"H1 R1 - R2 = {h1['mean']:+.2f} pp [{h1['scene_975'][0]:+.2f}, {h1['scene_975'][1]:+.2f}] -> "
             f"{'HELD' if h1['held'] else 'NOT HELD'}  (95% scene [{h1['scene_95'][0]:+.2f}, {h1['scene_95'][1]:+.2f}];"
             f" 95% episode [{h1['episode_95'][0]:+.2f}, {h1['episode_95'][1]:+.2f}])")
    L.append(f"H2 R3 - R3F = {h2['mean']:+.2f} pp [{h2['scene_975'][0]:+.2f}, {h2['scene_975'][1]:+.2f}] -> "
             f"{h2['verdict'].upper()}  (95% scene [{h2['scene_95'][0]:+.2f}, {h2['scene_95'][1]:+.2f}];"
             f" 95% episode [{h2['episode_95'][0]:+.2f}, {h2['episode_95'][1]:+.2f}])")
    L.append('')
    for split in SPLITS:
        L.append(f'--- {split} split: worst-car RLF % [95% scene interval] ---')
        L.append('delay | ' + ' | '.join(p for p, _ in POLICIES) + ' | R1-R2 | R3-R3F')
        for hl in HLS:
            c = R['S2'][f'{split}_hl{hl}']
            vals = ' | '.join(f"{c[p]['worst']:.2f} [{c[p]['scene_95'][0]:.2f}, {c[p]['scene_95'][1]:.2f}]"
                              for p, _ in POLICIES)
            r12 = c['R1-R2']; r3f = c['R3-R3F']
            L.append(f"{hl * 0.1:.1f} s | {vals} | {r12['mean']:+.2f} [{r12['scene_95'][0]:+.2f}, {r12['scene_95'][1]:+.2f}]"
                     f" | {r3f['mean']:+.2f} [{r3f['scene_95'][0]:+.2f}, {r3f['scene_95'][1]:+.2f}]")
        s1 = R['S1'][split]; s3 = R['S3'][split]
        L.append(f"S1 residual growth 1.0 s minus 0.2 s: {s1['mean']:+.2f} [{s1['scene_95'][0]:+.2f}, {s1['scene_95'][1]:+.2f}]")
        L.append(f"S3 R3 cost of delay 1.0 s minus 0.2 s: {s3['mean']:+.2f} [{s3['scene_95'][0]:+.2f}, {s3['scene_95'][1]:+.2f}]")
        L.append('')
    L.append('--- failure causes, worst car, % of its active steps (A attach, L landed-full, F full-other, W waiting, B outage) ---')
    for split in SPLITS:
        for pol in ['R1', 'R2', 'R3']:
            cb = R['causes'][f'{split}_{pol}']
            for hl in HLS:
                r = cb['rates'][hl]['worst']
                L.append(f"{split} {pol} {hl * 0.1:.1f} s: " + '  '.join(f"{k} {r[k]:.2f}" for k in CAUSES))
            sh = cb['shares']
            L.append(f"{split} {pol} cost of delay (1.0 - 0.2 s) {cb['delta_total']:+.2f} pp; shares: " +
                     '  '.join(f"{k} {(sh[k] if sh[k] is not None else float('nan')):.2f}" for k in CAUSES))
        L.append('')
    L.append(f"DECISION (R3, train): {R['decision']}")
    if 'guess_check' in R:
        L.append(f"Recorded guess check: {R['guess_check']}")
    L.append('')
    L.append('--- blind starts (handover started toward a station full at landing), share of starts ---')
    for split in SPLITS:
        for hl in HLS:
            L.append(f"{split} {hl * 0.1:.1f} s: " + '  '.join(
                f"{p} {R['S4'][f'{split}_hl{hl}_{p}']['blind_share']:.3f}" for p, _ in POLICIES))
    open(path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')


def run():
    g = json.load(open(os.path.join(OUT, 'GATES.json')))
    assert g['_pass'], 'gates did not pass'
    assert md5(os.path.abspath(__file__)) == g['md5']['script'], 'script changed after the gates'
    for split in SPLITS:
        assert md5(seed_path(split)) == g['T1'][split]['md5'], f'seed list {split} changed'
    pool = pd.read_csv(POOL)
    seeds = {sp: list(pd.read_csv(seed_path(sp)).seed.astype(int)) for sp in SPLITS}
    csv = os.path.join(OUT, 'REMEASURE_EPISODES.csv')
    assert not os.path.exists(csv), 'fresh run already exists: it runs once'
    allrows = []
    status('RUN START')
    for split in SPLITS:
        for hl in HLS:
            env = AV.make_env(pool, split, hl)
            for name, fn in POLICIES:
                t0 = time.time()
                w, m, o, rows = run_eval_log(env, fn, seeds[split])
                for r in rows:
                    r.update(split=split, hl=hl, policy=name)
                allrows += rows
                pd.DataFrame(allrows).to_csv(csv + '.partial', index=False)
                status(f'cell {split} hl={hl} {name} done ({len(rows)} episodes, {time.time() - t0:.0f} s)')
    df = pd.DataFrame(allrows)
    df.to_csv(csv, index=False)
    os.replace(csv + '.partial', os.path.join(OUT, 'REMEASURE_EPISODES_partial_final.csv'))
    status('RUN DONE; analysis')
    R = analyse(df)
    R['_gates'] = {k: g[k] for k in ['md5', 'T1', 'G0_ok', 'G1_ok', 'G2_ok', '_pass', '_versions']}
    R['_episodes_csv_md5'] = md5(csv)
    json.dump(R, open(os.path.join(OUT, 'REMEASURE_RESULT.json'), 'w'), indent=1)
    write_txt(R, os.path.join(OUT, 'REMEASURE_RESULT.txt'))
    status('ANALYSIS DONE')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['gates', 'run'], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == 'gates':
        sys.exit(0 if gates() else 2)
    run()
