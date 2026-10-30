#!/usr/bin/env python3
"""a3o_crossfit.py -- exploratory, post hoc (30 Oct 2026): a cross-fit estimate of per-scene A3 tuning.

Spec: EXPLORATORY_A3O_CrossFit_HonestPerSceneTuning_Spec_20261030.md (fixed before running; Daya's OK). It reads only
the stored files of the new-baselines run (models/baselines_check/oracle, test) and LSRA's stored episodes; it runs no
episode and changes no file of that run.

For each scene (8 episodes) and each of the 35 ways to split its episodes into two halves of 4 (the first episode always
in half A): choose the setting with the lowest mean J on one half (ties: the earlier in grid order) and score it on the
other half, both ways. An episode's cross-fit J is the average over the 35 splits of the J it receives from the setting
chosen on the other half.
"""
import os, sys, json, time, hashlib, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import remeasure_ma as RM
import a3_check as A3C
import a3_general as AG

OUT = os.path.join(HERE, 'models', 'baselines_check')
SPEC = 'EXPLORATORY_A3O_CrossFit_HonestPerSceneTuning_Spec_20261030.md'
CELLS = [(4, 'A2', 13000, 13002), (5, 'A2', 13001, 13003)]
REGISTERED_A3O = {4: 3.0161450478272434, 5: 1.8068641731462238}     # BASE_RESULT.json A3O_bound, for a consistency check


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def load_grid(cap, A):
    per = []
    for p in AG.grid('measured'):
        d = pd.read_csv(os.path.join(OUT, 'oracle', f'C{cap}_{A}', AG.setting_id(p) + '.csv'))
        d['scene'] = d['scene'].astype(str)
        per.append(d.sort_values('seed').reset_index(drop=True))
    base = per[0]
    for d in per:
        assert len(d) == 480 and list(d.seed) == list(base.seed) and list(d.scene) == list(base.scene), 'grid files do not pair'
    return base[['seed', 'scene']].copy(), np.stack([d.worst.values.astype(float) for d in per])


def stored_lsra(cap, A):
    s = pd.read_csv(A3C.stored_path(A, cap, 10))
    s = s[(s['arm'] == 'METHOD') & (s['hl'] == 10)].copy()
    s['scene'] = s['scene'].astype(str)
    return s.sort_values('seed').reset_index(drop=True)


def tuned_a3(cap, A):
    d = pd.read_csv(os.path.join(OUT, 'test', f'A3T_C{cap}_{A}.csv'))
    d['scene'] = d['scene'].astype(str)
    return d.sort_values('seed').reset_index(drop=True)


def crossfit(scenes, W):
    """scenes: (n_ep,) scene of each episode in seed order; W: (n_settings, n_ep) J per setting and episode."""
    cf = np.zeros(W.shape[1])
    ins = np.zeros(W.shape[1])
    splits = [np.array(c) for c in itertools.combinations(range(8), 4) if 0 in c]
    assert len(splits) == 35
    for sc in sorted(set(scenes)):
        idx = np.flatnonzero(scenes == sc)
        assert len(idx) == 8, f'scene {sc} has {len(idx)} episodes'
        ins[idx] = W[int(np.argmin(W[:, idx].mean(axis=1))), idx]
        acc = np.zeros(8)
        for a in splits:
            b = np.array([j for j in range(8) if j not in set(a.tolist())])
            ka = int(np.argmin(W[:, idx[a]].mean(axis=1)))
            kb = int(np.argmin(W[:, idx[b]].mean(axis=1)))
            acc[b] += W[ka, idx[b]]
            acc[a] += W[kb, idx[a]]
        cf[idx] = acc / len(splits)
    return cf, ins


def main():
    res = {'spec': SPEC, 'label': 'exploratory, post hoc (not a registered test)', 'code_md5': md5(os.path.abspath(__file__)),
           'written': time.strftime('%Y-%m-%d %H:%M:%S')}
    lines = ['A3-O CROSS-FIT (exploratory, post hoc): per-scene A3 tuning chosen on half of a scene\'s episodes and scored',
             'on the other half (35 splits per scene, both ways); 4 and 5 places, coordinated admission, 1.0 s; J in %', '']
    for cap, A, s1, s2 in CELLS:
        keys, W = load_grid(cap, A)
        l, t = stored_lsra(cap, A), tuned_a3(cap, A)
        for x in (l, t):
            assert len(x) == 480 and list(x.seed) == list(keys.seed) and list(x.scene) == list(keys.scene), 'pairing failed'
        sc = keys.scene.values
        cf, ins = crossfit(sc, W)
        assert abs(ins.mean() * 100 - REGISTERED_A3O[cap]) < 1e-9, 'in-sample bound differs from the registered A3-O'
        d_l = (cf - l.worst.values) * 100.0
        d_t = (cf - t.worst.values) * 100.0
        e = dict(insample_bound=float(ins.mean() * 100), crossfit=float(cf.mean() * 100),
                 optimism=float((cf.mean() - ins.mean()) * 100), LSRA=float(l.worst.mean() * 100),
                 A3T=float(t.worst.mean() * 100),
                 crossfit_minus_LSRA=dict(mean=float(d_l.mean()), ci_95=RM.scene_boot(d_l, sc, s1)),
                 crossfit_minus_A3T=dict(mean=float(d_t.mean()), ci_95=RM.scene_boot(d_t, sc, s2)))
        lo, hi = e['crossfit_minus_LSRA']['ci_95']
        e['reading'] = ('per-scene tuning without hindsight does not close the gap' if lo > 0 else
                        'honest per-scene tuning beats LSRA' if hi < 0 else
                        'per-scene tuning without hindsight is level with LSRA')
        res[f'C{cap}_{A}'] = e
        lines.append(f"{cap} places: in-sample bound {e['insample_bound']:.2f}, cross-fit {e['crossfit']:.2f} "
                     f"(optimism {e['optimism']:+.2f}); LSRA {e['LSRA']:.2f}; tuned A3 {e['A3T']:.2f}")
        lines.append(f"  cross-fit minus LSRA {e['crossfit_minus_LSRA']['mean']:+.2f} "
                     f"[{lo:+.2f}, {hi:+.2f}] -> {e['reading']}")
        c2 = e['crossfit_minus_A3T']['ci_95']
        lines.append(f"  cross-fit minus tuned A3 {e['crossfit_minus_A3T']['mean']:+.2f} [{c2[0]:+.2f}, {c2[1]:+.2f}]")
    json.dump(res, open(os.path.join(OUT, 'A3O_CROSSFIT.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'A3O_CROSSFIT.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
