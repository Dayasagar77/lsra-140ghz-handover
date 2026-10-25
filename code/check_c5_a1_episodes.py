#!/usr/bin/env python3
"""Read-only check of one statement in the paper draft (22 Oct 2026).

Statement (draft Section 8.3): at 5 places and 1.0 s delay, the cost of per-request admission (A1)
to LSRA (+1.36 pp worst-car RLF against A0) comes from 16 of the 480 episodes (two scenes).

This script only reads the frozen episode files of the load check and prints what they contain.
Standard library only. Nothing is simulated, nothing is changed.
Inputs:  models/load_check/LOAD_EPISODES_C5_A0_hl10.csv, LOAD_EPISODES_C5_A1_hl10.csv (arm METHOD = LSRA)
Outputs: models/load_check/CHECK_C5_A1vsA0_LSRA_episodes_20261022.json and .txt
"""
import csv, hashlib, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'models', 'load_check')
FILES = {'A0': os.path.join(D, 'LOAD_EPISODES_C5_A0_hl10.csv'),
         'A1': os.path.join(D, 'LOAD_EPISODES_C5_A1_hl10.csv')}
ARM = 'METHOD'
OUT_J = os.path.join(D, 'CHECK_C5_A1vsA0_LSRA_episodes_20261022.json')
OUT_T = os.path.join(D, 'CHECK_C5_A1vsA0_LSRA_episodes_20261022.txt')


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def load(path):
    rows = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['arm'] != ARM:
                continue
            key = (r['seed'], r['scene'])
            assert key not in rows, f'duplicate episode {key} in {path}'
            rows[key] = r
    return rows


def main():
    for p in FILES.values():
        if not os.path.isfile(p):
            sys.exit(f'missing {p}')
    a0, a1 = load(FILES['A0']), load(FILES['A1'])
    assert set(a0) == set(a1), 'A0 and A1 do not hold the same episodes'
    for k in a0:
        assert a0[k]['cap'] == a1[k]['cap'] == '5' and a0[k]['hl'] == a1[k]['hl'] == '10'
        assert a0[k]['adm'] == 'A0' and a1[k]['adm'] == 'A1'
    keys = sorted(a0, key=lambda k: (int(k[1]), int(k[0])))
    n = len(keys)
    f = lambda r, c: float(r[c])
    tot = {m: dict(starts=0.0, requests=0.0, refused=0.0, p_act=0.0) for m in ('A0', 'A1')}
    for k in keys:
        for m, rows in (('A0', a0), ('A1', a1)):
            for c in tot[m]:
                tot[m][c] += f(rows[k], c)
    d_worst = {k: (f(a1[k], 'worst') - f(a0[k], 'worst')) * 100.0 for k in keys}
    d_starts = {k: f(a1[k], 'starts') - f(a0[k], 'starts') for k in keys}
    differ_worst = [k for k in keys if abs(d_worst[k]) > 1e-12]
    differ_starts = [k for k in keys if d_starts[k] != 0]
    refused_eps = [k for k in keys if f(a1[k], 'refused') > 0]
    differ_any = sorted(set(differ_worst) | set(differ_starts),
                        key=lambda k: (int(k[1]), int(k[0])))
    mean_cost = sum(d_worst.values()) / n
    cost_from_differ = sum(d_worst[k] for k in differ_worst) / n
    by_scene = defaultdict(lambda: dict(episodes=0, sum_d_worst_pp=0.0, refused=0.0, d_starts=0.0))
    for k in differ_any + [k for k in refused_eps if k not in differ_any]:
        s = by_scene[k[1]]
        s['episodes'] += 1
        s['sum_d_worst_pp'] += d_worst[k]
        s['refused'] += f(a1[k], 'refused')
        s['d_starts'] += d_starts[k]
    refused_in_differ = sum(f(a1[k], 'refused') for k in differ_any)
    per_ep = [dict(seed=k[0], scene=k[1], worst_A0_pct=f(a0[k], 'worst') * 100, worst_A1_pct=f(a1[k], 'worst') * 100,
                   d_worst_pp=d_worst[k], starts_A0=f(a0[k], 'starts'), starts_A1=f(a1[k], 'starts'),
                   requests_A1=f(a1[k], 'requests'), refused_A1=f(a1[k], 'refused'),
                   loop_violations_A1=f(a1[k], 'loop_violations'))
              for k in differ_any]
    R = dict(
        inputs={m: dict(path=os.path.relpath(p, HERE), md5=md5(p)) for m, p in FILES.items()},
        script_md5=md5(os.path.abspath(__file__)), arm=ARM, episodes=n,
        totals=tot,
        mean_cost_A1_minus_A0_pp=mean_cost,
        episodes_where_worst_differs=len(differ_worst),
        episodes_where_starts_differ=len(differ_starts),
        episodes_with_any_refusal_A1=len(refused_eps),
        episodes_where_anything_differs=len(differ_any),
        scenes_where_anything_differs=sorted({k[1] for k in differ_any}, key=int),
        cost_from_differing_episodes_pp=cost_from_differ,
        share_of_cost_from_differing_episodes=(cost_from_differ / mean_cost) if mean_cost else None,
        refused_total_A1=tot['A1']['refused'], refused_in_differing_episodes=refused_in_differ,
        refused_per_differing_episode_max=max((f(a1[k], 'refused') for k in differ_any), default=0.0),
        refused_per_differing_episode_mean=(refused_in_differ / len(differ_any)) if differ_any else 0.0,
        by_scene=dict(by_scene), per_episode=per_ep)
    with open(OUT_J, 'w') as fh:
        json.dump(R, fh, indent=1)
    L = [f"C5, 1.0 s, LSRA (arm {ARM}): A1 against A0, paired by episode ({n} episodes)",
         f"inputs: A0 md5 {R['inputs']['A0']['md5']}, A1 md5 {R['inputs']['A1']['md5']}; script md5 {R['script_md5']}",
         f"starts: A0 {tot['A0']['starts']:.0f}, A1 {tot['A1']['starts']:.0f}; requests under A1 {tot['A1']['requests']:.0f}, "
         f"refused {tot['A1']['refused']:.0f} ({100 * tot['A1']['refused'] / max(tot['A1']['requests'], 1):.1f} %)",
         f"mean cost A1 - A0 (worst-car RLF): {mean_cost:+.4f} pp",
         f"episodes where worst-car RLF differs: {len(differ_worst)}; where starts differ: {len(differ_starts)}; "
         f"with any refusal under A1: {len(refused_eps)}; where anything differs: {len(differ_any)}",
         f"scenes where anything differs: {', '.join(R['scenes_where_anything_differs'])}",
         f"cost carried by those episodes: {cost_from_differ:+.4f} pp "
         f"({100 * R['share_of_cost_from_differing_episodes']:.1f} % of the mean cost)" if mean_cost else 'no cost',
         f"refusals in those episodes: {refused_in_differ:.0f} of {tot['A1']['refused']:.0f}; "
         f"per episode max {R['refused_per_differing_episode_max']:.0f}, mean {R['refused_per_differing_episode_mean']:.1f}",
         '', 'per episode (scene, seed): worst A0 % -> A1 %, d pp, starts A0/A1, requests/refused under A1']
    for e in per_ep:
        L.append(f"  {e['scene']} {e['seed']}: {e['worst_A0_pct']:.2f} -> {e['worst_A1_pct']:.2f} ({e['d_worst_pp']:+.2f}), "
                 f"starts {e['starts_A0']:.0f}/{e['starts_A1']:.0f}, requests {e['requests_A1']:.0f}, refused {e['refused_A1']:.0f}")
    with open(OUT_T, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
