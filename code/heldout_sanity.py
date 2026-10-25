#!/usr/bin/env python3
"""Environment-level sanity of the held-out pool (no policy is run, no outcome is computed).

For each scene of a pool: build the multi-agent geometry (cached as usual), and count frames,
active cars per frame (of 16) and the share of frames with all 16 present, for the held-out pool and
for the 45-scene design pool, so the two can be compared before any test.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import paper2ma_env as MA


def stats(pool_csv, split):
    pool = pd.read_csv(pool_csv)
    env = MA.MultiAgentRSUBlockageEnv(pool, n_agents=16, n_subbands=3, gnb_cap=4, seed=0, split=split,
                                      compute_anticipation=True, ho_latency=10)
    out = []
    for sid in env.scene_ids:
        obj = env._precompute(sid)
        act = np.array([fr['active'].sum() for fr in obj['frames']])
        out.append(dict(scene=str(sid), frames=len(act), mean_active=float(act.mean()),
                        full16=float((act == 16).mean()), min_active=int(act.min())))
    df = pd.DataFrame(out)
    return dict(n_scenes=int(len(df)), frames_median=float(df.frames.median()),
                mean_active=float(df.mean_active.mean()), full16_share=float(df.full16.mean()),
                scenes_mean_active_below_8=int((df.mean_active < 8).sum())), df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--heldout', default='dair_geometry_pool_heldout60_20261020.csv')
    a = ap.parse_args()
    h, hdf = stats(a.heldout, None)
    d, ddf = stats('dair_geometry_pool.csv', None)
    print('held-out:', json.dumps(h)); print('design 45:', json.dumps(d))
    hdf.to_csv('HELDOUT60_SCENE_STATS.csv', index=False)
