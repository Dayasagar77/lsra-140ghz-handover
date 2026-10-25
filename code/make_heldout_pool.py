#!/usr/bin/env python3
"""Held-out scene pool for the test of the landing-state method (20 Oct 2026).

Daya chose new DAIR scenes for the method test (20 Oct). Source: DAIR-V2X-Seq TFD, single-vehicle
view, official VAL split. The 45-scene design pool (dair_geometry_pool.csv) comes from the TRAIN
split; July's 300-scene subset (DAIR DATASET/sv_subset.tar.gz) used 240 train and 60 val scenes, so
those 60 val scenes are excluded, as is any id that also appears in the train folder.

Rule, fixed before any outcome on these scenes is computed: sort the remaining val scene ids
numerically, draw K = 60 without replacement with numpy default_rng(20261020), copy those CSVs
unchanged (byte-identical, md5 checked) into data/heldout_val60_20261020/single-vehicle/trajectories/,
and write a manifest. The unchanged extractor (extract_dair_geometry.py) then builds the pool.
No research number is computed here.
"""
import os, sys, csv, tarfile, hashlib, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.dirname(HERE)
SV_ROOT = os.path.join(P2, 'DAIR DATASET', 'V2X-Seq-TFD', 'single-vehicle', 'trajectories')
VAL = os.path.join(SV_ROOT, 'val')
TRAIN = os.path.join(SV_ROOT, 'train')
JULY = os.path.join(P2, 'DAIR DATASET', 'sv_subset.tar.gz')
POOL45 = os.path.join(HERE, 'dair_geometry_pool.csv')
CACHE = os.path.join(HERE, 'geom_cache_ma')
DEST_ROOT = os.path.join(P2, 'data', 'heldout_val60_20261020')
DEST = os.path.join(DEST_ROOT, 'single-vehicle', 'trajectories')
K, SEED = 60, 20261020


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def main():
    val_ids = sorted((int(stem(f)) for f in os.listdir(VAL) if f.endswith('.csv')))
    train_ids = {int(stem(f)) for f in os.listdir(TRAIN) if f.endswith('.csv')}
    july = set()
    with tarfile.open(JULY) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith('.csv'):
                july.add(int(stem(m.name)))
    pool45 = set()
    with open(POOL45, newline='') as f:
        for r in csv.DictReader(f):
            pool45.add(int(r['scene_id']))
    cache_ids = set()
    if os.path.isdir(CACHE):
        for f in os.listdir(CACHE):
            if f.endswith('.pkl'):
                cache_ids.add(f.split('_')[0])
    cand = [i for i in val_ids if i not in july and i not in pool45 and i not in train_ids]
    rng = np.random.default_rng(SEED)
    pick = sorted(int(x) for x in rng.choice(np.array(cand), size=K, replace=False))
    assert not (set(map(str, pick)) & cache_ids), 'a picked id already has cached geometry'
    os.makedirs(DEST, exist_ok=False)
    rows = []
    for i in pick:
        src = os.path.join(VAL, f'{i}.csv'); dst = os.path.join(DEST, f'{i}.csv')
        shutil.copy2(src, dst)
        m1, m2 = md5(src), md5(dst)
        assert m1 == m2, f'copy mismatch for {i}'
        rows.append(dict(scene_id=i, md5=m1, bytes=os.path.getsize(dst)))
    man = os.path.join(DEST_ROOT, 'MANIFEST_heldout_val60_20261020.csv')
    with open(man, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scene_id', 'md5', 'bytes']); w.writeheader(); w.writerows(rows)
    print(f'val ids {len(val_ids)}; excluded July {len(july & set(val_ids))}, in train {len(set(val_ids) & train_ids)}, '
          f'in 45-pool {len(pool45 & set(val_ids))}; candidates {len(cand)}; picked {len(pick)} (seed {SEED})')
    print('picked:', pick)
    print('manifest md5', md5(man))


if __name__ == '__main__':
    main()
