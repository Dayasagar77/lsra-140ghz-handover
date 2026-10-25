#!/usr/bin/env python3
"""Dry run of link_check.analyse on the admission check's finished A2 runs (default link), 22 Oct 2026.
Purpose: exercise every line of the analysis before the freeze, on finished data only. No variant episode
exists or is created here. Output: models/_dry_link/ (a code check, not a result).
Expected: the HB1-HB3 means equal the admission check's HR2 means (every 'variant' is the default-link data);
every 'variant minus default' contrast is exactly 0; the must-hold checks pass, as in the admission check."""
import os
import pandas as pd
import link_check as L
L.OUT = os.path.join(L.HERE, 'models', '_dry_link')
os.makedirs(L.OUT, exist_ok=True)
for var in L.VARIANTS:
    for h in L.HLS:
        dst = L.cell_path(var, h)
        if not os.path.exists(dst):
            d = pd.read_csv(os.path.join(L.HERE, 'models', 'admission_check', f'ADMIT_EPISODES_A2_hl{h}.csv'))
            d['var'] = var
            d.to_csv(dst, index=False)
L.check_link_freeze = lambda: {'beta': 0.05, 'dry_run': 'no freeze; the admission check A2 data under every variant name'}
L.analyse()
