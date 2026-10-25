#!/usr/bin/env python3
"""Tables T1-T5 of the multi-car paper (20 Oct 2026). No simulation and no new statistics.
v3 (22 Oct 2026): T1 adds the delay-difference symbols xi and xi_max; the T5 footnote names HR3 a
pre-registered secondary test (95 % interval), as in its frozen pre-registration. Nothing else changed.
v4 (22 Oct 2026): "car" -> "vehicle" in every label and description written; numbers and CSV column names unchanged.
v5 (22 Oct 2026): adds the third load, 6 places per station (LOAD6_RESULT.json). T4 is turned (one row per load and
admission model, one column per arm) so that nine columns of results fit a page; T5 gains two columns. Every CSV row
of v4 is written again unchanged, in the same order relative to the others; the new rows are the 6-place ones.
v6 (22 Oct 2026): one sentence under T4 and T5: each interval is its own scene bootstrap, so identical runs can show
slightly different intervals. Nothing else changed.
v7 (23 Oct 2026): T3 names the method "landing-state risk assignment" (LSRA), as chosen by Daya on 23 Oct. Nothing else changed.

T1  notation (as in the theory note v3)
T2  parameters, read from the code at run time (each row names the constant it comes from)
T3  arms and admission models
T4  main results at 1.0 s: three admission models x three loads, with the registered verdicts
T5  ablations and room for forecasting at 1.0 s
Sources (md5 checked): ADMIT_RESULT.json, LOAD_RESULT.json, LOAD6_RESULT.json, HELDOUT_RESULT.json,
method_heldout FREEZE.json.
Outputs (tables_ma/): TABLES_MA.md, T2_PARAMETERS.csv, T4_MAIN_1s.csv, T5_ABLATIONS_1s.csv.
"""
import os, sys, json, csv, hashlib, platform
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import paper2_env as PE
import paper2ma_env as MA
import blockage3d as B
import remeasure_ma as RM
import method_heldout as MHO

OUT = os.path.join(HERE, 'tables_ma')
SRC = {'ADMIT': ('models/admission_check/ADMIT_RESULT.json', '803e1017a39ef1dcd4a6e9ea10ade9ad'),
       'LOAD': ('models/load_check/LOAD_RESULT.json', '0fb5845ca6b234261d7ca78363c5e730'),
       'LOAD6': ('models/load_check_c6/LOAD6_RESULT.json', '8e1860b938f69560610da9619e7bbc00'),
       'HELDOUT': ('models/method_heldout/HELDOUT_RESULT.json', '9254d3e8964a4e49cd018d283d0837ef')}
NAME = {'METHOD': 'LSRA', 'R0': 'R0', 'R1': 'R1', 'R1i': 'R1i', 'R2': 'R2', 'R3': 'R3', 'R3F': 'R3F',
        'METHOD_oracle': 'LSRA, perfect forecast', 'ABL_nolanding': 'LSRA without the landing state',
        'ABL_worstcarfirst': 'LSRA with the worst-vehicle-first stage'}


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def load():
    R = {}
    for k, (rel, m) in SRC.items():
        p = os.path.join(HERE, rel)
        assert md5(p) == m, f'{rel} changed'
        R[k] = json.load(open(p))
    return R


def ci(e):
    return f"{e['worst']:.2f} [{e['scene_95'][0]:.2f}, {e['scene_95'][1]:.2f}]"


def md_table(header, rows):
    L = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    L += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return L


T1 = [('Δ', 'slot length, 0.1 s; t, τ are slot indices'),
      ('M, G, C', 'vehicles per scene (16), stations (4), places per station (4, 5 or 6)'),
      ('A(t)', 'vehicles present in slot t'),
      ('σᵢ(t), κᵢ(t), θᵢ(t)', 'serving station, handover timer, committed target of vehicle i'),
      ('L', 'handover delay in slots: a request made at t lands in slot t + L + 1'),
      ('ξ, ξ_max', 'delay-difference check: a handover takes L + ξ slots, ξ uniform on {−ξ_max, …, ξ_max} (ξ_max = 1, 2)'),
      ('F(t), P(t)', 'free vehicles (κ = 0) and vehicles in flight (κ > 0)'),
      ('λᵢ(t)', 'landing station: θᵢ(t) if in flight, else σᵢ(t)'),
      ('n_g(t), n̄_g(t)', 'vehicles at station g now; landing load (vehicles in flight counted where they land)'),
      ('S, n̄ˢ_g(t)', 'request set of one slot; landing load after S'),
      ('landing-feasible', 'n̄ˢ_g(t) ≤ max(C, n̄_g(t)) at every station'),
      ('γ_out, N', 'outage threshold (−13.8 dB SINR); consecutive failed slots for a radio-link failure (3)'),
      ('ψᵢ(τ), J', 'radio-link failure of vehicle i in slot τ; worst-vehicle RLF rate, maxᵢ of the mean of ψᵢ'),
      ('π̂_{i,g}(τ | t)', 'forecast probability that vehicle i fails on station g in slot τ'),
      ('r̄_{i,g}(t), H', 'risk after landing: mean of π̂ over the H = L + 1 slots after landing'),
      ('β', 'cost of one handover in LSRA (0.05)'),
      ('q_g(t)', 'places left for free vehicles at g: C minus the vehicles in flight landing at g'),
      ('A0, A1, A2', 'admission at landing only; per request at preparation (today); coordinated, as a set')]

T3_ARMS = [
    ('R0', 'A vehicle below γ_out hands over to the less crowded of its two strongest other stations (current load).', 'rule, no capacity check'),
    ('R1', 'R0, but a station at or above C (current load) is not a candidate; with no candidate the vehicle stays.', 'rule with capacity check'),
    ('R1i', 'R1 counting vehicles on their way (landing load), updated after each decision in the slot.', 'rule; R1 with the landing count'),
    ('R2', 'Joint assignment of exactly the vehicles in outage (R0\'s movers) to the places left; others stay.', 'restricted joint assignment'),
    ('R3', 'Joint assignment of every present vehicle to the G·C places, each slot, on the current geometry and current stations.', 'full joint assignment'),
    ('R3F', 'R3 scoring each station by its mean received power at the landing slot, from the true future geometry.', 'reference; uses the future'),
    ('LSRA', 'Landing-state risk assignment: free vehicles only, quotas q_g, cost r̄ + β per move, one assignment per slot.', 'proposed method'),
    ('LSRA, perfect forecast', 'LSRA with the true future geometry in the risk window.', 'reference; uses the future'),
    ('LSRA without the landing state', 'LSRA planning on the current stations with every vehicle movable.', 'ablation'),
    ('LSRA with the worst-vehicle-first stage', 'LSRA with a min-max stage before the min-cost stage (dropped at the design stage).', 'ablation')]
T3_ADM = [('A0', 'Every request starts; a station serves its C strongest vehicles at landing, the rest are unserved.'),
          ('A1', 'Per request at preparation (today\'s rule, most generous reading): strongest first, admitted if the landing load at the target is below C; departures credited; repeated passes.'),
          ('A2', 'Coordinated: the largest landing-feasible subset of the slot\'s requests is admitted as a set (the 6G requirement proposed here).')]


def t2_rows(beta, n_design, n_held, n_eps):
    CH, CM = PE.CH, PE.CM
    return [
        ('Carrier frequency', f'{CH.FREQ / 1e9:.0f} GHz', 'channel_model_nyu.FREQ'),
        ('Bandwidth', f'{CM.BW / 1e6:.2f} MHz', 'channel_model.BW (66 RBs x 12 x 480 kHz)'),
        ('Noise figure; noise power', f'{CM.NF_DB:.0f} dB; {CM.N0_DBM:.1f} dBm', 'channel_model.NF_DB, N0_DBM'),
        ('Path loss', f'CI model, FSPL(1 m) = {CH.fspl_d0_db():.2f} dB, exponent {CH.N_LOS}', 'channel_model_nyu (142 GHz UMi, LOS)'),
        ('Blockage loss', f'3D boxes, ITU-R P.526 knife edge, cap {B.CAP_DB_DEFAULT:.0f} dB, blocked if >= {B.BLOCK_THR_DB:.0f} dB', 'blockage3d'),
        ('Shadow fading (std)', f'{CH.SIG_LOS} dB clear, {CH.SIG_NLOS} dB blocked', 'channel_model_nyu.SIG_LOS, SIG_NLOS'),
        ('Heights', f'vehicle antenna {B.H_UE_DEFAULT} m, station {B.H_GNB_DEFAULT} m', 'blockage3d.H_UE_DEFAULT, H_GNB_DEFAULT'),
        ('Stations', f'{MA.N_GNB}, at the corners, 5 m outside the traffic extent', 'blockage3d.scene_gnb (margin 5 m)'),
        ('Transmit power (every arm)', f'{PE.PT_MAX:.0f} dBm', 'paper2_env.PT_MAX'),
        ('Serving beam gain; side lobe', f'{PE.G_SERV:.0f} dBi; {PE.G_SIDE:.0f} dBi', 'paper2_env.G_SERV, G_SIDE (assumption)'),
        ('Background interferers', f'{MA.K_BG} static vehicles per station at {PE.INTERF_PT_DBM:.0f} dBm', 'paper2ma_env.K_BG, paper2_env.INTERF_PT_DBM'),
        ('Sub-bands', f'{MA.N_SUBBANDS}, assigned round robin, fixed', 'paper2ma_env.N_SUBBANDS'),
        ('Outage threshold γ_out', f'{PE.SINR_OUT_DB:.1f} dB SINR (= {PE.RLF_THRESH_DBM:.0f} dBm minus the noise power)', 'paper2_env.SINR_OUT_DB'),
        ('Radio-link failure', f'N310 = {PE.N310} consecutive failed slots', 'paper2_env.N310'),
        ('Slot', '0.1 s (DAIR-V2X frame rate)', 'pool timestamps'),
        ('Vehicles, places', '16 vehicles per scene; C = 4 (full load), C = 5 (load 0.8) and C = 6 (load 0.67)', 'run scripts'),
        ('Handover delay', ', '.join(f'{h * 0.1:.1f} s' for h in MHO.HLS) + f' (L = {", ".join(str(h) for h in MHO.HLS)} slots)', 'method_heldout.HLS'),
        ('LSRA', f'β = {beta}, H = L + 1, forecast held from slot t', 'FREEZE.json, lsra.py'),
        ('Scenes, episodes', f'{n_design} design and {n_held} held-out DAIR-V2X intersection scenes; {n_eps} episodes per cell', 'pools, SEEDS_heldout60.csv'),
        ('Statistics', f'scene bootstrap, {RM.BOOT:,} resamples; 98.75 % for four-baseline families', 'remeasure_ma.BOOT')]


def main():
    os.makedirs(OUT, exist_ok=True)
    R = load()
    fz = MHO.check_freeze()
    TA, TL, TL6 = R['ADMIT']['secondary']['table'], R['LOAD']['secondary']['table'], R['LOAD6']['secondary']['table']
    TABS = ((TA, 4), (TL, 5), (TL6, 6))
    n_held = pd.read_csv(MHO.POOL_H, usecols=['scene_id'])['scene_id'].nunique()
    n_design = pd.read_csv(RM.POOL, usecols=['scene_id'])['scene_id'].nunique()
    n_eps = len(pd.read_csv(os.path.join(MHO.OUT, 'SEEDS_heldout60.csv')))
    L = ['# Tables T1–T5 (multi-vehicle paper)', '',
         'Made by `make_tables_ma.py` from the code and the finished result files only. Sources: ' +
         '; '.join(f'`{v[0]}` (md5 {v[1]})' for v in SRC.values()) + '.', '']
    L += ['## T1. Notation', ''] + md_table(['Symbol', 'Meaning'], T1) + ['']
    t2 = t2_rows(fz['beta'], n_design, n_held, n_eps)
    L += ['## T2. Parameters (read from the code)', ''] + md_table(['Parameter', 'Value', 'From'], t2) + ['']
    with open(os.path.join(OUT, 'T2_PARAMETERS.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['parameter', 'value', 'from']); w.writerows(t2)
    L += ['## T3. Arms and admission models', ''] + md_table(['Arm', 'What it does', 'Role'], T3_ARMS) + ['']
    L += md_table(['Admission model', 'Rule'], T3_ADM) + ['']
    # T4 (one row per load and admission model; the CSV keeps its long form, arm by arm)
    arms = ['R1', 'R1i', 'R2', 'R3', 'METHOD']
    csvrows = []
    for a in arms:
        for T, c in TABS:
            for A in ('A0', 'A1', 'A2'):
                e = T[f'{A}_hl10_{a}']
                csvrows.append([NAME[a], c, A, e['worst'], e['scene_95'][0], e['scene_95'][1], e['ho_per_car_step'],
                                e.get('refused_share', '')])
    fams = {(4, 'A0'): R['HELDOUT']['HM1'], (4, 'A1'): R['ADMIT']['HR1_A1'], (4, 'A2'): R['ADMIT']['HR2_A2'],
            (5, 'A1'): R['LOAD']['HL1_A1'], (5, 'A2'): R['LOAD']['HL2_A2'],
            (6, 'A1'): R['LOAD6']['HL3_A1'], (6, 'A2'): R['LOAD6']['HL4_A2']}
    tests = {(4, 'A0'): 'HM1', (4, 'A1'): 'HR1', (4, 'A2'): 'HR2', (5, 'A1'): 'HL1', (5, 'A2'): 'HL2',
             (6, 'A1'): 'HL3', (6, 'A2'): 'HL4'}
    rows = []
    for T, c in TABS:
        for A in ('A0', 'A1', 'A2'):
            f = fams.get((c, A))
            ver = '—' if f is None else (tests[(c, A)] + ': ' + ('held' if f['held'] else 'not held (' + ', '.join(
                b for b, v in f['per_baseline'].items() if not v['beats']) + ' not shown)'))
            rows.append([f'{c} places, {A}'] + [ci(T[f'{A}_hl10_{a}']) for a in arms] + [ver])
    hdr = ['Places, admission'] + [NAME[a] for a in arms] + ['Registered test: LSRA beats R1, R1i, R2, R3']
    L += ['## T4. Worst-vehicle RLF at 1.0 s (%), 95 % scene intervals', '',
          '60 held-out scenes, 480 episodes per cell. Registered tests use 98.75 % intervals (Fig. 5). Each interval is '
          'its own scene bootstrap, so identical runs, such as LSRA under A0 and A2, can show slightly different intervals.', '']
    L += md_table(hdr, rows) + ['']
    with open(os.path.join(OUT, 'T4_MAIN_1s.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['arm', 'places', 'model', 'worst', 'lo95', 'hi95', 'ho_per_car_step', 'refused_share'])
        w.writerows(csvrows)
    # T5
    arms5 = ['METHOD', 'ABL_nolanding', 'ABL_worstcarfirst', 'METHOD_oracle', 'R3', 'R3F']
    rows, csvrows = [], []
    for a in arms5:
        r = [NAME[a]]
        for T, c in TABS:
            for A in ('A0', 'A2'):
                e = T[f'{A}_hl10_{a}']
                d = e.get('minus_METHOD_same_admission')
                r.append(ci(e) + ('' if d is None else f"; {d['mean']:+.2f} vs LSRA"))
                csvrows.append([NAME[a], c, A, e['worst'], e['scene_95'][0], e['scene_95'][1],
                                '' if d is None else d['mean'], '' if d is None else d['scene_95'][0],
                                '' if d is None else d['scene_95'][1]])
        rows.append(r)
    h3 = R['ADMIT']['HR3_A2_nolanding_minus_METHOD']
    L += ['## T5. Ablations and room for forecasting, 1.0 s (worst-vehicle RLF %, 95 % scene interval; difference to LSRA)', '',
          'Each interval is its own scene bootstrap (see T4).', '']
    L += md_table(['Arm'] + [f'{c} places, {A}' for _, c in TABS for A in ('A0', 'A2')], rows) + ['']
    L += [f"Pre-registered secondary test HR3 (95 % interval; 4 places, A2: LSRA without the landing state minus LSRA): {h3['mean']:+.2f} pp "
          f"[{h3['scene_95'][0]:+.2f}, {h3['scene_95'][1]:+.2f}], {'held' if h3['held'] else 'not held'}.", '']
    with open(os.path.join(OUT, 'T5_ABLATIONS_1s.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['arm', 'places', 'model', 'worst', 'lo95', 'hi95', 'minus_LSRA', 'lo95_d', 'hi95_d'])
        w.writerows(csvrows)
    L += [f'_make_tables_ma.py md5 {md5(os.path.abspath(__file__))}; python {platform.python_version()}, '
          f'numpy {np.__version__}, pandas {pd.__version__}._']
    open(os.path.join(OUT, 'TABLES_MA.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
