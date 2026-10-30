#!/usr/bin/env python3
"""Tables 7-13 of the LSRA manuscript v6, built only from the frozen result files (30 Oct 2026).
Repository version: the same code as make_tables_v6b.py, with the inputs read where this package keeps them
(models/baselines_check/, models/stored_checks/ and ../realism_deviation/reanalysis/) and the outputs written to
tables_ma/. Its TABLES_V6.md must be byte-identical to the one used for the article.
v6b: Table 10 (the changed models and the verdicts) added (wording of B1, S3, DI, F1 and the note fixed after the
independent check of 30 Oct); the beam models relabelled for the paper
(A1a, A1b, A2, A3 of the result files -> G10, G20, DI, MA, since A1, A2 and A3 already name admission
models and the 3GPP trigger); Table 13 shows the cost against ideal coordination as a positive number.

Inputs (results/):
  BASE_RESULT.json                         new-baselines run (md5 67bdec4c778f28e75a6a0afd68e07b75)
  A3O_CROSSFIT.json                        exploratory cross-fit of the per-scene A3 bound (md5 81bfbdc6...)
  STORED_CHECKS_RESULT.json                planner / admission split (md5 ef278bfd2763f318791ad92131fc6336)
  REALISM_REAL_RESULT_reanalysis_20261030.json   realism run, the analysis as re-run once (md5 bef337dafcd5135ba6ce69a120461cd9)
Outputs (tables/): TABLES_V6.md and TABLE_VALUES_V6.json (every number placed in a table, for the checks).
"""
import os, json, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tables_ma')
SRC = {'BASE_RESULT.json': os.path.join(HERE, 'models', 'baselines_check', 'BASE_RESULT.json'),
       'A3O_CROSSFIT.json': os.path.join(HERE, 'models', 'baselines_check', 'A3O_CROSSFIT.json'),
       'STORED_CHECKS_RESULT.json': os.path.join(HERE, 'models', 'stored_checks', 'STORED_CHECKS_RESULT.json'),
       'REALISM_REAL_RESULT_reanalysis_20261030.json': os.path.join(HERE, '..', 'realism_deviation', 'reanalysis', 'REAL_RESULT.json')}
WANT = {'BASE_RESULT.json': '67bdec4c778f28e75a6a0afd68e07b75',
        'STORED_CHECKS_RESULT.json': 'ef278bfd2763f318791ad92131fc6336',
        'REALISM_REAL_RESULT_reanalysis_20261030.json': 'bef337dafcd5135ba6ce69a120461cd9'}
MINUS = '−'


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


for f, h in WANT.items():
    assert md5(SRC[f]) == h, f + ' changed'
B = json.load(open(SRC['BASE_RESULT.json']))
X = json.load(open(SRC['A3O_CROSSFIT.json']))
S = json.load(open(SRC['STORED_CHECKS_RESULT.json']))
R = json.load(open(SRC['REALISM_REAL_RESULT_reanalysis_20261030.json']))
assert not B['problems'] and not S['problems'] and not R['problems'] and R['ok'] is True
VALS = {}


def sg(x):
    s = f'{x:+.2f}'
    if s == '-0.00':
        s = '+0.00' if x == 0 else MINUS + '0.00'
    return s.replace('-', MINUS)


def lv(x):
    return f'{x:.2f}'


def ci(c):
    return f'[{sg(c[0])}, {sg(c[1])}]'


def d(m, c):
    return f'{sg(m)} {ci(c)}'


def put(key, v):
    VALS[key] = v
    return v


L = []
# ------------------------------------------------------------------ Table 7: new baselines at full load
LS = {'A2': 3.593273715702598, 'A1': 8.75995183456838}
names = [('A3T', 'Tuned A3 (A3-T)'), ('A3TNF', 'Tuned A3, noise-free signal (A3-T-NF)'),
         ('A3LB', 'Load-balancing A3 (A3-LB)'), ('PHO', 'Na et al., perfect forecast (PH-O)'),
         ('PHCV', 'Na et al., straight-line forecast (PH-CV)'),
         ('PHOR', 'Na et al., perfect forecast and room check (PH-OR)')]
L.append('| Arm | J, A2 (%) | Arm − LSRA, A2 (HN1) | J, A1 (%) | Arm − LSRA, A1 (HN2) | Handovers per vehicle-minute, A2 | Requests refused, A2 |')
L.append('|----------------|----------|----------------------|----------|----------------------|----------|----------|')
for k, nm in names:
    m2, m1 = B['mechanisms'][f'{k}_C4_A2'], B['mechanisms'][f'{k}_C4_A1']
    h1, h2 = B['HN1_A2_C4']['per_arm'][k], B['HN2_A1_C4']['per_arm'][k]
    assert abs(m2['worst'] - h1['mean'] - LS['A2']) < 1e-9 and abs(m1['worst'] - h2['mean'] - LS['A1']) < 1e-9
    put(f'T7_{k}_A2', [m2['worst'], h1['mean'], h1['ci_9917']])
    put(f'T7_{k}_A1', [m1['worst'], h2['mean'], h2['ci_9917']])
    put(f'T7_{k}_ho', m2['ho_per_vehicle_min'])
    put(f'T7_{k}_ref', m2['refused_share'])
    L.append(f"| {nm} | {lv(m2['worst'])} | {d(h1['mean'], h1['ci_9917'])} | {lv(m1['worst'])} | "
             f"{d(h2['mean'], h2['ci_9917'])} | {m2['ho_per_vehicle_min']:.1f} | {100 * m2['refused_share']:.1f} % |")
m2, m1 = B['mechanisms']['LSRAM_C4_A2'], B['mechanisms']['LSRAM_C4_A1']
a2, a1 = B['LSRAM']['C4_A2'], B['LSRAM']['C4_A1']
put('T7_LSRAM', [m2['worst'], a2['LSRAM_minus_LSRA'], a2['ci_95'], m1['worst'], a1['LSRAM_minus_LSRA'], a1['ci_95']])
L.append(f"| LSRA fed measurement reports only (LSRA-M)\\* | {lv(m2['worst'])} | {d(a2['LSRAM_minus_LSRA'], a2['ci_95'])}\\* | "
         f"{lv(m1['worst'])} | {d(a1['LSRAM_minus_LSRA'], a1['ci_95'])}\\* | {m2['ho_per_vehicle_min']:.1f} | {100 * m2['refused_share']:.1f} % |")
L.append(f"| LSRA | {lv(LS['A2'])} | — | {lv(LS['A1'])} | — | 1.7 | 0.0 % |")
L.append('')
L.append(': **Table 7.** The new baselines at full load (4 places, 1.0 s): worst-vehicle RLF J and difference to LSRA under the same admission, pp, with 99.17 % scene-bootstrap intervals (Bonferroni over six).')
L.append('')
L.append('::: {custom-style="TableNote"}')
hn1, hn2 = B['HN1_A2_C4'], B['HN2_A1_C4']
put('T7_HN1_held', hn1['held']); put('T7_HN2_held', hn2['held'])
L.append(f"HN1 (A2) {'held' if hn1['held'] else 'not held'}; HN2 (A1) {'held' if hn2['held'] else 'not held'}, not shown against "
         f"{', '.join('A3-T-NF' if x == 'A3TNF' else x for x in hn2['not_shown'])}. "
         "\\*LSRA-M is a variant of LSRA, not a baseline: its difference to LSRA has a 95 % interval and is not part of HN1 or HN2. "
         "A3-LB chose a load-balancing gain of 0 at 4 places, so its episodes equal A3-T's; its interval differs only through the bootstrap. "
         "LSRA's handover rate is the stored one (Section 8.4).")
L.append(':::')
L.append('')
L.append('')
# ------------------------------------------------------------------ Table 8: new baselines with spare places
L.append('| Places, admission | LSRA J (%) | A3-T | A3-T-NF | A3-LB | PH-O | PH-CV | PH-OR |')
L.append('|----------|--------|--------------|--------------|--------------|--------------|--------------|--------------|')
lsra_lv = {'C5_A1': 3.97, 'C5_A2': 2.61, 'C6_A1': 2.32, 'C6_A2': 2.32}
for cell in ['C5_A1', 'C5_A2', 'C6_A1', 'C6_A2']:
    so = B['secondary_other_loads'][cell]
    row = [f"{cell[1]} places, {cell[3:]}", lv(lsra_lv[cell])]
    for k, _ in names:
        put(f'T8_{cell}_{k}', [so[k]['mean'], so[k]['ci_95']])
        row.append(d(so[k]['mean'], so[k]['ci_95']))
    L.append('| ' + ' | '.join(row) + ' |')
L.append('')
L.append(': **Table 8.** The new baselines with spare places (1.0 s): baseline minus LSRA under the same admission, pp, with 95 % scene-bootstrap intervals (pre-registered secondary results).')
L.append('')
L.append('')
# ------------------------------------------------------------------ Table 9: split
C = S['check_A']['cells']
rows = [('4', 'R2', C['C4_R2_hl10']), ('4', 'R1i', C['C4_R1i_hl10']), ('4', 'R3', C['C4_R3_hl10']),
        ('4', 'A3, Set 5', C['C4_A3S5_hl10']), ('4', 'Tuned A3 (A3-T)', B['split']['C4_A3T']),
        ('4', 'Na et al. (PH-O)', B['split']['C4_PHO']),
        ('5', 'R2', C['C5_R2_hl10']), ('5', 'R3', C['C5_R3_hl10']), ('5', 'A3, Set 5', C['C5_A3S5_hl10']),
        ('6', 'R2', C['C6_R2_hl10']), ('6', 'R3', C['C6_R3_hl10']), ('6', 'A3, Set 5', C['C6_A3S5_hl10'])]
L.append('| Places | Baseline | Baseline under A1 → LSRA under A2 (%) | T, total | P1, planner alone | Ab, admission alone | I, needs both |')
L.append('|----|------------|------------|--------------|--------------|--------------|--------------|')
for pl, nm, c in rows:
    def g(k):
        e = c[k]
        cc = e.get('scene_95', e.get('ci_95'))
        return e['mean'], cc
    vals = {k: g(k) for k in ('T', 'P1', 'Ab', 'I')}
    lvls = c['levels']
    put(f'T9_{pl}_{nm}', {k: [v[0], v[1]] for k, v in vals.items()} | {'B_A1': lvls['B_A1'], 'L_A2': lvls['L_A2']})
    assert abs(vals['T'][0] - (vals['P1'][0] + vals['Ab'][0] + vals['I'][0])) < 1e-9
    assert abs(vals['T'][0] - (lvls['B_A1'] - lvls['L_A2'])) < 1e-9

    def cellv(k):
        m, cc = vals[k]
        if abs(m) < 1e-12 and abs(cc[0]) < 1e-12 and abs(cc[1]) < 1e-12:
            return '0.00 (exactly)'
        return d(m, cc)
    L.append(f"| {pl} | {nm} | {lv(lvls['B_A1'])} → {lv(lvls['L_A2'])} | {cellv('T')} | {cellv('P1')} | {cellv('Ab')} | {cellv('I')} |")
L.append('')
L.append(': **Table 9.** Where the gain over today\'s practice comes from (1.0 s): T = P1 + Ab + I, pp, with 95 % scene-bootstrap intervals.')
L.append('')
L.append('::: {custom-style="TableNote"}')
L.append('T is the baseline under per-request admission (A1) minus LSRA under coordinated admission (A2); P1 is the baseline minus LSRA, both under A1; Ab is the baseline under A1 minus the same baseline under A2; I = T − P1 − Ab is the part that needs both. Set 5 is the best A3 setting under A2 at every load. The tuned A3 was tuned separately for A1 and A2, so its Ab includes the re-tuning. All baselines and delays are in the result files.')
L.append(':::')
L.append('')
L.append('')
# ------------------------------------------------------------------ Table 11: RW1 per world
W = R['worlds']
LAB = {'A1a': 'G10', 'A1b': 'G20', 'A2': 'DI', 'A3': 'MA'}
lab = lambda w: LAB.get(w, w)
order = ['V0', 'S1', 'S2', 'S3', 'B1', 'B2', 'B3', 'A1a', 'A1b', 'A2', 'A3', 'F1', 'H1', 'H2a', 'H2b', 'ALL']
assert list(W.keys()) == order
# ------------------------------------------------------------------ Table 10: the changed models and the verdicts
DESC = {
 'V0': ('nothing: the model of Sections 3–8', '—'),
 'S1': ('shadowing correlated along each vehicle\'s path, independently per station; the A3 rules measure it', 'correlation distance 3.8 m, measured at 142 GHz [Ju2021]'),
 'S2': ('as S1', '10 m on clear and 13 m on blocked links, the 3GPP street-canyon values [TR38901]'),
 'S3': ('as S1', '50 m, a stress case: the 3GPP urban-macro value without line of sight [TR38901], not our road'),
 'B1': ('knife-edge loss also for a roof just below the path (−0.78 < v ≤ 0); such grazing links also take the blocked-link shadowing spread', 'the full knife-edge curve [P526]'),
 'B2': ('a blocker counts over 1.3 times its length if taller than 2.5 m and 0.75 times if lower', 'measured at 28 GHz [Rodriguez2024]'),
 'B3': ('the losses of all crossing blockers summed in dB, capped at 60 dB', 'blockage model B [TR38901]'),
 'A1a': ('serving-beam gain 10 dBi instead of 15 dBi, known to every planner', 'assumption'),
 'A1b': ('serving-beam gain 20 dBi, known to every planner', 'assumption'),
 'A2': ('co-channel interference through the gain 15 − min(12(θ/30°)², 15) dBi, θ the angle between the two vehicles seen from the station, instead of 0 dBi; the planners still assume 0 dBi', 'parabolic form of the 3GPP pattern [TR38901]; width and depth assumed'),
 'A3': ('the serving beam is re-pointed every 0.5 s, at landing and when service resumes, with a loss of min(12(Δθ/30°)², 30) dB in between, Δθ the change of angle', 'assumption'),
 'F1': ('fading in every slot on every link: Rician with a Rice factor of 9 dB if clear, Rayleigh if blocked; the A3 rules\' filtered measurement does not see it', 'assumption'),
 'H1': ('the vehicle is unserved in the slot in which it lands', '0.1 s; measured interruptions 43–134 ms [Goh2025; Vicario2026]'),
 'H2a': ('a handover fails if the source link reaches radio-link failure while it is in flight, or if the SINR at the target is below −13.8 dB at landing; the vehicle is then unserved for 0.2 s and re-established at the strongest station with a place', '[TR36839; TS38300; Park2020]'),
 'H2b': ('as H2a, with −8 dB and 0.5 s', '[TR36839; Park2020]'),
 'ALL': ('S2, B2, DI and H2a together', '—'),
}
assert set(DESC) == set(order)
L.append('| Model | What changes | Value and source | RW1 | RW2 | RW3 |')
L.append('|------|------------------------------|--------------------|------|------|------|')
for w in order:
    v = W[w]
    held = {t: v[t]['held'] for t in ('RW1', 'RW2', 'RW3')}
    for t in ('RW1', 'RW2', 'RW3'):
        assert held[t] == (w in R['robustness'][t]), (w, t)
    put(f'T10_{w}', held | {'RW1_not_beaten': v['RW1']['not_beaten']})
    rw1 = 'held' if held['RW1'] else 'not held (' + ', '.join(v['RW1']['not_beaten']) + ')'
    wt, src = DESC[w]
    L.append(f"| {lab(w)} | {wt} | {src} | {rw1} | {'held' if held['RW2'] else 'not held'} | {'held' if held['RW3'] else 'not held'} |")
L.append('')
L.append(': **Table 10.** The changed models of the sensitivity check, each set against the model of Sections 3–8 (V0), and the registered tests that held in each (4 places, 1.0 s).')
L.append('')
L.append('::: {custom-style="TableNote"}')
L.append('RW1: LSRA with coordinated admission beats R1i, R2, R3, A3 Set 5, the tuned A3 and PH-O under the same admission (99.17 % intervals, Bonferroni over six). RW2: coordinated admission is worth something to LSRA (95 %). RW3: LSRA with coordinated admission beats R2 and the tuned A3 under per-request admission (97.5 %, Bonferroni over two). No planner was re-tuned; Section 7 states what the planners knew of each change. In the result files, G10, G20, DI and MA are named A1a, A1b, A2 and A3.')
L.append(':::')
L.append('')
L.append('')
L.append('| Model | LSRA J (%) | R1i | R2 | R3 | A3 Set 5 = A3-T | PH-O | RW1 |')
L.append('|------|------|--------------|--------------|--------------|--------------|--------------|------|')
for w in order:
    v = W[w]
    pb = v['RW1']['per_baseline']
    assert abs(pb['A3S5']['mean'] - pb['A3T']['mean']) < 1e-12
    row = [lab(w), lv(v['levels']['LSRA_A2']['worst'])]
    for b in ('R1i', 'R2', 'R3', 'A3S5', 'PHO'):
        put(f'T11_{w}_{b}', [pb[b]['mean'], pb[b]['ci_9917']])
        row.append(d(pb[b]['mean'], pb[b]['ci_9917']))
    put(f'T11_{w}_A3T', [pb['A3T']['mean'], pb['A3T']['ci_9917']])
    put(f'T11_{w}_held', v['RW1']['held'])
    put(f'T11_{w}_LSRA', v['levels']['LSRA_A2']['worst'])
    nb = v['RW1']['not_beaten']
    row.append('held' if v['RW1']['held'] else 'not held (' + ', '.join(nb) + ')')
    L.append('| ' + ' | '.join(row) + ' |')
L.append('')
L.append(': **Table 11.** RW1 in each model: the baseline minus LSRA, both under coordinated admission (4 places, 1.0 s), pp, with 99.17 % scene-bootstrap intervals (Bonferroni over six).')
L.append('')
L.append('::: {custom-style="TableNote"}')
L.append('Under A2 at 4 places the tuned A3 chose the Set 5 setting, so its episodes equal Set 5\'s in every model; the table shows them once, with Set 5\'s interval (the tuned A3\'s, from its own bootstrap, is in the result files and leads to the same verdicts). V0 is the model of Sections 3–8, computed from the stored episodes.')
L.append(':::')
L.append('')
L.append('')
# ------------------------------------------------------------------ Table 12: RW2 and RW3 per world
L.append('| Model | LSRA, A1 (%) | RW2: LSRA A1 − LSRA A2 (95 %) | R2, A1 (%) | RW3: R2 A1 − LSRA A2 (97.5 %) | A3-T, A1 (%) | RW3: A3-T A1 − LSRA A2 (97.5 %) |')
L.append('|------|------|--------------|------|--------------|------|--------------|')
for w in order:
    v = W[w]
    lvls = v['levels']
    r2 = v['RW2']; r3 = v['RW3']['per_baseline']
    put(f'T12_{w}', {'RW2': [r2['mean'], r2['ci_95'], r2['held']], 'R2': [r3['R2']['mean'], r3['R2']['ci_975']],
                     'A3T': [r3['A3T']['mean'], r3['A3T']['ci_975']], 'RW3_held': v['RW3']['held'],
                     'L_A1': lvls['LSRA_A1']['worst'], 'R2_A1': lvls['R2_A1']['worst'], 'A3T_A1': lvls['A3T_A1']['worst']})
    L.append(f"| {lab(w)} | {lv(lvls['LSRA_A1']['worst'])} | {d(r2['mean'], r2['ci_95'])} | {lv(lvls['R2_A1']['worst'])} | "
             f"{d(r3['R2']['mean'], r3['R2']['ci_975'])} | {lv(lvls['A3T_A1']['worst'])} | {d(r3['A3T']['mean'], r3['A3T']['ci_975'])} |")
L.append('')
L.append(': **Table 12.** RW2 and RW3 in each model (4 places, 1.0 s): levels in %, differences in pp with scene-bootstrap intervals. RW2 and RW3 held in all 16 models.')
L.append('')
L.append('')
# ------------------------------------------------------------------ Table 13: imperfect coordination
CO = R['coordination']
labels = [('C1a', 'Landings spread by up to ±0.5 s'), ('C1b', 'Late landings only, up to +0.5 s'),
          ('C2a', '5 % of admitted moves not run'), ('C2b', '20 % of admitted moves not run'),
          ('C3a', '2 % of admitted moves fail at landing'), ('C3b', '5 % fail at landing'), ('C3c', '10 % fail at landing'),
          ('C4a', 'Coordination 0.1 s slower'), ('C4b', 'Coordination 0.2 s slower'), ('C4c', 'Coordination 0.5 s slower')]
L.append('| Setting | What changes | LSRA J (%) | LSRA A1 − LSRA (95 %) | R2 A1 − LSRA (95 %) | Cost: LSRA − ideal LSRA A2 (95 %) |')
L.append('|------|------------------|------|--------------|--------------|--------------|')
for c, txt in labels:
    v = CO[c]
    put(f'T13_{c}', {'J': v['LSRA_worst'], 'L1': [v['vs_LSRA_A1']['mean'], v['vs_LSRA_A1']['ci_95']],
                     'R2': [v['vs_R2_A1']['mean'], v['vs_R2_A1']['ci_95']],
                     'ideal': [v['vs_LSRA_A2_ideal']['mean'], v['vs_LSRA_A2_ideal']['ci_95']]})
    L.append(f"| {c} | {txt} | {lv(v['LSRA_worst'])} | {d(v['vs_LSRA_A1']['mean'], v['vs_LSRA_A1']['ci_95'])} | "
             f"{d(v['vs_R2_A1']['mean'], v['vs_R2_A1']['ci_95'])} | {d(-v['vs_LSRA_A2_ideal']['mean'], [-v['vs_LSRA_A2_ideal']['ci_95'][1], -v['vs_LSRA_A2_ideal']['ci_95'][0]])} |")
L.append('')
L.append(': **Table 13.** Imperfect coordination: LSRA under coordinated admission with each imperfection (4 places, 1.0 s), against LSRA and R2 under per-request admission and against ideal coordination, in pp.')
L.append('')
L.append('::: {custom-style="TableNote"}')
be = R['break_even']
assert all(x is None for s in be.values() for x in s.values())
put('T13_break_even', be)
L.append('In the two middle columns, positive values favour LSRA with imperfect coordinated admission; the last column is positive when the imperfection costs LSRA (negative: LSRA did slightly better than with ideal coordination). References: LSRA under A1 at 8.76 %, R2 under A1 at 9.11 % and ideal LSRA under A2 at 3.59 %. Break-even (the first setting of a sequence whose interval does not lie above 0): none in C1, C2, C3 or C4, against either reference.')
L.append(':::')
L.append('')
# ------------------------------------------------------------------ extra values used in the text
put('A3O', B['A3O_bound'])
put('CROSSFIT', X)
put('realism_last_boot_seed', R['last_boot_seed'])
put('robustness', R['robustness'])
put('guesses', R['guesses'])
put('base_last_boot_seed', B['last_boot_seed'])
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'TABLES_V6.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
json.dump(VALS, open(os.path.join(OUT, 'TABLE_VALUES_V6.json'), 'w'), indent=1, default=str)
print('tables written:', len(L), 'lines;', len(VALS), 'values')
