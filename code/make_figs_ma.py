#!/usr/bin/env python3
"""Result figures F4-F7 of the multi-car paper (20 Oct 2026).
v2 (22 Oct 2026): "car" -> "vehicle" in every label drawn, PNG at 600 dpi. Data keys in FIG_DATA.csv are unchanged.
v3 (22 Oct 2026): adds the third load, 6 places per station (LOAD6_RESULT.json), to every figure: a third row in
F4 and F7, a third panel in F5 and F6. Shorter panel titles in F5 and F6, and one x-axis label per figure there.
Every row FIG_DATA.csv held in v2 is written again unchanged; the new rows are the 6-place ones.
v4 (22 Oct 2026): F7 only: the A1 triangle is drawn above the A2 square, so that a triangle hidden by a nearly
coinciding square shows; an A1 label moves to the left of its mark when the A0 ring lies just to its right
(R3 at 6 places). No number changes.

Reads three finished result files only. No simulation and no new statistics: every number drawn is
read from the files, and every number drawn is written to FIG_DATA.csv.
  models/admission_check/ADMIT_RESULT.json   4 places per station (16 cars, 16 places)
  models/load_check/LOAD_RESULT.json         5 places per station (16 cars, 20 places)
  models/load_check_c6/LOAD6_RESULT.json     6 places per station (16 cars, 24 places)
Every md5 is checked against the RESULT documents before anything is drawn.

F4  worst-car RLF against handover delay: three admission models x three loads (95 % scene intervals)
F5  baseline minus LSRA at 1.0 s: the six registered families (98.75 % scene intervals)
F6  causes of the worst car's failures at 1.0 s (stacked), per admission model and load
F7  handovers per car-minute against worst-car RLF at 1.0 s, one panel per arm and load
Outputs (figures_ma/): F4..F7 as .pdf and .png (600 dpi), FIG_DATA.csv, FIG_CHECK.txt.
"""
import os, sys, json, csv, hashlib, platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_ma')
SRC = {4: ('models/admission_check/ADMIT_RESULT.json', '803e1017a39ef1dcd4a6e9ea10ade9ad'),
       5: ('models/load_check/LOAD_RESULT.json', '0fb5845ca6b234261d7ca78363c5e730'),
       6: ('models/load_check_c6/LOAD6_RESULT.json', '8e1860b938f69560610da9619e7bbc00')}
LOADS = (4, 5, 6)
FAM = {(4, 'A1'): 'HR1_A1', (4, 'A2'): 'HR2_A2', (5, 'A1'): 'HL1_A1', (5, 'A2'): 'HL2_A2',
       (6, 'A1'): 'HL3_A1', (6, 'A2'): 'HL4_A2'}
HLS = [2, 5, 10, 15]
MODELS = ['A0', 'A1', 'A2']
MODEL_TITLE = {'A0': 'A0  admission at landing only', 'A1': 'A1  per request (today)',
               'A2': 'A2  coordinated (proposed)'}
LOAD_TITLE = {4: '4 places per station (full load)', 5: '5 places per station (load 0.8)',
              6: '6 places per station (load 0.67)'}
LOAD_SHORT = {4: '4 places (full load)', 5: '5 places (load 0.8)', 6: '6 places (load 0.67)'}
ARMS = ['METHOD', 'R3', 'R2', 'R1i']                       # fixed order: legend, bars, panels
NAME = {'METHOD': 'LSRA', 'R3': 'R3', 'R2': 'R2', 'R1i': 'R1i', 'R1': 'R1'}
LEGEND = {'METHOD': 'LSRA (proposed)', 'R3': 'R3 joint assignment', 'R2': 'R2 restricted assignment',
          'R1i': 'R1i capacity rule'}
# categorical slots 1-4 of the validated palette (arms); slots 5-8 for the causes (F6)
COLOR = {'METHOD': '#2a78d6', 'R3': '#eb6834', 'R2': '#1baf7a', 'R1i': '#eda100'}
MARK = {'METHOD': 'o', 'R3': 's', 'R2': '^', 'R1i': 'D'}
DASH = {'METHOD': '-', 'R3': (0, (5, 2)), 'R2': (0, (1, 1.4)), 'R1i': (0, (5, 1.6, 1.2, 1.6))}
CAUSE_ORDER = ['B', 'W', 'L', 'AF']
CAUSE_COLOR = {'B': '#e87ba4', 'W': '#008300', 'L': '#6250d6', 'AF': '#e34948'}
CAUSE_LABEL = {'B': 'Outage, no handover in flight (B)', 'W': 'Waiting, handover in flight (W)',
               'L': 'No place after landing (L)', 'AF': 'No place, other (A, F)'}
MODEL_MARK = {'A0': 'o', 'A1': '^', 'A2': 's'}
INK, INK2, AXIS, GRID = '#0b0b0b', '#52514e', '#9a9993', '#e7e6e2'
STEP_S = 0.1                                                 # one step = 0.1 s

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white'})

DATA = []          # every number drawn: figure, places, model, delay_s, arm, quantity, value, lo, hi, interval


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def load():
    R = {}
    for c, (rel, m) in SRC.items():
        p = os.path.join(HERE, rel)
        got = md5(p)
        assert got == m, f'{rel}: md5 {got} differs from the recorded {m}'
        R[c] = json.load(open(p))
    return R


R = load()
T = {c: R[c]['secondary']['table'] for c in R}


def e(c, A, hl, arm):
    return T[c][f'{A}_hl{hl}_{arm}']


def note(fig, c, A, hl, arm, q, v, lo=None, hi=None, iv=''):
    DATA.append([fig, c, A, '' if hl is None else hl * STEP_S, NAME.get(arm, arm), q, v,
                 '' if lo is None else lo, '' if hi is None else hi, iv])


def style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid == 'y':
        ax.yaxis.grid(True, color=GRID, lw=0.6)
    elif grid == 'x':
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def letter(ax, k, x=0.02, y=0.98):
    ax.text(x, y, f'({"abcdefghijkl"[k]})', transform=ax.transAxes, ha='left', va='top', fontsize=8,
            fontweight='bold', color=INK)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ F4: RLF against delay
def f4():
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.5), sharex=True, sharey=True)
    off = {'METHOD': -0.036, 'R3': -0.012, 'R2': 0.012, 'R1i': 0.036}
    top = max(e(c, A, h, a)['scene_95'][1] for c in LOADS for A in MODELS for h in HLS for a in ARMS)
    for r, c in enumerate(LOADS):
        for k, A in enumerate(MODELS):
            ax = axs[r, k]
            style(ax)
            for a in ARMS:
                x = np.array([h * STEP_S for h in HLS]) + off[a]
                y = np.array([e(c, A, h, a)['worst'] for h in HLS])
                lo = np.array([e(c, A, h, a)['scene_95'][0] for h in HLS])
                hi = np.array([e(c, A, h, a)['scene_95'][1] for h in HLS])
                ax.vlines(x, lo, hi, color=COLOR[a], lw=0.7, alpha=0.45, zorder=2)
                ax.plot(x, y, color=COLOR[a], lw=1.4, ls=DASH[a], marker=MARK[a], ms=4.6, mfc=COLOR[a],
                        mec='white', mew=0.7, zorder=4 if a == 'METHOD' else 3, solid_capstyle='round',
                        dash_capstyle='round')
                for h, v, l_, u in zip(HLS, y, lo, hi):
                    note('F4', c, A, h, a, 'worst_car_RLF_pct', v, l_, u, 'scene 95%')
            letter(ax, 3 * r + k)
            if r == 0:
                ax.set_title(MODEL_TITLE[A], loc='left', color=INK, pad=4)
            if r == len(LOADS) - 1:
                ax.set_xlabel('Handover delay (s)')
            if k == 0:
                ax.set_ylabel(f'{LOAD_TITLE[c]}\nWorst-vehicle RLF (%)')
    axs[0, 0].set_xticks([0.2, 0.5, 1.0, 1.5])
    axs[0, 0].set_xlim(0.08, 1.62)
    axs[0, 0].set_ylim(0, 5 * np.ceil(top / 5))
    axs[0, 0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    hs = [Line2D([], [], color=COLOR[a], lw=1.4, ls=DASH[a], marker=MARK[a], ms=4.6, mfc=COLOR[a],
                 mec='white', mew=0.7) for a in ARMS]
    fig.legend(hs, [LEGEND[a] for a in ARMS], loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=3.0, columnspacing=1.6)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.07, top=0.915, wspace=0.08, hspace=0.12)
    save(fig, 'F4_RLF_vs_delay_three_admission_models')


# ------------------------------------------------------------------ F5: baseline minus LSRA at 1.0 s
def f5():
    order = ['R1', 'R1i', 'R2', 'R3']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.0), sharex=True, sharey=True)
    ys, labels, y = {}, [], 0
    for A in ('A1', 'A2'):
        for b in order:
            ys[(A, b)] = y
            labels.append(NAME[b])
            y += 1
        y += 1.6
    ylast = max(ys.values())
    vals = [R[c][FAM[(c, A)]]['per_baseline'][b] for c in LOADS for A in ('A1', 'A2') for b in order]
    lo_all = min(v['scene_9875'][0] for v in vals)
    hi_all = max(v['scene_9875'][1] for v in vals)
    for k, c in enumerate(LOADS):
        ax = axs[k]
        style(ax, grid='x')
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for A in ('A1', 'A2'):
            fam = R[c][FAM[(c, A)]]
            for b in order:
                v = fam['per_baseline'][b]
                yy = ys[(A, b)]
                ax.hlines(yy, v['scene_9875'][0], v['scene_9875'][1], color=COLOR['METHOD'], lw=1.4, zorder=2,
                          capstyle='round')
                ax.plot([v['mean']], [yy], marker='o', ms=5.2, mec=COLOR['METHOD'], mew=1.2,
                        mfc=COLOR['METHOD'] if v['beats'] else 'white', zorder=3, ls='none')
                note('F5', c, A, 10, b, 'baseline_minus_LSRA_pp', v['mean'], v['scene_9875'][0],
                     v['scene_9875'][1], 'scene 98.75%')
            ax.annotate(('held' if fam['held'] else 'not held'),
                        xy=(1.0, ys[(A, order[0])] - 0.8), xycoords=('axes fraction', 'data'), ha='right',
                        va='center', fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_title(f'({"abc"[k]})  {LOAD_SHORT[c]}', loc='left', color=INK, pad=4)
    fig.supxlabel('Baseline minus LSRA, worst-vehicle RLF at 1.0 s (pp)', fontsize=8, x=0.58, y=0.015)
    axs[0].set_yticks([ys[(A, b)] for A in ('A1', 'A2') for b in order])
    axs[0].set_yticklabels(labels)
    axs[0].set_ylim(ylast + 0.6, -1.4)
    for A in ('A1', 'A2'):
        yb = (ys[(A, order[0])] + ys[(A, order[-1])]) / 2
        axs[0].annotate({'A1': 'A1\nper request', 'A2': 'A2\ncoordinated'}[A], xy=(-0.185, yb),
                        xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=7.5, color=INK,
                        annotation_clip=False)
    axs[0].set_xlim(np.floor(lo_all - 0.6), np.ceil(hi_all + 0.6))
    hs = [Line2D([], [], color=COLOR['METHOD'], lw=1.4, marker='o', ms=5.2, mfc=COLOR['METHOD'],
                 mec=COLOR['METHOD'], mew=1.2),
          Line2D([], [], color=COLOR['METHOD'], lw=1.4, marker='o', ms=5.2, mfc='white', mec=COLOR['METHOD'],
                 mew=1.2)]
    fig.legend(hs, ['LSRA better (98.75 % interval above 0)', 'No clear difference (98.75 % interval includes 0)'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.56, 1.0), handlelength=2.6)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.16, top=0.82, wspace=0.10)
    save(fig, 'F5_baseline_minus_LSRA_registered_families_1s')


# ------------------------------------------------------------------ F6: causes at 1.0 s
def causes(c, A, a):
    cw = e(c, A, 10, a)['causes_worst']
    return {'B': cw['B'], 'W': cw['W'], 'L': cw['L'], 'AF': cw['A'] + cw['F']}


def f6():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.6), sharey=True)
    ys, labels, y = {}, [], 0
    for A in MODELS:
        for a in ARMS:
            ys[(A, a)] = y
            labels.append(NAME[a])
            y += 1
        y += 0.9
    for k, c in enumerate(LOADS):
        ax = axs[k]
        style(ax, grid='x')
        top = 0.0
        for A in MODELS:
            for a in ARMS:
                cs = causes(c, A, a)
                left = 0.0
                for q in CAUSE_ORDER:
                    ax.barh(ys[(A, a)], cs[q], left=left, height=0.72, color=CAUSE_COLOR[q], edgecolor='white',
                            linewidth=0.8, zorder=2)
                    note('F6', c, A, 10, a, f'cause_{q}_pct', cs[q])
                    left += cs[q]
                w = e(c, A, 10, a)['worst']
                ax.text(left + 0.15, ys[(A, a)], f'{w:.1f}', va='center', ha='left', fontsize=7, color=INK2)
                note('F6', c, A, 10, a, 'worst_car_RLF_pct', w)
                top = max(top, left)
        ax.set_xlim(0, np.ceil(top + 1.2))
        ax.set_title(f'({"abc"[k]})  {LOAD_SHORT[c]}', loc='left', color=INK, pad=4)
    fig.supxlabel('Worst-vehicle RLF at 1.0 s (%), split by cause', fontsize=8, x=0.57, y=0.015)
    axs[0].set_yticks([ys[(A, a)] for A in MODELS for a in ARMS])
    axs[0].set_yticklabels(labels)
    axs[0].invert_yaxis()
    for A in MODELS:
        yb = (ys[(A, ARMS[0])] + ys[(A, ARMS[-1])]) / 2
        axs[0].annotate({'A0': 'A0\nat landing', 'A1': 'A1\nper request', 'A2': 'A2\ncoordinated'}[A],
                        xy=(-0.185, yb), xycoords=('axes fraction', 'data'), ha='right', va='center',
                        fontsize=7.5, color=INK, annotation_clip=False)
    hs = [Patch(facecolor=CAUSE_COLOR[q], edgecolor='white') for q in CAUSE_ORDER]
    fig.legend(hs, [CAUSE_LABEL[q] for q in CAUSE_ORDER], loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.0), handlelength=1.4, columnspacing=1.8)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.14, top=0.83, wspace=0.10)
    save(fig, 'F6_causes_of_worst_car_failures_1s')


# ------------------------------------------------------------------ F7: handovers against RLF at 1.0 s
def f7():
    fig, axs = plt.subplots(3, 4, figsize=(7.0, 5.9), sharex=True, sharey=True)
    pts = {}
    for c in LOADS:
        for a in ARMS:
            for A in MODELS:
                v = e(c, A, 10, a)
                pts[(c, a, A)] = (v['ho_per_car_step'] / STEP_S * 60.0, v['worst'], v.get('refused_share'))
    xmax = max(p[0] for p in pts.values()); ymax = max(p[1] for p in pts.values())
    xlim = np.ceil(xmax * 1.25)
    for r, c in enumerate(LOADS):
        for k, a in enumerate(ARMS):
            ax = axs[r, k]
            style(ax, grid='y')
            ax.xaxis.grid(True, color=GRID, lw=0.6)
            for A in MODELS:
                x, y, rs = pts[(c, a, A)]
                if A == 'A0':
                    ax.plot([x], [y], marker='o', ms=10.5, ls='none', mfc='white', mec=COLOR[a], mew=1.3, zorder=3)
                else:
                    ax.plot([x], [y], marker=MODEL_MARK[A], ms=6.2, ls='none', mfc=COLOR[a], mec='white',
                            mew=0.6, zorder=5 if A == 'A1' else 4)
                note('F7', c, A, 10, a, 'handovers_per_car_minute', x)
                note('F7', c, A, 10, a, 'worst_car_RLF_pct', y)
                if A != 'A0' and rs is not None:
                    note('F7', c, A, 10, a, 'refused_share_pct', 100 * rs)
                    x1, y1, r1 = pts[(c, a, 'A1')]
                    both = (A == 'A2' and abs(x - x1) < 1e-12 and abs(y - y1) < 1e-12 and r1 is not None
                            and round(100 * r1) == round(100 * rs))
                    if rs >= 0.01 and not both:
                        right = x < 0.55 * xlim
                        x0, y0, _ = pts[(c, a, 'A0')]
                        if A == 'A1' and x0 > x and abs(y0 - y) < 2.0 and x > 0.45 * xlim:
                            right = False                    # keep the label off the A0 ring just to the right
                        up = (A == 'A1')
                        x2, y2, r2 = pts[(c, a, 'A2')]
                        same = (A == 'A1' and abs(x - x2) < 1e-12 and abs(y - y2) < 1e-12 and r2 is not None
                                and round(100 * r2) == round(100 * rs))
                        ax.annotate(f'{100 * rs:.0f} % refused' + (' (A1 and A2)' if same else ''), xy=(x, y),
                                    xytext=((5 if right else -5), (3 if up else -3)), textcoords='offset points',
                                    fontsize=6.5, color=INK2, ha='left' if right else 'right',
                                    va='bottom' if up else 'top')
            ax.set_title(f'({"abcdefghijkl"[4 * r + k]})  {LEGEND[a]}', loc='left', color=INK, pad=3, fontsize=7.5)
            if r == len(LOADS) - 1:
                ax.set_xlabel('Handovers per vehicle-minute')
            if k == 0:
                ax.set_ylabel(f'{c} places per station\nWorst-vehicle RLF (%)')
    axs[0, 0].set_xlim(0, xlim)
    axs[0, 0].set_ylim(0, 4 * np.ceil(ymax * 1.12 / 4))
    axs[0, 0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    gray = INK2
    hs = [Line2D([], [], marker='o', ms=10.5, ls='none', mec=gray, mfc='white', mew=1.3),
          Line2D([], [], marker=MODEL_MARK['A1'], ms=6.2, ls='none', mec='white', mfc=gray, mew=0.6),
          Line2D([], [], marker=MODEL_MARK['A2'], ms=6.2, ls='none', mec='white', mfc=gray, mew=0.6)]
    fig.legend(hs, ['A0 admission at landing only', 'A1 per request (today)', 'A2 coordinated (proposed)'],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.8)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.91, wspace=0.08, hspace=0.30)
    save(fig, 'F7_handovers_vs_RLF_1s')


# ------------------------------------------------------------------ checks
def checks():
    L = []
    ok = True
    for c in LOADS:
        for A in MODELS:
            for a in ARMS + ['R1']:
                for h in HLS:
                    v = e(c, A, h, a)
                    s = sum(v['causes_worst'].values())
                    if abs(s - v['worst']) > 1e-6:
                        ok = False
                        L.append(f'FAIL causes do not add up: {c} places {A} hl{h} {a}: {s} vs {v["worst"]}')
    L.append(f'causes add up to the worst-vehicle RLF in every entry drawn: {ok}')
    same = all(e(c, 'A0', h, a)['worst'] == e(c, 'A2', h, a)['worst'] for c in LOADS for h in HLS
               for a in ('METHOD', 'R1i'))
    L.append(f'LSRA and R1i identical under A0 and A2 (all three loads, all delays): {same}')
    ok = ok and same
    for c in LOADS:
        L.append(f'{c} places, 1.0 s, R1 against R1i (R1 is not drawn in F4, F6, F7):')
        for A in MODELS:
            L.append(f'  {A}: R1 {e(c, A, 10, "R1")["worst"]:.2f}  R1i {e(c, A, 10, "R1i")["worst"]:.2f}')
    exp = {(4, 'A1'): False, (4, 'A2'): True, (5, 'A1'): False, (5, 'A2'): True, (6, 'A1'): False, (6, 'A2'): False}
    for key, h in exp.items():
        got = R[key[0]][FAM[key]]['held']
        L.append(f'{FAM[key]} ({key[0]} places) held = {got} (RESULT document: {h})')
        ok = ok and (got == h)
    spot = {(4, 'A1'): 8.76, (4, 'A2'): 3.59, (5, 'A1'): 3.97, (5, 'A2'): 2.61, (5, 'A0'): 2.61,
            (6, 'A0'): 2.32, (6, 'A1'): 2.32, (6, 'A2'): 2.32}
    for (c, A), v in spot.items():
        got = round(e(c, A, 10, 'METHOD')['worst'], 2)
        L.append(f'LSRA {c} places {A} 1.0 s = {got:.2f} (RESULT document: {v:.2f})')
        ok = ok and abs(got - v) < 0.005
    L.append(f'ALL CHECKS PASS: {ok}')
    return ok, L


def main():
    os.makedirs(OUT, exist_ok=True)
    ok, L = checks()
    f4(); f5(); f6(); f7()
    with open(os.path.join(OUT, 'FIG_DATA.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['figure', 'places', 'model', 'delay_s', 'arm', 'quantity', 'value', 'lo', 'hi', 'interval'])
        w.writerows(DATA)
    head = [f'make_figs_ma.py md5 {md5(os.path.abspath(__file__))}',
            f'python {platform.python_version()}, numpy {np.__version__}, matplotlib {matplotlib.__version__}',
            f'font used: {matplotlib.font_manager.findfont(plt.rcParams["font.sans-serif"][0])}',
            f'sources: ' + '; '.join(f'{SRC[c][0]} md5 {SRC[c][1]}' for c in SRC),
            f'numbers written to FIG_DATA.csv: {len(DATA)}', '']
    with open(os.path.join(OUT, 'FIG_CHECK.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(head + L) + '\n')
    print('\n'.join(head + L))
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
