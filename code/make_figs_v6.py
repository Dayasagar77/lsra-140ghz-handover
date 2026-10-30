#!/usr/bin/env python3
"""New figures of the LSRA manuscript v6 (30 Oct 2026), in the style of F1-F7 (Arial 8 pt, 600 dpi PNG and PDF).

F4_reserve_and_commit_protocol   schematic message chart of the coordination protocol (Section 5.1); no data
F9_imperfect_coordination        LSRA with imperfect coordinated admission (Section 8.10, Table 13)

F9 reads one frozen result file only, the realism run's analysis as re-run once on 30 Oct 2026:
  realism_deviation/reanalysis/REAL_RESULT.json   md5 bef337dafcd5135ba6ce69a120461cd9
Every number drawn is written to FIG_DATA_v6.csv and checked against the file (FIG_CHECK_v6.txt).
Usage: python make_figs_v6.py <path to REAL_RESULT.json> <output folder>
"""
import os, sys, json, csv, hashlib, platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

SRC, OUT = sys.argv[1], sys.argv[2]
MD5 = 'bef337dafcd5135ba6ce69a120461cd9'
os.makedirs(OUT, exist_ok=True)

INK, INK2, AXIS, GRID = '#0b0b0b', '#52514e', '#9a9993', '#e7e6e2'
BLUE, GREEN = '#2a78d6', '#1baf7a'          # LSRA and R2, as in F4-F7
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white'})
DATA = []


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=600)
    plt.close(fig)


# ------------------------------------------------------------------ F4: the reserve-and-commit protocol
def f4():
    W, H = 5.7, 4.1
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    X = {'co': 1.05, 'X': 2.40, 'Y': 3.50, 'a': 4.42, 'b': 5.24}
    head = {'co': 'Coordinator\n(near-RT RIC xApp\nor gNB-CU)', 'X': 'Station X', 'Y': 'Station Y',
            'a': 'Vehicle a\nX → Y', 'b': 'Vehicle b\nY → X'}
    ytop, ybot = 3.70, 0.50
    for k, x in X.items():
        hw = 0.55 if k == 'co' else 0.37
        ax.add_patch(FancyBboxPatch((x - hw, ytop), 2 * hw, 0.34, boxstyle='round,pad=0.02,rounding_size=0.05',
                                    fc='white', ec=AXIS, lw=0.6))
        ax.text(x, ytop + 0.17, head[k], ha='center', va='center', fontsize=6.8 if k == 'co' else 7.2, color=INK)
        ax.plot([x, x], [ybot, ytop], color=AXIS, lw=0.6, zorder=0)

    def msg(y, a, b, label, color=BLUE, ls='-', lab_side='above', fs=7, lab_x=None):
        x0, x1 = X[a], X[b]
        ax.annotate('', xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=0.9, ls=ls, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))
        if label:
            xm = (x0 + x1) / 2 if lab_x is None else lab_x
            ax.text(xm, y + (0.035 if lab_side == 'above' else -0.035), label, ha='center',
                    va='bottom' if lab_side == 'above' else 'top', fontsize=fs, color=color)

    # decision on the landing state
    ax.add_patch(FancyBboxPatch((X['co'] - 0.70, 3.32), 1.40, 0.28, boxstyle='round,pad=0.02,rounding_size=0.04',
                                fc='#eaf2fb', ec=BLUE, lw=0.7))
    ax.text(X['co'], 3.46, 'Set admitted on the landing state\nS = {a: X → Y, b: Y → X}', ha='center', va='center',
            fontsize=6.6, color=INK)
    # phase 1: reserve
    msg(3.02, 'co', 'X', 'RESERVE (in b, out a, lease)', lab_x=1.72)
    msg(2.85, 'co', 'Y', 'RESERVE (in a, out b, lease)', lab_x=2.30)
    msg(2.66, 'X', 'co', 'RESERVED', lab_x=1.72)
    msg(2.50, 'Y', 'co', 'RESERVED', lab_x=2.30)
    # phase 2: commit
    msg(2.26, 'co', 'X', 'COMMIT', lab_x=1.72)
    msg(2.12, 'co', 'Y', 'COMMIT', lab_x=2.30, lab_side='below')
    # unchanged per-UE procedures
    msg(1.82, 'X', 'Y', 'preparation of a (Xn)', color=INK2, ls=(0, (3, 2)), fs=6.6)
    msg(1.64, 'Y', 'X', 'preparation of b (Xn)', color=INK2, ls=(0, (3, 2)), fs=6.6, lab_side='below')
    msg(1.38, 'X', 'a', 'command', color=INK2, ls=(0, (3, 2)), fs=6.6, lab_x=3.96)
    msg(1.24, 'Y', 'b', 'command', color=INK2, ls=(0, (3, 2)), fs=6.6, lab_x=4.83, lab_side='below')
    # landings at different times
    msg(1.02, 'a', 'Y', 'a lands', color=INK, fs=6.8)
    ax.add_patch(FancyBboxPatch((X['Y'] - 0.035, 0.74), 0.07, 0.28, boxstyle='square,pad=0', fc=INK2, ec='none'))
    ax.text(X['Y'] - 0.08, 0.885, 'C + 1 until b leaves:\nserves its C strongest', ha='right', va='center',
            fontsize=6.2, color=INK2)
    msg(0.74, 'b', 'X', 'b lands, later', color=INK, fs=6.8, lab_x=4.83)
    # completion
    msg(0.60, 'X', 'co', 'COMPLETE (each station)', lab_x=1.72, lab_side='below')
    # phase brackets on the left
    for y0, y1, t in ((3.10, 2.45, 'Phase 1\nreserve'), (2.31, 2.07, 'Phase 2\ncommit'),
                      (1.88, 1.19, 'Today\'s\nper-UE\nhandover'), (1.08, 0.50, 'Landing\nand release')):
        ax.plot([0.12, 0.12], [y1, y0], color=INK2, lw=0.6)
        ax.plot([0.12, 0.17], [y0, y0], color=INK2, lw=0.6)
        ax.plot([0.12, 0.17], [y1, y1], color=INK2, lw=0.6)
        ax.text(0.20, (y0 + y1) / 2, t, ha='left', va='center', fontsize=6.4, color=INK2)
    # legend
    hs = [Line2D([], [], color=BLUE, lw=0.9), Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2))),
          Line2D([], [], color=INK, lw=0.9)]
    fig.legend(hs, ['New: reserve and commit, one transaction', 'Existing per-UE procedure', 'Landing'],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.2,
               fontsize=6.8, columnspacing=1.2)
    save(fig, 'F4_reserve_and_commit_protocol')


# ------------------------------------------------------------------ F9: imperfect coordination
def f9(R):
    CO = R['coordination']
    V0 = R['worlds']['V0']
    ideal_l1 = V0['RW2']['mean']                          # LSRA A1 minus ideal LSRA A2
    ideal_r2 = V0['RW3']['per_baseline']['R2']['mean']    # R2 A1 minus ideal LSRA A2
    groups = [('Landings spread', [('C1a', '±0.5 s'), ('C1b', 'late only, ≤ +0.5 s')]),
              ('Moves not run', [('C2a', '5 %'), ('C2b', '20 %')]),
              ('Moves failing at landing', [('C3a', '2 %'), ('C3b', '5 %'), ('C3c', '10 %')]),
              ('Slower coordination', [('C4a', '+0.1 s'), ('C4b', '+0.2 s'), ('C4c', '+0.5 s')])]
    ys, labels, y, gy = {}, [], 0.0, []
    for g, rows in groups:
        gy.append((g, y - 0.75))
        for c, t in rows:
            ys[c] = y; labels.append(f'{c}  {t}'); y += 1.0
        y += 1.1
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.5), sharey=True, gridspec_kw={'width_ratios': [1.55, 1.0]})
    ax = axs[0]
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False); axs[1].spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, lw=0.6); axs[1].xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True); axs[1].set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(ideal_l1, color=BLUE, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.axvline(ideal_r2, color=GREEN, lw=0.7, ls=(0, (2, 2)), zorder=1)
    for c in ys:
        for key, col, mk, dy in (('vs_LSRA_A1', BLUE, 'o', -0.17), ('vs_R2_A1', GREEN, '^', 0.17)):
            v = CO[c][key]
            m, (lo, hi) = v['mean'], v['ci_95']
            ax.hlines(ys[c] + dy, lo, hi, color=col, lw=1.3, zorder=2, capstyle='round')
            ax.plot([m], [ys[c] + dy], marker=mk, ms=4.6, mec=col, mew=1.1, mfc=col if lo > 0 else 'white',
                    ls='none', zorder=3)
            DATA.append(['F9', c, key, m, lo, hi, '95 %'])
    ax.set_title('(a)  Lead over per-request admission, pp', loc='left', color=INK, pad=4)
    ax.set_xlim(-1.0, 11.0)
    a2 = axs[1]
    a2.axvline(0, color=INK2, lw=0.8, zorder=1)
    for c in ys:
        v = CO[c]['vs_LSRA_A2_ideal']
        m, lo, hi = -v['mean'], -v['ci_95'][1], -v['ci_95'][0]     # cost = imperfect minus ideal
        a2.hlines(ys[c], lo, hi, color=BLUE, lw=1.3, zorder=2, capstyle='round')
        a2.plot([m], [ys[c]], marker='s', ms=4.2, mec=BLUE, mew=1.1, mfc=BLUE, ls='none', zorder=3)
        DATA.append(['F9', c, 'cost_vs_ideal', m, lo, hi, '95 %'])
    a2.set_title('(b)  Cost vs ideal, pp', loc='left', color=INK, pad=4)
    a2.set_xlim(-0.4, 2.6)
    ax.set_yticks([ys[c] for c in ys]); ax.set_yticklabels(labels)
    ax.set_ylim(max(ys.values()) + 0.7, -1.6)
    for g, yy in gy:
        ax.annotate(g, xy=(-0.02, yy), xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=7,
                    color=INK, fontweight='bold', annotation_clip=False)
    hs = [Line2D([], [], color=BLUE, lw=1.3, marker='o', ms=4.6, mfc=BLUE, mec=BLUE),
          Line2D([], [], color=GREEN, lw=1.3, marker='^', ms=4.6, mfc=GREEN, mec=GREEN),
          Line2D([], [], color=INK2, lw=0.7, ls=(0, (2, 2)))]   # the dashed lines are drawn in each colour
    fig.legend(hs, ['Against LSRA under per-request admission', 'Against R2 under per-request admission',
                    'Lead with ideal coordination'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.60, 1.0), fontsize=6.8,
               handlelength=2.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.30, right=0.975, bottom=0.07, top=0.835, wspace=0.08)
    save(fig, 'F9_imperfect_coordination')
    return ideal_l1, ideal_r2


def main():
    got = hashlib.md5(open(SRC, 'rb').read()).hexdigest()
    assert got == MD5, f'{SRC}: md5 {got} differs from {MD5}'
    R = json.load(open(SRC))
    assert R['ok'] is True and not R['problems']
    f4()
    il1, ir2 = f9(R)
    # checks: every drawn value equals the file, every interval of (a) lies above 0 (no break-even)
    CO = R['coordination']
    lines = [f'source {SRC} md5 {got}', f'python {platform.python_version()} matplotlib {matplotlib.__version__}',
             f'font used: {matplotlib.font_manager.findfont(plt.rcParams["font.sans-serif"][0])}',
             f'ideal leads drawn: {il1:.6f} (LSRA A1) and {ir2:.6f} (R2 A1)']
    n = 0
    for fig, c, key, m, lo, hi, iv in DATA:
        if key == 'cost_vs_ideal':
            v = CO[c]['vs_LSRA_A2_ideal']
            assert m == -v['mean'] and lo == -v['ci_95'][1] and hi == -v['ci_95'][0]
        else:
            v = CO[c][key]
            assert m == v['mean'] and (lo, hi) == tuple(v['ci_95']) and lo > 0, (c, key)
        n += 1
    be = R['break_even']
    assert all(x is None for s in be.values() for x in s.values())
    lines.append(f'{n} drawn values equal the file; every interval of panel (a) lies above 0; break-even: none')
    with open(os.path.join(OUT, 'FIG_DATA_v6.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['figure', 'setting', 'quantity', 'value_pp', 'lo', 'hi', 'interval']); w.writerows(DATA)
    open(os.path.join(OUT, 'FIG_CHECK_v6.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
