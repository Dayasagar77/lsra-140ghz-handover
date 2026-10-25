#!/usr/bin/env python3
"""Figures F1-F3 of the multi-car paper (20 Oct 2026). No simulation and no result numbers.
v8 (22 Oct 2026): "car" -> "vehicle" in every label drawn, PNG at 600 dpi, F1 prints the loss to 0.1 dB. F1_EXAMPLE.json keys are unchanged.
v9 (22 Oct 2026): F1 panel (b) is titled "40 m x 40 m window", as in the caption. Nothing else changed.

F1  one DAIR crossing at one instant: the four roadside stations, every vehicle as a box (top view), and
    one blocked link with its height profile (side view). The example is chosen by a fixed rule, not by
    eye: scenes in the order of the frozen held-out seed list, frames in time order, the 16 simulated
    cars in the environment's order; the first car whose link to its NEAREST station is blocked by a
    vehicle at least 2.5 m tall with a knife-edge loss of at least 15 dB, in a frame with at least 10
    vehicles. Geometry and loss come from the environment's own functions (blockage3d.py).
F2  timeline sketch: decide, in flight for L slots, land; two decisions that land together (Theorem 1)
    against planning on the landing state.
F3  closed-loop sketch at full load: a swap refused request by request (today) and admitted as a set
    (coordinated admission).
Outputs (figures_ma/): F1..F3 as .pdf and .png (600 dpi) and F1_EXAMPLE.json (the example's numbers).
"""
import os, sys, json, math, hashlib, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import blockage3d as B
import method_heldout as MHO
import admission_check as AC

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_ma')
INK, INK2, AXIS, GRID = '#0b0b0b', '#52514e', '#9a9993', '#e7e6e2'
BLUE, ORANGE = '#2a78d6', '#eb6834'
GOOD, CRIT = '#0ca30c', '#d03b3b'            # status colours, always with a mark and a label
LIGHT, MID = '#d9d8d3', '#7d7c77'
MIN_VEH, MIN_H, MIN_DB = 10, 2.5, 15.0

plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=600)
    plt.close(fig)


def corners(x, y, L, W, th):
    c, s = math.cos(th), math.sin(th)
    pts = [(+L / 2, +W / 2), (-L / 2, +W / 2), (-L / 2, -W / 2), (+L / 2, -W / 2)]
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in pts]


# ------------------------------------------------------------------ F1: pick the example by rule
def pick_example():
    assert md5(MHO.POOL_H) == MHO.POOL_H_MD5, 'held-out pool changed'
    pool = pd.read_csv(MHO.POOL_H)
    env = AC.make_env3(pool, 10, 'none')
    seeds = pd.read_csv(os.path.join(MHO.OUT, 'SEEDS_heldout60.csv'))
    order = list(dict.fromkeys(seeds['scene'].astype(str)))
    names = ['SW', 'SE', 'NW', 'NE']
    for sid in order:
        sdf = env.pool[sid]
        agents = env._select_agents(sdf)
        gnbs = [B.scene_gnb(sdf, B.H_GNB_DEFAULT, mode=m) for m in names]
        for ts, fr in sdf.groupby('timestamp'):
            if len(fr) < MIN_VEH:
                continue
            boxes_all = fr[['x', 'y', 'length', 'width', 'height', 'theta', 'agent_id', 'sub_type']].to_dict('records')
            present = {r['agent_id']: r for r in boxes_all}
            for aid in agents:
                if aid is None or aid not in present:
                    continue
                v = present[aid]
                boxes = [b for b in boxes_all if not (abs(b['x'] - v['x']) < 1e-9 and abs(b['y'] - v['y']) < 1e-9)]
                dists = [math.hypot(v['x'] - g[0], v['y'] - g[1]) for g in gnbs]
                gi = int(np.argmin(dists))
                g = gnbs[gi]
                r = B.link_blockage((v['x'], v['y']), B.H_UE_DEFAULT, g[:2], g[2], boxes)
                if r['blocked'] and r['depth_db'] >= MIN_DB and float(boxes[r['blocker']]['height']) >= MIN_H:
                    others = []
                    for gj, gg in enumerate(gnbs):
                        rr = B.link_blockage((v['x'], v['y']), B.H_UE_DEFAULT, gg[:2], gg[2], boxes)
                        others.append(dict(station=names[gj], dist_m=dists[gj], depth_db=rr['depth_db']))
                    return dict(scene=sid, timestamp=float(ts), car=str(aid), car_box=v, boxes=boxes, frame=boxes_all,
                                gnbs=gnbs, names=names, station=names[gi], result=r, links=others, sdf=sdf)
    raise RuntimeError('no frame matches the rule')


def f1(ex):
    v, boxes, gnbs, r = ex['car_box'], ex['boxes'], ex['gnbs'], ex['result']
    blk = boxes[r['blocker']]
    gi = ex['names'].index(ex['station'])
    g = gnbs[gi]
    x0 = min(q[0] for q in gnbs); y0 = min(q[1] for q in gnbs)
    same = lambda a, b: abs(a['x'] - b['x']) < 1e-9 and abs(a['y'] - b['y']) < 1e-9
    D = math.hypot(g[0] - v['x'], g[1] - v['y'])
    fig = plt.figure(figsize=(7.0, 3.1))
    axa = fig.add_axes([0.055, 0.14, 0.24, 0.66])
    axb = fig.add_axes([0.36, 0.14, 0.24, 0.66])
    axc = fig.add_axes([0.69, 0.14, 0.30, 0.66])
    for ax in (axa, axb, axc):
        for sd in ('top', 'right'):
            ax.spines[sd].set_visible(False)
    cx, cy = v['x'] - x0, v['y'] - y0
    # (a) the whole crossing: stations, every vehicle as a dot, the four links
    for b in ex['frame']:
        if same(b, v):
            continue
        axa.plot([b['x'] - x0], [b['y'] - y0], marker='o', ms=2.6, ls='none', zorder=2,
                 color=MID if same(b, blk) else '#b9b8b2')
    for j, q in enumerate(gnbs):
        blocked = ex['links'][j]['depth_db'] >= B.BLOCK_THR_DB
        for ax in (axa, axb):
            ax.plot([cx, q[0] - x0], [cy, q[1] - y0], color=CRIT if blocked else AXIS, lw=1.2 if blocked else 0.8,
                    ls=(0, (4, 2)) if blocked else '-', zorder=3)
        axa.plot([q[0] - x0], [q[1] - y0], marker='^', ms=6.5, color=INK, ls='none', zorder=5)
        west, north = (q[0] - x0) < 1, (q[1] - y0) > 1
        axa.annotate(ex['names'][j], (q[0] - x0, q[1] - y0), xytext=(5 if west else -5, -2 if north else 2),
                     textcoords='offset points', ha='left' if west else 'right', va='top' if north else 'bottom',
                     fontsize=7, color=INK)
    axa.plot([cx], [cy], marker='o', ms=4.5, color=BLUE, ls='none', zorder=6)
    Wz = 40.0
    axa.add_patch(Rectangle((cx - Wz / 2, cy - Wz / 2), Wz, Wz, fill=False, ec=INK2, lw=0.7, zorder=4))
    Wa = max(q[0] for q in gnbs) - x0; Ha = max(q[1] for q in gnbs) - y0
    axa.set_xlim(-0.06 * Wa, 1.06 * Wa); axa.set_ylim(-0.05 * Ha, 1.05 * Ha); axa.set_aspect('equal')
    axa.set_xlabel('x (m)'); axa.set_ylabel('y (m)')
    axa.set_title('(a)  The crossing', loc='left', color=INK, pad=4)
    # (b) a 40 m x 40 m window centred on the vehicle: every vehicle as its true-size box
    for b in ex['frame']:
        if same(b, v):
            continue
        pts = [(px - x0, py - y0) for px, py in corners(b['x'], b['y'], b['length'], b['width'], b['theta'])]
        axb.add_patch(Polygon(pts, closed=True, fc=MID if same(b, blk) else LIGHT, ec='white', lw=0.5, zorder=2))
    pts = [(px - x0, py - y0) for px, py in corners(v['x'], v['y'], v['length'], v['width'], v['theta'])]
    axb.add_patch(Polygon(pts, closed=True, fc=BLUE, ec='white', lw=0.5, zorder=4))
    axb.set_xlim(cx - Wz / 2, cx + Wz / 2); axb.set_ylim(cy - Wz / 2, cy + Wz / 2); axb.set_aspect('equal')
    axb.set_xlabel('x (m)')
    axb.set_title('(b)  40 m \u00d7 40 m window', loc='left', color=INK, pad=4)
    # (c) side view of the blocked link, first 35 m
    xmax = 35.0
    for b in boxes:
        tr = B._seg_rect_trange((v['x'], v['y']), g[:2], b['x'], b['y'], b['length'], b['width'], b['theta'])
        if tr is None or tr[0] * D > xmax:
            continue
        axc.add_patch(Rectangle((tr[0] * D, 0), (min(tr[1] * D, xmax) - tr[0] * D), float(b['height']),
                                fc=MID if same(b, blk) else LIGHT, ec='white', lw=0.5, zorder=2))
    axc.add_patch(Rectangle((-v['length'] / 2, 0), v['length'], float(v['height']), fc=BLUE, ec='white', lw=0.5, zorder=2))
    slope = (g[2] - B.H_UE_DEFAULT) / D
    axc.plot([0, xmax], [B.H_UE_DEFAULT, B.H_UE_DEFAULT + slope * xmax], color=CRIT, lw=1.2, ls=(0, (4, 2)), zorder=3)
    d1 = r['d1']; h_ray = B.H_UE_DEFAULT + slope * d1
    axc.annotate('', xy=(d1, float(blk['height'])), xytext=(d1, h_ray),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=0.7, shrinkA=0, shrinkB=0), zorder=5)
    axc.annotate(f'{blk.get("sub_type", "vehicle")!s}'.lower() + f' {float(blk["height"]):.1f} m tall,\n'
                 f'{r["dh"]:.1f} m above the ray:\n{r["depth_db"]:.1f} dB knife-edge loss',
                 (d1, float(blk['height'])), xytext=(10, 4), textcoords='offset points', fontsize=7, color=INK,
                 ha='left', va='bottom')
    axc.text(xmax - 0.5, 0.25, f'the link continues to {ex["station"]}:\n{D:.0f} m away, 6 m high \u2192',
             ha='right', va='bottom', fontsize=7, color=INK2)
    axc.set_xlim(-v['length'], xmax); axc.set_ylim(0, 4.5)
    axc.set_xlabel(f'distance along the link to {ex["station"]} (m)'); axc.set_ylabel('height (m)')
    axc.set_title('(c)  Side view of the blocked link', loc='left', color=INK, pad=4)
    hs = [Line2D([], [], marker='s', ms=7, ls='none', color=BLUE), Line2D([], [], marker='s', ms=7, ls='none', color=MID),
          Line2D([], [], marker='s', ms=7, ls='none', color=LIGHT), Line2D([], [], marker='^', ms=6.5, ls='none', color=INK),
          Line2D([], [], color=CRIT, lw=1.2, ls=(0, (4, 2))), Line2D([], [], color=AXIS, lw=0.8)]
    fig.legend(hs, ['the vehicle', 'blocking vehicle', 'other vehicles', 'roadside station', 'blocked link', 'clear link'],
               loc='upper center', bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=6, handletextpad=0.4,
               columnspacing=1.2)
    save(fig, 'F1_scene_top_and_side_view')
    rec = dict(rule=f'first match: >= {MIN_VEH} vehicles in the frame, link to the nearest station blocked by a '
                    f'vehicle >= {MIN_H} m tall, knife-edge loss >= {MIN_DB} dB; scenes in frozen seed-list order',
               scene=ex['scene'], timestamp=ex['timestamp'], car=ex['car'], vehicles_in_frame=len(ex['frame']),
               station=ex['station'], link_length_m=D, loss_db=r['depth_db'], height_above_ray_m=r['dh'],
               knife_edge_distance_m=d1,
               blocker=dict(type=str(blk.get('sub_type')), length=float(blk['length']), width=float(blk['width']),
                            height=float(blk['height'])),
               car_height_m=float(v['height']), links=ex['links'],
               ue_antenna_m=B.H_UE_DEFAULT, station_height_m=B.H_GNB_DEFAULT, margin_m=5.0)
    json.dump(rec, open(os.path.join(OUT, 'F1_EXAMPLE.json'), 'w'), indent=1)
    return rec


# ------------------------------------------------------------------ F2: timeline sketch
def f2():
    xs = {'t': 0, 't+1': 1, 't+2': 2, '...': 3, 't+L': 4, 't+L+1': 5, 't+L+2': 6}
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.7), sharex=True)
    for k, ax in enumerate(axs):
        for sd in ('top', 'right', 'left'):
            ax.spines[sd].set_visible(False)
        ax.set_yticks([])
        rows = [('vehicle a', 1.6, 0, 'Y'), ('vehicle b', 0.9, 1, 'Y' if k == 0 else 'Z')]
        for name, y, d0, dest in rows:
            ax.text(-0.75, y, name, ha='right', va='center', fontsize=7.5, color=INK)
            ax.plot([d0], [y], marker='D', ms=6, color=INK, zorder=3)
            ax.add_patch(Rectangle((d0 + 0.6, y - 0.13), 3.8, 0.26, fc=GRID, ec=AXIS, lw=0.6, hatch='////', zorder=2))
            if name == 'vehicle a':
                ax.text(d0 + 2.5, y + 0.22, 'in flight: L slots', ha='center', va='bottom', fontsize=7, color=INK2)
            clash = (k == 0)
            ax.plot([d0 + 5], [y], marker='o', ms=7, color=CRIT if (clash and name == 'vehicle b') else BLUE, zorder=3)
            ax.text(d0 + 5.25, y, f'lands at {dest}', ha='left', va='center', fontsize=7.5, color=INK)
        yl = 0.15
        ax.text(-0.75, yl, 'station Y', ha='right', va='center', fontsize=7.5, color=INK)
        if k == 0:
            ax.text(0.0, yl, 'one free place now; at t+1 vehicle a is in flight and not counted at Y, so Y still shows room',
                    ha='left', va='center', fontsize=7, color=INK2)
            ax.text(6.0, yl - 0.5, '\u2717  C + 1 vehicles at Y: one has no place', ha='center', va='center', fontsize=7.5,
                    color=CRIT, fontweight='bold')
            ax.set_title('(a)  Planning on where vehicles are now: vehicle b is sent to the place vehicle a will take (Theorem 1)',
                         loc='left', color=INK, pad=4)
        else:
            ax.text(0.0, yl, 'at t+1 the landing load of Y already counts vehicle a: C, so vehicle b is given Z',
                    ha='left', va='center', fontsize=7, color=INK2)
            ax.text(6.0, yl - 0.5, '\u2713  C vehicles at Y: every vehicle has a place', ha='center', va='center', fontsize=7.5,
                    color=GOOD, fontweight='bold')
            ax.set_title('(b)  Planning on the landing state (LSRA): vehicles in flight count where they will land',
                         loc='left', color=INK, pad=4)
        ax.set_ylim(-0.55, 2.2)
    axs[1].set_xticks(list(xs.values()))
    axs[1].set_xticklabels(list(xs.keys()))
    axs[1].set_xlim(-0.9, 7.9)
    axs[1].set_xlabel('slot (0.1 s each); L = handover delay in slots')
    hs = [Line2D([], [], marker='D', ms=6, ls='none', color=INK), Rectangle((0, 0), 1, 1, fc=GRID, ec=AXIS, hatch='////'),
          Line2D([], [], marker='o', ms=7, ls='none', color=CRIT), Line2D([], [], marker='o', ms=7, ls='none', color=BLUE)]
    fig.legend(hs, ['decision', 'in flight (still served by the old station)', 'landing on a full station', 'landing with room'],
               loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.5, columnspacing=1.4)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.86, hspace=0.35)
    save(fig, 'F2_timeline_decide_fly_land')


# ------------------------------------------------------------------ F3: closed-loop sketch
def station(ax, cx, cy, name, cars, hi):
    ax.add_patch(FancyBboxPatch((cx - 0.75, cy - 0.32), 1.5, 0.64, boxstyle='round,pad=0.04,rounding_size=0.12',
                                fc='white', ec=INK2, lw=0.9, zorder=2))
    ax.text(cx, cy + 0.52, f'station {name} (full: C = 2)', ha='center', va='bottom', fontsize=7.5, color=INK)
    for k, c in enumerate(cars):
        px = cx - 0.36 + 0.72 * k
        ax.add_patch(Circle((px, cy), 0.2, fc=BLUE if c in hi else LIGHT, ec='white', lw=0.8, zorder=3))
        ax.text(px, cy, c, ha='center', va='center', fontsize=7.5, color='white' if c in hi else INK, zorder=4,
                fontweight='bold')


def f3():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.25))
    for k, ax in enumerate(axs):
        ax.set_xlim(0, 4.2); ax.set_ylim(-0.75, 1.35); ax.set_aspect('equal'); ax.axis('off')
        station(ax, 0.9, 0.35, 'X', ['a', 'c'], {'a', 'b'})
        station(ax, 3.3, 0.35, 'Y', ['b', 'd'], {'a', 'b'})
        ok = (k == 1)
        col = GOOD if ok else CRIT
        ax.add_patch(FancyArrowPatch((1.72, 0.47), (2.48, 0.47), arrowstyle='-|>', mutation_scale=9, lw=1.2,
                                     color=col, zorder=1))
        ax.add_patch(FancyArrowPatch((2.48, 0.23), (1.72, 0.23), arrowstyle='-|>', mutation_scale=9, lw=1.2,
                                     color=col, zorder=1))
        ax.text(2.1, 0.53, 'a', ha='center', va='bottom', fontsize=7, color=INK)
        ax.text(2.1, 0.17, 'b', ha='center', va='top', fontsize=7, color=INK)
        if not ok:
            ax.text(2.1, 1.22, '\u2717  a \u2192 Y refused: Y is full;  b \u2192 X refused: X is full',
                    ha='center', va='center', fontsize=7.5, color=CRIT)
            ax.text(2.1, -0.32, 'Each request is checked alone, so both stay.', ha='center', va='center',
                    fontsize=7.5, color=INK)
            ax.text(2.1, -0.6, 'Their links keep failing (cause B in Fig. 6).', ha='center', va='center',
                    fontsize=7, color=INK2)
            ax.set_title('(a)  Per-request admission (today)', loc='left', color=INK, pad=2)
        else:
            ax.text(2.1, 1.22, '\u2713  {a \u2192 Y, b \u2192 X} admitted together', ha='center', va='center',
                    fontsize=7.5, color=GOOD)
            ax.text(2.1, -0.32, 'After both moves, X and Y still hold C = 2.', ha='center', va='center',
                    fontsize=7.5, color=INK)
            ax.text(2.1, -0.6, 'Assumes both land in the same slot (tested by the delay check).', ha='center',
                    va='center', fontsize=7, color=INK2)
            ax.set_title('(b)  Coordinated admission (proposed for 6G)', loc='left', color=INK, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93, wspace=0.05)
    save(fig, 'F3_closed_loop_swap_two_admission_models')


def main():
    os.makedirs(OUT, exist_ok=True)
    ex = pick_example()
    rec = f1(ex)
    f2(); f3()
    print(json.dumps(rec, indent=1))
    print(f'make_figs_ma_f1f3.py md5 {md5(os.path.abspath(__file__))}; python {platform.python_version()}, '
          f'matplotlib {matplotlib.__version__}')


if __name__ == '__main__':
    main()
