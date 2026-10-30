#!/usr/bin/env python3
"""Diagnostic replay for the realism run's stopped must-hold (30 Oct 2026; Daya's OK: "Diagnose, then re-analyse").

The registered analysis of the realism run stopped because the harness count dec_over ("landing excess created by a
decision", method_harness.run_eval_m) was above 0 in 17 cells of H2a, H2b and ALL: 98 episodes, one event each.

This script replays only those 98 episodes, read-only, with the frozen code, through a subclass of RealismEnv that
only records (it changes no behaviour). For every slot it computes:
  - the harness's own count: each handover started in the slot charged to env._ho_target read after the slot;
  - the same count with each started handover charged to the station its decision chose, that is, the target it had
    before any re-establishment in that slot (realism_env._start_reest);
  - every re-establishment in the slot, with the target it replaced;
  - the environment's own landing load after the slot (what _start_reest checks: active vehicles at their target if
    in flight, else at their serving station).

Must hold (else 'confirmed' is false):
  1. the 235 frozen files and the 163 stored cells are unchanged;
  2. every replayed episode reproduces its stored row exactly (all columns, and the CSV text line);
  3. the replayed harness count equals the stored dec_over in every episode;
  4. in every slot where the harness count rises, a handover started in that slot was re-established too late in the
     same slot and so charged to another station; recounted on the decision's own target, that slot has no excess;
     and the environment's own landing load at the charged station is within the 4 places;
  5. the recount on the decision's own target is 0 in all 98 episodes.

It writes DIAG_DECOVER.json next to itself and nothing else. It changes no frozen file and no stored cell.
"""
import os
import sys
import io
import json
import time
import hashlib
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', 'code'))
sys.path.insert(0, CODE)
import numpy as np
import pandas as pd
import realism_check as RC
import realism_env as RE

OUT = os.path.join(HERE, 'DIAG_DECOVER.json')
REAL = os.path.join(CODE, 'models', 'realism_check')
CELLS = os.path.join(REAL, 'cells')
FREEZE_PATH = os.path.join(REAL, 'REAL_FREEZE.json')
FREEZE_MD5 = '5d78f39449460efa68924399107e7030'
STOPPED_RESULT_MD5 = '20b00d61fe02132f13d52dbf2f5caf68'
MANIFEST = os.path.join(HERE, 'REALISM_CELLS_MD5_20261030.txt')
MANIFEST_MD5 = '704b7e84cf64ccbfa04c17c18f7567ec'
WORLDS = ('H2a', 'H2b', 'ALL')
N_FLAGGED = 98


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_inputs():
    assert md5(FREEZE_PATH) == FREEZE_MD5, 'REAL_FREEZE.json changed'
    fz = json.load(open(FREEZE_PATH))
    bad = [k for k, v in fz['md5'].items() if md5(os.path.normpath(os.path.join(CODE, k))) != v]
    assert not bad, 'frozen files changed: ' + ', '.join(bad[:10])
    assert md5(os.path.join(REAL, 'REAL_RESULT.json')) == STOPPED_RESULT_MD5, 'the stopped REAL_RESULT.json changed'
    assert md5(MANIFEST) == MANIFEST_MD5, 'the cell manifest changed'
    want = {}
    for line in open(MANIFEST, encoding='utf-8'):
        h, name = line.split()
        want[name] = h
    assert len(want) == 163
    bad = [n for n, h in want.items() if md5(os.path.join(CELLS, n)) != h]
    assert not bad, 'stored cells changed: ' + ', '.join(bad[:10])
    return len(fz['md5']), len(want)


class DiagEnv(RE.RealismEnv):
    """RealismEnv with read-only records of every re-establishment and of the landing counts of every slot."""

    def _start_reest(self, i, slots, kind):
        rec = getattr(self, '_diag_re', None)
        if rec is not None:
            rec.append((int(i), str(kind), int(self._ho_target[i]), int(self._ho_timer[i]), int(self.serv[i])))
        return super()._start_reest(i, slots, kind)

    def reset(self, *, seed=None, options=None):
        self.diag_slots = []
        self.diag_harness = 0
        self.diag_decision = 0
        return super().reset(seed=seed, options=options)

    def step(self, action, ho_targets=None):
        t0 = int(self.t)
        act_pre = np.asarray(self.frames[t0]['active'], dtype=bool).copy()
        serv_pre = self.serv.copy()
        tim_pre = self._ho_timer.copy()
        tgt_pre = self._ho_target.copy()
        self._diag_re = []
        out = super().step(action, ho_targets)
        re_now = list(self._diag_re)
        self._diag_re = None
        info = out[4]
        M, G, cap = self.M, self.n_gnb, int(self.gnb_cap)
        started = [int(j) for j in np.where(info['ho'])[0]]
        land = np.zeros(G)
        for j in range(M):
            if act_pre[j]:
                land[tgt_pre[j] if tim_pre[j] > 0 else serv_pre[j]] += 1
        before = land.copy()
        land_h = land.copy()
        land_d = land.copy()
        first = {}
        for r in re_now:
            first.setdefault(r[0], r)
        moved = {}
        for j in started:
            g_from = int(serv_pre[j])
            g_h = int(self._ho_target[j])
            g_d = int(first[j][2]) if j in first else g_h
            land_h[g_h] += 1
            land_h[g_from] -= 1
            land_d[g_d] += 1
            land_d[g_from] -= 1
            if g_d != g_h:
                moved[j] = (g_d, g_h, first[j][1])
        dec_h = int(np.maximum(land_h - np.maximum(before, cap), 0).sum())
        dec_d = int(np.maximum(land_d - np.maximum(before, cap), 0).sum())
        self.diag_harness += dec_h
        self.diag_decision += dec_d
        if dec_h > 0 or dec_d > 0:
            act_post = np.asarray(self.frames[self.t]['active'], dtype=bool)
            env_post = np.zeros(G)
            for j in range(M):
                if act_post[j]:
                    env_post[self._ho_target[j] if self._ho_timer[j] > 0 else self.serv[j]] += 1
            over = [int(g) for g in range(G) if land_h[g] > max(before[g], cap)]
            pos_pre = {j: int(tgt_pre[j] if tim_pre[j] > 0 else serv_pre[j]) for j in range(M) if act_pre[j]}
            why = {}
            for g in over:
                why[g] = dict(
                    charged_after_same_slot_too_late=[j for j, v in moved.items() if v[1] == g and v[2] == 'too_late'],
                    left_the_scene=[j for j, p in pos_pre.items() if p == g and not act_post[j]],
                    moved_away_by_other_reestablishment=[r[0] for r in re_now if r[0] not in started and r[2] == g
                                                        and int(self._ho_target[r[0]]) != g],
                    harness_load=float(land_h[g]), env_load_after=float(env_post[g]), cap=cap)
            self.diag_slots.append(dict(
                slot=t0, started=started, reestablished=[list(r) for r in re_now],
                moved_started={str(j): list(v) for j, v in moved.items()},
                before=before.tolist(), harness_after=land_h.tolist(), decision_after=land_d.tolist(),
                env_after=env_post.tolist(), harness_count=dec_h, decision_count=dec_d, over=over,
                why={str(g): v for g, v in why.items()}))
        return out


def make_diag_env(world, A):
    """realism_check.make_env with DiagEnv in place of RealismEnv (the same arguments and switches)."""
    cfg = RC.config(world, None)
    hl = int(cfg.pop('HL', RC.HL))
    gain = cfg.pop('GAIN', None)
    env = DiagEnv(RC.pool(), n_agents=16, n_subbands=3, gnb_cap=RC.CAP, seed=0, split=None,
                  compute_anticipation=True, ho_latency=hl)
    env.ATTACH = True
    env.ADMIT = RC.ADM[A]
    for k, v in cfg.items():
        if not hasattr(RE.RealismEnv, k) and k != 'JITTER':
            raise KeyError(k)
        setattr(env, k, v)
    return env, gain


def csv_lines(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().splitlines()


def same_value(u, v):
    if pd.isna(u) and pd.isna(v):
        return True
    if isinstance(u, str) or isinstance(v, str):
        return str(u) == str(v)
    return float(u) == float(v)


def main():
    t_start = time.time()
    assert not os.path.exists(OUT), 'DIAG_DECOVER.json exists: it runs once'
    n_frozen, n_cells = check_inputs()
    problems = json.load(open(os.path.join(REAL, 'REAL_RESULT.json')))['problems']
    flagged_cells = []
    for p in problems:
        key = p.split(':')[0].strip()
        w, arm, A = [s.strip(" '()") for s in key.split(',')]
        flagged_cells.append((w, arm, A))
    assert all(w in WORLDS for w, _, _ in flagged_cells)
    rec = dict(written=None, versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
               code=dict(this=md5(os.path.abspath(__file__)), realism_check=md5(RC.__file__), realism_env=md5(RE.__file__)),
               inputs=dict(frozen_files_unchanged=n_frozen, cells_unchanged=n_cells), cells={})
    # every H2a, H2b and ALL cell is read, so that a flagged episode outside the stopped list would be seen
    all_flagged = 0
    for w in WORLDS:
        for arm, A in RC.ARMS:
            name = RC.cell_name(w, None, arm, A)
            d = pd.read_csv(os.path.join(CELLS, name + '.csv'))
            n = int((d.dec_over > 0).sum())
            all_flagged += n
            assert (n > 0) == ((w, arm, A) in flagged_cells), f'{name}: flagged episodes do not match the stopped list'
    assert all_flagged == N_FLAGGED, f'{all_flagged} flagged episodes, not {N_FLAGGED}'
    ok_rows = ok_text = ok_count = ok_slots = True
    total_h = total_d = 0
    n_ep = 0
    tally = dict(slots=0, charged_after_same_slot_too_late=0, left_the_scene=0, moved_away_by_other_reestablishment=0,
                 env_within_cap=0)
    for w, arm, A in flagged_cells:
        name = RC.cell_name(w, None, arm, A)
        path = os.path.join(CELLS, name + '.csv')
        stored = pd.read_csv(path)
        text = open(path, encoding='utf-8').read().splitlines()
        header, body = text[0], text[1:]
        assert len(body) == len(stored) == RC.N_EP
        idx = [i for i in range(len(stored)) if stored.dec_over.iloc[i] > 0]
        seeds = [int(stored.seed.iloc[i]) for i in idx]
        env, gain = make_diag_env(w, A)
        RE.set_beam_gain(gain)
        eps = []
        try:
            pol = RC.policy(arm, A)
            for i, s in zip(idx, seeds):
                rows = RC.run_real(env, pol, [s])
                for x in rows:
                    x.update(world=w, coord='', arm=arm, adm=A, hl=int(env.ho_lat), cap=RC.CAP)
                lines = csv_lines(pd.DataFrame(rows))
                back = pd.read_csv(io.StringIO('\n'.join(lines)))
                srow = stored.iloc[i]
                cols_same = list(back.columns) == list(stored.columns)
                vals_same = cols_same and all(same_value(back[c].iloc[0], srow[c]) for c in stored.columns)
                text_same = lines[0] == header and lines[1] == body[i]
                slots = env.diag_slots
                slot_ok = True
                for sl in slots:
                    if sl['harness_count'] > 0:
                        tally['slots'] += 1
                        good = sl['decision_count'] == 0 and len(sl['over']) > 0
                        for g in sl['over']:
                            wy = sl['why'][str(g)]
                            good = good and len(wy['charged_after_same_slot_too_late']) > 0 and wy['env_load_after'] <= wy['cap']
                            tally['charged_after_same_slot_too_late'] += int(len(wy['charged_after_same_slot_too_late']) > 0)
                            tally['left_the_scene'] += int(len(wy['left_the_scene']) > 0)
                            tally['moved_away_by_other_reestablishment'] += int(len(wy['moved_away_by_other_reestablishment']) > 0)
                            tally['env_within_cap'] += int(wy['env_load_after'] <= wy['cap'])
                        slot_ok = slot_ok and good
                    elif sl['decision_count'] > 0:
                        slot_ok = False
                ok_rows = ok_rows and vals_same
                ok_text = ok_text and text_same
                ok_count = ok_count and env.diag_harness == int(srow.dec_over)
                ok_slots = ok_slots and slot_ok
                total_h += env.diag_harness
                total_d += env.diag_decision
                n_ep += 1
                eps.append(dict(seed=s, scene=str(srow.scene), row_same=bool(vals_same), text_same=bool(text_same),
                                stored_dec_over=int(srow.dec_over), harness_count=int(env.diag_harness),
                                decision_count=int(env.diag_decision), slots=slots, slots_explained=bool(slot_ok)))
        finally:
            RE.set_beam_gain(None)
        rec['cells'][name] = eps
        print(f'{name}: {len(eps)} episodes replayed', flush=True)
    confirmed = bool(n_ep == N_FLAGGED and ok_rows and ok_text and ok_count and ok_slots and total_d == 0)
    rec['summary'] = dict(episodes=n_ep, rows_reproduced=bool(ok_rows), csv_text_reproduced=bool(ok_text),
                          harness_count_reproduced=bool(ok_count), every_slot_explained=bool(ok_slots),
                          harness_total=int(total_h), decision_total=int(total_d), tally=tally,
                          confirmed=confirmed, seconds=round(time.time() - t_start, 1))
    rec['written'] = time.strftime('%Y-%m-%d %H:%M:%S')
    assert not os.path.exists(OUT), 'DIAG_DECOVER.json exists: it runs once'
    json.dump(rec, open(OUT, 'w'), indent=1)
    print(json.dumps(rec['summary']), flush=True)


if __name__ == '__main__':
    main()
