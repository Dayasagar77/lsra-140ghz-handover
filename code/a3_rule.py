#!/usr/bin/env python3
"""The 3GPP event-A3 handover rule as a baseline for paper2ma_env (DRAFT, 25 Oct 2026; nothing run).

Pre-registration: PREREG_MA_A3Baseline_DRAFT_v1_20261025.md (needs Daya's OK).

Sources:
  * 3GPP TS 38.331, clause 5.5.4.4 (event A3, "neighbour becomes offset better than SpCell"):
      entering condition  Mn + Ofn + Ocn - Hys > Mp + Ofp + Ocp + Off.
    Here Ofn = Ocn = Ofp = Ocp = 0 and Hys = 0 (TR 36.839 gives only the A3 offset), so the condition is
    Fn > Fp + Off on the layer-3 filtered results.
  * 3GPP TS 38.331, clause 5.5.3.2 (layer-3 filtering):  F_n = (1 - a) F_(n-1) + a M_n,  a = 1/2^(k/4), in
    dBm; the first measurement initialises the filter; k = 0 means no filtering; the filter keeps its time
    characteristics at other input rates. TR 36.839 (Table 5.2.4.1) takes the L1 measurement period as
    200 ms and notes that K is scaled when the sample period is shorter. Our input is one measurement per
    0.1 s slot, so per slot  a_slot = 1 - (1 - a)^(100/200).
  * 3GPP TR 36.839 V11.1.0, Table 5.3.2.1 ("Configuration parameter sets for simulation calibration"):
      Set 1: TTT 480 ms, A3 offset  3 dB, K 4      Set 4: TTT  80 ms, A3 offset  1 dB, K 1
      Set 2: TTT 160 ms, A3 offset  3 dB, K 4      Set 5: TTT  40 ms, A3 offset -1 dB, K 0
      Set 3: TTT 160 ms, A3 offset  2 dB, K 1      (L1 to L3 period 200 ms in every set)

The rule, every slot, for every present vehicle i:
  1. Measure: the received power of i at every station g in this slot, the quantity the simulator's own channel
     uses for a serving link (paper2ma_env._sinr_db): transmit power + serving-beam gain - close-in path loss -
     blockage loss - the slot's shadowing draw z[t, i, g] times its standard deviation (8.2 dB if blocked,
     2.7 dB otherwise). No new random numbers are drawn and no measurement error is added.
  2. Filter it (layer 3, above). A vehicle that (re)appears starts a new filter.
  3. If i is in flight, its counts are cleared and it requests nothing (as for every arm).
  4. Otherwise, for every neighbour n of its serving station p: count the consecutive slots with F_n > F_p + Off
     (a slot that fails resets the count). Counts restart when the serving station changes.
  5. Time to trigger: the condition must hold at every measurement instant (slot start) inside the TTT
     window, that is in floor(TTT / 0.1 s) + 1 consecutive slots (5, 2, 2, 1, 1 for Sets 1-5). The request is
     made at the last of them, 40-80 ms before the TTT itself would end.
  6. If some neighbours have reached that count, request a handover to the one with the strongest filtered
     power. A refused request may be repeated in the next slot, as for every arm. Full transmit power and a
     fixed sub-band, as for every arm.
"""
import math
import numpy as np
import paper2ma_env as MA

SETS = {1: (480.0, 3.0, 4), 2: (160.0, 3.0, 4), 3: (160.0, 2.0, 1), 4: (80.0, 1.0, 1), 5: (40.0, -1.0, 0)}
SLOT_MS = 100.0
L1_L3_MS = 200.0


def ttt_slots(ttt_ms):
    """Consecutive slots in which the entering condition must hold: the slot starts inside [0, TTT]."""
    return int(math.floor(ttt_ms / SLOT_MS + 1e-9)) + 1


def alpha_slot(k):
    a = 1.0 / (2.0 ** (k / 4.0))
    return 1.0 - (1.0 - a) ** (SLOT_MS / L1_L3_MS)


def measured_dbm(env, i):
    """Received power (dBm) of vehicle i at every station in the current slot, with the slot's shadowing draw."""
    a = env.frames[env.t]['ag'][i]
    dist = np.maximum(np.asarray(a['dist'], dtype=float), 1.0)
    depth = np.asarray(a['depth'], dtype=float)
    pl = MA.CH.ci_pathloss_db(dist, True)
    sig = np.where(depth > 0, MA.CH.SIG_NLOS, MA.CH.SIG_LOS)
    return float(env.pt[i]) + MA.G_SERV - pl - np.asarray(env._z[env.t, i, :], dtype=float) * sig - depth


class A3Rule:
    """Event A3 with one TR 36.839 parameter set. Holds per-episode state; a new episode is recognised by
    t = 0 (every episode of the harness starts at slot 0), and slots must then follow one by one."""

    def __init__(self, set_id):
        ttt, off, k = SETS[int(set_id)]
        self.set_id = int(set_id)
        self.n_ttt = ttt_slots(ttt)
        self.off = float(off)
        self.a = alpha_slot(k)
        self._z = None

    def _new_episode(self, env):
        M, G = env.M, env.n_gnb
        self.F = np.zeros((M, G))
        self.cnt = np.zeros((M, G), dtype=int)
        self.p = np.full(M, -1, dtype=int)
        self.seen = np.zeros(M, dtype=bool)
        self.last_t = -1
        self._z = env._z

    def __call__(self, env, obs, sinr, rng, ag, temp):
        if env.t == 0:
            self._new_episode(env)
        if self._z is not env._z or env.t != self.last_t + 1:
            raise RuntimeError('A3 rule: slots of one episode must be given in order, starting at t = 0')
        self.last_t = env.t
        M = env.M
        fr = env.frames[env.t]
        a = np.ones((M, 3), int); a[:, 1] = 2; a[:, 2] = 1
        tgt = -np.ones(M, int)
        for i in range(M):
            if not fr['active'][i]:
                self.seen[i] = False
                a[i, 0] = 0
                continue
            m = measured_dbm(env, i)
            if not self.seen[i]:
                self.F[i] = m
                self.cnt[i] = 0
                self.seen[i] = True
            else:
                self.F[i] = (1.0 - self.a) * self.F[i] + self.a * m
            p = int(env.serv[i])
            if p != self.p[i] or env._ho_timer[i] > 0:
                self.cnt[i] = 0
                self.p[i] = p
            if env._ho_timer[i] > 0:
                a[i, 0] = 0
                continue
            cond = self.F[i] > self.F[i, p] + self.off
            cond[p] = False
            self.cnt[i] = np.where(cond, self.cnt[i] + 1, 0)
            trig = np.flatnonzero(self.cnt[i] >= self.n_ttt)
            if trig.size:
                g = int(trig[np.argmax(self.F[i, trig])])
                a[i, 0] = 1
                tgt[i] = g
            else:
                a[i, 0] = 0
        return a, tgt


def arms():
    """The five A3 arms, fresh objects (each holds its own state)."""
    return [(f'A3S{k}', A3Rule(k)) for k in sorted(SETS)]
