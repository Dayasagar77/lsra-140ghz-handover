#!/usr/bin/env python3
"""Capacity-aware attachment for the multi-agent environment (20 Oct 2026).

Pre-registered rule D0 (PREREG_MA_REMEASURE_FROZEN_20261020.md) fired: in the base environment
(paper2ma_env.py, unchanged) the episode starts from a best-signal attachment that ignores station
capacity, and a car that appears joins station 0 (SW), or the station it had before it left, again
with no capacity check. This subclass changes ONLY those two things:

  * at the start of an episode, the cars present in frame 0 are attached strongest first (by each
    car's best mean received power), each to its best station that still has room;
  * a car that appears (absent in the previous frame, present now) is attached, strongest first, to
    its best station that has room among the cars it currently serves; a handover it had in flight
    when it left is cancelled.

"Best" uses the environment's own mean received power (_desired_dbm, as in its admission rule).
Everything else is inherited unchanged. With 16 cars and 4 x 4 places there is always a station with
room; if there were not, the class raises instead of silently overfilling.

ATTACH = False switches the change off (used to show the subclass is otherwise identical).
"""
import numpy as np
import paper2ma_env as MA


class CapacityAttachEnv(MA.MultiAgentRSUBlockageEnv):
    ATTACH = True

    def __init__(self, *args, **kwargs):
        self.n_attach = 0          # attachments decided by this class
        self.n_changed = 0         # ... that differ from what the base environment would have done
        super().__init__(*args, **kwargs)

    def _strength(self, i):
        return max(self._desired_dbm(i, g) for g in range(self.n_gnb))

    def _best_with_room(self, i, occ, cap):
        order = sorted(range(self.n_gnb), key=lambda g: (-self._desired_dbm(i, g), g))
        for g in order:
            if occ[g] < cap:
                return g
        raise RuntimeError('no station has room: active cars exceed total capacity')

    def _compute_admission(self):
        if self.ATTACH and self.gnb_cap is not None:
            fr = self.frames[self.t]
            cap = int(self.gnb_cap)
            occ = np.zeros(self.n_gnb, dtype=int)
            if self.t == 0:
                movers = [i for i in range(self.M) if fr['active'][i]]
            else:
                prev = self.frames[self.t - 1]['active']
                movers = [i for i in range(self.M) if fr['active'][i] and not prev[i]]
                if movers:
                    ms = set(movers)
                    for j in range(self.M):
                        if fr['active'][j] and j not in ms:
                            occ[self.serv[j]] += 1
            movers.sort(key=lambda i: (-self._strength(i), i))
            for i in movers:
                g = self._best_with_room(i, occ, cap)
                self.n_attach += 1
                self.n_changed += int(g != int(self.serv[i]))
                self.serv[i] = g; self._ho_target[i] = g; self._ho_timer[i] = 0
                occ[g] += 1
            if self.t == 0:
                assert (occ <= cap).all(), 'start attachment exceeds capacity'
        return super()._compute_admission()
