#!/usr/bin/env python3
"""Handover delays that differ from car to car, for the multi-car environment (DRAFT, 20 Oct 2026; not run).

Pre-registration: PREREG_MA_DelayDifference_Jitter_DRAFT_v1_20261020.md (needs Daya's OK).

In every earlier run a handover started in slot t lands after exactly L slots, so the cars of a
coordinated swap (or longer loop) all land in the same slot. Real handovers do not finish at exactly
the same instant. This subclass gives each started handover its own delay L + d, where d is drawn
uniformly from {-J, ..., +J} slots, and changes nothing else (channel, attachment, admission at
preparation and at landing, RLF).

Common random numbers: d is read from a table D[t, i] drawn at reset from the episode seed alone, so a
car that starts a handover in slot t gets the same d whatever the policy. J = 0 gives D = 0, which must
reproduce the earlier runs exactly (gate G0).

Per-episode record (one per reset, in self.jit_log):
  starts                 handovers started;
  early, late            how many got d < 0 and d > 0;
  min_delay, max_delay   the shortest and longest delay given (slots; must lie in [L - J, L + J]);
  excess_car_steps       sum over slots and stations of max(0, cars attached - C), counted after each
                         slot: cars that landed before the cars they replace had left, plus any other
                         over-filling (for example a car that appeared at a station that was full at
                         landing). Its level at J = 0 is the reference.
  flight_mismatch        landings whose flight time (active slots) differs from the delay given; must be 0.
With DEBUG = True each start and landing is also listed in self.events (unit test U1).
"""
import numpy as np
import paper2ma_env_admit as AD

JIT_SEED = 20261020


class JitterEnv(AD.AdmissionEnv):
    JITTER = 0
    DEBUG = False

    def __init__(self, *args, **kwargs):
        self.jit_log = []
        self.events = []
        self._D = None
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        J = int(self.JITTER)
        if J < 0:
            raise ValueError('JITTER must be >= 0')
        if J > 0 and seed is None:
            raise ValueError('a seed is needed for the delay table')
        F = len(self.frames)
        if J == 0:
            self._D = np.zeros((F, self.M), dtype=int)
        else:
            rng = np.random.default_rng([JIT_SEED, int(seed)])
            self._D = rng.integers(-J, J + 1, size=(F, self.M))
        self.jit_log.append(dict(starts=0, early=0, late=0, min_delay=-1, max_delay=-1,
                                 excess_car_steps=0, flight_mismatch=0))
        self._delay = np.zeros(self.M, dtype=int)
        self._flown = np.zeros(self.M, dtype=int)
        return out

    def step(self, action, ho_targets=None):
        t0 = self.t
        act0 = np.asarray(self.frames[t0]['active'], dtype=bool).copy()
        timer0 = self._ho_timer.copy()
        out = super().step(action, ho_targets)
        info = out[4]
        rec = self.jit_log[-1]
        L = int(self.ho_lat)
        # flights in progress: one more active slot flown; a landing ends the flight
        for i in range(self.M):
            if act0[i] and timer0[i] > 0:
                self._flown[i] += 1
                if timer0[i] == 1:                               # landed in this slot
                    if self._flown[i] != self._delay[i]:
                        rec['flight_mismatch'] += 1
                    if self.DEBUG:
                        self.events.append(('land', int(i), t0, int(self._flown[i]), int(self._delay[i])))
        for i in np.where(info['ho'])[0]:
            d = int(self._D[t0, i])
            delay = max(1, L + d)
            self._ho_timer[i] = delay
            self._delay[i] = delay
            self._flown[i] = 0
            rec['starts'] += 1
            rec['early'] += int(d < 0)
            rec['late'] += int(d > 0)
            rec['min_delay'] = delay if rec['min_delay'] < 0 else min(rec['min_delay'], delay)
            rec['max_delay'] = max(rec['max_delay'], delay)
            if self.DEBUG:
                self.events.append(('start', int(i), t0, delay))
        fr = self.frames[self.t]
        occ = np.zeros(self.n_gnb, dtype=int)
        for j in range(self.M):
            if fr['active'][j]:
                occ[int(self.serv[j])] += 1
        rec['excess_car_steps'] += int(np.maximum(occ - int(self.gnb_cap), 0).sum())
        return out
