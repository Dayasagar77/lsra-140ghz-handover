#!/usr/bin/env python3
"""Admission at handover preparation for the multi-car environment (DRAFT, 20 Oct 2026; not run yet).

Pre-registration: PREREG_MA_AdmissionAtPreparation_Robustness_DRAFT_v1_20261020.md (needs Daya's OK).

In the base environment a station only admits cars when they land: in every slot it serves its C
strongest cars and the rest are unserved. 3GPP TS 38.300 (intra-NR handover, C-plane step 4) says
admission control may be performed by the target gNB when the handover is prepared. This subclass
adds that step as an option, OFF by default:

  ADMIT = 'none'         as before: identical to CapacityAttachEnv (checked by gate G0).
  ADMIT = 'per_request'  (A1, today's per-request admission, most generous reading) each target admits
                         a request only if it has room when the handover lands: its landing load (cars on
                         their way counted where they will land, departures already committed taken off)
                         is below C. Requests are taken strongest first (mean received power at the
                         target, the same ranking as the landing rule). A refused request is tried again
                         after other departures in the same slot are committed, until nothing changes.
                         So chains always get through and closed loops never do (maths note, Theorem 2).
  ADMIT = 'coordinated'  (A2, what a RIC could do) the slot's requests are admitted together: the whole
                         set if it creates no landing excess, otherwise the largest subset that creates
                         none (ties: stronger requests first). Exact, by one assignment problem.

"Creates landing excess" means some station's landing load rises above max(C, its load before the
slot's requests), the same definition as dec_over in method_harness.py.

A refused request means the car stays where it is: no handover starts, and it may ask again next slot.
Nothing else changes (channel, attachment, admission at landing as the backstop, RLF).

Per-episode record (one per reset, in self.adm_log): requests, refusals, and loop_violations. A loop
violation is a slot where (A1 only) the full request set creates no excess, no station is over capacity
before the slot, and yet the refused requests are not closed loops. The pre-registration proves this
cannot happen, so the count must stay 0 (check M3).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment as LSA
import paper2ma_env_attach as PA

BIG = 1e9
EPS = 1e-3
MODES = ('none', 'per_request', 'coordinated')


def landing_after(nbar, reqs, admitted):
    """Landing load after the admitted requests. reqs: list of (car, from, to)."""
    n = np.array(nbar, dtype=int).copy()
    for k in admitted:
        _, s, g = reqs[k]
        n[g] += 1
        n[s] -= 1
    return n


def creates_excess(nbar, reqs, admitted, cap):
    nb = np.array(nbar, dtype=int)
    return bool((landing_after(nb, reqs, admitted) > np.maximum(nb, cap)).any())


def admit_per_request(reqs, nbar, cap, order):
    """A1. order: request indices, strongest first. Returns the sorted admitted indices."""
    n = np.array(nbar, dtype=int).copy()
    pending = list(order)
    admitted = []
    changed = True
    while changed and pending:
        changed = False
        keep = []
        for k in pending:
            _, s, g = reqs[k]
            if n[g] < cap:
                n[g] += 1
                n[s] -= 1
                admitted.append(k)
                changed = True
            else:
                keep.append(k)
        pending = keep
    return sorted(admitted)


def admit_coordinated(reqs, nbar, cap, order):
    """A2. The whole set if it creates no landing excess; otherwise the largest subset that creates
    none, ties broken towards stronger requests (order = strongest first). Exact."""
    allk = list(range(len(reqs)))
    if not reqs or not creates_excess(nbar, reqs, allk, cap):
        return allk
    nb = np.array(nbar, dtype=int)
    G = len(nb)
    req_at = np.zeros(G, dtype=int)
    for _, s, _ in reqs:
        req_at[s] += 1
    room = np.maximum(nb, cap) - (nb - req_at)        # places left for the requesting cars at each station
    slot_g = np.array([g for g in range(G) for _ in range(int(room[g]))], dtype=int)
    rank = {k: r for r, k in enumerate(order)}
    nr = len(reqs)
    cost = np.full((nr, len(slot_g)), BIG)
    for k, (_, s, g) in enumerate(reqs):
        w = 1.0 - rank[k] / nr                          # strongest request: w = 1
        cost[k, slot_g == s] = 0.0                      # refused: stays
        cost[k, slot_g == g] = -(1.0 + EPS * w)         # admitted: moves
    r, c = LSA(cost)
    if cost[r, c].max() >= BIG / 2:
        raise RuntimeError('no feasible arrangement (cannot happen: refusing everything is feasible)')
    adm = sorted(int(k) for k, j in zip(r, c) if slot_g[j] == reqs[k][2])
    if creates_excess(nbar, reqs, adm, cap):
        raise RuntimeError('coordinated admission produced landing excess')
    return adm


def refused_is_closed_loops(reqs, refused, G):
    bal = np.zeros(G, dtype=int)
    for k in refused:
        _, s, g = reqs[k]
        bal[g] += 1
        bal[s] -= 1
    return bool((bal == 0).all())


class AdmissionEnv(PA.CapacityAttachEnv):
    ADMIT = 'none'

    def __init__(self, *args, **kwargs):
        self.adm_log = []
        super().__init__(*args, **kwargs)

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.adm_log.append(dict(requests=0, refused=0, loop_violations=0))
        return out

    def _requests(self, action, ho_targets):
        """The handovers the base step would start this slot, as (car, from, to)."""
        fr = self.frames[self.t]
        reqs = []
        for i in range(self.M):
            if not fr['active'][i] or self._ho_timer[i] > 0 or int(action[i, 0]) != 1:
                continue
            if ho_targets is not None and int(ho_targets[i]) >= 0:
                cand = int(ho_targets[i])
            else:                                       # the base step's own choice, same expression
                cand = int(np.argmax([self._desired_dbm(i, g) if g != self.serv[i] else -1e9
                                      for g in range(self.n_gnb)]))
            if cand != int(self.serv[i]):
                reqs.append((i, int(self.serv[i]), cand))
        return reqs

    def step(self, action, ho_targets=None):
        if self.ADMIT not in MODES:
            raise ValueError(self.ADMIT)
        if self.ADMIT == 'none':
            return super().step(action, ho_targets)
        action = np.array(action).reshape(self.M, 3).copy()
        fr = self.frames[self.t]
        cap = int(self.gnb_cap)
        nbar = np.zeros(self.n_gnb, dtype=int)
        for j in range(self.M):
            if fr['active'][j]:
                nbar[int(self._ho_target[j]) if self._ho_timer[j] > 0 else int(self.serv[j])] += 1
        reqs = self._requests(action, ho_targets)
        if reqs:
            order = sorted(range(len(reqs)),
                           key=lambda k: (-self._desired_dbm(reqs[k][0], reqs[k][2]), reqs[k][0]))
            if self.ADMIT == 'per_request':
                adm = admit_per_request(reqs, nbar, cap, order)
            else:
                adm = admit_coordinated(reqs, nbar, cap, order)
            a_set = set(adm)
            refused = [k for k in range(len(reqs)) if k not in a_set]
            for k in refused:
                action[reqs[k][0], 0] = 0
            rec = self.adm_log[-1]
            rec['requests'] += len(reqs)
            rec['refused'] += len(refused)
            if (self.ADMIT == 'per_request' and refused and (nbar <= cap).all()
                    and not creates_excess(nbar, reqs, list(range(len(reqs))), cap)
                    and not refused_is_closed_loops(reqs, refused, self.n_gnb)):
                rec['loop_violations'] += 1
        return super().step(action, ho_targets)
