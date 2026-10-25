#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_controller.py — Paper-2 controller: branched QR-DQN + CVaR + optional LSTM, with
n-step prioritized sequence replay, trained on the training scenes and evaluated on held-out
scenes under common random numbers.

Architecture:
  obs(11) -> FC encoder -> {LSTM | feed-forward} -> per-branch DUELING QR-DQN heads emitting N
  quantiles per action, for the branched action {handover(2), power(3), spectrum(3)}.
Risk-sensitive control: action selection maximises CVaR_alpha of the return quantiles (the worst
alpha fraction of outcomes — the deep-blockage tail), NOT the mean. Loss = quantile-Huber (QR-DQN)
against an n-step Double-DQN target. This realises cliff -> tail -> CVaR: because deep blockage is
a tail event, optimise the tail.

R3 REVISION — four changes, all of which alter the reported numbers:

  C1  PRIORITIZED REPLAY IS NOW REAL.  The previous loop learned from `buf[-8:]`, the eight most
      recently collected episodes in collection order: a sliding on-policy window with no buffer,
      no priorities and no importance-sampling correction, while the manuscript claimed n-step
      prioritized experience replay.  Training now draws IS-weighted minibatches from
      paper2_per.SequencePER and writes back per-sequence TD errors as priorities.

  C2  TRUNCATION IS NO LONGER TREATED AS TERMINATION.  When the n-step window reached the end of
      an episode the target was set to the undiscounted partial return regardless of WHY the
      episode ended.  For episodes that ended at the time limit (the scene ran out of frames)
      that silently teaches the agent that the world stops — a value bias against surviving.  The
      target now bootstraps from the final state whenever the episode was truncated rather than
      terminated by sustained RLF.

  C3  EVALUATION IS OUT-OF-SAMPLE AND PAIRED.  Training uses `split='train'` (30 scenes) and every
      reported metric is measured on `split='test'` (15 scenes never seen in training), with
      reset seeds EVAL_SEED_BASE + i shared by every configuration, so all configurations are
      compared on byte-identical scenarios (common random numbers).  In-sample performance is
      also measured, so the generalisation gap can be reported rather than hidden.

  C4  RESOLUTION.  The RLF rate was estimated from 60 rollouts, i.e. it could only take multiples
      of 1.67 percentage points.  The default is now 500 rollouts.

Also added: per-decision inference latency measurement (single observation, batch 1, one CPU
thread), for comparison against the O-RAN Near-RT RIC 10 ms - 1 s control budget.

INTEGRITY: standard algorithms on the real-geometry environment; nothing fabricated.
"""
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from paper2_per import SequencePER

BRANCHES = [2, 3, 3]        # handover, power, spectrum

# --- pin-release (added 21 Sep 2026) ---------------------------------------------------------
# Holds the POWER branch at its no-change action for the first PIN_EPISODES training episodes.
# The environment resets pt to PT_MAX every episode, so holding keeps power at maximum
# throughout the pinned phase. The forced action is the one RECORDED in the transition, so
# replay stays honest: the agent is not told that a decrease did nothing.
# Off by default (PIN_EPISODES = 0 -> PIN_ON never becomes True -> one bool test per step).
PIN_EPISODES = 0
PIN_ACTION = None
PIN_ON = False
N_QUANT = 32
# --- FD-QR, added 31 Aug 2026. CVaR-constrained feasibility selection over the branched
# action space. Read at MODULE SCOPE and OFF unless P2_FDQR=1, so every path below is
# bypassed by default and the published controller reproduces bit-for-bit.
FDQR = os.environ.get('P2_FDQR', '0') == '1'
# PREREG_07: EXACT joint selection. Enumerates all prod(BRANCHES) joint actions and scores
# each with the LAST branch's cost head at a complete prefix, which the decomposition theory
# states is exact on the joint action. This is the CEILING any factored rule could reach; it
# is affordable only because the space is 18 here. OFF unless P2_FDQR_EXACT=1, and every path
# below is untouched when off.
FDQR_EXACT = os.environ.get('P2_FDQR_EXACT', '0') == '1'
# Budget on CVaR_alpha of the discounted per-step outage cost. 2.0 sits between the two
# measured basins: with gamma=0.99 a sustained outage rate p gives cost ~ 100p, so the
# high-reliability basin (0.96%) scores ~0.96 and the low basin (8.90%) ~8.9.
FDQR_D = float(os.environ.get('P2_FDQR_D', '2.0'))
# Per-run QUANTILE budget. The absolute d is not defensible: the learned cost SCALE differs by
# run (median 0.997 on seed 14 vs 0.880 on seed 19), so a fixed d binds on 15% of decisions in
# one run and under 1% in another. A quantile of the run's OWN observed cost-CVaR distribution
# is scale-free and binds at a controlled rate everywhere. Unset -> absolute d, unchanged.
FDQR_Q = os.environ.get('P2_FDQR_Q')
FDQR_Q = float(FDQR_Q) if FDQR_Q is not None else None
# --- stage18 Part B control, 14 Sep 2026. Matched RANDOM veto: with this probability the branch's
# action is drawn uniformly from the NON-greedy actions, ignoring the cost head. The cost head is
# still built, still trained and still updates the reservoir, so the auxiliary half is held
# constant and the only varying quantity is whether the substituted action is cost-informed or
# arbitrary. 0.0 -> off, and every path below is bypassed unchanged.
FDQR_RAND = float(os.environ.get('P2_FDQR_RAND', '0') or 0)
# --- stage19 Part C control, 14 Sep 2026. Cost as a PREFERENCE rather than a veto: the branch takes
# argmax(r - lam*c), with lam adapted online to hold the decision-change rate at this value -- the
# same rate the veto binds at. B0 and C1 then differ only in HOW the estimate enters the choice,
# not in how often it changes one. 0.0 -> off, and the existing chain is reached unchanged.
FDQR_PEN = float(os.environ.get('P2_FDQR_PEN', '0') or 0)
PEN_GAIN = 50.0        # multiplicative feedback gain on the rate error; fixed before any run
PEN_STEP = 1.0         # max |log-step| per adaptation window
# --- stage20 Part C' control, 14 Sep 2026 (amended same day, see stage20/AMENDMENT1). Cost as
# a PRICE on exactly the veto's flagged set: the branch takes argmax(r - lam_eff*1[c > thr]),
# where thr is B0's own per-run q-threshold, computed by the same code on the same schedule.
# The flagged set is therefore B0's BY CONSTRUCTION and only the CONSEQUENCE differs -- B0
# removes a flagged action, this charges it. lam_eff = this value x the median OVERRIDE GAP:
# the return the hard veto gives up when it binds. Amendment 1: the unconditional return
# spread was the wrong denominator, because bind opportunities are precisely the high-spread
# decisions, so a price set at the typical spread is too small where it is actually charged
# (smoke test: override 0.92, near-vacuous). It stays a descriptive statistic of the return
# head with no target, no error signal, no gain and no clip -- the property the stage19
# rate-matched controller lacked, and why that one diverged over eight orders of magnitude.
# 0.0 -> off, and the existing chain is reached unchanged.
FDQR_SOFT = float(os.environ.get('P2_FDQR_SOFT', '0') or 0)
GAMMA = 0.99
NSTEP = 3
# stage13 (18 Sep 2026): delay-matched backup on the handover branch. Defaults reproduce
# the previous behaviour exactly; DELAY_BACKUP=False means NSTEP is used everywhere.
DELAY_BACKUP = False
HO_LAT = 2
CVAR_ALPHA = 0.25          # optimise the worst 25% (the deep-blockage tail)
EVAL_SEED_BASE = 900_000_000   # fixed forever: every configuration evaluates on the same scenarios


class BranchedQRDQN(nn.Module):
    def __init__(self, obs_dim, branches=BRANCHES, n_quant=N_QUANT, hidden=128, lstm_hidden=128, use_lstm=True,
                 fdqr=False):
        # obs_dim is deliberately required: it is read from env.observation_space at every call
        # site, so the network can never silently disagree with the environment it trains on.
        super().__init__()
        self.branches, self.n_quant, self.use_lstm = branches, n_quant, use_lstm
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if use_lstm:
            self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        else:                                          # memory ablation: feed-forward, no recurrence
            self.ff = nn.Sequential(nn.Linear(hidden, lstm_hidden), nn.ReLU())
        self.val = nn.ModuleList([nn.Linear(lstm_hidden, n_quant) for _ in branches])
        self.adv = nn.ModuleList([nn.Linear(lstm_hidden, nb * n_quant) for nb in branches])
        # FD-QR cost heads. Constructed ONLY when enabled: creating them unconditionally would
        # draw from the RNG during init and the default path would stop being bit-identical.
        self.fdqr = fdqr
        if fdqr:
            self.pref_dim = [int(sum(branches[:i])) for i in range(len(branches))]
            self.cval = nn.ModuleList([nn.Linear(lstm_hidden + p, n_quant) for p in self.pref_dim])
            self.cadv = nn.ModuleList([nn.Linear(lstm_hidden + p, nb * n_quant)
                                       for p, nb in zip(self.pref_dim, branches)])

    def trunk(self, x, hidden=None):
        """Shared encoder + recurrence. Split out so the cost heads can reuse it without a
        second forward pass. The operations and their order are unchanged from the original
        forward(), which is what keeps the default path bit-identical."""
        h = self.enc(x)
        if self.use_lstm:
            out, hidden = self.lstm(h, hidden)
        else:
            out, hidden = self.ff(h), None
        return out, hidden

    def cost_quantiles(self, out, i, prefix):
        """Cost quantiles for branch i, conditioned on the already-committed prefix actions.

        prefix: (B,T,sum(branches[:i])) one-hot of branches 0..i-1, or None for branch 0.
        Branch B-1 therefore conditions on a COMPLETE assignment of every other branch, so its
        feasibility test is exact on the joint action -- that is the whole point of Prop 3.
        """
        z = out if prefix is None else torch.cat([out, prefix], dim=-1)
        B, T, _ = out.shape
        v = self.cval[i](z).view(B, T, 1, self.n_quant)
        a = self.cadv[i](z).view(B, T, self.branches[i], self.n_quant)
        return v + a - a.mean(dim=2, keepdim=True)

    def heads_from_trunk(self, out):
        """Return-quantile heads applied to an already-computed trunk output."""
        B, T, _ = out.shape
        qs = []
        for i, nb in enumerate(self.branches):
            v = self.val[i](out).view(B, T, 1, self.n_quant)
            a = self.adv[i](out).view(B, T, nb, self.n_quant)
            qs.append(v + a - a.mean(dim=2, keepdim=True))
        return qs

    def forward(self, x, hidden=None):
        """x: (B,T,obs). Returns list of per-branch quantiles (B,T,n_act_b,n_quant) and LSTM hidden."""
        out, hidden = self.trunk(x, hidden)
        B, T, _ = out.shape
        qs = []
        for i, nb in enumerate(self.branches):
            v = self.val[i](out).view(B, T, 1, self.n_quant)
            a = self.adv[i](out).view(B, T, nb, self.n_quant)
            qs.append(v + a - a.mean(dim=2, keepdim=True))
        return qs, hidden


def cvar_scores(q_branch, alpha):
    """q_branch: (..., n_act, n_quant) -> CVaR_alpha per action (mean of lowest alpha-fraction quantiles)."""
    k = max(1, int(round(alpha * q_branch.shape[-1])))
    low, _ = torch.sort(q_branch, dim=-1)
    return low[..., :k].mean(dim=-1)


def cvar_cost_scores(c_branch, alpha):
    """CVaR_alpha of a COST distribution: mean of the HIGHEST alpha-fraction of quantiles.

    cvar_scores() takes the LOWEST fraction, which is the risk-averse view of a RETURN. For a
    cost the bad tail is the upper one, so reusing cvar_scores() here scores each action by its
    OPTIMISTIC cost -- the opposite of what a safety constraint needs. That defect made the
    31 Aug FD-QR run inert: the filter compared a best-case cost against a worst-case budget and
    never excluded anything in 459,162 decisions.
    """
    k = max(1, int(round(alpha * c_branch.shape[-1])))
    hi, _ = torch.sort(c_branch, dim=-1)
    return hi[..., -k:].mean(dim=-1)


def greedy_actions(qs, alpha):
    """Per-branch CVaR-greedy actions from quantiles qs[i]: (B,T,n_act,n_quant)."""
    return [cvar_scores(q, alpha).argmax(dim=-1) for q in qs]


def quantile_huber(pred, target, taus, kappa=1.0):
    """pred,target: (B,n_quant). Quantile-Huber loss (QR-DQN)."""
    u = target.unsqueeze(1) - pred.unsqueeze(2)                     # (B, Npred, Ntar)
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    rho = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs() * huber / kappa
    return rho.sum(dim=1).mean(dim=1).mean()


class Agent:
    def __init__(self, obs_dim, alpha=CVAR_ALPHA, lr=5e-4, device='cpu', use_lstm=True):
        self.net = BranchedQRDQN(obs_dim, use_lstm=use_lstm, fdqr=FDQR).to(device)
        self.tgt = BranchedQRDQN(obs_dim, use_lstm=use_lstm, fdqr=FDQR).to(device); self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.taus = torch.tensor([(i + 0.5) / N_QUANT for i in range(N_QUANT)], device=device)
        self.alpha, self.device = alpha, device
        # FD-QR binding instrumentation. The FD-QR result of 31 Aug could not distinguish
        # "the constraint was active but ineffective" from "the constraint never bound at all",
        # because nothing recorded whether the feasible set ever excluded the greedy action.
        # These counters answer that directly and cost one comparison per branch per decision.
        self._fd_dec = 0        # decisions taken (branch-steps)
        self._fd_bind = 0       # ... where the constraint CHANGED the chosen action
        self._fd_excl = 0       # actions excluded as infeasible, summed
        self._fd_empty = 0      # ... where NO action was feasible (fallback fired)
        self._fd_c_n = 0        # cost-CVaR samples, for the range of the constraint statistic
        self._fd_c_sum = 0.0
        self._fd_c_min = float('inf')
        self._fd_c_max = float('-inf')
        # Reservoir of recent cost-CVaR values, for the per-run quantile budget. Bounded, so
        # the threshold tracks the critic as it learns instead of being fixed by early noise.
        self._fd_res = []
        self._fd_res_cap = 20000
        self._fd_thr = float('inf')      # no exclusion until the reservoir has warmed up
        self._fd_warm = 2000
        self._fd_every = 200             # decisions between threshold recomputations
        self._fd_since = 10 ** 9         # force a recompute as soon as the reservoir is warm
        # Dedicated RNG for the Part B control, built only when the control is on so the default
        # path draws no extra RNG and stays bit-identical. Seeded from the global stream, which the
        # driver has already seeded from the run seed, so the arm is reproducible.
        self._fd_rng = (np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
                        if FDQR_RAND > 0.0 else None)
        # stage19 Part C: online lam for the preference control. lam stays 0 until the cost
        # reservoir is warm, mirroring B0, where the threshold is +inf and nothing binds before then.
        self._pen_lam = 0.0
        self._pen_win = 0
        self._pen_chg = 0
        # stage20 Part C': bounded reservoir of OVERRIDE GAPS -- the return the hard veto
        # gives up each time it binds. That is the one quantity the price has to compete
        # with, so it is what lam_eff is denominated in; it also tracks the return scale as
        # the critic learns. No warm-up special case is needed: _fd_thr starts at +inf, so
        # nothing is flagged until the cost reservoir warms and the price term is identically
        # zero before then -- which is exactly what B0 does over the same span.
        self._sf_gap = []
        # Amendment 1a: 300, not the cost reservoir's 20000. Override opportunities are ~140x
        # rarer than cost samples, so 20000 never rolls over on a 1500-episode run and the
        # median is dragged down by early-training gaps -- the window would not be trailing at
        # all, which is the one thing the design requires of it.
        self._sf_cap = 300
        self._sf_warm = 100      # override-gap samples before lam_eff goes live
        self._sf_every = 50      # new opportunities between lam_eff recomputations
        self._sf_since = 10 ** 9
        self._sf_lam = 0.0
        self._sf_med = 0.0
        self._sf_opp = 0        # decisions where the HARD veto WOULD have bound (lam live)
        self._sf_over = 0       # ... where the price was paid and the flagged action kept

    @torch.no_grad()
    def _act_exact(self, out, qs):
        """PREREG_07 EXACT joint selection — the ceiling.

        Enumerate every joint action. Score each by the LAST branch's cost head conditioned on a
        COMPLETE prefix of the others, which the decomposition theory states is exact on the joint
        action. Keep those within budget; take the return-argmax over that feasible set, a joint
        action scored by the sum of its per-branch return CVaRs. Fall back to the cost-argmin joint
        action when nothing is feasible.

        Reservoir feed. The threshold must stay comparable to the sequential arm, whose reservoir
        receives one full cost vector per branch at the prefix it committed to. This does the same:
        after the joint action is chosen, the three cost vectors ALONG THE CHOSEN PATH are computed
        and fed. Nothing from the enumeration enters the reservoir, so the threshold population has
        the same structure as the sequential arm's.
        """
        import itertools
        dev = self.device
        rets = [cvar_scores(qs[i], self.alpha)[0, 0] for i in range(len(BRANCHES))]

        def prefix_of(j, upto):
            parts = []
            for k in range(upto):
                oh = torch.zeros(1, 1, BRANCHES[k], device=dev); oh[0, 0, j[k]] = 1.0
                parts.append(oh)
            return torch.cat(parts, dim=-1) if parts else None

        last = len(BRANCHES) - 1
        joints = list(itertools.product(*[range(nb) for nb in BRANCHES]))
        costs, scores = [], []
        for j in joints:
            cq = self.net.cost_quantiles(out, last, prefix_of(j, last))
            costs.append(float(cvar_cost_scores(cq, self.alpha)[0, 0][j[last]]))
            scores.append(float(sum(rets[i][j[i]] for i in range(len(BRANCHES)))))
        costs_t = torch.tensor(costs)
        thr = FDQR_D if FDQR_Q is None else self._fd_thr
        feas = costs_t <= thr
        if bool(feas.any()):
            pick = int(torch.tensor(scores).masked_fill(~feas, float('-inf')).argmax())
        else:
            pick = int(costs_t.argmin())
            self._fd_empty += 1
        a = list(joints[pick])

        # instrumentation and reservoir, mirroring the sequential arm branch for branch
        unconstrained = [int(r.argmax()) for r in rets]
        for i, nb in enumerate(BRANCHES):
            c = cvar_cost_scores(self.net.cost_quantiles(out, i, prefix_of(a, i)), self.alpha)[0, 0]
            self._fd_dec += 1
            self._fd_excl += int((c > thr).sum())
            cf = c.float()
            self._fd_c_n += int(cf.numel()); self._fd_c_sum += float(cf.sum())
            self._fd_c_min = min(self._fd_c_min, float(cf.min()))
            self._fd_c_max = max(self._fd_c_max, float(cf.max()))
            if a[i] != unconstrained[i]:
                self._fd_bind += 1
            if FDQR_Q is not None:
                self._fd_res.extend(c.tolist())
                if len(self._fd_res) > self._fd_res_cap:
                    del self._fd_res[:len(self._fd_res) - self._fd_res_cap]
                self._fd_since += 1
                if len(self._fd_res) >= self._fd_warm and self._fd_since >= self._fd_every:
                    self._fd_thr = float(np.quantile(np.asarray(self._fd_res), FDQR_Q))
                    self._fd_since = 0
        return a

    def act(self, obs, hidden, eps):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        if not FDQR:
            qs, hidden = self.net(x, hidden)
            if eps > 0.0 and np.random.random() < eps:
                a = [np.random.randint(nb) for nb in BRANCHES]
            else:
                a = [int(g[0, 0]) for g in greedy_actions(qs, self.alpha)]
            return a, hidden
        # ---- FD-QR: sequential feasibility-first selection -------------------------------
        # Branches are committed IN ORDER. Each branch's cost head conditions on the actions
        # already committed, so the last branch tests feasibility on the true joint action.
        # Parallel per-branch filtering is unsound here (see the Prop 2 counterexample and the
        # RIGM counterexamples of Qiu et al., NeurIPS 2023); the ordering is what repairs it.
        out, hidden = self.net.trunk(x, hidden)
        if eps > 0.0 and np.random.random() < eps:
            return [np.random.randint(nb) for nb in BRANCHES], hidden
        qs = self.net.heads_from_trunk(out)
        if FDQR_EXACT:
            return self._act_exact(out, qs), hidden
        a, prefix = [], None
        for i, nb in enumerate(BRANCHES):
            cq = self.net.cost_quantiles(out, i, prefix)             # (1,1,nb,nq)
            c = cvar_cost_scores(cq, self.alpha)[0, 0]               # (nb,) UPPER-tail CVaR of cost
            r = cvar_scores(qs[i], self.alpha)[0, 0]                 # (nb,) CVaR of return
            if FDQR_Q is None:
                thr = FDQR_D
            else:
                # update the reservoir with this decision's costs, then threshold at the quantile
                self._fd_res.extend(c.tolist())
                if len(self._fd_res) > self._fd_res_cap:
                    del self._fd_res[:len(self._fd_res) - self._fd_res_cap]
                # Recompute periodically, not every decision: np.quantile over the reservoir is
                # O(n log n) and at 8 calls per step it dominated the decision latency (2.9 ms
                # against a 0.37 ms baseline). The threshold moves slowly, so a stale one is fine.
                self._fd_since += 1
                if len(self._fd_res) >= self._fd_warm and self._fd_since >= self._fd_every:
                    self._fd_thr = float(np.quantile(np.asarray(self._fd_res), FDQR_Q))
                    self._fd_since = 0
                thr = self._fd_thr
            feas = c <= thr
            unconstrained = int(r.argmax())        # what the UNCONSTRAINED controller would pick
            self._fd_dec += 1
            self._fd_excl += int((~feas).sum())
            cf = c.float()
            self._fd_c_n += int(cf.numel()); self._fd_c_sum += float(cf.sum())
            self._fd_c_min = min(self._fd_c_min, float(cf.min()))
            self._fd_c_max = max(self._fd_c_max, float(cf.max()))
            if FDQR_SOFT > 0.0:
                # stage20 Part C': the flag is a PRICE, not a ban. Same flagged set as B0,
                # same threshold, same schedule -- only the consequence of being flagged.
                # A bind OPPORTUNITY: the best-returning action is flagged, but something
                # else is not -- exactly the decisions where B0's veto changes the action.
                opp = (not bool(feas[unconstrained])) and bool(feas.any())
                if opp:
                    self._sf_gap.append(float(
                        r[unconstrained] - r.masked_fill(~feas, float('-inf')).max()))
                    if len(self._sf_gap) > self._sf_cap:
                        del self._sf_gap[:len(self._sf_gap) - self._sf_cap]
                    self._sf_since += 1
                    if len(self._sf_gap) >= self._sf_warm and self._sf_since >= self._sf_every:
                        self._sf_med = float(np.median(np.asarray(self._sf_gap)))
                        self._sf_lam = FDQR_SOFT * self._sf_med
                        self._sf_since = 0
                ai = int((r - self._sf_lam * (~feas).to(r.dtype)).argmax())
                # Softness diagnostic -- MEASURED, not controlled, which is the difference
                # from stage19 that matters. Counted only once lam_eff is live, so the
                # calibration span (lam 0, hence trivially overriding) is excluded from it.
                if opp and self._sf_lam > 0.0:
                    self._sf_opp += 1
                    if ai == unconstrained:
                        self._sf_over += 1
            elif FDQR_PEN > 0.0:
                # stage19 Part C control: the estimate enters as a PREFERENCE. Same information and
                # (by adaptation) the same decision-change rate as the veto -- only the form differs.
                if len(self._fd_res) >= self._fd_warm:
                    ai = int((r - self._pen_lam * c).argmax())
                    self._pen_win += 1
                    if ai != unconstrained:
                        self._pen_chg += 1
                    if self._pen_win >= self._fd_every:
                        err = FDQR_PEN - self._pen_chg / self._pen_win
                        step = float(np.clip(PEN_GAIN * err, -PEN_STEP, PEN_STEP))
                        self._pen_lam = float(np.clip(max(self._pen_lam, 1e-3) * np.exp(step),
                                                      1e-4, 1e6))
                        self._pen_win = self._pen_chg = 0
                else:
                    ai = unconstrained
            elif FDQR_RAND > 0.0:
                # stage18 Part B control: selection ignores the cost head entirely. Every firing
                # changes the decision by construction, so the realised binding rate equals
                # FDQR_RAND -- the control is matched to B0 by design, not by tuning.
                if nb > 1 and self._fd_rng.random_sample() < FDQR_RAND:
                    alt = [k for k in range(nb) if k != unconstrained]
                    ai = int(alt[self._fd_rng.randint(len(alt))])
                else:
                    ai = unconstrained
            elif bool(feas.any()):
                masked = r.masked_fill(~feas, float('-inf'))
                ai = int(masked.argmax())
            else:
                self._fd_empty += 1
                ai = int(c.argmin())                                 # least-infeasible fallback
            if ai != unconstrained:
                self._fd_bind += 1                 # the constraint actually changed the decision
            a.append(ai)
            oh = torch.zeros(1, 1, nb, device=self.device); oh[0, 0, ai] = 1.0
            prefix = oh if prefix is None else torch.cat([prefix, oh], dim=-1)
        return a, hidden

    # ------------------------------------------------------------------ learning
    def _seq_loss(self, ep):
        """Quantile-Huber loss and mean |n-step TD error| for one episode (sequence).

        ep: dict(obs:(T+1,obs), acts:(T,3), rews:(T,), term:bool).
        `term` is True only when the episode ended in sustained RLF (a true absorbing state).  When
        it is False the episode ended at the scene time limit, so the n-step target must bootstrap
        from the final state instead of pretending the return stopped there (fix C2).
        """
        obs = torch.tensor(ep['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)   # (1,T+1,obs)
        acts = torch.tensor(ep['acts'], dtype=torch.long, device=self.device)                 # (T,3)
        rews = torch.tensor(ep['rews'], dtype=torch.float32, device=self.device)              # (T,)
        term = bool(ep.get('term', True))
        T = acts.shape[0]
        hostart = ep.get('ho_started', None)
        qs_all, _ = self.net(obs)                                    # per-branch (1,T+1,n_act,nq)
        with torch.no_grad():
            qs_tgt, _ = self.tgt(obs)
            qs_on, _ = self.net(obs)
            astar = greedy_actions(qs_on, self.alpha)                # Double-DQN: online picks a*
        loss, td = 0.0, 0.0
        for i, nb in enumerate(BRANCHES):
            pred = qs_all[i][0, torch.arange(T), acts[:, i]]         # (T, nq) chosen-action quantiles at s_t
            tgt = torch.zeros(T, N_QUANT, device=self.device)
            for t in range(T):
                # stage13: the handover branch actuates HO_LAT steps later, so for a
                # transition where a handover actually started, bootstrap from the LANDING
                # state. Power and spectrum actuate immediately and keep NSTEP.
                n_t = NSTEP
                if DELAY_BACKUP and i == 0 and hostart is not None and t < len(hostart) and bool(hostart[t]):
                    n_t = HO_LAT
                R = 0.0; g = 1.0; k = 0
                while k < n_t and t + k < T:
                    R += g * rews[t + k].item(); g *= GAMMA; k += 1
                if t + k < T:                                        # bootstrap from an interior state
                    tgt[t] = R + g * qs_tgt[i][0, t + k, astar[i][0, t + k]]
                elif not term:                                       # C2: time-limit truncation
                    tgt[t] = R + g * qs_tgt[i][0, T, astar[i][0, T]]
                else:                                                # true terminal (sustained RLF)
                    tgt[t] = R
            tgt = tgt.detach()
            loss = loss + quantile_huber(pred, tgt, self.taus)
            # priority signal: distributional TD error reduced to a scalar via the quantile means
            td = td + float((tgt.mean(dim=1) - pred.detach().mean(dim=1)).abs().mean())

        if FDQR and 'costs' in ep:
            # Cost critic. Evaluated ON-POLICY against the recorded actions: the constraint is
            # on the cost of the policy being followed, not on a greedy cost-minimiser.
            costs = torch.tensor(ep['costs'], dtype=torch.float32, device=self.device)
            last = torch.tensor([int(astar[i][0, T]) for i in range(len(BRANCHES))],
                                device=self.device).view(1, -1)
            afull = torch.cat([acts, last], dim=0)                   # (T+1, B), teacher forcing
            ohs = []
            for i, nb in enumerate(BRANCHES):
                oh = torch.zeros(1, T + 1, nb, device=self.device)
                oh[0, torch.arange(T + 1), afull[:, i]] = 1.0
                ohs.append(oh)
            out_on, _ = self.net.trunk(obs)
            with torch.no_grad():
                out_tg, _ = self.tgt.trunk(obs)
            for i, nb in enumerate(BRANCHES):
                prefix = None if i == 0 else torch.cat(ohs[:i], dim=-1)
                cq = self.net.cost_quantiles(out_on, i, prefix)
                with torch.no_grad():
                    cq_t = self.tgt.cost_quantiles(out_tg, i, prefix)
                cpred = cq[0, torch.arange(T), acts[:, i]]
                ctgt = torch.zeros(T, N_QUANT, device=self.device)
                for t in range(T):
                    R = 0.0; g = 1.0; k = 0
                    while k < NSTEP and t + k < T:
                        R += g * costs[t + k].item(); g *= GAMMA; k += 1
                    if t + k < T:
                        ctgt[t] = R + g * cq_t[0, t + k, afull[t + k, i]]
                    elif not term:
                        ctgt[t] = R + g * cq_t[0, T, afull[T, i]]
                    else:
                        ctgt[t] = R
                loss = loss + quantile_huber(cpred, ctgt.detach(), self.taus)
        return loss, td / len(BRANCHES)

    def learn(self, episodes, is_w=None):
        """One optimizer step on an importance-weighted minibatch of sequences.

        Returns (weighted mean loss, per-sequence |TD error|) so the caller can write the TD errors
        back to the replay buffer as priorities.
        """
        self.net.train()
        if is_w is None:
            is_w = np.ones(len(episodes), dtype=np.float64)
        is_w = np.asarray(is_w, dtype=np.float64)
        wsum = float(is_w.sum()) or 1.0
        self.opt.zero_grad()
        total, tds = 0.0, []
        for ep, w in zip(episodes, is_w):
            l, td = self._seq_loss(ep)
            (float(w) / wsum * l).backward()          # accumulate; graph freed per sequence
            total += float(l.detach()) * float(w) / wsum
            tds.append(td)
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return total, np.array(tds, dtype=np.float64)

    def sync(self):
        self.tgt.load_state_dict(self.net.state_dict())


def _sinr_out_db():
    """Outage threshold, read from the already-imported environment module."""
    return float(sys.modules['paper2_env'].SINR_OUT_DB)


def rollout(env, agent, eps, learn_buf=None, seed=None, latency=None, trace=None):
    """One episode. `seed` makes the scene and the whole fading realisation deterministic (CRN).

    `latency`, if a list, receives the wall-clock cost in milliseconds of every act() call.
    `trace`, if a list, receives one per-episode record of the actuator state (mean/final
    transmit power, mean sub-band, handover count). Added 28 Aug 2026 because the
    convergence basin is DEFINED by mean transmit power and it was never recorded per
    episode -- only per-episode return was, and return carries the basin only weakly
    (within-config AUC 0.714 at episode 500 against a 0.80 bar).
    Both are pure observers: default None reproduces the previous behaviour exactly,
    and neither consumes randomness, so seeds are unaffected.
    Returns (episode return, terminated) where terminated == sustained RLF.
    """
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    hidden = None
    O, Ac, Re = [obs], [], []
    Co = []                                    # FD-QR per-step cost: the link-outage indicator
    Hs = []                                    # stage13: did the env actually START a handover this step
    pts, rbs, hos = [], [], 0
    done = trunc = False
    while not (done or trunc):
        if latency is None:
            a, hidden = agent.act(obs, hidden, eps)
        else:
            t0 = time.perf_counter()
            a, hidden = agent.act(obs, hidden, eps)
            latency.append((time.perf_counter() - t0) * 1000.0)
        if PIN_ON:                      # pin-release: power branch held, see PIN_EPISODES
            a = list(a); a[1] = PIN_ACTION
        obs, r, done, trunc, info = env.step(a)
        Hs.append(bool(info.get('ho', False)))   # pure observer; consumes no randomness
        if trace is not None:
            pts.append(float(env.pt)); rbs.append(float(env.rb)); hos += int(a[0] == 1)
        O.append(obs); Ac.append(a); Re.append(r)
        if FDQR:
            # cost = 1 while the link is below the outage threshold. Derived from info the
            # environment already returns, so paper2_env.py needs no change at all.
            Co.append(1.0 if float(info['sinr']) < _sinr_out_db() else 0.0)
    ep = dict(obs=np.array(O, dtype=np.float32), acts=np.array(Ac), rews=np.array(Re, dtype=np.float32),
              term=bool(done), ho_started=np.array(Hs, dtype=bool))
    if FDQR:
        ep['costs'] = np.array(Co, dtype=np.float32)
    if learn_buf is not None:
        (learn_buf.add if hasattr(learn_buf, 'add') else learn_buf.append)(ep)
    if trace is not None:
        trace.append(dict(pt_mean=float(np.mean(pts)) if pts else float('nan'),
                          pt_last=float(pts[-1]) if pts else float('nan'),
                          rb_mean=float(np.mean(rbs)) if rbs else float('nan'),
                          ho=int(hos), term=bool(done), steps=len(Re)))
    return sum(Re), bool(done)                # done == terminated == sustained RLF this episode


# ---------------------------------------------------------------------- evaluation
def evaluate(env, agent, n, seed_base=EVAL_SEED_BASE, latency=None):
    """Greedy evaluation over n episodes with common random numbers.

    Episode i always uses reset seed `seed_base + i`, so every configuration and every training
    seed is scored on exactly the same n scenarios with the same fading realisations. Returns a
    per-episode DataFrame — the paired statistics are computed from it downstream.
    """
    import pandas as pd
    rows = []
    for i in range(n):
        obs, info = env.reset(seed=seed_base + i)
        hidden, R, steps = None, 0.0, 0
        pts, rbs, nho = [], [], 0
        done = trunc = False
        while not (done or trunc):
            if latency is None:
                a, hidden = agent.act(obs, hidden, 0.0)
            else:
                t0 = time.perf_counter()
                a, hidden = agent.act(obs, hidden, 0.0)
                latency.append((time.perf_counter() - t0) * 1000.0)
            obs, r, done, trunc, i2 = env.step(a)
            R += r; steps += 1
            pts.append(i2['pt']); rbs.append(i2['rb']); nho += int(i2['ho'])
        rows.append(dict(idx=i, eval_seed=seed_base + i, scene=info['scene'],
                         ret=round(float(R), 6), rlf=int(bool(done)), steps=steps,
                         mean_pt=round(float(np.mean(pts)), 4), mean_rb=round(float(np.mean(rbs)), 4),
                         n_ho=nho))
    return pd.DataFrame(rows)


def summarize(df, tag=''):
    r = df['ret'].to_numpy(dtype=float)
    k = max(1, int(round(0.25 * len(r))))
    p = (tag + '_') if tag else ''
    out = {p + 'return_mean': round(float(r.mean()), 4),
           p + 'return_std': round(float(r.std(ddof=1)), 4) if len(r) > 1 else 0.0,
           p + 'cvar25': round(float(np.sort(r)[:k].mean()), 4),
           p + 'rlf_rate': round(float(100.0 * df['rlf'].mean()), 4),
           p + 'n_eval': int(len(df))}
    # resource efficiency, measured rather than asserted: what the policy actually spends
    for c, nm in (('mean_pt', 'pt_dbm'), ('mean_rb', 'rb'), ('n_ho', 'ho_per_ep'), ('steps', 'steps')):
        if c in df.columns:
            out[p + nm] = round(float(df[c].mean()), 4)
    return out


def latency_stats(lat):
    a = np.asarray(lat, dtype=float)
    if a.size == 0:
        return {}
    return {'lat_mean_ms': round(float(a.mean()), 4), 'lat_p50_ms': round(float(np.percentile(a, 50)), 4),
            'lat_p95_ms': round(float(np.percentile(a, 95)), 4), 'lat_p99_ms': round(float(np.percentile(a, 99)), 4),
            'lat_max_ms': round(float(a.max()), 4), 'lat_n': int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--p1', default=None)
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--alpha', type=float, default=CVAR_ALPHA,
                    help='CVaR level; RISK ablation: --alpha 1.0 = expected-value (mean) Dn-DQN')
    ap.add_argument('--ff', action='store_true', help='MEMORY ablation: feed-forward (no LSTM)')
    ap.add_argument('--no_ant', action='store_true', help='ANTICIPATION ablation: zero the N1 features')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None, help='save trained model (.pt) + return history + per-episode eval + metrics')
    ap.add_argument('--eval_n', type=int, default=500, help='held-out evaluation rollouts (common random numbers)')
    ap.add_argument('--eval_n_train', type=int, default=200, help='in-sample rollouts, for the generalisation gap')
    ap.add_argument('--seed', type=int, default=0, help='random seed (multi-seed runs)')
    ap.add_argument('--nstep', type=int, default=3, help='stage13: n-step return horizon')
    ap.add_argument('--ho_latency', type=int, default=2, help='stage13: handover actuation latency (steps)')
    ap.add_argument('--delay_backup', action='store_true',
                    help='stage13: delay-matched backup on the handover branch (horizon = ho_latency)')
    # --- prioritized replay
    ap.add_argument('--per_capacity', type=int, default=500, help='replay capacity in episodes')
    ap.add_argument('--per_alpha', type=float, default=0.6)
    ap.add_argument('--per_beta0', type=float, default=0.4)
    ap.add_argument('--batch', type=int, default=8, help='sequences per gradient step')
    ap.add_argument('--warmup', type=int, default=32, help='episodes collected before learning starts')
    ap.add_argument('--updates_per_ep', type=int, default=2, help='gradient steps per collected episode')
    ap.add_argument('--target_sync', type=int, default=0, help='target-net sync period in episodes (0 = episodes//6)')
    ap.add_argument('--no_per', action='store_true', help='replay ablation: uniform sampling, no IS weights')
    ap.add_argument('--threads', type=int, default=1, help='torch CPU threads (1 = deployment condition)')
    ap.add_argument('--fdqr_soft', type=float, default=None,
                    help="stage20 Part C' control: PRICE the veto's flagged set instead of\n"
                         'removing it. argmax(r - lam*1[c>thr]) with lam = this multiple of\n'
                         'the median OVERRIDE GAP -- the return the veto gives up when it\n'
                         'binds. Identical flagged set to the veto; only\n'
                         'the consequence differs. 1.0 = one typical spread.')
    ap.add_argument('--fdqr_pen', type=float, default=None,
                    help='stage19 Part C control: cost as a PREFERENCE, argmax(r - lam*c), with\n'
                         'lam adapted online to hold the decision-change rate at this value.')
    ap.add_argument('--fdqr_rand', type=float, default=None,
                    help='stage18 Part B control: matched RANDOM veto at this probability. The\n'
                         'cost head still trains and still updates the reservoir; only selection\n'
                         'ignores it. Every firing changes the decision, so the realised binding\n'
                         'rate equals this value.')
    ap.add_argument('--fdqr', action='store_true',
                    help='FD-QR: CVaR-constrained sequential feasibility selection over the\n'
                         'branched action space, with per-branch cost quantile heads.')
    ap.add_argument('--fdqr_d', type=float, default=None,
                    help='ABSOLUTE CVaR budget on the discounted outage cost (default 2.0).')
    ap.add_argument('--fdqr_q', type=float, default=None,
                    help='PER-RUN QUANTILE budget in (0,1): exclude actions above this quantile\n'
                         'of the cost CVaR the run has itself observed. Scale-free; overrides\n'
                         '--fdqr_d. e.g. 0.80 excludes roughly the worst fifth of actions.')
    ap.add_argument('--pin_release', type=int, default=0,
                    help='hold transmit power at maximum for the first N training episodes, '
                         'then release it. 0 = off (default, fully inert).')
    ap.add_argument('--pwr_act', default=None, choices=['inc3', 'inc7', 'direct8', 'hybrid11', 'hold30'],
                    help='transmit-power actuator parameterisation. All modes share the\n'
                         'same reachable power set {23..30} dBm; they differ only in how\n'
                         'many steps it takes to reach a level. Mechanism test, 30 Aug 2026.')
    ap.add_argument('--pwr_form', default=None, choices=['linear', 'quad', 'sqrt'],
                    help='shape of the transmit-power penalty in the reward. All forms are\n'
                         'normalised so the rail policies price identically; only interior\n'
                         'power levels are repriced. Mechanism test, 30 Aug 2026.')
    ap.add_argument('--w_pwr', type=float, default=None,
                    help='override the reward transmit-power penalty weight (paper2_env.W_PWR, default 0.15). '
                         'Reward-weight sensitivity sweep, 30 Aug 2026.')
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.fdqr_soft is not None:
        if not (0.0 <= a.fdqr_soft <= 100.0):
            sys.exit('ABORT: --fdqr_soft must be in [0,100]; got %r' % a.fdqr_soft)
        globals()['FDQR_SOFT'] = float(a.fdqr_soft)
    if a.fdqr_pen is not None:
        if not (0.0 <= a.fdqr_pen < 1.0):
            sys.exit('ABORT: --fdqr_pen must be in [0,1); got %r' % a.fdqr_pen)
        globals()['FDQR_PEN'] = float(a.fdqr_pen)
    if a.fdqr_rand is not None:
        if not (0.0 <= a.fdqr_rand < 1.0):
            sys.exit('ABORT: --fdqr_rand must be in [0,1); got %r' % a.fdqr_rand)
        globals()['FDQR_RAND'] = float(a.fdqr_rand)
    if a.fdqr:
        globals()['FDQR'] = True
    if a.fdqr_d is not None:
        globals()['FDQR_D'] = float(a.fdqr_d)
    if a.fdqr_q is not None:
        if not (0.0 < a.fdqr_q < 1.0):
            sys.exit('ABORT: --fdqr_q must be strictly between 0 and 1; got %r' % a.fdqr_q)
        globals()['FDQR_Q'] = float(a.fdqr_q)
    if a.p1:
        sys.path.insert(0, a.p1)
    # set BEFORE the env is imported: paper2_env reads P2_W_PWR at module scope
    if a.pwr_act is not None:
        os.environ['P2_PWR_ACT'] = a.pwr_act
    if a.pwr_form is not None:
        os.environ['P2_PWR_FORM'] = a.pwr_form
    if a.w_pwr is not None:
        os.environ['P2_W_PWR'] = repr(float(a.w_pwr))
    import pandas as pd, paper2_env as PE
    # G1: fail in the first second, not after six minutes of training on the wrong reward
    # G1a: the actuator must have taken, AND the controller's branch widths must follow it.
    if a.pwr_act is not None and PE._PWR_ACT != a.pwr_act:
        sys.exit(f'ABORT: asked for pwr_act={a.pwr_act!r} but paper2_env._PWR_ACT is '
                 f'{PE._PWR_ACT!r}. The override did not take -- refusing to train.')
    BRANCHES[1] = int(PE.PWR_N_ACT)          # in-place: paper2ma_train binds the same list
    BRANCHES[2] = int(getattr(PE, 'SPEC_N_ACT', 3))   # PREREG_05: hold66r reduces the head too
    if a.pwr_form is not None and PE._PWR_FORM != a.pwr_form:
        sys.exit(f'ABORT: asked for pwr_form={a.pwr_form!r} but paper2_env._PWR_FORM is '
                 f'{PE._PWR_FORM!r}. The override did not take -- refusing to train.')
    if a.w_pwr is not None and abs(PE.W_PWR - a.w_pwr) > 1e-12:
        sys.exit(f'ABORT: asked for W_PWR={a.w_pwr!r} but paper2_env.W_PWR is {PE.W_PWR!r}. '
                 'The override did not take -- refusing to train.')
    pool = pd.read_csv(a.pool)
    # C3: train on the training scenes only; every reported number comes from the held-out scenes
    globals()['NSTEP'] = int(a.nstep)
    globals()['HO_LAT'] = int(a.ho_latency)
    globals()['DELAY_BACKUP'] = bool(a.delay_backup)
    print(f"stage13: nstep={a.nstep} ho_latency={a.ho_latency} delay_backup={a.delay_backup}", flush=True)
    env = PE.MultiRSUBlockageEnv(pool, seed=a.seed, mask_anticipation=a.no_ant, split='train', ho_latency=a.ho_latency)
    env_te = PE.MultiRSUBlockageEnv(pool, seed=a.seed + 10_000, mask_anticipation=a.no_ant, split='test', ho_latency=a.ho_latency)
    # G1b: the network's branch widths must equal the environment's action space. A mismatch
    # here would train a head that cannot address every action, silently and for six minutes.
    _nvec = [int(x) for x in env.action_space.nvec]
    if _nvec != list(BRANCHES):
        sys.exit(f'ABORT: env action space {_nvec} != controller BRANCHES {list(BRANCHES)}.')
    if [int(x) for x in env_te.action_space.nvec] != _nvec:
        sys.exit('ABORT: train and held-out envs disagree on the action space.')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    agent = Agent(obs_dim=env.observation_space.shape[0], alpha=a.alpha, use_lstm=not a.ff)
    # G3: if FD-QR was asked for, the cost heads must actually exist, and there must be one
    # per branch with the right prefix width. Fail in the first second, not after six minutes.
    if FDQR:
        if not getattr(agent.net, 'fdqr', False) or not hasattr(agent.net, 'cval'):
            sys.exit('ABORT: --fdqr requested but the network has no cost heads.')
        if len(agent.net.cval) != len(BRANCHES):
            sys.exit('ABORT: %d cost heads for %d branches.' % (len(agent.net.cval), len(BRANCHES)))
        want = [int(sum(BRANCHES[:i])) for i in range(len(BRANCHES))]
        if list(agent.net.pref_dim) != want:
            sys.exit('ABORT: prefix widths %s != %s' % (list(agent.net.pref_dim), want))
        print('FD-QR ON | cost heads %s | prefix widths %s | budget d=%g'
              % ([c.out_features for c in agent.net.cval], want, FDQR_D))
    buf = SequencePER(capacity=a.per_capacity, alpha=(0.0 if a.no_per else a.per_alpha),
                      beta0=(1.0 if a.no_per else a.per_beta0), beta_steps=max(a.episodes, 1),
                      rng=np.random.default_rng(a.seed + 777))
    sync_every = a.target_sync if a.target_sync > 0 else max(a.episodes // 6, 1)
    print(f"config: {'FF' if a.ff else 'LSTM'} | {'mean' if a.alpha >= 0.999 else f'CVaR{a.alpha}'} | "
          f"anticipation={'OFF' if a.no_ant else 'ON'} | replay={'uniform' if a.no_per else 'PER'} "
          f"(cap {a.per_capacity}, batch {a.batch}, {a.updates_per_ep} upd/ep, warmup {a.warmup})")
    print(f"reward weights: W_REL={PE.W_REL} RLF_PEN={PE.RLF_PEN} W_PWR={PE.W_PWR} "
          f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO} form={PE._PWR_FORM} act={PE._PWR_ACT}({PE.PWR_N_ACT})" + ("   <-- W_PWR OVERRIDDEN" if a.w_pwr is not None else ""))
    print(f"train scenes={len(env.scene_ids)} held-out scenes={len(env_te.scene_ids)} | "
          f"quant={N_QUANT} nstep={NSTEP} gamma={GAMMA} | target sync every {sync_every} ep", flush=True)

    global PIN_EPISODES, PIN_ACTION, PIN_ON
    PIN_EPISODES = max(0, int(a.pin_release))
    if PIN_EPISODES > 0:
        if PIN_EPISODES >= a.episodes:
            sys.exit('ABORT: --pin_release %d must be less than --episodes %d, otherwise the '
                     'power branch is never released.' % (PIN_EPISODES, a.episodes))
        holds = [k for k in range(PE.PWR_N_ACT)
                 if abs(PE._apply_pwr(PE.PT_MAX, k) - PE.PT_MAX) < 1e-9]
        if not holds:
            sys.exit('ABORT: actuator %r has no action that holds power at PT_MAX; '
                     'pin-release is undefined for it.' % PE._PWR_ACT)
        PIN_ACTION = holds[0]
        if PE.PWR_N_ACT != BRANCHES[1]:
            sys.exit('ABORT: power branch has %d actions but the network expects %d.'
                     % (PE.PWR_N_ACT, BRANCHES[1]))
        print('PIN-RELEASE: power held at %.2f dBm (action %d of %d) for episodes 1-%d, '
              'free from episode %d.' % (PE.PT_MAX, PIN_ACTION, PE.PWR_N_ACT,
                                         PIN_EPISODES, PIN_EPISODES + 1), flush=True)

    hist, loss, t0 = [], float('nan'), time.time()
    ptrace = []                       # per-episode actuator trace (see rollout docstring)
    for ep in range(a.episodes):
        PIN_ON = (ep < PIN_EPISODES)
        eps = max(0.05, 1.0 - ep / max(a.episodes * 0.7, 1))
        ret, _ = rollout(env, agent, eps, learn_buf=buf, trace=ptrace)
        hist.append(ret)
        if len(buf) >= a.warmup:
            for _ in range(a.updates_per_ep):
                eps_b, idx, w = buf.sample(a.batch)
                loss, td = agent.learn(eps_b, w)
                buf.update(idx, td)
            buf.anneal()
        if (ep + 1) % sync_every == 0:
            agent.sync()
            st = buf.stats()
            print(f"  ep {ep+1:5d} | eps {eps:.2f} | loss {loss:8.3f} | recent return "
                  f"{np.mean(hist[-25:]):7.2f} | buf {st['n']:4d} beta {st['beta']:.2f} "
                  f"p[{st['p_min']:.3f},{st['p_max']:.3f}] | {time.time()-t0:6.0f}s", flush=True)

    PIN_ON = False                    # evaluation is NEVER pinned
    lat = []
    ev_te = evaluate(env_te, agent, a.eval_n, latency=lat)
    ev_tr = evaluate(env, agent, a.eval_n_train)
    m = dict(seed=a.seed, alpha=a.alpha, lstm=int(not a.ff), anticipation=int(not a.no_ant),
             per=int(not a.no_per), episodes=a.episodes, pin_release=PIN_EPISODES, w_pwr=float(PE.W_PWR), pwr_form=PE._PWR_FORM, pwr_act=PE._PWR_ACT, spec_act=PE._SPEC_ACT,
             fdqr=int(FDQR), fdqr_d=(FDQR_D if FDQR else None),
             fdqr_q=(FDQR_Q if FDQR else None),
             fd_thr_final=(round(agent._fd_thr, 6) if FDQR and agent._fd_thr != float('inf') else None),
             fd_bind_rate=(round(agent._fd_bind / max(agent._fd_dec, 1), 6) if FDQR else None),
             fd_excl_rate=(round(agent._fd_excl / max(agent._fd_c_n, 1), 6) if FDQR else None),
             fd_empty_rate=(round(agent._fd_empty / max(agent._fd_dec, 1), 6) if FDQR else None),
             fd_cost_mean=(round(agent._fd_c_sum / max(agent._fd_c_n, 1), 6) if FDQR else None),
             fd_cost_min=(round(agent._fd_c_min, 6) if FDQR and agent._fd_c_n else None),
             fd_cost_max=(round(agent._fd_c_max, 6) if FDQR and agent._fd_c_n else None),
             fd_decisions=(agent._fd_dec if FDQR else None),
             fd_rand=(FDQR_RAND if FDQR else None),
             fd_pen=(FDQR_PEN if FDQR else None),
             fd_pen_lam=(round(agent._pen_lam, 6) if FDQR and FDQR_PEN > 0.0 else None),
             fd_soft=(FDQR_SOFT if FDQR else None),
             fd_soft_lam=(round(agent._sf_lam, 6) if FDQR and FDQR_SOFT > 0.0 else None),
             fd_soft_gap_med=(round(agent._sf_med, 6) if FDQR and FDQR_SOFT > 0.0 else None),
             fd_soft_opp=(agent._sf_opp if FDQR and FDQR_SOFT > 0.0 else None),
             fd_soft_over=(agent._sf_over if FDQR and FDQR_SOFT > 0.0 else None),
             fd_soft_over_rate=(round(agent._sf_over / max(agent._sf_opp, 1), 6)
                                if FDQR and FDQR_SOFT > 0.0 else None),
             pwr_n_act=int(PE.PWR_N_ACT),
             train_s=round(time.time() - t0, 1))
    m.update(summarize(ev_te))                       # held-out: the reported numbers
    m.update(summarize(ev_tr, 'insample'))
    m['gap_return'] = round(m['insample_return_mean'] - m['return_mean'], 4)
    m['gap_rlf'] = round(m['rlf_rate'] - m['insample_rlf_rate'], 4)
    m.update(latency_stats(lat))
    print(f"HELD-OUT  return={m['return_mean']:.2f}  CVaR25={m['cvar25']:.2f}  RLF={m['rlf_rate']:.2f}%  "
          f"(n={m['n_eval']})")
    print(f"IN-SAMPLE return={m['insample_return_mean']:.2f}  CVaR25={m['insample_cvar25']:.2f}  "
          f"RLF={m['insample_rlf_rate']:.2f}%  (n={m['insample_n_eval']}) | gap {m['gap_return']:+.2f} return, "
          f"{m['gap_rlf']:+.2f} pp RLF")
    print(f"latency/decision: mean {m.get('lat_mean_ms', float('nan')):.3f} ms  "
          f"p95 {m.get('lat_p95_ms', float('nan')):.3f} ms  p99 {m.get('lat_p99_ms', float('nan')):.3f} ms")
    if FDQR:
        # The diagnostic the 31 Aug FD-QR run could not answer: did the constraint bind?
        print('FD-QR binding: constraint changed the action in %.1f%% of %d decisions | '
              'no feasible action in %.1f%% | actions excluded %.1f%% | '
              'cost CVaR min %.3f mean %.3f max %.3f  (effective threshold %.3f)'
              % (100 * agent._fd_bind / max(agent._fd_dec, 1), agent._fd_dec,
                 100 * agent._fd_empty / max(agent._fd_dec, 1),
                 100 * agent._fd_excl / max(agent._fd_c_n, 1),
                 agent._fd_c_min, agent._fd_c_sum / max(agent._fd_c_n, 1),
                 agent._fd_c_max,
                 (agent._fd_thr if FDQR_Q is not None else FDQR_D)), flush=True)
        if FDQR_SOFT > 0.0:
            print("FD-QR soft: lam_eff %.4f (= %.2f x median override-gap %.4f) | "
                  "override %d/%d = %.4f"
                  % (agent._sf_lam, FDQR_SOFT, agent._sf_med, agent._sf_over,
                     agent._sf_opp, agent._sf_over / max(agent._sf_opp, 1)), flush=True)
    if a.out:
        torch.save(agent.net.state_dict(), a.out + '.pt')
        _rt = pd.DataFrame({'episode': range(len(hist)), 'return': hist})
        if len(ptrace) == len(hist):      # pt_mean is the quantity that DEFINES the basin
            _rt = pd.concat([_rt, pd.DataFrame(ptrace)], axis=1)
        _rt.to_csv(a.out + '_returns.csv', index=False)
        ev_te.to_csv(a.out + '_eval.csv', index=False)
        ev_tr.to_csv(a.out + '_eval_train.csv', index=False)
        pd.DataFrame([m]).to_csv(a.out + '_metrics.csv', index=False)
        print(f"saved -> {a.out}.pt / _returns.csv / _eval.csv / _eval_train.csv / _metrics.csv")
    if a.smoke:
        print('SMOKE OK — QR-DQN + CVaR + prioritized sequence replay trains end-to-end, '
              'evaluated out-of-sample under common random numbers.')


if __name__ == '__main__':
    main()
