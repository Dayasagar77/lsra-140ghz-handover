# PREREG (FROZEN) — stronger baselines for LSRA: a tuned A3, a load-balancing A3, Na et al.'s predictive handover, and equal-information arms

**Date:** 28 Oct 2026. **Status:** FROZEN at about 7:35 PM IST on Daya's OK ("Full grid overnight with 6 workers: the plan as written, no code change"). This copy is draft v2 with only this status block and the number of workers (§4) changed. The unit tests, gates and dry run had passed by 11:12 AM IST with the frozen code; nothing registered had run.

**What changed from v1.** v2 adds the details fixed while building the code; no arm, grid, test or guess changes.
- The quick-return (ping-pong) count is defined for our 1.0 s delay (§5).
- The bootstrap seeds are fixed (§5).
- How the run is carried out is written out: workers, claim files and the stop rule (§4).
- The gates are listed in full, with the dry run and the freeze record (§7).
- Small rules the code needed: in-flight vehicles in the cell offset load, vehicles leaving the scene in Na et al.'s rule, and tie rules (§2, §3, §4).

**Answers:** the guide's correction 2 (plan items W2.2, W2.3, W2.4, W2.5 for the new arms, and the equal-information arms).

**Uses:** the parameter sources in `SEARCH_LSRA_GuideCorrections_ThreeRoutes_20261028.md`.

**Code:** `baselines_check.py`, `a3_general.py`, `ph_rule.py` and `lsra_m.py` in `Paper2/code`; their md5 values go into the freeze record.

## 1. What stays fixed

- **The environment and episodes.**
  - The admission environment of every earlier check (`paper2ma_env_admit.AdmissionEnv`, capacity-aware attachment on).
  - 16 vehicles, 4 stations, 3 sub-bands, full power and the same channel.
  - The 1.0 s delay (L = 10).
- **LSRA unchanged:** `lsra.make_lsra('now', landing=True, minmax=False, beta=0.05)`.
- **Design episodes** (tuning only): the 45 design scenes with the 360 episodes already used to fix β (`models/remeasure_ma/SEEDS_train.csv` and `SEEDS_test.csv`).
- **Test episodes:** the 60 held-out scenes with their 480 episodes (`models/method_heldout/SEEDS_heldout60.csv`). This is their eighth use; nothing is tuned on them.
- **Metric:** the worst-vehicle RLF J, as in the paper. Admission models A0, A1 and A2, as in the paper.

## 2. New arms

**A3-T, the tuned A3.**
- The 3GPP event-A3 rule of `a3_rule.py`, with its parameters free instead of fixed to a TR 36.839 set:
  - time-to-trigger n slots;
  - offset Off (dB);
  - layer-3 filter coefficient K;
  - measurement source:
    - "measured": the slot's received power with its shadowing draw, as in the paper;
    - "noise-free": the mean received power from the geometry, the same quantity R1–R3 and LSRA use.
- **Grid:**
  - n ∈ {1, 2, 3, 5, 7}, which are the 3GPP time-to-trigger values 40, 160, 256, 480 and 640 ms at our 0.1 s slot;
  - Off ∈ {−2, −1, 0, 1, 2, 3, 4, 6} dB;
  - K ∈ {0, 1, 4} for the measured source and {0, 4} for the noise-free source;
  - 200 settings in all.
- **Arms:** A3-T uses the best measured-source setting; A3-T-NF uses the best noise-free setting (the equal-information A3).

**A3-LB, A3 with load balancing through cell offsets.**
- A3-T's setting, with a cell individual offset added to each neighbour n of the serving station p in the entering condition:
  - F_n + O_pn > F_p + Off;
  - O_pn = clip(κ · (N_p − N_n) / C, −10, 10) dB;
  - N_g is the number of present vehicles attached to station g in the slot. A vehicle in flight counts at the station it is leaving, as in a measurement report.
- Among the triggered neighbours, the request goes to the one with the largest F_n + O_pn.
- A crowded serving station pushes its vehicles towards emptier neighbours, which is what load balancing through CIO does [C5-19, C5-8].
- κ ∈ {0, 2, 4, 8, 16} dB. κ = 0 is A3-T itself, so on the design scenes A3-LB is never worse than A3-T.

**A3-O, the best A3 setting per test scene, picked in hindsight. This is a bound, not a method.**
- The measured-source grid (120 settings) is run on the 60 test scenes, under A2, at 4 and 5 places.
- For each scene, the setting with the lowest mean J over its 8 episodes is kept. Ties go to the earlier setting in grid order (n, then Off, then K).
- It picks on the outcomes it is scored on, so it can only favour A3. It answers the question: "would any A3 tuning, even per scene, close the gap?"

**PH-O and PH-CV: Na et al.'s predictive handover** (IEEE TVT 2026, Algorithm 1, read in full).
- **Forecast:** each free vehicle forecasts the mean received power of every station over the window it would land in, W = slots t+L+1 … t+L+5 (their T_window = 0.5 s).
  - PH-O uses the true future geometry (a perfect forecast, the best any forecaster can give this rule). Past the last slot of a scene, the last slot is used.
  - PH-CV uses the straight-line forecast of the learned-forecast study (`cv_forecast.SceneGeom.cv_window`, every vehicle moving on at its recent velocity).
  - A vehicle that is absent from some slot of its window (it is leaving the scene) requests nothing.
- **Case 1, serving station predicted above τ throughout W.**
  - The eligible stations stay above τ throughout W. Their window average beats the serving station's by Δ_avg = 3 dB, and they beat it by Δ_init at the window start.
  - The vehicle must have stayed at least T_min = 0.5 s since its last landing (or since it appeared).
  - Then it requests the eligible station with the highest average (ties: the lower station number).
- **Case 2, serving station predicted to fall below τ.**
  - It requests the station with the highest average among those that stay above τ throughout W.
  - If there is none, it requests the station that stays above τ longest from the window start, with ties going to the higher average.
  - In both branches the chosen station must beat the serving station by Δ_init at the window start; otherwise the vehicle stays.
- **Values:**
  - Δ_avg = 3 dB and T_min = 0.5 s are their values. T_window = 0.5 s is their value, which is 5 slots here.
  - τ = −92 dBm: our outage threshold at the noise floor, the same threshold the paper uses.
  - Δ_init = 0 dB. Their paper gives no number for it.
- **Not tuned.** It requests whatever the rule says, ignoring load. That is their method.

**PH-OR, PH-O with a room check.**
- The same rule, but a station is a candidate only if it has room when the vehicle lands, counting the vehicles on their way. Decisions earlier in the same slot are counted, in vehicle order, as in R1i.
- This is the strongest one-vehicle-at-a-time predictive rule we can build: a perfect forecast plus the landing count.

**LSRA-M, LSRA fed only measurement reports (equal information the other way).**
- The same planner, but the risk comes from what a network measures, not from the geometry:
  - each vehicle's received power from every station is filtered by the 3GPP layer-3 filter with the default coefficient K = 4, as in A3 (restarted when a vehicle reappears);
  - interference at a station is estimated from the other vehicles' filtered powers through the side lobe;
  - the risk is Φ((γ_out − SINR_est)/σ), with σ = 8.2 dB on every link, since no geometry tells which links are blocked.
- β = 0.05 as in LSRA, the window held, nothing tuned.
- Its request sets are landing-feasible like LSRA's, so it runs under A1 and A2 only. A0 equals A2 for it.

## 3. Stage 1, tuning (design scenes only)

- **Cells:** 1.0 s; 4, 5 and 6 places; A1 and A2. Six cells, each tuned separately in favour of the baseline.
- **A3-T and A3-T-NF:** in each cell, run every setting of the grid on the 360 design episodes. Keep the setting with the lowest mean J. Ties go to fewer handovers, then to the order n, Off, K.
- **A3-LB:** in each cell, add κ to that cell's A3-T setting and keep the best κ, with ties going to the smaller κ.
- **Freeze:** the chosen settings are written to `TUNED.json`, whose md5 is recorded in `TUNED_FREEZE.json` before any test episode is run. Every test job checks it.
- **Cost:** 1,200 grid runs and 24 κ runs of 360 episodes. Gate G4 gives the time.

## 4. Stage 2, the test (held-out scenes), and how the run is carried out

- **A3 arms:** A3-T, A3-T-NF and A3-LB with their frozen settings, at 4, 5 and 6 places, under A1 and A2, 480 episodes per cell.
- **PH arms:** PH-O, PH-CV and PH-OR under A0, A1 and A2.
- **LSRA-M:** under A1 and A2.
- **Comparison data:** LSRA and the earlier arms are taken from their stored files (same episodes, paired by seed and scene).
- **A3-O:** the grid on the test scenes (A2; 4 and 5 places; 120 settings × 480 episodes).
- **How it runs.**
  - After the freeze, 6 workers are launched once through WMI, with the window hidden. Daya chose 6 after gate G4 measured about 65 CPU-hours (12–15 hours with 6 workers) and each worker was found to need about 0.6 GB of the 3.9 GB of free memory.
  - Each worker takes jobs one at a time in a fixed order: the grid, then the κ runs, then the test cells, then the A3-O grid. A job is one setting or one cell; it is taken through a claim file, so no job runs twice, and it writes its own file.
  - The choices of stage 1 are made once, by the first worker to find all their inputs; the others wait for them. The analysis runs once, the same way.
  - A worker that finds no new output for one hour writes STOP and leaves. A job that fails writes STOP with its error. Nothing is relaunched without Daya's OK.
  - The run is checked every 15–30 minutes.

## 5. Registered tests (1.0 s, 4 places; J of the baseline minus J of LSRA, in pp, paired by episode)

- **HN1, coordinated admission.**
  - LSRA under A2 against each of A3-T, A3-T-NF, A3-LB, PH-O, PH-CV and PH-OR under A2.
  - 99.17 % scene-bootstrap intervals (10,000 resamples; Bonferroni over six).
  - Held if all six lower ends are above 0.
- **HN2, per-request admission.** The same six under A1, against LSRA under A1, with the same rule.

**Secondary (95 % intervals, no family):**
- the same comparisons at 5 and 6 places;
- A3-O against LSRA (A2; 4 and 5 places);
- LSRA-M against LSRA, and LSRA-M against the best new baseline;
- each new baseline under A1 against LSRA under A2 ("today against the proposal");
- the planner/admission split of the stored-results check (T, P1, Ab, I with intervals; P2, Al and the two shares as means) for every new baseline. For the A3 arms the setting tuned for A1 can differ from the one tuned for A2; the result records whether it does;
- the level of J of every new cell, handovers and requests per vehicle-minute, refused shares, failure causes, and the quick-return rate.

**The quick-return (ping-pong) rate.**
- TR 36.839 counts a ping-pong when a vehicle returns to the station it left within the minimum time of stay, 1 s.
- With our 1.0 s delay, a return cannot land within 1 s of the landing it undoes. So the time of stay is measured from the landing to the start of the return.
- A quick return is a handover back to the station the vehicle came from, started less than 1 s (10 slots) after it landed. It is counted per episode and reported per vehicle-minute and as a share of handovers, for every arm.

**Bootstrap seeds.** 12000, 12001, … in the order of the analysis code: HN1, HN2, the other loads, A3-O, LSRA-M, the split, the levels. The last seed used is recorded.

## 6. Recorded guesses (before anything runs)

- HN1 held.
- HN2 not held: PH-OR under A1 behaves like R1i, which LSRA was not shown to beat under A1.
- The tuned A3 under A2 at 4 places stays above 6 %.
- A3-O at 4 places stays above LSRA's 3.59 %; at 5 places it may come within 1 pp.
- LSRA-M is worse than LSRA by 0.5–2 pp at 4 places, but still beats A3-T under A2.
- PH-O makes fewer handovers than the A3 arms, but most of its requests are refused at 4 places.

## 7. Checks before the run (gates)

**Unit tests** (hand-made cases, no scenes), `UNIT.json`:
- U1: A3-T with each TR 36.839 set equals the stored A3 rule, slot by slot, on random episodes;
- U2: the cell offset formula, its clip at ±10 dB, κ = 0, and the move to the emptiest neighbour;
- U3: every branch of Na et al.'s Algorithm 1 (Case 1 with and without an eligible station, the dwell time, Δ_avg, Δ_init; Case 2 with and without a reliable station, the longest-above fallback and its tie, Δ_init; no candidate);
- U4: the dwell count, and the room count within a slot;
- U5: LSRA-M's risk (equal to LSRA's in gate mode, and rebuilt by hand in measured mode), the K = 4 coefficient, the filter and its restart, and its decision equal to LSRA's;
- U6: the quick-return count (returns started 0 and 9 slots after landing count; 10 and 12 do not; a reappearing vehicle starts afresh);
- U7 and U8: the tie rules of stage 1 and of A3-O;
- U9: the grid sizes and the TR 36.839 slot counts.

**Gates on real scenes** (no new arm is compared with LSRA), `GATE.json`:
- **G1, identity:**
  - G1a: A3-T with the parameters of TR 36.839 Sets 1–5 (measured source, κ = 0) reproduces the stored A3S1–A3S5 episodes row for row (4 places, A2, 1.0 s, first 40 episodes);
  - G1b: LSRA reproduces its stored episodes (same cell);
  - G1c: the stage-1 design environment reproduces the learned-forecast study's stored LSRA episodes (4 and 6 places, A2, first 20 episodes of each split);
  - G1d: the straight-line forecast's geometry code rebuilds all 6,000 stored slots of the 60 held-out scenes exactly.
- **G2, LSRA-M:** with the noise-free mean, K = 0 and σ chosen by blockage, LSRA-M's risk equals LSRA's to 10⁻¹² in every slot, its decision equals LSRA's in every slot, and its episodes equal LSRA's stored episodes (first 40). This shows the only difference is the information.
- **G3:** every new arm runs in all 51 test cells and the A3 rule in all 12 stage-1 runs (first episode), with no created excess, starts equal to requests minus refusals, and no loop violation under A1.
- **G4, timing:** seconds per episode of every arm and of the design runs, and of the straight-line forecast per scene, to set the number of workers.
- **G5, analysis dry run** (`DRY.json`): the registered analysis on stored arms relabelled as the new arms (A3-T ← A3 Set 5, A3-T-NF ← R1, A3-LB ← R2, PH-O ← R3, PH-CV ← R0, PH-OR ← R1i, LSRA-M ← ABL_nolanding; A3-O ← the five stored A3 sets in turn). Every level, handover rate, cause share, request rate, refused share, difference from LSRA and split part must equal the earlier checks' result files to 10⁻⁹ (at least 500 numbers), and A3-O must equal an independent computation.

**If a check fails:** stop, find the cause, report to Daya; no freeze.

**Freeze** (`BASE_FREEZE.json`, on Daya's OK): md5 of every file the run reads (the code actually imported, the pools, the seed lists, the stored episodes and results, this plan as FROZEN, and the three check records). Each check record carries the md5 of the four new code files; the freeze refuses if the code changed after the checks. Every registered step checks the freeze first.

## 8. What goes into the paper (decided now)

- **HN1 held:** "with coordinated admission at full load, LSRA beats an A3 tuned for this scenario, A3 with load balancing, and Na et al.'s predictive handover even with a perfect forecast". If any comparison fails, it is named and the claim is narrowed to the others.
- **HN2:** reported as it falls, with the planner/admission split.
- **A3-O:** reported as a bound ("even the best A3 setting chosen per scene in hindsight …" or "… would close the gap at 5 places").
- **LSRA-M:** one sentence on what using standard measurements instead of geometry costs.
- **Justification of the TR 36.839 sets in §7:** they remain the standard reference (used as benchmarks in 2025 [C8-10]); the tuned and per-scene results show what tuning can and cannot recover.

## 9. Limits (to state)

- The tuned A3 is tuned per load and admission model on design scenes from the same dataset. A deployment would tune on its own data.
- MRO's end state is a tuned trigger, so A3-T and A3-O stand in for it; slower self-optimisation loops are not modelled.
- Na et al.'s Transformer is not retrained. The perfect forecast bounds what any forecaster can give their rule; the straight-line forecast is a simple deployable one.
- Δ_init is our choice (0 dB), because their paper gives no value.
- LSRA-M's single σ is a simplification.
- The quick-return count measures the time of stay to the start of the return, not to its landing (§5).
