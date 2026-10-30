# PREREG (FROZEN) — realism run for LSRA: correlated shadowing, blockage, beams, fading, handover failures, and imperfect coordination

**Date:** 30 Oct 2026. **Status:** FROZEN at 9:18 AM IST on 30 Oct 2026, on Daya's OK. It includes Daya's choices of 30 Oct: 4 places only, small-scale fading added as one more world, and "Build and test it". Every check of §7 passed before this freeze; nothing registered has run.

**What changed from v1:**
- Small-scale fading is added as world F1.
- The details fixed while building the code are written in: the timing of re-establishment, how a partial commit and a failure inside a set are drawn, the shadowing spread on near-miss links in B1, the one-off build of the B frames, the checks, the bootstrap seeds and the must-hold checks.
- The worlds, the arms, the tests and the guesses are otherwise unchanged.

**What changed from v2 (draft v2 md5 e3da82e21132432c20ae112407fff2b9):**
- S2's source was re-read, as v2 required before the freeze. TR 38.901 Table 7.5-6 Part-1 gives the shadow-fading correlation distance of UMi-Street Canyon as 10 m (LOS) and 13 m (NLOS), the values planned. The same table gives 50 m for UMa NLOS, which is now named as the origin of S3's value; S3 stays a stress case.
- The check results and their md5s are written into §7.
- A note on two arms that coincide (§4).
- The run time from gate G4 (§4).
- Nothing else changes: the worlds, the arms, the tests, the guesses and the code are those of v2.

**Answers:**
- the guide's correction 1: the 140 GHz model, which conclusions are robust (plan items W1.3 and W1.4);
- the evaluation part of correction 3: imperfect coordination (plan item W3.2).

**Uses:**
- the values fixed in `SEARCH_LSRA_GuideCorrections_ThreeRoutes_20261028.md` §2;
- the tested code patterns of the E10 channel re-test (`e10_channel.py`: correlated shadowing V1, fading V2, beam misalignment V3);
- the delay-jitter environment (`paper2ma_env_jitter.py`);
- the stored results, and the new baselines' frozen settings (`TUNED.json`, md5 7af67772…).

**Code:** `realism_env.py` (the worlds) and `realism_check.py` (checks, run and analysis) in `Paper2/code`.

## 1. What stays fixed

- **The environment, scenes and episodes.**
  - The admission environment of every earlier check, with capacity-aware attachment.
  - 16 vehicles, 4 stations, 3 sub-bands, full power, the 1.0 s delay (L = 10).
  - The 60 held-out scenes with their 480 episodes. This is their ninth use, and nothing is tuned on them.
- **Every planner keeps its settings.** Only the world changes; no planner is re-tuned or told about any change.
  - LSRA: β = 0.05, the 'now' forecast.
  - The TR 36.839 Set 5 rule, which was the best A3 set.
  - The tuned A3, at its frozen setting per cell.
  - Na et al.'s rule with the perfect forecast.
  - R1i, R2 and R3.
- **Operating point:** 4 places per station (full load). The metric is the worst-vehicle RLF J, as in the paper.
- **Common random numbers.** Every change draws its random numbers from its own generator, seeded by the episode seed. So the scene, the vehicles and the base shadowing draws are those of the paper.

## 2. The worlds (each changes one thing; V0 is the paper's world)

| World | What changes | Value and source |
|---|---|---|
| V0 | nothing | the stored episodes; the new environment with every switch off must reproduce them row for row |
| **S1** | shadowing correlated along each vehicle's path: the same standardised draws, filtered per vehicle and station as z′ₜ = ρₜ z′ₜ₋₁ + √(1−ρₜ²) zₜ, with ρₜ = exp(−vₜ·0.1 s / d) and vₜ the vehicle's speed. A stopped vehicle keeps its shadowing; a vehicle that reappears starts afresh; stations are independent. The A3 rules measure the filtered shadowing. | d = 3.8 m on every link (measured at 142 GHz [C1-1]); independent stations [C1-12] |
| **S2** | as S1 | d = 10 m clear, 13 m blocked (3GPP UMi-Street Canyon LOS and NLOS; TR 38.901 Table 7.5-6 Part-1, re-read 30 Oct) |
| **S3** | as S1 | d = 50 m (a stress case; the 3GPP UMa NLOS value of the same table, which is not our road, so an assumption, labelled as such) |
| **B1** | near-miss loss: the knife-edge loss also when a blocker's roof is just below the ray (−0.78 < v ≤ 0). The paper's rule that a link with any blockage loss has the blocked shadowing spread (8.2 dB) is kept, so a grazing link also gets the wider spread. | ITU-R P.526 curve (the paper's model skips v ≤ 0) |
| **B2** | blockage persistence: each blocker counts with its length × 1.3 if it is taller than 2.5 m, × 0.75 if lower | fade-causing length, measured at 28 GHz [I-F3] |
| **B3** | several blockers: the losses of all crossing blockers summed in dB; cap raised from 40 to 60 dB | TR 38.901 blockage model B rule [W-a] |
| **A1a / A1b** | serving-beam gain 10 dBi / 20 dBi (now 15 dBi), in the world and in every planner's estimate (an operator knows its antenna gain) | E10 record; TR 38.901 |
| **A2** | directional interference: a co-channel vehicle interferes with gain G(θ) = 15 − min(12 (θ/30°)², 15) dBi, θ the angle between it and the served vehicle seen from the station (now always 0 dBi). The planners still assume 0 dBi. | 3GPP parabolic pattern (TR 38.901 Table 7.3-1); 30° as in E10 |
| **A3** | beam misalignment on the serving link: the beam is re-pointed every 0.5 s, when a handover lands, and when service resumes after a gap. Between pointings the loss is min(12 (Δθ/30°)², 30) dB, Δθ the change of the station-to-vehicle angle. | E10 variant V3 |
| **F1** | small-scale fading on every vehicle-station link (desired and interfering), independent per slot. The power gain is \|h\|², with h = √(K/(K+1)) + √(1/(K+1)) w, w ~ CN(0, 1), K = 9 dB on a clear link and 0 (Rayleigh) on a blocked one; E\|h\|² = 1. The A3 rules' measurement (RSRP, an average) does not see it. | E10 variant V2 |
| **H1** | handover interruption: the vehicle is unserved in the slot it lands | 0.1 s (measured 43–134 ms; TR 36.839 execution 40 ms) |
| **H2a** | handover failure, the 3GPP way, then re-establishment. (a) Too late: an RLF on the source link (3 consecutive outage slots) while the handover is in flight cancels it. (b) The vehicle is served in the landing slot with SINR below Q_out = −13.8 dB (our outage threshold). Either way it is unserved in the next 2 slots (0.2 s), then served by the strongest station with a place in the landing state (its own reserved place counts). | TR 36.839 §5.2.1.3; TS 38.300 §9.2.7; recovery "a few hundred ms" [I-F2] |
| **H2b** | as H2a, with Q_out = −8 dB (TR 36.839) and 5 slots (0.5 s) unserved | the same sources |
| **ALL** | S2 + B2 + A2 + H2a together, the harshest combination | — |

**Notes:**
- **B worlds:** every slot's geometry is rebuilt with the changed rule, including the background interferers.
  - The frames are built once (step "prep") into `geom_cache_realism` and frozen by md5.
  - The anticipation features, which no planner here reads, are copied from the stored frames.
- **Re-establishment:** a vehicle being re-established is shown to the planners as a vehicle in flight to the station that will serve it, so LSRA plans on the actual landing state.
- **Scheduling:** an unserved vehicle does not take a scheduling place at its station.

## 3. Imperfect coordination (W3.2): in the paper's world, under coordinated admission

| Setting | What changes | Values |
|---|---|---|
| C1a | landing spread: each admitted handover lands after L + d slots, d uniform (the delay-jitter environment's own generator) | d ∈ {−5, …, +5} (±0.5 s) |
| C1b | late landings only | d ∈ {0, …, +5} |
| C2a, C2b | partial commit: each admitted move is independently cancelled right after it starts (it never runs); the others go ahead; the next decision re-plans. Counted as "dropped". | 5 %, 20 % |
| C3a, C3b, C3c | failure inside a set: each admitted move is independently marked, when it starts, to fail at landing; then re-establishment as in H2a (0.2 s unserved) | 2 %, 5 %, 10 % |
| C4a, C4b | coordination takes longer: coordinated admission with a longer delay than per-request admission | +0.1 s, +0.2 s (L = 11, 12); +0.5 s (L = 15) is already stored |

If a swap is broken (one vehicle lands before the other leaves, or one move does not run), the station serves its 4 strongest vehicles. This is the model already measured (Daya's choice, 28 Oct).

## 4. Arms and cells (4 places)

**In each of the 15 new worlds of §2** (V0's cells are already stored), there are 10 arm-cells:
- LSRA under A1 and under A2;
- R2 under A1 and under A2;
- the tuned A3 under A1 and under A2, each at its own frozen setting;
- R1i, R3, A3 Set 5 and Na et al.'s rule with the perfect forecast (PH-O), each under A2.

That is 150 cells × 480 episodes.

**Two arms that coincide.** Under coordinated admission at 4 places, the tuned A3's frozen setting (`TUNED.json`, C4_A2: time-to-trigger one slot, offset −1 dB, filter k = 0, measured signal) is the setting of TR 36.839 Set 5 on our 0.1 s slots (`a3_rule.py`: 40 ms, −1 dB, k = 0). So A3 Set 5 and the tuned A3 under A2 give the same episodes in every world, as they do in the stored results. Both stay in RW1, as registered.

**Imperfect coordination:**
- LSRA under A2 in all 9 settings;
- R3 and the tuned A3 under A2 in C2b and C3c, to show what the same imperfection costs a baseline.

That is 13 cells.

**How it runs.**
- 163 jobs (one per cell), 78,240 episodes. Gate G4 measured 0.52–0.78 s per episode and projects 12.6 CPU-hours, about 2.1 hours with 6 workers. Six workers on one laptop run slower than one, so 2½–3 hours is expected.
- After the freeze, 6 workers are launched once through WMI with the window hidden. They take jobs through claim files, as in the baselines run.
- The analysis runs once, by the first worker to finish. A worker that sees no new output for an hour writes STOP; nothing is relaunched without Daya's OK.
- Checked every 30 minutes.

## 5. Registered tests (J of the other arm minus J of LSRA under A2, pp, paired by episode; scene bootstrap, 10,000 resamples)

**RW1, per world: LSRA with coordinated admission beats every baseline under the same admission.**
- The baselines are R1i, R2, R3, A3 Set 5, the tuned A3 and PH-O, all under A2.
- 99.17 % intervals (Bonferroni over the six).
- It holds in a world if all six lower ends are above 0. Otherwise the baselines not beaten are named.

**RW2, per world: coordinated admission is worth something to LSRA.**
- LSRA under A1 minus LSRA under A2, 95 % interval. It holds if the lower end is above 0.

**RW3, per world: the proposal against today's practice.**
- R2 under A1 and the tuned A3 under A1, each minus LSRA under A2; 97.5 % intervals (Bonferroni over two).
- It holds if both lower ends are above 0.

**RC, per imperfect-coordination setting.**
- LSRA under A2 (imperfect) against LSRA under A1, R2 under A1 and the ideal LSRA under A2, all from V0, 95 % intervals.
- The **break-even** is reported: the first value in each sequence (C1a→C1b; C2a→C2b; C3a→C3b→C3c; C4a→C4b→+0.5 s) whose interval does not lie above 0.
- For R3 and the tuned A3 in C2b and C3c: the cost of the imperfection to them, and their gap to LSRA under the same imperfection.

**No pooling across worlds.** Each world answers its own question. The robustness table lists, for each conclusion, the worlds where it holds. V0 is included, computed from the stored files.

**Must-hold checks** (else the analysis stops and reports):
- every cell has 480 episodes paired with V0 by seed and scene;
- starts = requests − refused − dropped;
- no landing excess created by a decision, except in C2 (a partial commit can break a swap by design);
- no loop violation under A1.

**Bootstrap seeds:** 14000, 14001, … in the order of the analysis code. The last one used is recorded.

**Secondary:**
- the levels of J with 95 % intervals;
- handovers, quick returns and refusals;
- the failure causes;
- in H1 and H2, the counts of unserved landing slots, too-late failures, target failures and re-establishments;
- in C2 and C3, the counts of dropped and failed moves.

## 6. Recorded guesses (before anything runs)

- RW1 holds in V0, S1, S2, B1, B2, B3, A1a, A1b, H1, H2a and F1.
- It may fail in:
  - S3 (long shadowing memory makes the measured signal more useful, which helps A3);
  - A2 (LSRA's interference model is wrong there);
  - ALL.
- RW2 and RW3 hold in every world except possibly ALL.
- H2 lowers every arm's J gap, because failures add outages for everyone, but LSRA stays ahead.
- C2b (20 % of moves not run) and C3c (10 % failing) keep LSRA under A2 ahead of LSRA under A1 and of R2 under A1. C4: no break-even within +0.5 s.

## 7. Checks before the run

**Unit tests** (`UNIT.json`), hand-made cases:
- the shadowing filter: the recursion, restart on reappearance, a stopped vehicle, unit variance and lag-1 correlation;
- the B rules on hand-made boxes: near miss, tall and low blockers, two blockers summed and capped;
- the beam-gain switch reaching every module that reads it;
- the A2 pattern and the A3 loss and pointing clock;
- fading: unit mean, the Rayleigh spread, common random numbers;
- the break-even rule;
- the switch names and the cell counts.

**Prep** (`PREP.json`): the B1–B3 frames of the 60 scenes built once, with md5s and the share of blocked link-slots per world.

**Gates** (`GATE.json`):
- **G0, identity:** with every switch off, all 10 arm-cells reproduce their stored episodes row for row (first 40 episodes).
- **G1, geometry:** the frame builder with the paper's blockage rule rebuilds all 6,000 stored slots exactly.
- **G2, mechanics on real scenes** (LSRA under A2):
  - H1 makes landing slots unserved;
  - in H2a and H2b, every re-establishment station had a place, and the flight bookkeeping matches;
  - C2b drops about 20 % of started moves (±6 points), and C3c fails 4–16 % at landing;
  - C1a and C1b delays stay within L − 5…L + 5 and L…L + 5.
- **G3:** all 163 cells run (first episode), and the accounting adds up.
- **G4:** timing per world and setting.

**Dry run** (`DRY.json`): the registered analysis on stored V0 cells relabelled as every world and setting. RW1, RW2, RW3 and the RC differences must equal the earlier results files to 10⁻⁹ (at least 150 numbers).

**If a check fails:** stop, find the cause, report; no freeze.

**Results of the checks (30 Oct, on the laptop; Python 3.11.9, numpy 2.4.4, scipy 1.17.1, pandas 3.0.3).** Every record carries the code md5s `realism_env.py` c827b7f636f6635f6df987882f2eb056 and `realism_check.py` 28345fdf2abe97119debbc25ec0b71c2.
- **Unit tests** (`UNIT.json`, md5 f27da7b4084ba05f52575843851efd06): all 19 pass.
- **Prep** (`PREP.json`, md5 52039382d4517138144f3b76478e2aad): 180 frame files built. Over 328,972 link-slots, the share blocked and the mean loss when blocked are:
  - the paper's rule: 4.42 %, 24.9 dB;
  - B1: 5.20 %, 21.6 dB;
  - B2: 4.45 %, 26.1 dB;
  - B3: 4.42 %, 26.0 dB (the same links as the paper's rule, with the losses summed).
- **Gates** (`GATE.json`, md5 97ec1aa560c49895a42ed0ce1de1d716): all pass.
  - G0: all 10 arm-cells reproduce their stored episodes, 40 episodes each.
  - G1: 6,000 slots rebuilt, 0 mismatches.
  - G2: H1 left 46 landing slots unserved. H2a made 18 re-establishments (all too late), H2b 19 (16 too late, 3 target failures); every re-establishment station had a place, and the flight bookkeeping matched. C1a delays ran 5–15 slots and C1b 10–15. C2b dropped 18.7 % of admitted moves (14 of 75) and C3c failed 15.3 % at landing (9 of 59, inside the 4–16 % band).
  - G3: all 163 cells ran, and the accounting added up.
  - G4: as in §4.
- **Dry run** (`DRY.json`, md5 dd20a89301c3fe0a50d4f94a3cdd4b00): 171 numbers compared, 0 mismatches; its analysis output `REAL_RESULT.json` (dry) has md5 acdf6f6e3cddf56bfa7d15ba0c287d85.

**Freeze** (`REAL_FREEZE.json`, on Daya's OK): md5 of every file the run reads. That is the code actually imported, the pool, the seed list, the stored episodes and results, the 180 built frame files, this plan as FROZEN, and the four check records. Each record carries the md5 of the two new code files; the freeze refuses if the code changed after the checks.

## 8. What goes into the paper (decided now)

- **One robustness table:** each conclusion and the worlds in which it holds; the full tables go in the supplement.
- **One figure:** LSRA under ideal and imperfect coordination against today's practice, with the break-evens.
- **The claims are narrowed to what holds.** If RW1 fails in a world, the paper names the world and the baseline.
- **The framing stays** "simulation evidence on real trajectories, not 140 GHz field performance". The robustness table says which conclusions do not depend on the channel assumptions tested.

## 9. Limits (to state)

- Shadowing correlation and blockage persistence at 140 GHz on vehicle-to-roadside links were not found in the literature. S1 uses a pedestrian-route value at 142 GHz and B2 a 28 GHz measurement.
- S3 is an assumption.
- One combined world only.
- B1 keeps the paper's wider shadowing spread on any link with a blockage loss, so it also widens the spread on grazing links.
- The A3 rules do not see fading (RSRP averages it).
- No beam training delay.
- 4 places only (Daya's choice).
