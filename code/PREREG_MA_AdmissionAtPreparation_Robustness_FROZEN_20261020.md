# PRE-REGISTRATION (FROZEN) — robustness check: admission at handover preparation

**Date:** 20 Oct 2026. **Status:** FROZEN on Daya's "Freeze and run now" (20 Oct), after both gates passed. Nothing above §10 changed from `…_DRAFT_v1_20261020.md` except this title and status line. The run has not started at the time of freezing.
**Builds on:**
- the frozen held-out test (`PREREG_LSRA_MethodTest_Heldout60_FROZEN_20261020.md`) and its result (`RESULT_LSRA_Heldout60_HM1Held_MethodBeatsAllFourAt1s_20261020.md`);
- the maths note (`MATH_LandingStateRotationAssignment_DRAFT_v1_20261020.md`, v2 text), Theorem 2 and its general form.

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

- **The main result.** At a 1 s handover delay, the best existing planner (R3) fails mostly because several handovers land on a full station at once (cause L: 6.33 of its 8.56 % worst-car RLF on the 60 held-out scenes). Our method never does that.
- **The objection a reviewer will raise.** 3GPP TS 38.300 (intra-NR handover, C-plane step 4) says "Admission Control may be performed by the target gNB" when the handover is prepared. So a reviewer can say R3's overflow is an artefact of the simulator, which admits only at landing.
- **The check.** Give every method that admission step and ask whether our method still wins. This is the one result that could change the paper's story, so it comes before the write-up.

## 1. What changes: one new option, off by default

New file `paper2ma_env_admit.py`, class `AdmissionEnv(CapacityAttachEnv)`. Option `ADMIT`:

| Value | Name | What happens when cars ask for handovers in a slot |
|---|---|---|
| `none` | A0 | As now. Stations admit only at landing (the frozen test). |
| `per_request` | **A1**, today's per-request admission | Each target admits a request only if it has room when the handover lands. Room is counted on the landing load: cars on their way are counted where they will land, and departures already committed are taken off. Requests are taken strongest first (mean received power at the target, the same ranking as the landing rule). A refused request is tried again after other departures in the same slot are committed, until nothing changes. |
| `coordinated` | **A2**, admission coordinated by the RIC | The slot's requests are admitted together: the whole set if it creates no landing excess. Otherwise, the largest subset that creates none is admitted (ties go to stronger requests). This is exact: one assignment problem. |

- "Creates landing excess" means some station's landing load rises above max(C, its load before the slot's requests). This is the same definition as `dec_over` in the frozen harness.
- **A refused request:** the car stays where it is. No handover starts, and the car may ask again in the next slot.
- **Nothing else changes:** channel, attachment, admission at landing (kept as the backstop), RLF counting, power, sub-bands.
- **What admission cannot stop:** cars that appear can still land on a full station (the a_g(τ) term of G1). This is the same for every arm.
- **A1 is the most generous per-request reading.** The target credits departures already committed, and a refused request is retried after other departures in the same slot. A stricter target that gives no credit for departures is not tested. At full load it would block even chains.

**The maths behind the check (from Theorem 2, general form).**
- A1 lets chains through (moves that end at a station with room), but never a closed loop.
- **Corollary (what per-request admission refuses).** Suppose no station is over capacity before the slot, and the slot's full request set creates no landing excess. Then the requests A1 refuses form closed loops: at every station, as many refused requests leave as arrive.
- *Proof.* Take a refused request with target g. At the end, g had no room: n_end(g) ≥ C ≥ n_S(g), where n_S is the landing load if the whole set S were admitted. Write P for the refused set. Then n_end(g) = n_S(g) − in_P(g) + out_P(g), so out_P(g) ≥ in_P(g) at every target of P. At any other station, in_P = 0 ≤ out_P. Both sides sum to |P| over all stations, so out_P(g) = in_P(g) everywhere. A balanced directed multigraph splits into closed loops. ∎
- Our method and R1i always send sets that create no excess. So under A1, every request of theirs that is refused (when no station starts over capacity) is part of a closed loop. METHOD under A1 therefore measures what closed loops are worth. This is checked in the run (M3).

## 2. Data: the same as the frozen test, so every episode is paired

- **Scenes and seeds:**
  - the 60 held-out scenes, `dair_geometry_pool_heldout60_20261020.csv` (md5 3a68d055876a733ee21de37700385a75);
  - the frozen seed list `models/method_heldout/SEEDS_heldout60.csv` (md5 ef0e8e0ce83cdb0369154bd35b9d22e4), 8 per scene, 480 episodes per cell.
- **Delays:** 0.2, 0.5, 1.0 and 1.5 s.
- **Why the same scenes:** each episode is paired with its frozen-run twin, so the only change is the admission step. Nothing is tuned. The method and every baseline are the frozen code (`method_heldout.arms`, β = 0.05). This is a sensitivity analysis of frozen code. It is not a second design round.
- **A0 is the frozen run itself.** It is not re-run, except for the gate G0.

## 3. Arms

All 10 frozen arms, code unchanged, each under A1 and under A2:
- R0, R1, R1i, R2, R3;
- R3F (reference, uses the future);
- METHOD;
- METHOD_oracle (reference, uses the future);
- ABL_nolanding;
- ABL_worstcarfirst.

**Under A2, two arms answer the reviewer directly:**
- **R3 under A2** is "R3 with coordinated admission", the fix a reviewer would propose.
- **ABL_nolanding under A2** is "our objective, planned on the current stations, with coordinated admission". It tests whether planning on the landing state helps beyond simply refusing what would overflow.

No arm is told about admission. A planner may keep asking for a move that keeps being refused.

## 4. Primary (1.0 s): two separate families, both reported, whatever they show

- **HR1 (A1, per-request admission).** For each B in {R1, R1i, R2, R3}: mean(B − METHOD) > 0 in per-episode worst-car RLF, both arms under A1. Each needs its **98.75 % scene interval** (bootstrap over the 60 scenes, 10,000 resamples, Bonferroni over four) above 0. HR1 needs all four.
- **HR2 (A2, coordinated admission).** The same, with both arms under A2.
- Each family is an "it beats all four" claim, tested exactly as HM1 was. Bonferroni is conservative for a claim that needs all four.
- **No claim of the form "it survives under at least one model" will be made.** Both families are reported side by side.

## 5. Secondary (95 % scene intervals)

- **HR3, the landing state beyond admission (A2, 1.0 s):** ABL_nolanding − METHOD > 0. Its consequence for the paper is fixed in §7.
- **Tables (A0 frozen, A1, A2) at the four delays:**
  - worst-car RLF;
  - handovers, requests and refusals per car-step;
  - causes of the worst car at 1.0 s;
  - every arm minus METHOD under the same admission.
- **What closed loops are worth:** METHOD under A1 minus METHOD under A2, at each delay.
- **What admission does for R3:** R3 under A1 minus R3 under A0, and R3 under A2 minus R3 under A0, at each delay.
- **Robustness to delay under A1 and under A2:** R3's cost of delay (1.0 s minus 0.2 s) minus METHOD's.

## 6. Checks

**Gates (before the freeze; if one fails, I stop, fix it and report before going on):**
- **G0 (admission off = frozen run).** With `ADMIT = none`, the new class reproduces the frozen episodes exactly: first 40 seeds, 1.0 s, all 10 arms, every column.
- **G1 (unit tests, no scenes):**
  - six hand-made cases:
    - a swap between two full stations: A1 refuses both, A2 admits both;
    - a chain into a free place, listed in the worst order: both models admit all;
    - two requests for one free place: one is admitted, the stronger;
    - a request into a station over capacity after an appearance: refused;
    - a closed loop plus a chain: A1 admits only the chain, and what it refuses is a closed loop;
    - an infeasible set: A2 keeps its largest feasible part.
  - 500 random small cases checked against brute force:
    - A2's subset is the largest feasible one, with the stated tie-break;
    - neither model ever creates landing excess;
    - A1 never admits more than A2;
    - the corollary of §1 holds wherever it applies.
  - 0 mismatches allowed.

**Must hold in the run (if any fails, the results are void: stop, find the cause, report, and get a new OK before any re-run):**
- **M1:** no arm creates landing excess under A1 or A2 (`dec_over` = 0 in every episode).
- **M2:** under A2, the four arms whose frozen runs never created excess reproduce their frozen episodes exactly, at every delay, with 0 refusals. They are R1i, METHOD, METHOD_oracle and ABL_worstcarfirst; in the frozen run their `dec_over` was 0 in all 1,920 episodes each. This checks the A2 code against 7,680 known episodes.
- **M3:** under A1, the corollary holds at every slot where it applies (`loop_violations` = 0).
- **M4:** in every episode, handovers started = requests − refusals (the environment starts exactly the admitted requests).

## 7. Decisions fixed now

| Outcome | What follows |
|---|---|
| HR1 held and HR2 held | The main result survives admission at preparation, with or without coordination. The main claim stands, and this check becomes its robustness section. |
| HR1 not held, HR2 held | The gain needs coordinated admission. The abstract and the claims must state that condition, and Theorem 2 explains why. Before the paper says an O-RAN RIC can do this, a standards check is needed (the E2 service models). METHOD under A1 minus under A2 is reported as the value of closed loops. |
| HR2 not held against R3 | Most of the gain over R3 was overflow that admission would prevent. The claim against R3 is withdrawn in its current form. The write-up pauses and we discuss before going on. |
| HR2 not held against R1, R1i or R2 only | Report as is and discuss. |
| HR3 interval not above 0 | The paper must not claim that planning on the landing state helps beyond coordinated admission. It credits the risk-after-landing objective plus admission, and the landing state stays as the analysis tool (T1, T2, G1). |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

No re-tuning of the method or of any baseline, whatever the outcome.

**Recorded guesses** (coded before the run; not hypotheses):
- METHOD under A1 at 1.0 s between 3.6 and 5.0 %;
- R3 under A1 at 1.0 s between 4 and 7 %;
- R3 under A2 at 1.0 s between 3.5 and 6 %;
- ABL_nolanding under A2 at 1.0 s between 4 and 6 %;
- HR1, HR2 and HR3 all held.

I am least sure about R3 under A2. It hands over about six times as often as METHOD, and most of its worst-car failures were L, which A2 removes.

## 8. Threats and limits

- **A1 is generous.** It credits departures and retries within the slot; a stricter real target would block more.
- **A2 is an ideal coordinator** (largest feasible subset, exact). Whether today's O-RAN interfaces let a RIC make admission atomic across several targets has not been checked.
- **Admission uses the landing state at decision time.** Cars that appear later can still collide; this is the same for every arm.
- **The planners are not told about admission.** An admission-aware METHOD would be a new design: design scenes first, then fresh test scenes. It is not part of this check.
- **The 60 held-out scenes are used a second time.** This is a paired sensitivity analysis of frozen code, with no tuning.
- **Model limits:** simulator only; load at most 1; full power and fixed sub-bands for every arm; one machine. The code is deterministic, so a rerun reproduces bit for bit.

## 9. Code and compute

- **New:**
  - `paper2ma_env_admit.py`: the option, plus the pure admission functions;
  - `admission_check.py`: modes `unit`, `gate`, `freeze`, `test --adm A1|A2 --hl H`, `analyse`.
  - Both are written, not run. Their md5s are recorded at the freeze.
- **Unchanged and checked before every step:**
  - the 20 files in `models/method_heldout/FREEZE.json` (method, harness, environment, policies, held-out pool, seed list);
  - the four frozen episode files (the A0 reference).
- **Compute (laptop):**
  - gates: a few minutes;
  - run: 8 cells (A1 and A2 × 4 delays), 10 arms × 480 episodes each, as 4 jobs (one per delay, A1 then A2), about 30 minutes;
  - then the analysis, once.
- **Outputs** (`models/admission_check/`): `UNIT.json`, `GATE.json`, `ADMIT_FREEZE.json`, `ADMIT_EPISODES_A{1,2}_hl{2,5,10,15}.csv`, `ADMIT_RESULT.json`, `ADMIT_RESULT.txt`, `STATUS.txt`.

## 10. Freeze record

- **Approval:** Daya, 20 Oct, "Freeze and run now".
- **Code at the freeze:**
  - `paper2ma_env_admit.py` md5 5a644059a82ba06a73ad2f1f47968c90;
  - `admission_check.py` md5 1b16d561d8a1e14a715cc5b0dc5e8e95.
- **G1 (unit tests), 1:42 PM IST: passed.**
  - All six hand-made cases gave the expected admitted sets.
  - 500 random cases, 0 mismatches. The closed-loop corollary applied in 161 of them.
  - `UNIT.json` md5 2048018cf432dfbb8f0a9e7bc95d999a.
- **G0 (admission off = frozen run), 1:42–1:43 PM IST: passed.** All 10 arms reproduced their frozen episodes exactly (first 40 seeds, 1.0 s, every column). `GATE.json` md5 f4e3ace8a1f4dbbbc1bbba73ee8b941c.
- **Versions:** Python 3.11.9, numpy 2.4.4, scipy 1.17.1, pandas 3.0.3.
- **`models/admission_check/ADMIT_FREEZE.json`** is written by `--mode freeze` after this file is in the folder. It records the md5 of:
  - this file;
  - the two code files;
  - `UNIT.json` and `GATE.json`;
  - the main test's `FREEZE.json`;
  - the four frozen episode files.

  Its own md5 and the run times go in the RESULT document, because writing them here would change this file's md5.
