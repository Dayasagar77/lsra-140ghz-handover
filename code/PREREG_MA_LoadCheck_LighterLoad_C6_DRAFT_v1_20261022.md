# PRE-REGISTRATION (DRAFT v1) — load check, third level: the three admission models at 6 places per station

**Date:** 22 Oct 2026, about 9:00 PM IST.
**Status:** DRAFT, for Daya's OK. Nothing has run at 6 places.
**Decision it serves:** on 22 Oct, Daya chose to add a third, lighter traffic level to Paper 2 ("Yes, plan it first"). It was offered as the one gap still worth closing: the Vehicular Communications reviewers asked for more traffic conditions (R5.5).
**Builds on:**
- the first load check (5 places): `PREREG_MA_LoadCheck_SparePlaces_C5_FROZEN_20261020.md` and `RESULT_MA_LoadCheck_C5_SparePlacesStillNeedCoordinatedAdmission_HL1NotHeld_HL2Held_20261020.md`;
- the admission check (4 places): `PREREG_MA_AdmissionAtPreparation_Robustness_FROZEN_20261020.md` and its RESULT;
- Theorem 2 (general form) in the theory note.

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

- **4 places per station (full load).** The method needs coordinated admission, and with it beats all four baselines clearly (HR2).
- **5 places (load 0.8).**
  - With coordinated admission it still beats all four, but only narrowly against the simple rules (HL2; lower bounds +0.01 to +0.09 pp).
  - Under today's per-request admission it is not shown to beat them (HL1).
- **With only two loads, a reader cannot see the trend.** A third point, 6 places (load 0.67), shows whether the method's lead fades, holds or reverses as the stations empty.

## 1. What changes: one variable

- **Every station has 6 places:** 16 vehicles and 24 places, load 0.67. At least 8 places are always free.
- **Why change the places and not the vehicles.** This is the same reason as at 5 places. The scenes, vehicles, geometry, interference and seeds stay identical, so every episode is paired with its 4-place and 5-place twins.
- **Everything else is as in the first load check:**
  - the environment class `paper2ma_env_admit.AdmissionEnv`, with capacity-aware attachment and admission at landing as the backstop;
  - the models A0 (admission only at landing), A1 (per request) and A2 (coordinated);
  - the frozen arms and harness.
- **Every policy reads the number of places from the environment** (`env.gnb_cap`). β stays at 0.05, the value chosen at 4 places. **No re-tuning.**

## 2. Data

- **Scenes and seeds:** the 60 held-out scenes (`dair_geometry_pool_heldout60_20261020.csv`, md5 3a68d055…) and the frozen seed list `SEEDS_heldout60.csv` (md5 ef0e8e0c…). That gives 480 episodes per cell.
- **Delays:** 0.2, 0.5, 1.0 and 1.5 s.
- **Reuse:** this is the fifth use of these scenes, after the held-out test, the admission check, the first load check and the delay-difference check. It is a paired sensitivity analysis of frozen code, with no tuning.

## 3. Arms

All 10 frozen arms (R0, R1, R1i, R2, R3, R3F, METHOD, METHOD_oracle, ABL_nolanding, ABL_worstcarfirst), at 6 places, under A0, A1 and A2. That is 12 cells.

## 4. Primary (1.0 s): two registered tests, both reported, whatever they show

- **HL3 (A1, per-request admission, 6 places).**
  - For each B in {R1, R1i, R2, R3}: mean(B − METHOD) > 0 in per-episode worst-vehicle RLF.
  - Each comparison needs its 98.75 % scene interval (10,000 resamples, Bonferroni over four) above 0.
  - HL3 needs all four.
- **HL4 (A2, coordinated admission, 6 places).** The same.
- No claim of the form "holds under at least one model".
- With these two, the paper reports nine registered tests.

## 5. Secondary (95 % scene intervals; reported, not tested)

- **Value of coordinated admission for METHOD:** A1 minus A2, at each delay.
- **What admission does for R3:** A1 minus A0, and A2 minus A0, at each delay.
- **Every arm minus METHOD under each model,** including the landing-state ablation (ABL_nolanding).
- **What the extra places do:** 6 minus 5 places, and 6 minus 4 places. Same seeds, 1.0 s, for METHOD, R1, R1i, R2 and R3 under each model.
- **The load curve at 1.0 s (descriptive):** each simple rule and R3 minus METHOD, at 4, 5 and 6 places, under each model.
- **Tables:** all arms, models and delays; handovers, requests and refusals per vehicle-step; causes of the worst vehicle at 1.0 s.

## 6. Checks

- **Gate G0a (before the freeze).** At 4 places with admission off, the runner reproduces the frozen held-out episodes exactly. This uses the first 40 seeds, 1.0 s, all 10 arms and every column.
- **Gate G0b (before the freeze).** At 5 places under A1, it reproduces the first load check's episodes exactly, on the same 40 seeds, 1.0 s, all 10 arms and every column. This tests the code path that sets the number of places.
- **If either gate fails,** I stop and report.
- **Must hold in the run.** If any of these fails, the results are void: stop, find the cause, report, and get a new OK before any re-run.
  - **M1:** no arm creates landing excess under A1 or A2.
  - **M2:** under A2, every arm whose A0 run never created excess at a given delay reproduces that A0 run exactly, with 0 refusals. At least one arm must qualify.
  - **M3:** 0 closed-loop violations under A1.
  - **M4:** handovers started = requests − refusals in every episode.
- **Code check, done before this draft was sent (22 Oct, 8:51–8:52 PM IST).**
  - The analysis was dry-run on the first load check's finished 5-place data (`_dry_load6.py`; output `models/_dry_load6/`, not a result). No 6-place episode existed.
  - It reproduced the first load check exactly:
    - the HL1 and HL2 means for all four baselines;
    - all 120 table entries and their minus-METHOD means;
    - the must-hold checks.
  - Every "same load minus same load" contrast was exactly 0 (15 of 15). The 4-place contrasts equalled the first check's "5 places minus 4 places".
  - `LOAD6_RESULT.json` of the dry run: md5 445458a8b4a8d1d5eb61128fad60e4cb.

## 7. Decisions fixed now

| Outcome | What follows |
|---|---|
| HL3 held and HL4 held | At load 0.67 the gain holds under both admission models. The paper says coordinated admission is needed at full load, and its need fades as load falls, while the method's lead remains. |
| HL3 not held, HL4 held | The gain needs coordinated admission at all three loads. The paper states the condition without a load qualifier. |
| HL4 not held | With a third of the places free, the method is not shown to beat all four baselines even with coordinated admission. The paper limits its claim to full and near-full load (1.0 and 0.8) and shows the load curve. |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

- **No re-tuning** of the method or of any baseline, whatever the outcome.
- **The abstract and Sections 7–8** are updated with whichever outcome occurs.

**Recorded guesses** (coded before the run; not hypotheses):
- METHOD under A2 at 1.0 s between 1.5 and 3.0 %;
- R1i under A2 at 1.0 s between 3.0 and 5.0 %;
- METHOD under A1 minus under A2, at 1.0 s, below 0.5 pp;
- HL3 held; HL4 held.

**Why these guesses.**
- With spare places, the simple rules already fail mostly while they wait for a handover they started after the outage began (cause W: R1i 4.15 of 4.50 pp at 5 places). More places do not fix that.
- The method plans ahead, so its lead may persist.
- Both tests are close calls: at 5 places the 98.75 % lower bounds against the simple rules were within 0.1 pp of zero.
- My guesses about per-request admission were wrong before. These are recorded so that any surprise is visible.

## 8. Threats and limits

- **Load is lowered by adding places only.** A different number of vehicles is not tested.
- **More places make the problem easier for every method.** The comparison of interest is within the 6-place runs.
- **β was chosen at 4 places.** It is not re-tuned.
- **A1 is the most generous per-request reading, and A2 an ideal coordinator.**
- **Simulator only; one machine; deterministic code.**
- **This is the fifth use of the test scenes.**
- **Prior art.** This check adds no new method and no new claim. The method's prior-art search is unchanged: two routes done, the Consensus route from 1 Nov. The standing rule asks for a prior-art check before every experiment. Because this is a sensitivity analysis of the same method, none is planned for it. **Daya to confirm.**

## 9. Code and compute

- **New code:** `load_check_c6.py` (md5 404f81e19e8f247dd5a0bef59252e86d), with modes `gate`, `freeze`, `test --adm A0|A1|A2 --hl H` and `analyse`.
  - It reuses `admission_check.py` (1b16d561…) and `paper2ma_env_admit.py` (5a644059…), both unchanged.
  - `load_check.py` (c9251e84…) is not changed.
- **Checked before every step:**
  - `models/method_heldout/FREEZE.json`;
  - `models/admission_check/ADMIT_FREEZE.json`;
  - `models/load_check_c6/LOAD6_FREEZE.json`. It is written at the freeze and records the md5 of:
    - the runner and the two admission files;
    - this pre-registration (frozen copy) and the gate record;
    - the two freeze files;
    - every finished file read for the 5-place and 4-place comparisons. These are `LOAD_FREEZE.json`, `LOAD_EPISODES_C5_{A0,A1,A2}_hl10.csv`, `HELDOUT_EPISODES_hl10.csv` and `ADMIT_EPISODES_{A1,A2}_hl10.csv`.
- **Compute (laptop):**
  - gates: about 4 minutes;
  - run: 12 cells as 4 jobs, one per delay (A0, then A1, then A2), about 40 minutes (the 5-place run took 38 minutes);
  - analysis: once, about 2 minutes.
- **Outputs (`models/load_check_c6/`):** `GATE.json`, `LOAD6_FREEZE.json`, `LOAD_EPISODES_C6_{A0,A1,A2}_hl{2,5,10,15}.csv`, `LOAD6_RESULT.json`, `LOAD6_RESULT.txt`, `STATUS.txt`.

## 10. Freeze record

Filled at the freeze.
