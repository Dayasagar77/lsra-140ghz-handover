# PRE-REGISTRATION (FROZEN) — load check: the three admission models when stations have spare places

**Date:** 20 Oct 2026. **Status:** FROZEN on Daya's "Freeze and run now" (20 Oct), after the gate passed. Nothing above §10 changed from `…_DRAFT_v1_20261020.md` except this title and status line. The run had not started when this was frozen.
**Decision it serves:** Daya chose "Load check first" after the admission check (20 Oct).
**Builds on:**
- the admission check: `PREREG_MA_AdmissionAtPreparation_Robustness_FROZEN_20261020.md` and `RESULT_MA_AdmissionAtPreparation_GainNeedsCoordinatedAdmission_HR1NotHeld_HR2Held_20261020.md`;
- Theorem 2 (general form) in the maths note.

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

- **What we found.** With 4 places per station (16 cars, 16 places), today's per-request admission blocks almost every rescuing move. Every method then fails about 9 % at 1 s, and about 7 % even at 0.2 s. With coordinated admission, our method reaches 3.6 %.
- **Why, according to the maths.** All 16 places are taken in 48 % of frames. Per-request admission then admits nothing, because the only possible moves are closed loops (Theorem 2). With free places, chains become possible, and per-request admission lets chains through.
- **The question.** Is the lock-up only a full-load effect? If the stations have spare places, does our method keep its gain under today's per-request admission? The answer decides how strongly the paper must state the condition "coordinated admission is needed".

## 1. What changes: one variable

- **Every station gets 5 places instead of 4:** 16 cars, 20 places, load 0.8. So at least 4 places are always free, and per-request admission can always form chains (Theorem 2, general form, with free landing capacity Φ ≥ 4).
- **Why change the places and not the cars.** Fewer cars would also change interference and the pairing of episodes. With 5 places, the scenes, the cars, the geometry, the interference and the seeds are all the same, so every episode is paired with its 4-place twin.
- **Everything else is as in the admission check:**
  - the environment class `paper2ma_env_admit.AdmissionEnv`, with capacity-aware attachment and admission at landing as the backstop;
  - the models A0 (admission only at landing), A1 (per request) and A2 (coordinated);
  - the frozen arms and harness.
- **Every policy reads the number of places from the environment** (checked in the code: R1, R2, R3, R3F, R1i and the method all use `env.gnb_cap`). β stays at 0.05, the value chosen at 4 places. **No re-tuning.**

## 2. Data

- **Scenes and seeds:** the 60 held-out scenes (`dair_geometry_pool_heldout60_20261020.csv`, md5 3a68d055…) and the frozen seed list `SEEDS_heldout60.csv` (md5 ef0e8e0c…), 480 episodes per cell. The scene a seed picks does not depend on the number of places.
- **Delays:** 0.2, 0.5, 1.0 and 1.5 s.
- **Reuse:** this is the third use of these scenes, for a paired sensitivity analysis of frozen code with no tuning.

## 3. Arms

All 10 frozen arms (R0, R1, R1i, R2, R3, R3F, METHOD, METHOD_oracle, ABL_nolanding, ABL_worstcarfirst), at 5 places, under A0, A1 and A2. That is 12 cells.

## 4. Primary (1.0 s): two families, both reported, whatever they show

- **HL1 (A1, per-request admission, 5 places).** For each B in {R1, R1i, R2, R3}: mean(B − METHOD) > 0 in per-episode worst-car RLF. Each needs its 98.75 % scene interval (10,000 resamples, Bonferroni over four) above 0. HL1 needs all four.
- **HL2 (A2, coordinated admission, 5 places).** The same.
- No claim of the form "holds under at least one model".

## 5. Secondary (95 % scene intervals)

- **Value of closed loops at 5 places:** METHOD under A1 minus under A2, at each delay. At 4 places it was +5.17 pp at 1 s.
- **What admission does for R3 at 5 places:** R3 under A1 minus A0, and under A2 minus A0, at each delay.
- **The no-admission comparison at 5 places:** every arm minus METHOD under A0.
- **What the spare places do (5 minus 4 places, same seeds, 1.0 s),** for METHOD, R1, R1i, R2 and R3, under each model.
- **Tables:** all arms, models and delays; handovers, requests and refusals per car-step; causes of the worst car at 1.0 s.

## 6. Checks

- **Gate G0 (before the freeze):** this runner at 4 places with admission off reproduces the frozen held-out episodes exactly (first 40 seeds, 1.0 s, all 10 arms, every column). If it fails, I stop and report.
- **Must hold in the run** (if any fails, the results are void: stop, find the cause, report, and get a new OK before any re-run):
  - **M1:** no arm creates landing excess under A1 or A2.
  - **M2:** under A2, every arm whose A0 run at 5 places never created excess at a given delay reproduces that A0 run exactly, with 0 refusals. At least one arm must qualify; expected: METHOD, METHOD_oracle, ABL_worstcarfirst and R1i.
  - **M3:** 0 closed-loop violations under A1.
  - **M4:** handovers started = requests − refusals in every episode.

## 7. Decisions fixed now

| Outcome | What follows |
|---|---|
| HL1 held and HL2 held | With spare places, the gain survives today's per-request admission. The paper says: coordinated admission is needed only near full load, and the measured lock-up at full load is Theorem 2 in action. |
| HL1 not held, HL2 held | Even with spare places, the gain needs coordinated admission. The paper states the condition without a load qualifier. |
| HL2 not held | Unexpected after the 4-place result. Report as is and discuss before the write-up continues. |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

No re-tuning of the method or of any baseline, whatever the outcome.

**Recorded guesses** (coded before the run; not hypotheses):
- METHOD under A1 at 1.0 s between 2 and 4 %;
- METHOD under A1 minus under A2 at 1.0 s below 1 pp;
- R1 under A1 at 1.0 s between 4 and 7 %;
- R3 under A1 at 1.0 s between 4 and 7 %;
- HL1 held; HL2 held.

My guesses for per-request admission were badly wrong last time. These are my honest best guesses again, and they are recorded so that any surprise is visible.

## 8. Threats and limits

- **One extra load level only (0.8).** A full load curve (for example 6 places, or fewer cars) is not part of this check.
- **More places also makes the whole problem easier.** Every arm should fail less than at 4 places, so the comparison of interest is within the 5-place runs.
- **β was chosen at 4 places,** which could disadvantage the method. It is not re-tuned.
- **A1 is the most generous per-request reading, and A2 an ideal coordinator** (as in the admission check).
- **Simulator only; one machine;** deterministic code.

## 9. Code and compute

- **New:** `load_check.py`, with modes `gate`, `freeze`, `test --adm A0|A1|A2 --hl H` and `analyse`. It reuses the frozen `admission_check.py` helpers and `paper2ma_env_admit.py`, both unchanged. It is written, not run.
- **Checked before every step:**
  - `models/method_heldout/FREEZE.json` (the 20 frozen files);
  - `models/admission_check/ADMIT_FREEZE.json`;
  - `models/load_check/LOAD_FREEZE.json`, written at the freeze with the md5 of the runner, both admission files, this pre-registration (frozen copy), the gate record and the two freeze files.
- **Compute (laptop):**
  - gate: about 2 minutes;
  - run: 12 cells of 10 arms × 480 episodes, as 4 jobs (one per delay, A0 then A1 then A2), about 45 minutes;
  - then the analysis, once (about 5 minutes).
- **Outputs (`models/load_check/`):** `GATE.json`, `LOAD_FREEZE.json`, `LOAD_EPISODES_C5_{A0,A1,A2}_hl{2,5,10,15}.csv`, `LOAD_RESULT.json`, `LOAD_RESULT.txt`, `STATUS.txt`.

## 10. Freeze record

- **Approval:** Daya, 20 Oct, "Freeze and run now".
- **Code at the freeze:**
  - `load_check.py` md5 c9251e84de8d670adb802a85ed4d6c34;
  - `admission_check.py` 1b16d561… (unchanged);
  - `paper2ma_env_admit.py` 5a644059… (unchanged).
- **Gate G0, 2:26–2:27 PM IST: passed.** At 4 places with admission off, all 10 arms reproduced their frozen episodes exactly (first 40 seeds, 1.0 s, every column). `GATE.json` md5 08e6a7066234f0f8ff29b535d4f257fe.
- **`models/load_check/LOAD_FREEZE.json`** is written by `--mode freeze` after this file is in the folder, and it records this file's md5. Its own md5 and the run times go in the RESULT document.
