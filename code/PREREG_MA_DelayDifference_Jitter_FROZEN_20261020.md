# PRE-REGISTRATION (FROZEN) — delay-difference check: do coordinated swaps still work when handovers finish at different times?

**Date:** 20 Oct 2026. **Status:** FROZEN on Daya's "Freeze and run now" (20 Oct, about 9:15 PM), after the unit tests and the gate passed. Nothing above §10 changed from `…_DRAFT_v1_20261020.md` except this title and status line. The run had not started when this was frozen.
**Decision it serves:** Daya chose "all the options" (20 Oct, about 9:10 PM). This includes the delay-difference check, decision D3 in the outline.
**Builds on:**
- the admission check (`RESULT_MA_AdmissionAtPreparation_…`) and the load check (`RESULT_MA_LoadCheck_C5_…`);
- the assumption stated in outline v2 §2 and in the prior-art record: "a loop under coordinated admission lands within one 0.1 s slot".

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

- **What we assumed.** Coordinated admission (A2) accepts a swap or a longer loop as a whole. In every run so far, each handover took exactly 1.0 s, so the cars of a loop landed in the same 0.1 s slot.
- **Why that may not hold.** Real handovers do not finish at exactly the same moment. If a car lands before the car it replaces has left, the station has one car too many for a few slots. The weakest car there is unserved, and after N310 = 3 slots in a row it counts as a radio-link failure.
- **The question.** Does LSRA still beat the baselines under A2 when each handover takes 1.0 s ± up to 0.2 s, drawn separately for each car?
- **What the answer decides.** Whether the paper's 6G requirement is "admit the set together" or "admit the set together and make it land together".

## 1. What changes: one variable

- **Delays differ from car to car.** Each started handover takes L + d slots:
  - L = 10 slots (1.0 s);
  - d is uniform on {−J, …, +J};
  - J = 1 (±0.1 s) and J = 2 (±0.2 s).
  
  So two cars of a swap can land up to 0.2 s (J = 1) or 0.4 s (J = 2) apart.
- **Paired draws.** d comes from a table fixed by the episode seed, per car and slot. Every method gets the same d for a car that starts a handover in the same slot, so the comparison stays paired.
- **The planners keep planning with the nominal L.** They do not know d.
- **Everything else is as in the admission and load checks:**
  - `AdmissionEnv` with coordinated admission (A2);
  - capacity-aware attachment, with admission at landing as the backstop;
  - the frozen arms and harness, β = 0.05.
- **No re-tuning.**

## 2. Data

- The 60 held-out scenes and the 480 frozen seeds. This is the fourth use of these scenes, for a paired sensitivity analysis of frozen code, with no tuning.
- 4 and 5 places per station.
- 1.0 s nominal delay only.

## 3. Arms and cells

- All 10 frozen arms under A2, at J = 1 and J = 2, with 4 and 5 places: four cells.
- **J = 0 is the frozen A2 runs:**
  - `ADMIT_EPISODES_A2_hl10.csv` (4 places);
  - `LOAD_EPISODES_C5_A2_hl10.csv` (5 places).
  
  Gate G0 checks that the new class with J = 0 reproduces them.

## 4. Primary: two families, both reported whatever they show

- **HJ1 (4 places, J = 2).** For each B in {R1, R1i, R2, R3}: mean(B − METHOD) > 0 in per-episode worst-car RLF.
  - Each needs its 98.75 % scene interval above 0 (10,000 resamples, Bonferroni over four).
  - HJ1 needs all four.
- **HJ2 (5 places, J = 2).** The same.
- **J = 2 is primary because it is the harder case;** J = 1 is secondary.

## 5. Secondary (95 % scene intervals)

- The same four comparisons at J = 1, at both loads.
- **Cost of the delay differences for each arm:** the arm at J minus the same arm at J = 0, paired by episode.
- Excess car-steps per car-step (cars attached beyond capacity), and the share of handovers that land early or late.
- **Tables:** all arms and cells; handovers and refusals per car-step; causes of the worst car.

## 6. Checks

**Before the freeze** (if any fails, I stop and report):
- **Unit tests:**
  - **U1:** delays lie in [L − J, L + J], and a handover lands after exactly that many active slots.
  - **U2:** the delay table depends on the seed only.
  - **U3:** d is uniform (pooled over the 480 seeds, each value within ±0.01 of 1/(2J + 1)).
  - **U4:** J = 0 changes no delay.
- **Gate G0:** with J = 0, the new class reproduces the frozen A2 runs at 4 and 5 places (first 40 seeds, all 10 arms, every column, plus requests and refusals).

**Must hold in the run** (if any fails, the results are void: stop, find the cause, report, and get a new OK before any re-run):
- **M1:** no landing excess is created by the decisions (A2).
- **M4:** handovers started = requests − refusals.
- **M5:** every landing came after exactly its delay.
- **M6:** every delay lies within [L − J, L + J].
- **M7:** the new class and the harness count the same handover starts.

## 7. Decisions fixed now

| Outcome | What follows |
|---|---|
| HJ1 and HJ2 held | The gain survives handovers that finish up to 0.4 s apart. The paper states the simultaneity assumption together with this result. |
| HJ1 held, HJ2 not held | At full load the gain survives. With spare places the narrow margin does not. Report both. |
| HJ1 not held | Coordinated admission must also make the cars of a loop land together (synchronised execution). The paper states this as part of the 6G requirement and reports the measured cost. |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

No re-tuning of the method or of any baseline, whatever the outcome.

**Recorded guesses** (not hypotheses):
- METHOD at 4 places, J = 2 minus J = 0, below +0.5 pp;
- METHOD at 4 places, J = 1 minus J = 0, below +0.2 pp;
- HJ1 held;
- HJ2 not held (the margins at 5 places were narrow: lower bounds +0.01 to +0.09 pp).

## 8. Threats and limits

- **Simple delay model.** Delay differences are uniform and independent per car. Real delays may be correlated, or skewed (mostly late).
- **No delay knowledge.** The planners do not know d; a real RIC might learn the delays. That is not tested.
- **Depends on N310.** With N310 = 3, an excess shorter than 3 slots causes no radio-link failure, so the result depends on N310 (stated in the paper).
- **Reused scenes.** The held-out scenes are reused for the fourth time.
- **Simulator only.**

## 9. Code and compute

- **New:**
  - `paper2ma_env_jitter.py` (`JitterEnv`, a subclass of `AdmissionEnv`);
  - `jitter_check.py` (modes `unit`, `gate`, `freeze`, `test --cap C --j J`, `analyse`).
  
  Both are written, not run.
- **Checked before every step:**
  - `models/method_heldout/FREEZE.json`;
  - `ADMIT_FREEZE.json` and `LOAD_FREEZE.json`;
  - the two frozen A2 episode files;
  - `models/jitter_check/JITTER_FREEZE.json`, written at the freeze with the md5 of the new code, the frozen copy of this file, the unit and gate records and the files above.
- **Compute (laptop):**
  - unit tests: about 2 minutes;
  - gate: about 5 minutes;
  - run: four jobs, one per cell, about 15 minutes;
  - analysis: once, about 5 minutes.
- **Outputs (`models/jitter_check/`):** `UNIT.json`, `GATE.json`, `JITTER_FREEZE.json`, `JITTER_EPISODES_C{4,5}_J{1,2}.csv`, `JITTER_RESULT.json`, `JITTER_RESULT.txt`, `STATUS.txt`.

## 10. Freeze record

- **Approval:** Daya, 20 Oct, "Freeze and run now".
- **Code at the freeze:**
  - `jitter_check.py` md5 62caddb202043b8bb5ea065803391206;
  - `paper2ma_env_jitter.py` md5 3bbfba25b54b28aca1f18023ccf57f58;
  - `admission_check.py` 1b16d561… and `paper2ma_env_admit.py` 5a644059… (both unchanged).
- **Unit tests, 9:18 PM IST: passed.** `UNIT.json` md5 7c3f15b8d47212887f8594e688a60c0e.
  - **U1:** every delay in range; early and late delays both present; every landing after exactly its delay.
    - J = 1: METHOD 36 starts and 34 landings, R3 144 and 113;
    - J = 2: METHOD 36 and 34, R3 137 and 108.
    
    The other flights were still in the air at the end of an episode, or were cancelled when their car left and came back.
  - **U2:** the delay table depends on the seed only.
  - **U3:** frequencies were 0.333 / 0.333 / 0.334 (J = 1) and 0.200 / 0.200 / 0.200 / 0.200 / 0.201 (J = 2).
  - **U4:** J = 0 gives all 42 delays equal to L, and no mismatch.
- **Gate G0, 9:18–9:20 PM IST: passed.** All 10 arms at 4 places and all 10 at 5 places reproduced the frozen A2 runs exactly (first 40 seeds, every column, requests and refusals). `GATE.json` md5 7d5d918200a201d9119176ec38433bb6.
  - Recorded only: at J = 0, LSRA had 30 excess car-steps at 4 places and 2 at 5 places over these 40 seeds. These come from cars that appear at a station that is full at landing.
- **`models/jitter_check/JITTER_FREEZE.json`** is written by `--mode freeze` after this file is in the folder, and it records this file's md5. Its own md5 and the run times go in the RESULT document.
