# PRE-REGISTRATION — stage 2: frozen test of the landing-state method on 60 new DAIR scenes

**Date:** 20 Oct 2026. **Status:** FROZEN at 12:58 PM IST, on Daya's "Freeze and run now". Nothing above §8 changed after his OK; the freeze record is §8.
**Builds on:**
- `PREREG_LSRA_MethodTest_Dev45_Heldout60_DRAFT_v1_20261020.md`;
- stage 1 and its follow-up (`RESULT_LSRA_Stage1_Dev45_LandingStateWorks_WorstCarStepHurts_20261020.md`);
- the maths note (`MATH_LandingStateRotationAssignment_DRAFT_v1_20261020.md`, v2 text).

**Change from the draft, made on design data only and documented:** the method no longer minimises the worst car's risk first, because on the 45 design scenes that step caused churn and was worse at every delay. The method is now: landing-state assignment minimising total risk after landing plus β per move, with β = 0.05 set by the pre-declared rule. The worst-car-first version is kept as an ablation.
**Where:** the laptop, Windows Python. Nothing in the cloud. **No policy has been run on the held-out scenes.**

## 1. Data

- **Scenes:** `dair_geometry_pool_heldout60_20261020.csv` (md5 3a68d055876a733ee21de37700385a75). 60 scenes drawn with seed 20261020 from DAIR-V2X-Seq single-vehicle **val**, excluding July's 60 val scenes and anything in train; manifest md5 1ef354d3beb8e306bee61fd83c4c81ac. The unchanged extractor was used (eb4cc1ce…).
- **Seeds:** `models/method_heldout/SEEDS_heldout60.csv` (md5 ef0e8e0ce83cdb0369154bd35b9d22e4). 8 per scene, the first seeds of 930,000,000 + j that pick each scene; 480 episodes; checked by `env.reset` with 0 mismatches; largest j 1,028.
- **Environment:** `paper2ma_env_attach.CapacityAttachEnv` with capacity-aware attachment, split None (all 60 scenes); 16 cars, 4 stations × 4 places; delays 0.2, 0.5, 1.0 and 1.5 s.

## 2. Arms (`method_heldout.py`, md5 39118bbceafde03a0549982968b4d4d5)

| Name | What it is |
|---|---|
| R0, R1, R1i, R2, R3 | the existing methods (R1i = capacity check counting cars on their way) |
| R3F | R3 with perfect foresight at landing (reference; uses the future) |
| **METHOD** | **landing-state assignment, total risk after landing (H = L+1, current geometry) + β 0.05 per move** |
| METHOD_oracle | the same with the true future geometry in the window (reference; uses the future) |
| ABL_nolanding | the same planned on current stations, every car movable (ablation) |
| ABL_worstcarfirst | the version-1 method, worst car first, β 0.05 (ablation) |

## 3. Primary claim (1.0 s)

**HM1, the method beats every deployable existing method:** for each B in {R1, R1i, R2, R3}, mean(B − METHOD) > 0 in per-episode worst-car RLF. Each needs its **98.75 % scene interval** (bootstrap over the 60 scenes, 10,000 resamples, Bonferroni over four) above 0. HM1 needs all four.

## 4. Secondary (95 % scene intervals; reported, no claims)

- **HM3:** R3's cost of delay (1.0 s minus 0.2 s) minus METHOD's.
- **Short-delay check:** METHOD − R3 at 0.2 s.
- **All arms at all delays,** with every arm minus METHOD.
- **Room for forecasting:** METHOD_oracle − METHOD.
- **Ablations:** ABL_nolanding − METHOD and ABL_worstcarfirst − METHOD.
- **Mechanisms:** causes A/L/F/W/B, landing excess created per arm, handovers per car-step.

**Mechanism check (must hold):** METHOD's created and total landing excess are 0 in every episode.

## 5. What follows (fixed now)

- **HM1 held:** the method is Paper 2's main result. Next: write-up; the Consensus prior-art route (1 Nov) before any novelty claim; the learned forecast (deep-RL part) aimed at the remaining "waiting" failures.
- **HM1 not held against some baselines:** report as is and discuss. No re-tuning on these scenes.
- **ABL_nolanding within 1 pp of METHOD with its interval covering 0:** the landing state is not what helps; discuss.

**Recorded guesses** (coded in `method_heldout.py` before stage-1's follow-up finished; not hypotheses):
- METHOD at 1.0 s between 3 and 5 %;
- R3 between 9 and 11 %;
- HM1 held;
- ABL_nolanding within 1 pp of R3;
- METHOD within 0.5 pp of R3 at 0.2 s;
- METHOD hands over more often than R1 and R2.

## 6. Threats and limits

- METHOD uses the environment's propagation model at mean fading (no future information). The forecast references show the extra value of knowing the future.
- Interference model as in the environment. No admission at handover preparation. Load at most 1 (16 cars, 16 places; some scenes sparser).
- The 45 design scenes shaped the method, including the drop of the worst-car step and β; the 60 held-out scenes did not.
- One machine. Deterministic code: a rerun reproduces bit for bit.

## 7. Compute

10 arms × 4 delays × 480 episodes, run as 4 jobs (one per delay), about 15–20 minutes; then the analysis once. `test` refuses to run without `FREEZE.json`, if any frozen file changed, or if a delay already ran. `analyse` runs once.

## 8. Freeze record

- **Approval:** Daya, 20 Oct, "Freeze and run now".
- **`models/method_heldout/FREEZE.json`:** written 07:28:20 UTC (12:58 PM IST), md5 **76f010f740673ef3fa66f4cf8914b0a5**. It records β = 0.05 and the md5 of 20 files the run depends on, none missing:
  - the method, harness, runner and environment files;
  - the policies;
  - the channel and blockage modules;
  - the held-out pool;
  - the held-out seed list;
  - `DEV2_RESULT.json`.

  The values checked at writing include: `method_heldout.py` 39118bbc…, `lsra.py` d344b861…, `SEEDS_heldout60.csv` ef0e8e0c…, `DEV2_RESULT.json` d06e4b7b….
- **Run:** 4 jobs (one per delay) started at 12:58 PM IST on the laptop.
