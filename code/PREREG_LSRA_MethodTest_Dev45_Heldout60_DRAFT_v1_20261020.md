# PRE-REGISTRATION (DRAFT v1) — test of the landing-state rotation method on 60 new DAIR scenes

**Date:** 20 Oct 2026. **Status:** DRAFT. Two stages, each needs Daya's OK:
- **(1) development check** on the 45 design scenes;
- **(2) the frozen test** on the 60 held-out scenes, which no policy has touched.

**For:** Paper 2, multi-car version. The method is in `MATH_LandingStateRotationAssignment_DRAFT_v1_20261020.md` (§4); the motivation is `RESULT_MA_Remeasure2_…_D1_20261020.md` (rule D1).
**Where:** the laptop, Windows Python. Nothing in the cloud.

## 1. Code (written and tested 20 Oct; no failure-rate number has been looked at for the new arms)

- **`lsra.py`** (md5 d344b8614629917c6461e517e2bca8c3):
  - the method LSRA: landing state, risk after landing over H = L+1 steps, worst-car-first then total risk plus β per move, exact;
  - the one-line fix R1i (R1 counting cars on their way, decision by decision);
  - two ablations.
- **`method_harness.py`** (md5 3b3d3ed06eed670b8d6fca0a6c02c0fa): the 20 Oct harness plus two read-only measures:
  - `dec_over`: excess over capacity at landing created by a step's decisions;
  - `land_over`: total excess at landing after the step.

  The old "blind start" count is kept but is known to overstate blindness for loop-moving planners.
- **Environment:** `paper2ma_env_attach.CapacityAttachEnv` (capacity-aware attachment, 15039749…); the base environment is unchanged (42c9dba3…).
- **Tests passed** (`lsra_tests.py` md5 9091ffeaa769abb01080a5073067f163, 11:53 AM IST; `models/method_dev/LSRA_TESTS.json`):
  - H: the new harness equals the 20 Oct harness on every shared output.
  - U1: the solver equals brute force on 500 random instances with ties (0 mismatches).
  - U2: on 12 design episodes at 0.2 s and 1.0 s, LSRA (current and perfect forecast, with and without min-max) creates no landing excess and leaves none (dec_over = land_over = 0). R1i creates none.
  - For comparison, in the same episodes R3 created 17 and 34 units of landing excess at 0.2 s and 1.0 s, and the "landing state off" ablation 53 and 89.
  - About 0.08–0.11 s per episode.
  - An earlier run of these tests (archived in `models/method_dev/_superseded/`) used only the total measure, which counts excess left by a car that appears; it was replaced by the created-excess measure before any conclusion was drawn.

## 2. Arms

| Arm | What it is | Role |
|---|---|---|
| R0 | load-aware rule (no capacity check) | baseline |
| R1 | R0 + capacity check on current load | baseline |
| R1i | R1 counting cars on their way | baseline (the one-line fix) |
| R2 | joint assignment of the cars in outage | baseline |
| R3 | joint assignment of every car, every step | baseline (best at 0.2 s) |
| R3F | R3 with perfect foresight at landing | reference (uses the future) |
| **LSRA** | the method, current geometry, no prediction | **the method** |
| LSRA-oracle | LSRA with the true future geometry in the window | reference (uses the future) |
| LSRA−landing | LSRA planned on the current stations, every car movable | ablation |
| LSRA−minmax | LSRA without the worst-car stage | ablation |

**Fixed now:**
- H = L+1;
- the risk model as in `lsra.py`: mean SINR from the environment's channel model at mean shadow fading, with interference from present cars on the same sub-band; p = Φ((SINR_out − mean SINR)/σ(depth));
- full power and unchanged sub-bands for every arm;
- β is fixed by the rule in §3.

## 3. Stage 1: development check (45 design scenes; needs OK)

- **Data:** the 20 Oct seed lists (train 240 + test 120 episodes), capacity-aware attachment, delays 0.2, 0.5, 1.0 and 1.5 s, all ten arms. These scenes are design data; nothing from them is a claim.
- **β rule (fixed now):** LSRA is run with β ∈ {0.001, 0.01, 0.05}. The held-out value is the largest β whose worst-car RLF at 1.0 s (all 45 scenes) is within 0.5 pp of the best of the three. The aim is fewer handovers at no reliability cost.
- **Report to Daya:** the table of all arms, handovers per car-step, causes, created landing excess, and the chosen β. Any other change to the method after this stage is written down with its reason before freezing. **No held-out scene is run in stage 1.**

## 4. Stage 2: frozen test (60 held-out scenes; needs OK after stage 1)

- **Data:**
  - pool: `dair_geometry_pool_heldout60_20261020.csv` (md5 3a68d055876a733ee21de37700385a75; 60 scenes drawn with seed 20261020 from DAIR-V2X-Seq single-vehicle **val**, excluding July's 60; manifest md5 1ef354d3…);
  - seeds: 8 per scene, balanced, the first seeds of 930,000,000 + j that pick each scene (480 episodes);
  - environment: `CapacityAttachEnv`, split None.
- **Primary claim (1.0 s), HM1, LSRA beats every deployable existing method:** for each B in {R1, R1i, R2, R3}, mean(B − LSRA) > 0. Each difference needs its 98.75 % scene interval above 0 (Bonferroni over four). The claim needs all four.
- **Secondary (95 % scene intervals; reported, no claims):**
  - HM3, robustness: the cost of delay (1.0 − 0.2 s) is smaller for LSRA than for R3;
  - short-delay check: LSRA − R3 at 0.2 s;
  - all arms × delays, against R0 and R3F;
  - LSRA-oracle − LSRA (room for forecasting);
  - each ablation − LSRA at 1.0 s (does each part matter);
  - causes A/L/F/W/B;
  - created landing excess per arm;
  - handovers per car-step.
- **Mechanism check (must hold, not a statistic):** LSRA's created and total landing excess are 0 in every episode.
- **What follows (fixed now):**
  - HM1 held: the method is Paper 2's main result. Next comes the write-up, the Consensus check (1 Nov), and the learned forecast as the deep-RL part.
  - HM1 not held against some baselines: report as is and discuss. No re-tuning on the held-out scenes.
  - LSRA−landing within 1 pp of LSRA, with an interval covering 0: the landing state is not what helps. Discuss.

## 5. Recorded guesses (not hypotheses)

- LSRA at 1.0 s on the held-out scenes: 3–5 %; R3 9–11 %.
- LSRA beats all four baselines at 1.0 s.
- LSRA−landing is close to R3.
- LSRA within 0.5 pp of R3 at 0.2 s.
- LSRA hands over more often than R1 and R2.

## 6. Threats and limits

- LSRA knows the environment's propagation model (not the future fading, not the future geometry). This is standard for a model-based planner, and LSRA-oracle shows the extra value of forecasts.
- The interference model is the environment's (from every present car on the same sub-band). Admission at handover preparation is not modelled.
- The held-out scenes are a little less crowded (13.7 of 16 cars present on average, against 14.4).
- One machine. β and H are fixed before the test.
- The design data (45 scenes) shaped the method; the held-out scenes did not.

## 7. Compute

- Stage 1: about 10 arms (12 with the β variants) × 4 delays × 360 episodes, at about 0.1 s per episode: about 30 minutes.
- Stage 2: 10 arms × 4 delays × 480 episodes: about 35 minutes.
