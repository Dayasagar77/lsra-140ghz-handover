# PRE-REGISTRATION (FROZEN) — link-budget check: the main comparison at 23 dBm and with the 3GPP −8 dB threshold

**Date:** 22 Oct 2026, written from about 10:00 PM IST.
**Status:** FROZEN on 23 Oct 2026, about 9:55 PM IST, from DRAFT v1 (md5 4c6da60838200d44f9b2daa3841a8948), the version Daya approved. His answers on the morning of 23 Oct: "Freeze and run now"; for the literature, "Do one first", then, with Consensus out of searches until 1 Nov, "After web + IEEE". The plan itself is unchanged from DRAFT v1; only the records below were added (WARM's result, the literature routes, this status and §10). No episode has run with a changed link.
**Decision it serves:** on 22 Oct, Daya chose "Power and threshold check" among the steps before 1 Nov.
**Builds on:**
- the admission check (`PREREG_MA_AdmissionAtPreparation_Robustness_FROZEN_20261020.md` and its RESULT): the registered test HR2;
- the parameter sources (`SOURCES_ChannelAndLinkParameters_140GHz_…`, §5: "the main comparison at 23 dBm … and with the −8 dB threshold. Evaluation only, no tuning");
- the reviews of the earlier Vehicular Communications submission: reviewer 3, points 9 (hardware support for 30 dBm), 12 (a lower power) and 14 (results at −8 dB).

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

The paper's link is generous, in two ways.
- **Transmit power.** Every vehicle transmits at 30 dBm. 3GPP's vehicular devices in FR2 are limited to 23 dBm total radiated power [TS38101-2].
- **Failure threshold.** A slot is out of sync below an SINR of −13.8 dB. 3GPP mobility studies use −8 dB [TR36839], with a 1 s timer (T310) after one out-of-sync indication. Our timer is stricter (3 consecutive 0.1 s slots, no T310) and is not changed here.

The earlier reviewers asked about both, and our parameter sources note proposed this check (evaluation only, no tuning). It asks whether the main result still holds:
- **at 23 dBm;**
- **at −8 dB;**
- **with both together.**

The main result is HR2: at full load (4 places) with coordinated admission, LSRA beats all four baselines at a 1 s delay.

## 1. What changes: one variant at a time

| Variant | Transmit power (every vehicle, including the background interferers) | Out-of-sync threshold |
|---|---|---|
| P23 | 23 dBm | −13.8 dB |
| T8 | 30 dBm | −8 dB |
| P23T8 | 23 dBm | −8 dB |

- **The default threshold is −13.82 dB** (−92 dBm minus the −78.18 dBm noise floor), written −13.8 dB in the paper. T8 raises it by 5.82 dB.
- **N310 = 3, the beam gains (15 dBi serving, 0 dBi side lobe), the channel and the shadowing are unchanged.**
- **How the change is made.** The runner sets the constants of `paper2ma_env` before it builds the environment: `PT_MAX`, `INTERF_PT_DBM` and `SINR_OUT_DB`.
  - Every arm, the harness and the method's forecast read these constants at run time. A source scan of every module the runner imports found no copy by value (check U1d), and no episode reads Paper 1's copies of them (check U1e).
  - The rules' trigger and LSRA's risk forecast therefore follow the new threshold automatically, just as the simulator's own out-of-sync test does.
- **`PT_MIN` keeps its 7 dB distance to `PT_MAX`** (16 dBm at 23 dBm). It only scales the observation and the reward and clips the power. Every arm asks for full power in every slot (power action "up", clipped at `PT_MAX`), so every controlled vehicle transmits at `PT_MAX` throughout.
- **The cache trap.** The background interference is computed once per scene and cached on disk, under a key that does not include the power. A 23 dBm run reading the existing cache would silently keep the 30 dBm background. The 23 dBm variants therefore use their own cache folder (`geom_cache_ma_p23`).
  - It is filled once, in one process, before any run (WARM), and checked there scene by scene against the existing cache.
  - Its content is fingerprinted (one md5 over the 60 scene files) at WARM and again at the freeze. Every run checks the fingerprint and refuses to start on an incomplete or changed cache.
  - T8 keeps the 30 dBm background, so it uses the existing cache, which holds all 60 scenes and is fingerprinted the same way.
  - Nothing is deleted: a cache folder to be rebuilt is renamed aside.
- **Everything else is as in HR2:**
  - 4 places per station, under coordinated admission (A2);
  - the environment class `paper2ma_env_admit.AdmissionEnv`;
  - the 10 frozen arms and the harness;
  - β = 0.05.
- **No re-tuning.**

## 2. Data

- **Scenes and seeds:** the 60 held-out scenes and the frozen seed list `SEEDS_heldout60.csv`, 480 episodes per cell.
- **Delays:** 0.2, 0.5, 1.0 and 1.5 s.
- **Reuse:** this is the sixth use of these scenes. It is a paired sensitivity analysis of frozen code, with no tuning.

## 3. Arms

All 10 frozen arms (R0, R1, R1i, R2, R3, R3F, METHOD, METHOD_oracle, ABL_nolanding, ABL_worstcarfirst), under A2 at 4 places, for each variant and delay. That is 12 cells.

## 4. Primary (1.0 s): three registered tests, all reported, whatever they show

- **HB1 (P23), HB2 (T8) and HB3 (P23T8).**
  - For each B in {R1, R1i, R2, R3}: mean(B − METHOD) > 0 in per-episode worst-vehicle RLF.
  - Each comparison needs its 98.75 % scene interval (10,000 resamples, Bonferroni over four) above 0.
  - A test holds only if all four do.
- No claim of the form "holds under at least one variant".

## 5. Secondary (95 % scene intervals; reported, not tested)

- **Levels** for every arm and delay; handovers per vehicle-step; the causes of the worst vehicle's failures at 1.0 s.
- **Every arm minus METHOD,** at 1.0 s.
- **What the link change does** to each arm: variant minus default link, at 1.0 s. This is paired with the admission check's A2 runs, which use the same seeds.
- **Room for forecasts** (METHOD_oracle against METHOD) under each variant.

## 6. Checks

All checks before the freeze produce no new result: they use no changed link in an episode, or only reproduce stored episodes.

- **U1, link-patch unit checks (before the freeze).**
  - **(a)** On the first 5 held-out scenes, both computed fresh: the scene geometry is identical at 30 and 23 dBm, and the background interference is scaled by exactly 10^(−0.7), to a relative error below 10^−12.
  - **(b)** After reset at 23 dBm, every controlled vehicle transmits at 23 dBm, and the desired signal is exactly 7 dB lower than at 30 dBm (same scene and seed).
  - **(c)** LSRA's risk forecast is never lower at −8 dB than at −13.8 dB, and is higher somewhere.
  - **(d)** A source scan of every local module the runner imports (directly or not) finds no copy of the four link constants by value: no `from … import` of them, no default argument, and no use at import time outside their own definitions.
  - **(e)** With Paper 1's copies of the four constants (module `paper2_env`) set to "not a number", the first 4 held-out seeds at 1.0 s, under the default link, reproduce the admission check's stored A2 episodes exactly, for all 10 arms. So no episode reads them.
- **WARM (before the freeze).**
  - The 23 dBm cache is built for all 60 scenes, in one process, in a fresh folder.
  - Scene by scene, against the existing cache: the geometry is identical, and the background is scaled by exactly 10^(−0.7) (relative error below 10^−12).
  - The fingerprints of both caches are recorded.
- **G0 (before the freeze).** With the default link and a fresh, empty cache folder, the runner reproduces the admission check's A2 episodes at 4 places exactly (first 40 seeds, 1.0 s, all 10 arms, every column). This also tests that the scene precomputation, run fresh with today's code and packages, gives the same frames as the cached ones.
- **Same code.** Each of the U1, WARM and G0 records stores the md5 of the runner that wrote it. The freeze refuses unless all three match the runner being frozen.
- **If U1, WARM or G0 fails,** I stop and report.
- **First attempt (10:13 PM).** All four checks (U1, WARM, G0 and the dry run) stopped at import, before doing anything: the runner asserted the default threshold equal to −13.8 dB to within 10^−9, but it is −13.822 dB. The assertion now checks the threshold against its definition in Paper 1's module and its rounding to −13.8. No cache, record or result was written; the error files are kept in `models/link_check/_superseded_attempt1_2213/`, and the first runner version in `_superseded/`.
- **Results before the OK was asked (second attempt, launched 10:14:59 PM; runner md5 b3195de3…):**
  - **U1 passed** at 10:16:33 PM: geometry identical; background ratio error 2.8 × 10^−15 over 1,708 entries; desired signal 7.0 dB lower for every active vehicle; 0 copies by value in 23 modules; with Paper 1's constants set to "not a number", all 10 arms reproduced the stored episodes.
  - **G0 passed** at 10:21:21 PM: with a fresh cache folder, all 10 arms reproduced the admission check's A2 episodes exactly (40 seeds, every column).
  - **The dry run of the analysis passed** at 10:17:49 PM: the HB1–HB3 means equal the HR2 means (R1 5.67, R1i 5.48, R2 4.64, R3 3.11 pp); all 30 "variant minus default" contrasts are exactly 0; all 120 levels equal the admission check's; the must-hold checks pass on the default data.
  - **WARM** was still running (47 of 60 scenes at 10:21 PM) when the OK was asked.
- **WARM passed** at 10:23:31 PM on 22 Oct (read on 23 Oct, 9:43 PM, after the laptop link was restored): 60 scenes in 490.6 s; the geometry identical to the existing cache in every scene; the background ratio error 2.9 × 10^−15 over 19,068 entries; cache fingerprints 014cb3fc… (23 dBm) and 99ae7a99… (default).
- **Dry run of the analysis (before the plan is frozen).** The analysis is run on the admission check's finished A2 runs, with every "variant" set to that default-link data. It must reproduce the HR2 means for all four baselines and give exactly 0 for every "variant minus default" contrast. Output: `models/_dry_link/` (a code check, not a result).
- **Must hold in the run.** If any of these fails, the results are void: stop, find the cause, report, and get a new OK before any re-run.
  - **M1:** no landing excess under A2.
  - **M2:** 0 refusals for METHOD, METHOD_oracle, ABL_worstcarfirst and R1i in every cell. They plan on the landing state.
  - **M4:** handovers started = requests − refusals in every episode.

## 7. Decisions fixed now

| Outcome | What follows |
|---|---|
| HB1, HB2 and HB3 all held | The main result is not an artefact of the generous link. It holds at the 3GPP vehicular power limit, at the 3GPP threshold, and with both. The paper says so in a new subsection of Section 8, and the limits paragraph of Section 3.2 is updated. |
| some not held | The paper states under which link the main result holds, and reports where it does not, with the levels. The main result stays labelled with its link (30 dBm, −13.8 dB). |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

No re-tuning, whatever the outcome.

**Recorded guesses** (coded before the run; not hypotheses):
- METHOD at 1.0 s between 4 and 10 % under P23;
- METHOD at 1.0 s between 4 and 10 % under T8;
- METHOD at 1.0 s between 8 and 25 % under P23T8;
- HB1, HB2 and HB3 held.

**Why these guesses.** Weaker links make every method fail more often, mostly by waiting through handovers (cause W). The capacity failures that LSRA avoids at full load (causes L and B) should remain. My guesses about the lighter load were wrong the same evening, so these are recorded to make any surprise visible.

## 8. Threats and limits

- **One load (4 places) and one admission model (A2) only.** This is the setting of the main claim. The 6-place check showed that the gain is a full-load effect.
- **T8 is a stress test, not a replica of the 3GPP setting.** It combines the −8 dB threshold with our stricter timer, so it declares failures sooner than the 3GPP studies would.
- **The link changes apply to every vehicle alike.** The beam gain, the channel model and the shadowing are unchanged.
- **β was chosen with the default link.** It is not re-tuned.
- **Sixth use of the test scenes.**
- **Simulator only; one machine; deterministic code.**
- **Prior art.** The check adds no new method and no new claim. Daya asked for a literature check before the run (23 Oct). The open-web and IEEE Xplore routes are done, and nothing found changes this plan (`LITCHECK_LinkBudgetSensitivity_P23_T8_…_20261023.md`). The Consensus route follows on 1 Nov, when its searches reset.

## 9. Code and compute

- **New code:**
  - `link_check.py`, with modes `unit`, `warm`, `gate`, `freeze`, `test --var P23|T8|P23T8 --hl H` and `analyse`. Its md5 is given at the freeze.
  - `_dry_link.py`, the dry run of the analysis (§6).
  - It reuses `admission_check.py` (1b16d561…) and `paper2ma_env_admit.py` (5a644059…) unchanged.
  - It sets constants of `paper2ma_env` in memory. No simulator file is changed.
- **Checked before every step:** `models/method_heldout/FREEZE.json` and `models/link_check/LINK_FREEZE.json`. The latter is written at the freeze. It records:
  - the md5 of the runner, the simulator and policy modules it uses (14 files), this pre-registration (frozen copy), the U1, WARM and G0 records, the two earlier freeze files, and the admission check's four A2 episode files used for the paired contrasts;
  - the fingerprints of the two caches.
- **Compute (laptop):**
  - U1: about 3 minutes;
  - WARM: about 10–20 minutes;
  - G0: about 5–10 minutes;
  - the dry run: about 2 minutes;
  - these four run in parallel before the plan is sent;
  - run: 12 cells as 4 jobs, one per delay, each running the three variants in turn, about 45 minutes;
  - analysis: once, about 2 minutes.
- **Outputs (`models/link_check/`):** `UNIT.json`, `WARM.json`, `GATE.json`, `LINK_FREEZE.json`, `LINK_EPISODES_{P23,T8,P23T8}_hl{2,5,10,15}.csv`, `LINK_RESULT.json`, `LINK_RESULT.txt`, `STATUS.txt`.
- **New cache folders:** `geom_cache_ma_p23` (used by the run), `geom_cache_ma_gate` and `geom_cache_ma_unit` (checks only).

## 10. Freeze record

- Frozen by `link_check.py --mode freeze` (runner md5 b3195de3a26e204b7bdbb7a34abc4c49), which writes `models/link_check/LINK_FREEZE.json` with the md5 of this file, the runner, the 14 simulator and policy modules, the U1, WARM and G0 records, the two earlier freeze files and the admission check's four A2 episode files, and the fingerprints of both caches.
- U1, WARM and G0 were all written by that same runner (checked by the freeze).
- First attempt of the checks (22 Oct, 10:13 PM): stopped at import; see §6. No other deviation.
