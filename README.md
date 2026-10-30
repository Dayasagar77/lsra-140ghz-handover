# Code and results for the landing-state assignment paper (Paper 2)

**Article:** "Landing State Risk Assignment with Coordinated Admission for Delayed Vehicle Handover at 140 GHz" (title chosen by Daya on 25 Oct 2026).
**Authors:** Daya Sagar G and Deepa Nivethika S, School of Computer Science and Engineering, VIT Chennai.
**Package prepared:** 22 Oct 2026, for the article's data availability statement. **25 Oct 2026:** MIT licence added; section numbers follow the submitted article. **Updated 23 Oct 2026:** the link-budget check (the weaker link, §8.6), and Figures 4–7 and Tables 4–5 with 6 places. **Updated 25 Oct 2026 (release v1.1):** the 3GPP A3 baseline at the five TR 36.839 settings (§7, §8.1–8.2 and Table 5 of the article; `models/a3_check/`); the article's ablations table is now Table 6. **Updated 30 Oct 2026 (release v1.2), for the revised article (v6):** the stronger baselines (§7, §8.7, Tables 7–8; `models/baselines_check/`), the split of the gain into the planner's part and coordinated admission's part, and the blockage statistics (§3.1, §8.8, Table 9; `models/stored_checks/`), the sensitivity to the link and handover model and imperfect coordination (§8.9–8.10, Tables 10–13, Figure 9; `models/realism_check/` and `realism_deviation/`), and the protocol figure (§5.1, Figure 4). Figures 4–7 of release v1.1 are Figures 5–8 of the revised article; their files keep their names.
**Licence:** MIT (see `LICENSE`). **Repository and archive DOI:** added by the authors when they publish this package.

This package holds everything needed to check every number, figure and table in the article. It does **not** hold the V2X-Seq trajectory data or anything derived from them (see "Data").

## Contents

| Path | What it is |
|---|---|
| `code/*.py` | The simulator (16 vehicles, 4 roadside stations, 140 GHz uplink, 3D vehicle blockage), the method (LSRA, `lsra.py`), the baselines, the three admission models, every experiment runner and analysis, and the figure and table scripts. |
| `code/PREREG_*_FROZEN_*.md`, `code/PROTOCOL_Timing_*.md` | The frozen pre-registrations and the timing protocol. The freeze records (`models/*/…FREEZE.json`) hold their md5. |
| `code/PREREG_LSRA_MethodTest_Dev45_Heldout60_DRAFT_v1_20261020.md` | The design-stage plan, including the rule that chose β = 0.05 before any test scene was used. |
| `code/models/<test>/` | For each test: the per-episode results (CSV), the result files (`*_RESULT.json` and `.txt`), the freeze, gate and unit-test records, and the run logs. |
| `code/figures_ma/` | The article's figures (PDF with embedded fonts; PNG at 600 dpi). Files `F1`–`F3` are Figures 1–3, `F4_reserve_and_commit_protocol` is Figure 4, `F4_RLF_vs_delay…`, `F5_…`, `F6_…` and `F7_…` are Figures 5–8, and `F9_imperfect_coordination` is Figure 9. `FIG_DATA.csv` (Figures 5–8) and `FIG_DATA_v6.csv` (Figure 9) hold every plotted number, and `FIG_CHECK.txt` and `FIG_CHECK_v6.txt` the checks run when the figures were drawn. |
| `code/tables_ma/` | Tables 1–6 (`TABLES_MA.md`) and their CSV files, and Tables 7–13 (`TABLES_V6.md`; every value also in `TABLE_VALUES_V6.json`). In `TABLES_MA.md`, T5 is the article's Table 6 (ablations) and T6 its Table 5 (the A3 trigger). In Tables 10–12 the beam models named A1a, A1b, A2 and A3 in the result files are called G10, G20, DI and MA. |
| `code/EXPLORATORY_A3O_CrossFit_HonestPerSceneTuning_Spec_20261030.md` | The specification of the exploratory cross-fit of §8.7, written before it ran but after the bound A3-O had been seen. |
| `realism_deviation/` | The replay and the single re-analysis of the realism run (the first deviation in §7 of the article): `realism_diag_decover.py` with its record `DIAG_DECOVER.json`, and `realism_reanalyse_h2.py` with its outputs in `reanalysis/`. `reanalysis/REAL_RESULT.json` is the result file used in the article; the stopped output of the registered analysis is `code/models/realism_check/REAL_RESULT.json`. |
| `learned_forecast_cv_check/cv_forecast.py` | The straight-line forecast of the authors' learned-forecast study, read by the baselines run (PH-CV) and by the realism run. Their freeze records list it at `../../LEARNED_FORECAST_PAPER/cv_check/cv_forecast.py` relative to `code/`, so copy it there before a run or a freeze check. |
| `code/data_ids/MANIFEST_heldout_val60_20261020.csv` | The 60 held-out scene ids, with the md5 and size of each raw trajectory file. |
| `requirements.txt` | The exact package versions used. |
| `MD5SUMS.txt` | The md5 of every file in this package. |

**Not included from the link-budget check:** the dry run's output (`models/_dry_link/`, which holds the default-link data written under the variant names; a code check only, rebuilt by `_dry_link.py`), the geometry caches, and the `.partial` copies of the episode files (each is byte-identical to its final file).

**Not included from the A3 check:** the dry run's output (`models/_dry_a3/`, which holds the stored R0, R1, R1i, R2 and R3 episodes relabelled as the five A3 settings; a code check only, rebuilt by `a3_check.py --mode dry`) and the `.partial` copies of the episode files (each is byte-identical to its final file).

**Not included from the revision runs:** the dry-run outputs (`models/_dry_bc/` and `models/_dry_rc/`, stored episodes relabelled as the new arms for a code check, rebuilt by `baselines_check.py dry` and `realism_check.py dry`, and the first one, `models/_superseded_dry_bc_v1_20261028/`), the blockage frames that the realism run builds from the V2X-Seq geometry (`geom_cache_realism/`, which may not be redistributed; rebuilt by `realism_check.py prep` and checked against the md5s in `models/realism_check/PREP.json`), and the scratch folder of the stored checks' unit tests (`models/stored_checks/_unit_tmp/`).

### Which folder supports which result

| Article | Test | Folder | Plan |
|---|---|---|---|
| §6 (Design), design stage | β and the dropped stage, on 45 design scenes | `models/method_dev/` | `PREREG_LSRA_MethodTest_Dev45_Heldout60_DRAFT_v1_20261020.md` |
| §8.2, HM1 | admission at landing (A0), 4 places, 60 held-out scenes | `models/method_heldout/` | `PREREG_LSRA_MethodTest_Heldout60_FROZEN_20261020.md` |
| §8.1–8.2, HR1, HR2, HR3 | per-request (A1) and coordinated (A2) admission, 4 places | `models/admission_check/` | `PREREG_MA_AdmissionAtPreparation_Robustness_FROZEN_20261020.md` |
| §8.3, HL1, HL2 | the same at 5 places | `models/load_check/` | `PREREG_MA_LoadCheck_SparePlaces_C5_FROZEN_20261020.md` |
| §8.3, HL3, HL4 | the same at 6 places | `models/load_check_c6/` | `PREREG_MA_LoadCheck_LighterLoad_C6_FROZEN_20261022.md` |
| §8.6 (handovers that finish at different times), HJ1, HJ2 | handovers that land at different times | `models/jitter_check/` | `PREREG_MA_DelayDifference_Jitter_FROZEN_20261020.md` |
| §8.6 (a weaker link), HB1, HB2, HB3 | the main comparison (A2, 4 places) with a weaker link: 23 dBm, −8 dB, and both | `models/link_check/` | `PREREG_MA_LinkBudget_P23_T8_FROZEN_20261022.md` |
| §7, §8.1–8.2, Table 5, HA1, HA2 | the 3GPP A3 trigger at the five TR 36.839 settings, under A0, A1 and A2, at 4, 5 and 6 places and four delays | `models/a3_check/` | `PREREG_MA_A3Baseline_FROZEN_20261025.md` |
| §6 (complexity and decision time) | decision time per step | `models/timing/` | `PROTOCOL_Timing_DecisionTimePerStep_20261020.md` |
| §8.3, 16-episode statement | A1 against A0 at 5 places, per episode | `models/load_check/CHECK_C5_A1vsA0_LSRA_episodes_20261022.*` | — (`check_c5_a1_episodes.py`, read-only) |
| §3.1 (blockage statistics), §8.8, Table 9 | where the gain over today's practice comes from (the planner alone, coordinated admission alone, both), and how long blockages last, from the stored episodes and the geometry | `models/stored_checks/` | `PREREG_LSRA_StoredChecks_PlannerAdmissionSplit_BlockageDurations_FROZEN_20261028.md` |
| §7, §8.7, Tables 7–8, HN1, HN2 | the tuned A3 (and on the noise-free signal), A3 with load balancing, Na et al.'s predictive rule (perfect and straight-line forecasts, room check), LSRA-M, and the per-scene bound A3-O; stage-1 tuning on the design scenes in `tune/` and `tune_lb/` | `models/baselines_check/` | `PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation_FROZEN_20261028.md` |
| §8.7, cross-fit (exploratory) | per-scene A3 tuning chosen on half of each scene's episodes and scored on the other half | `models/baselines_check/A3O_CROSSFIT.*` | `EXPLORATORY_A3O_CrossFit_HonestPerSceneTuning_Spec_20261030.md` (`a3o_crossfit.py`) |
| §8.9–8.10, Tables 10–13, Figure 9, RW1–RW3, RC | 15 changed link and handover models, and ten imperfect-coordination settings, at 4 places | `models/realism_check/` (episodes, checks, freeze) and `realism_deviation/reanalysis/` (the analysis used) | `PREREG_LSRA_Realism_ChannelBlockageBeamsHandoverFailures_ImperfectCoordination_FROZEN_20261030.md` |

## Requirements

All reported numbers were produced on one laptop (Intel i9-12900H, Windows 11) with:
- Python 3.11.9;
- numpy 2.4.4, scipy 1.17.1, pandas 3.0.3, matplotlib 3.10.9;
- torch 2.5.1 and gymnasium 1.2.3.

torch and gymnasium are imported by shared modules of the code base. LSRA itself trains nothing.

`requirements.txt` pins these versions. Figures use the Arial font; another font changes only the look.

## Data (not included)

The vehicle geometry comes from the trajectory-forecasting data of V2X-Seq [Yu et al., CVPR 2023, doi:10.1109/CVPR52729.2023.00531], part of the DAIR-V2X family. Its providers distribute it under their own terms, which require registration. **This package therefore contains neither the trajectories nor the geometry pools built from them.** To rebuild the pools:

1. **Design pool (45 scenes).**
   - Source: the V2X-Seq-TFD example release, single-vehicle view.
   - Run from `code/`:
     ```
     python extract_dair_geometry.py --tfd_root <V2X-Seq-TFD-Example> --view single-vehicle --report --out dair_geometry_pool.csv
     ```
   - The file used for the article has md5 `3de813a13293b9810b21fe7683a71c26`.
2. **Held-out pool (60 scenes).**
   - Source: the full V2X-Seq-TFD, single-vehicle view, validation split.
   - Copy the 60 trajectory files listed in `data_ids/MANIFEST_heldout_val60_20261020.csv` into `<X>/single-vehicle/trajectories/`, checking each md5 against the manifest.
   - Then run:
     ```
     python extract_dair_geometry.py --tfd_root <X> --view single-vehicle --out dair_geometry_pool_heldout60_20261020.csv
     ```
   - The file used has md5 `3a68d055876a733ee21de37700385a75`, and `method_heldout.py` checks this md5.
   - The scenes were drawn by `make_heldout_pool.py` with a fixed rule, written in its header. Re-running that draw also needs the authors' list of scenes used in earlier work, which is not included.

If a rebuilt pool has a different md5, the freeze checks stop every runner. Please report it to the authors.

## How to reproduce

Run everything from `code/`. Each analysis refuses to overwrite its own result file, so move the stored file aside before re-running one. The freeze files reject any changed code, plan or pool.

1. **The result figures, without any data:**
   ```
   python make_figs_ma.py
   ```
   This redraws the article's Figures 5–8 (files `F4_…` to `F7_…`) from `models/admission_check/ADMIT_RESULT.json`, `models/load_check/LOAD_RESULT.json` and `models/load_check_c6/LOAD6_RESULT.json`. The new `figures_ma/FIG_DATA.csv` must be byte-identical to the stored one (md5 `dd7fadd71209be1dd635f6eafdfe8b35`).
   Figures 4 and 9 and Tables 7–13 of the revised article need no data either:
   ```
   python make_figs_v6.py ../realism_deviation/reanalysis/REAL_RESULT.json figures_v6_check
   python make_tables_v6.py
   ```
   The first writes into a new folder; its `FIG_DATA_v6.csv` must be byte-identical to the stored one (md5 `a4da6a711f1f785718900bbf6b0e9ad6`). The second rewrites `tables_ma/TABLES_V6.md`, which must keep md5 `d501718f4e44689af85bad0da6b3f3af`.
2. **Figures 1–3 and the tables (need both pools):**
   ```
   python make_figs_ma_f1f3.py
   python make_tables_ma.py
   ```
3. **The analyses, from the stored episodes (need the pools for the freeze checks):**
   ```
   python method_heldout.py --mode analyse
   python admission_check.py --mode analyse
   python load_check.py --mode analyse
   python load_check_c6.py --mode analyse
   python jitter_check.py --mode analyse
   python link_check.py --mode analyse
   python a3_check.py --mode analyse
   python stored_checks.py analyse
   python a3o_crossfit.py
   ```
   `a3o_crossfit.py` does not refuse: it rewrites `A3O_CROSSFIT.json` and `.txt`, so copy the stored ones aside to compare. The runners of the revision take the mode as a plain argument. Their analysis runs at the end of `pipeline`: with every episode file present, `python baselines_check.py pipeline` and `python realism_check.py pipeline` check the freeze, skip the stored cells and run the analysis, if its result file and `analyse.claim` have been moved aside. The registered realism analysis stops on its landing-excess check, as reported; the analysis used in the article is then `python ../realism_deviation/realism_reanalyse_h2.py`, which needs `../realism_deviation/DIAG_DECOVER.json` to say 'confirmed'.
4. **The experiments themselves (need the pools).** Each runner has the modes listed in §9 of its pre-registration, run in order: `unit` (where present), `gate`, `freeze`, then `test`, one call per cell. Examples:
   ```
   python load_check_c6.py --mode test --adm A1 --hl 10
   python jitter_check.py --mode test --cap 4 --j 2
   python link_check.py --mode test --var P23 --hl 10
   python a3_check.py --mode test --adm A2 --cap 4 --hl 10
   ```
   - The link-budget check's 23 dBm runs read their own geometry cache (`geom_cache_ma_p23`), which `link_check.py --mode warm` builds. Caches are not included. Each test run checks the cache's fingerprint against `models/link_check/LINK_FREEZE.json` and refuses to start on a different one.
   - The A3 check's runner has the modes `unit`, `dry`, `gate0`, `gate`, `freeze`, `test` and `analyse` (§6 and §9 of its plan); its gate G0 reproduced stored episodes of all three earlier load levels before any A3 episode was run.
   - The baselines and realism runners have the modes `unit`, `gate`, `dry`, `freeze` and `pipeline` (and `prep` for the realism run), in the order of §7 of their plans; six workers ran `pipeline --worker 0` to `--worker 5` in parallel, taking cells through claim files.
   - The code is deterministic: a re-run on the same machine and versions reproduces the stored episodes bit for bit.
   - The gates of the later runners (admission, load and delay-difference checks) confirmed this against the earlier runs before anything new was run.

## Integrity notes

- **Pre-registration.** Every registered test was pre-registered and frozen (md5 recorded) before it ran. Tests that did not support the method are reported: HR1, HL1, HJ2, HL3, HL4 and HN2, and RW1 in the models H2b and ALL. At 6 places, the method ties the simple rules.
- **Clerical incidents.** These are recorded in the logs and in the plan files. Examples:
  - the first gate attempt of the 6-place check stopped with a ValueError, was fixed and was run again (§6 of its plan);
  - an interrupted first timing run (`models/timing/_attempt1_killed_2050/`);
  - the first attempt of the link-budget check's pre-run checks stopped at import, before writing anything, on an assertion about the default threshold (`models/link_check/_superseded_attempt1_2213/`; §6 of its plan).
- **Deviations in the revision runs** (also in §7 of the article and in the plans and status files):
  - the realism run's registered analysis stopped on its landing-excess check in 98 episodes of H2a, H2b and ALL; the replay (`realism_deviation/DIAG_DECOVER.json`) showed a counting artefact, a re-establishment in the same slot that changed the target after the decision, and the analysis was run once more with that one check widened (`realism_deviation/realism_reanalyse_h2.py`, outputs in `realism_deviation/reanalysis/`);
  - the baselines run was cut by a Windows update restart and resumed; the stale claim file is kept in `models/_superseded_stale_claims_20261029_0221/`, and each episode depends only on its seed;
  - before the baselines freeze, the freeze record was fixed to skip two torch module names that are not files, and each worker's environment cache was cut from 4 to 2; the first check records are in `models/_superseded_baselines_check_checks_v1_20261028/`;
  - the cross-fit of the per-scene A3 bound is exploratory: it was decided after the bound had been seen.
- **Generative AI.** The authors used Claude (Anthropic) to assist with developing the method, writing the code and drafting the documents in this package, as stated in the article's Acknowledgments. The authors directed and reviewed all of it, and the first author approved every pre-registration before its run.
