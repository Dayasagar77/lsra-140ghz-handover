# Code and results for the landing-state assignment paper (Paper 2)

**Article:** "Landing State Risk Assignment with Coordinated Admission for Delayed Vehicle Handover at 140 GHz" (title chosen by Daya on 25 Oct 2026).
**Authors:** Daya Sagar G and Deepa Nivethika S, School of Computer Science and Engineering, VIT Chennai.
**Package prepared:** 22 Oct 2026, for the article's data availability statement. **25 Oct 2026:** MIT licence added; section numbers follow the submitted article. **Updated 23 Oct 2026:** the link-budget check (the weaker link, §8.6), and Figures 4–7 and Tables 4–5 with 6 places.
**Licence:** MIT (see `LICENSE`). **Repository and archive DOI:** added by the authors when they publish this package.

This package holds everything needed to check every number, figure and table in the article. It does **not** hold the V2X-Seq trajectory data or anything derived from them (see "Data").

## Contents

| Path | What it is |
|---|---|
| `code/*.py` | The simulator (16 vehicles, 4 roadside stations, 140 GHz uplink, 3D vehicle blockage), the method (LSRA, `lsra.py`), the baselines, the three admission models, every experiment runner and analysis, and the figure and table scripts. |
| `code/PREREG_*_FROZEN_*.md`, `code/PROTOCOL_Timing_*.md` | The frozen pre-registrations and the timing protocol. The freeze records (`models/*/…FREEZE.json`) hold their md5. |
| `code/PREREG_LSRA_MethodTest_Dev45_Heldout60_DRAFT_v1_20261020.md` | The design-stage plan, including the rule that chose β = 0.05 before any test scene was used. |
| `code/models/<test>/` | For each test: the per-episode results (CSV), the result files (`*_RESULT.json` and `.txt`), the freeze, gate and unit-test records, and the run logs. |
| `code/figures_ma/` | Figures 1–7 (PDF with embedded fonts; PNG at 600 dpi). `FIG_DATA.csv` holds every plotted number, and `FIG_CHECK.txt` the checks run when the figures were drawn. |
| `code/tables_ma/` | Tables 1–5 (`TABLES_MA.md`) and their CSV files. |
| `code/data_ids/MANIFEST_heldout_val60_20261020.csv` | The 60 held-out scene ids, with the md5 and size of each raw trajectory file. |
| `requirements.txt` | The exact package versions used. |
| `MD5SUMS.txt` | The md5 of every file in this package. |

**Not included from the link-budget check:** the dry run's output (`models/_dry_link/`, which holds the default-link data written under the variant names; a code check only, rebuilt by `_dry_link.py`), the geometry caches, and the `.partial` copies of the episode files (each is byte-identical to its final file).

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
| §6 (complexity and decision time) | decision time per step | `models/timing/` | `PROTOCOL_Timing_DecisionTimePerStep_20261020.md` |
| §8.3, 16-episode statement | A1 against A0 at 5 places, per episode | `models/load_check/CHECK_C5_A1vsA0_LSRA_episodes_20261022.*` | — (`check_c5_a1_episodes.py`, read-only) |

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
   This redraws Figures 4–7 from `models/admission_check/ADMIT_RESULT.json`, `models/load_check/LOAD_RESULT.json` and `models/load_check_c6/LOAD6_RESULT.json`. The new `figures_ma/FIG_DATA.csv` must be byte-identical to the stored one (md5 `dd7fadd71209be1dd635f6eafdfe8b35`).
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
   ```
4. **The experiments themselves (need the pools).** Each runner has the modes listed in §9 of its pre-registration, run in order: `unit` (where present), `gate`, `freeze`, then `test`, one call per cell. Examples:
   ```
   python load_check_c6.py --mode test --adm A1 --hl 10
   python jitter_check.py --mode test --cap 4 --j 2
   python link_check.py --mode test --var P23 --hl 10
   ```
   - The link-budget check's 23 dBm runs read their own geometry cache (`geom_cache_ma_p23`), which `link_check.py --mode warm` builds. Caches are not included. Each test run checks the cache's fingerprint against `models/link_check/LINK_FREEZE.json` and refuses to start on a different one.
   - The code is deterministic: a re-run on the same machine and versions reproduces the stored episodes bit for bit.
   - The gates of the later runners (admission, load and delay-difference checks) confirmed this against the earlier runs before anything new was run.

## Integrity notes

- **Pre-registration.** Every registered test was pre-registered and frozen (md5 recorded) before it ran. Tests that did not support the method are reported: HR1, HL1, HJ2, HL3 and HL4. At 6 places, the method ties the simple rules.
- **Clerical incidents.** These are recorded in the logs and in the plan files. Examples:
  - the first gate attempt of the 6-place check stopped with a ValueError, was fixed and was run again (§6 of its plan);
  - an interrupted first timing run (`models/timing/_attempt1_killed_2050/`);
  - the first attempt of the link-budget check's pre-run checks stopped at import, before writing anything, on an assertion about the default threshold (`models/link_check/_superseded_attempt1_2213/`; §6 of its plan).
- **Generative AI.** Claude (Anthropic) wrote most of this code and these documents at the authors' direction (see the article's Methods and Acknowledgments). The first author approved every pre-registration before its run.
