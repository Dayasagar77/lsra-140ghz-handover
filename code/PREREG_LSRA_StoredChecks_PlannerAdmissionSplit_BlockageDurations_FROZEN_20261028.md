# PREREG (FROZEN) — two checks on stored results: LSRA's part and coordinated admission's part, and how long blockages last

**Status: FROZEN, 28 Oct 2026, about 12:15 PM IST.** Daya approved "Freeze and run once" at about 12:10 PM IST. The text below is DRAFT v1 (md5 e96bc1645aefc1432d91d19e1cef3d6c) unchanged, apart from this status block, the title and the list of changes at the end. The md5 of the code, this plan and every input file is recorded by `stored_checks.py freeze` in `models/stored_checks/STORED_CHECKS_FREEZE.json`. The analysis runs once.

**Date:** 28 Oct 2026. **Status:** draft for Daya's OK. Nothing has been computed.

**What runs:** an analysis of files already on the laptop. There is no new simulation, nothing is tuned, and no hypothesis is tested.

**Answers:**
- the guide's correction 2: "Separate the benefit of LSRA from the benefit of coordinated admission in every key comparison" (plan W2.5);
- part of correction 1, blockage persistence (plan W1.2).

## 1. Check A — splitting each comparison into the planner's part and admission's part

**Data:** the stored per-episode files of the earlier frozen runs, read only.
- `models/admission_check/ADMIT_EPISODES_{A1,A2}_hl{2,5,10,15}.csv` (4 places);
- `models/load_check/LOAD_EPISODES_C5_{A1,A2}_hl*.csv`;
- `models/load_check_c6/LOAD_EPISODES_C6_{A1,A2}_hl*.csv`;
- `models/a3_check/A3_EPISODES_C{4,5,6}_{A1,A2}_hl*.csv`.

The md5 of every file is recorded when this plan is frozen.

**Cells:**
- Main: 1.0 s (hl = 10) at 4, 5 and 6 places. Supplement: 0.2, 0.5 and 1.5 s.
- Baselines B: R1, R1i, R2, R3, R0 (for reference) and A3 Sets 1–5.
- 480 paired episodes per cell, paired by seed and scene.

**Quantities.** J is the worst-vehicle RLF in %, per episode; L is LSRA. Each difference below is in pp and paired by episode.

| Name | Definition | Meaning |
|---|---|---|
| T | J(B, A1) − J(L, A2) | today's practice (baseline with per-request admission) against the proposal (LSRA with coordinated admission) |
| P1 | J(B, A1) − J(L, A1) | the planner alone, under today's admission |
| Ab | J(B, A1) − J(B, A2) | coordinated admission alone, given to the baseline |
| I | T − P1 − Ab | the part that needs both together |
| P2 | J(B, A2) − J(L, A2) | the planner, given coordinated admission |
| Al | J(L, A1) − J(L, A2) | coordinated admission, given LSRA |

- **Identities, checked exactly:** T = P1 + Ab + I = P1 + Al = Ab + P2.
- **Shares** (the average of the two orders): planner = (P1 + P2)/2; admission = (Ab + Al)/2. Their sum is T.

**Intervals:**
- 95 % scene bootstrap with 10,000 resamples, using the same function as every earlier result (`remeasure_ma.scene_boot`, percentile intervals).
- Seeds 9101, 9102, … in the fixed order: load, then baseline, then quantity, then delay.

**Reading rule** (descriptive; no test, no family correction):
- each quantity is reported with its interval;
- the text says "above zero" only when the whole 95 % interval is above zero.

**Must hold**, otherwise the analysis stops and nothing is reported:
- **M1** — every cell pairs by seed and scene (480 episodes each).
- **M2** — each of the four cell means equals the stored value in the frozen result files to within 1e-9:
  - `ADMIT_RESULT.json`, `LOAD_RESULT.json` and `LOAD6_RESULT.json` (`secondary.table`, key `{A}_hl{h}_{arm}`, field `worst`);
  - `A3_RESULT.json` (key `{A}_C{cap}_hl{h}_{arm}`).
- **M3** — the identities above hold for every episode to within 1e-9.
- **M4** — the LSRA files under A2 and under A0 hold identical worst values at all loads and delays (Corollary 1). Here A0 is `HELDOUT_EPISODES_hl*.csv` at 4 places and `LOAD_EPISODES_C{5,6}_A0_*.csv` at 5 and 6. This shows the split does not depend on whether A0 or A2 stands for "LSRA alone".

**What goes into the paper:**
- **Main text.** One compact table at 1.0 s with three rows per load: R2, R3 and the best A3 set, chosen by its stored A2 level. The columns are T, P1, Ab and I.
- **Abstract and Conclusion.** The headline comparison is stated with its parts, for example "of the 5.5 pp, 0.4 pp comes from the planner alone, 0.9 pp from coordinated admission alone, and 4.3 pp needs both". These are the stored values, to be confirmed by the run.
- **Supplement.** All baselines and all delays.

## 2. Check B — how long blockages last in our scenes, and how close links sit to the threshold

**Data:**
- the environment's own geometry cache for each of the 60 held-out scenes (`geom_cache_ma/{scene}_{hash}.pkl`, read only);
- the hash is computed exactly as `paper2ma_env._precompute` does, for 16 vehicles, 3 sub-bands, 4 background vehicles, stations SW, SE, NW, NE and anticipation on;
- if any file is missing, the check stops; nothing is recomputed.

**B1 — blockage events:**
- For each link (controlled vehicle i, station g), take the frames in which i is present.
- A frame is blocked if the knife-edge loss is at least 1 dB (the paper's definition).
- An event is a maximal run of consecutive present, blocked frames. An event touching the first or last frame of the scene, or an absence, is marked censored.
- **Reported:**
  - the number of links, and the share of present link-slots that are blocked;
  - the number of events, and their duration (median, mean, 90th percentile and maximum, in seconds) for all events and for uncensored ones;
  - the share of events lasting at least 0.5 s and at least 1 s;
  - the loss on blocked slots (median and 90th percentile);
  - per present vehicle-slot, how many of the four stations are clear (0–4).

**B2 — how often shadowing can decide an outage:**
- For each present vehicle-slot, take the best station by mean SNR. This counts the background interference and the blockage loss, with no co-channel interference and no shadowing, at 30 dBm.
- The margin m is that SNR minus −13.8 dB. σ is 8.2 dB if the link is blocked and 2.7 dB otherwise.
- **Reported:**
  - the shares with m < 0, 0 ≤ m < σ, σ ≤ m < 2σ and m ≥ 2σ;
  - the mean of Φ(−m/σ), the chance of outage from shadowing alone on the best station.
- This is a description of the geometry, independent of any planner. Real serving stations depend on the planner and on capacity.

**B3 — comparison with measurement** (text only, no computation):
- At 28 GHz, fade duration matched 0.75 × the vehicle's length for low vehicles and 1.3 × for vehicles taller than 2.5 m (Rodriguez et al., IEEE TVT 2024).
- In our model a vehicle blocks while its box crosses the path, so its blocking length is its geometric length.
- The realism run tests the measured 1.3 × for tall vehicles (variant B2 of the plan).

**Must hold:**
- all 60 cache files are found;
- the blocked share computed per link equals the share computed per slot;
- the vehicle-slots match the pool: 13.7 vehicles present in an average slot, as reported in §3.1 of the paper, to 0.1.

**What goes into the paper:** one or two sentences in §3.1 (how long blockages last; how often links are marginal), and the limitations paragraph.

## 3. Process

1. Code (`stored_checks.py`), with unit tests on small synthetic files built by hand.
2. Development tests in the laptop's Linux VM.
3. A code check; then the freeze, recording the md5 of the code, this plan and the input files.
4. One run on the laptop: Windows Python 3.11.9, numpy 2.4.4, through a `.cmd` launched by WMI.
5. A result note.

There are no other runs. A failed must-hold check stops the analysis, and it is reported to Daya. The analysis is not changed and re-run.


## Changes from DRAFT v1 (made before anything was computed)

1. **The scene check in Check B is stricter than the draft said.** Each scene's number of frames, mean number of vehicles present and share of frames with all 16 present must equal the stored `HELDOUT60_SCENE_STATS.csv` to within 1e-6. These are the per-scene values behind "13.7" and "47.8 %" in §3.1. The draft only asked for a match to 0.1.
2. **Unit tests** (in `stored_checks.py unit`), all passed in the laptop's Linux VM on 28 Oct:
   - the split on hand-made numbers;
   - pairing failures are caught;
   - the store filters arm, delay, admission and places;
   - Check A end to end on a small synthetic store: values, an M2 flag and an M1 flag;
   - blockage events and their censoring;
   - the best-link margin;
   - the cache-file hash;
   - all 60 cache files exist.
   They are re-run on Windows before the freeze.
