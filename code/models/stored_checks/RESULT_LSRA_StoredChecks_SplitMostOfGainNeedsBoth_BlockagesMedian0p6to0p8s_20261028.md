# RESULT — stored-result checks: LSRA's part, admission's part, and how long blockages last (28 Oct 2026)

**Plan:** `PREREG_LSRA_StoredChecks_PlannerAdmissionSplit_BlockageDurations_FROZEN_20261028.md` (md5 7e19a770a0fb80958f155906f7ffabf8), frozen on Daya's "Freeze and run once". There were no deviations.

**Correction of a time.**
- The status block of the frozen plan says the approval came "about 12:10 PM" and the freeze "about 12:15 PM IST". Both are wrong.
- By the laptop clock, Daya approved at about 10:05 AM. The freeze was written at 10:09:18 AM IST.
- The frozen file was left as it is, so its md5 still matches the freeze record. The correction is recorded here.

**Runs** (laptop, Windows Python 3.11.9, numpy 2.4.4, pandas 3.0.3; each launched through WMI with conhost hidden; no new Terminal process):

| Step | Time (IST) | Outcome |
|---|---|---|
| Unit | 10:08 AM | PASS, all 11 tests |
| Freeze | 10:09 AM | 138 input files recorded |
| Analysis | 10:10–10:19 AM | `analyse.err` empty |

**Files** (`Paper2/code/models/stored_checks/`):

| File | md5 |
|---|---|
| `STORED_CHECKS_RESULT.json` | ef278bfd2763f318791ad92131fc6336 |
| `STORED_CHECKS_RESULT.txt` | 9eebf1da049fdd9bd524f8237a2e0fab |
| `STORED_CHECKS_FREEZE.json` | ba78bacca3e611c81c78617ea14f0738 |
| `stored_checks.py` | e4f9fcd1d3f646682bdc9154cbcd27a9 |

**Must-hold checks:** all passed.
- M1: every cell pairs, with 480 episodes each.
- M2: every cell mean equals the stored value exactly (largest deviation 0.0).
- M3: the identities hold to 1.4 × 10⁻¹⁴.
- M4: LSRA's runs under A0 and A2 are identical in all 12 load–delay cells.
- The scene statistics equal `HELDOUT60_SCENE_STATS.csv`: 13.707 vehicles present on average, and all 16 present in 47.78 % of slots.
- The last bootstrap seed used was 10060, as planned (9101 + 960 − 1).

## 1. Check A — where the gain comes from (1.0 s; pp; 95 % scene intervals)

**How to read the columns.**
- **T** is today's practice (the baseline with per-request admission, A1) against the proposal (LSRA with coordinated admission, A2).
- **T = P1 + Ab + I:**
  - P1, the planner alone under today's admission;
  - Ab, coordinated admission alone, given to the baseline;
  - I, the part that needs both.

| Places | Baseline | B/A1 → L/A2 (%) | T | P1 planner alone | Ab admission alone | I needs both |
|---|---|---|---|---|---|---|
| 4 | R2 | 9.11 → 3.59 | +5.52 [+1.76, +10.25] | +0.35 [−0.03, +0.74] | +0.89 [−0.04, +2.52] | +4.28 [+0.99, +8.52] |
| 4 | R1i | 9.08 → 3.59 | +5.48 [+1.79, +10.19] | +0.32 [−0.18, +0.82] | 0.00 (exactly) | +5.17 [+1.48, +9.84] |
| 4 | R1 | 9.26 → 3.59 | +5.67 [+1.94, +10.59] | +0.50 [+0.05, +0.98] | 0.00 (exactly) | +5.17 [+1.44, +9.85] |
| 4 | R3 | 10.82 → 3.59 | +7.22 [+3.29, +12.00] | +2.06 [+0.53, +3.53] | +4.12 [+0.20, +9.17] | +1.05 [+0.29, +1.83] |
| 4 | A3 Set 5 | 11.75 → 3.59 | +8.16 [+4.31, +12.88] | +2.99 [+1.61, +4.24] | +3.44 [−0.16, +7.93] | +1.73 [+1.03, +2.49] |
| 5 | R2 | 4.46 → 2.61 | +1.85 [+0.09, +4.87] | +0.49 [−1.63, +2.90] | 0.00 | +1.36 [0.00, +3.65] |
| 5 | R3 | 6.11 → 2.61 | +3.50 [+1.65, +5.90] | +2.13 [+0.47, +3.63] | +1.01 [−0.14, +3.05] | +0.35 [−0.33, +1.38] |
| 5 | A3 Set 5 | 5.10 → 2.61 | +2.49 [+1.38, +3.48] | +1.12 [−1.61, +3.07] | **−0.32 [−0.56, −0.09]** | +1.68 [+0.28, +3.98] |
| 6 | R2 | 2.45 → 2.32 | +0.12 [−0.03, +0.36] | +0.12 [−0.03, +0.38] | 0.00 | 0.00 |
| 6 | R3 | 3.95 → 2.32 | +1.62 [+0.34, +2.97] | +1.62 [+0.32, +2.93] | +0.03 [−0.35, +0.47] | −0.03 [−0.48, +0.35] |
| 6 | A3 Set 5 | 3.94 → 2.32 | +1.61 [+0.53, +2.58] | +1.61 [+0.54, +2.57] | **−0.22 [−0.38, −0.07]** | +0.22 [+0.07, +0.38] |

Set 5 is the best A3 setting under A2 at every load. All baselines and all delays are in the JSON and TXT files.

**Plain reading.**
1. **At full load, most of LSRA's lead over the simple rules needs both the planner and coordinated admission.**
   - Against R2: neither the planner alone (0.35 pp) nor admission alone (0.89 pp) is shown. The 4.28 pp that needs both is above zero.
   - For R1 and R1i, coordinated admission changes nothing at all. They only ask for stations with room, so they never form a closed loop.
2. **Against R3 and A3, the planner alone already helps:**
   - +2.06 pp against R3 at 4 places;
   - +2.99 to +7.94 pp against the five A3 sets, all above zero.
   
   Coordinated admission alone also helps R3 (+4.12 pp), because it refuses R3's overfilling handovers.
3. **Coordinated admission is not a free gain for every planner.**
   - For A3 at 5 and 6 places it makes things slightly worse: −0.32 pp for Set 5 at 5 places, and −0.12 to −0.32 pp for Sets 2–5 at 6 places.
   - It admits more of A3's moves, and A3's moves are not good ones.
   - It pays when the planner's requests are good. This is the interaction the guide asked us to separate.
4. **With spare places the parts change.**
   - At 5 places, against the simple rules, the both-together part sits at the edge (+1.36 [0.00, +3.65]).
   - At 6 places, only the planner-alone part against R3 and A3 remains.
5. **Across delays at 4 places against R2,** the both-together part falls from +5.63 pp (0.2 s) to +3.68 pp (1.5 s), and the planner-alone part grows from +0.14 to +0.58 pp.

## 2. Check B — blockage in the 60 test scenes (geometry only)

**Blockage events.**
- 780 events on 3,764 vehicle–station links; 4.4 % of present link-slots are blocked.
- Durations:

| Events | Median | Mean | 90th percentile | Longest | Share ≥ 1 s |
|---|---|---|---|---|---|
| Uncensored (437) | 0.6 s | 0.88 s | 1.9 s | 8.3 s | 31 % |
| All (780; 44 % touch a scene edge or an absence) | 0.8 s | 1.87 s | 5.5 s | — | — |

- Loss on blocked slots: median 24.8 dB, 90th percentile 40 dB (the cap).

**Clear stations per present vehicle-slot.**

| Clear stations | 4 | 3 | 2 | 1 or 0 |
|---|---|---|---|---|
| Share of vehicle-slots | 84.2 % | 14.0 % | 1.7 % | 0.06 % |

**How close the best link sits to the threshold.**
- In 99.98 % of vehicle-slots, the best station's mean SNR is at least 2σ above the outage threshold. This counts the background interference and the blockage loss, but no co-channel interference.
- The mean chance of outage from shadowing alone on the best station is 0.013 %.

**Reading (inference, not a test).**
- A vehicle almost always has a clear, strong station.
- Failures arise when capacity and the handover delay keep it elsewhere, which agrees with the failure causes in §8.4.
- The typical blockage, 0.6–0.8 s, is shorter than the 1 s handover delay. This is why most of LSRA's remaining failures are vehicles in outage while their handover is in flight (cause W).
- **Comparison with measurements** (B3, text only):
  - Measured blockages last from a few hundred milliseconds to seconds [C2-6, C2-1].
  - At 28 GHz, tall vehicles shadow a link for about 1.3 times their length [I-F3]; in our model a vehicle blocks over its geometric length.
  - The realism run tests the measured 1.3 ×.
- **Caveat:** this check uses the best station and no co-channel interference. The serving station depends on the planner and on capacity, so the realism run measures the effect of the shadowing model directly.

## 3. What goes into the paper

- **New table** (Section 8, "LSRA's part and coordinated admission's part"): the rows above for R2, R3 and A3 Set 5 at 4, 5 and 6 places. All baselines and delays go in the supplement.
- **Abstract and Conclusion:** the headline lead is stated with its parts. For example: "of the 5.5 pp lead over the best simple rule under today's admission, 0.4 pp comes from the planner alone, 0.9 pp from coordinated admission alone and 4.3 pp needs both". The first two intervals include zero.
- **Section 8.2 or a new 8.x:** coordinated admission slightly hurts A3 at 5–6 places. It is not a free gain for every planner.
- **§3.1 and the limitations:** blockage durations (median 0.6–0.8 s; 90th percentile 1.9–5.5 s), four stations clear in 84 % of vehicle-slots, and the best link almost always well above the threshold.
