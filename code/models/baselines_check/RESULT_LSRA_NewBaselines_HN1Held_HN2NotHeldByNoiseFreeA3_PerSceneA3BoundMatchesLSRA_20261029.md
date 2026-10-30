# RESULT — stronger baselines for LSRA (tuned A3, load-balancing A3, Na et al.'s predictive handover, LSRA-M), 29 Oct 2026

**Plan:** `PREREG_LSRA_NewBaselines_TunedA3_LoadBalancingA3_PredictiveNa_EqualInformation_FROZEN_20261028.md` (md5 ffb04871…).

**Freeze:** `BASE_FREEZE.json` (md5 67279da9…, 66 files, written 7:43:16 PM IST, 28 Oct).

**Run** (laptop, Windows Python 3.11.9, numpy 2.4.4, pandas 3.0.3; 6 workers):
- 28 Oct, 7:44 PM to 2:21 AM: the stage-1 grid (1,200 runs).
- 2:25 AM: the laptop restarted for a Windows update, which cut off the stage-1 choice.
- 29 Oct, 9:24 PM: resumed on Daya's OK, with the stale claim file moved to `models/_superseded_stale_claims_20261029_0221/`.
- 11:48 PM: analysis done. Every `.err` file is empty.

**Deviations** (none touches an arm, an episode or the analysis):
1. Before the freeze, the freeze record was fixed to skip two torch module names that are not files, and each worker's environment cache was cut from 4 to 2. All checks were run again and passed.
2. The Windows-update interruption and resume. Every episode depends only on its seed and each job writes its own file, so the results are what an uninterrupted run would give.

**Files** (`Paper2/code/models/baselines_check/`):

| File | md5 |
|---|---|
| `BASE_RESULT.json` | 67bdec4c778f28e75a6a0afd68e07b75 |
| `BASE_RESULT.txt` | 6e01a28f03b97c0b952290c0700c9ac3 |
| `TUNED.json` (frozen 9:33:22 PM, before any test episode) | 7af6777289b7aeff008ec408f8b38207 |
| `TUNED_T.json` | cf7798d65e061966ce59deb1ae0b3041 |

**Must-hold checks:** all passed (`problems` empty).
- Every one of the 51 test cells has 480 episodes, paired with LSRA's stored episodes by seed and scene.
- Under A1 and A2, no created excess, and starts equal requests minus refusals.
- No loop violation under A1.

**Spot checks** (recomputed from the raw CSVs, outside the analysis code): the seven levels and differences at 4 places under A2, HN2's A3-T-NF difference and the A3-O level at 4 places are equal to the printed values. Bootstrap seeds 12000–12172 were used.

## 1. Stage 1: the settings chosen on the design scenes

| Cell | A3-T (measured signal) | A3-T-NF (noise-free signal) | A3-LB |
|---|---|---|---|
| 4 places, A1 | TTT 1 slot, Off 0 dB, K 0 | TTT 7, Off +6, K 4 | = A3-T (κ 0) |
| 4 places, A2 | TTT 1, Off −1, K 0 | TTT 7, Off +6, K 4 | = A3-T (κ 0) |
| 5 places, A1 | TTT 1, Off +1, K 0 | TTT 5, Off +3, K 4 | = A3-T (κ 0) |
| 5 places, A2 | TTT 5, Off +6, K 0 | TTT 5, Off +3, K 4 | κ 8 |
| 6 places, A1 | TTT 1, Off +3, K 0 | TTT 7, Off +6, K 4 | = A3-T (κ 0) |
| 6 places, A2 | TTT 3, Off +6, K 0 | TTT 7, Off +6, K 4 | κ 8 |

**Limit to report.** Several choices sit at the edge of the registered grid: Off +6 dB is the largest offset and 7 slots the longest TTT. This holds for A3-T-NF in four cells, and for A3-T at 5 and 6 places under A2. A still more cautious A3 might do slightly better on the design scenes.

## 2. Registered tests (4 places, 1.0 s; J of the baseline minus J of LSRA, pp; 99.17 % scene intervals)

**HN1, coordinated admission (A2): HELD.** LSRA (3.59 %) beats all six.

| Baseline | J (%) | Minus LSRA | 99.17 % interval |
|---|---|---|---|
| A3-T, tuned A3 | 8.31 | +4.72 | [+2.76, +6.58] |
| A3-T-NF, tuned A3, noise-free signal | 9.64 | +6.04 | [+1.16, +12.75] |
| A3-LB, load-balancing A3 (κ 0 here, so the same episodes as A3-T) | 8.31 | +4.72 | [+2.64, +6.65] |
| PH-O, Na et al., perfect forecast | 12.49 | +8.90 | [+2.98, +16.64] |
| PH-CV, Na et al., straight-line forecast | 12.90 | +9.31 | [+3.32, +16.71] |
| PH-OR, Na et al., perfect forecast and room check | 16.62 | +13.02 | [+5.52, +21.91] |

**HN2, per-request admission (A1): NOT HELD.**
- LSRA under A1 is at 8.76 %.
- It beats A3-T (+3.24 [+1.35, +5.03]), A3-LB (+3.24), PH-O (+3.69 [+0.11, +8.37]), PH-CV (+4.28 [+0.54, +8.80]) and PH-OR (+7.86 [+2.09, +15.01]).
- It does **not** beat the noise-free tuned A3: **A3-T-NF +0.88 [−0.39, +2.37]**.
- **Reading:** A3-T-NF's chosen setting almost never hands over (0.50 handovers per vehicle-minute). Under per-request admission at full load, LSRA is itself locked up: 95 % of its requests are refused. So a rule that stays put does about as well. This is the lock-up the paper already reports, now seen from the baseline's side.

## 3. Secondary results (95 % intervals; baseline minus LSRA, pp)

| Cell | A3-T | A3-T-NF | A3-LB | PH-O | PH-CV | PH-OR |
|---|---|---|---|---|---|---|
| 5 places, A1 | +1.15 [−1.46, +3.03] | +2.92 [+0.93, +5.50] | +1.15 [−1.42, +3.09] | +2.10 [−0.01, +4.72] | +1.76 [−0.12, +4.15] | +0.83 [−2.89, +4.71] |
| 5 places, A2 | +2.49 [+0.49, +5.69] | +4.27 [+1.31, +7.99] | +1.56 [+0.52, +3.17] | +3.24 [+0.17, +7.18] | +2.93 [+0.04, +6.75] | +2.20 [−0.94, +6.26] |
| 6 places, A1 | +1.61 [+0.65, +2.50] | +2.95 [+0.80, +5.73] | +1.61 [+0.62, +2.50] | +1.18 [−0.99, +3.95] | +0.97 [−1.13, +3.57] | −0.59 [−2.23, +1.35] |
| 6 places, A2 | +0.88 [+0.08, +1.74] | +2.95 [+0.81, +5.77] | +0.55 [−0.22, +1.25] | +0.85 [−1.04, +3.06] | +0.85 [−1.16, +3.31] | −0.59 [−2.24, +1.38] |

With spare places, several baselines are not shown to be worse than LSRA:
- Na et al.'s rule with a perfect forecast and the room check (PH-OR) is level with LSRA at 6 places (−0.59, interval across zero).
- At 5 places under A2, LSRA is ahead of every baseline except PH-OR.

## 4. A3-O, the best A3 setting per test scene, chosen in hindsight (a bound, A2)

| Places | A3-O | LSRA | A3-O minus LSRA (95 %) | Settings used |
|---|---|---|---|---|
| 4 | 3.02 % | 3.59 % | **−0.58 [−1.61, +0.32]** | 32 of 120 |
| 5 | 1.81 % | 2.61 % | **−0.80 [−1.66, −0.20]** | 30 of 120 |

- **This is the most important result to report honestly.** If the A3 setting is picked separately for each test scene, looking at that scene's own outcomes, A3 matches LSRA at 4 places and beats it at 5 places.
- **Why it is optimistic.** It picks the best of 120 settings on the same 8 episodes it is scored on, so luck counts in its favour. No deployed tuning can do this. How much of the 0.6–0.8 pp is luck is not measured yet.
- **Proposed follow-up (exploratory; stored files only; no run; needs Daya's OK):**
  - for each scene, choose the setting on 4 of its 8 episodes and score it on the other 4, then swap the halves;
  - this gives an honest per-scene-tuning estimate to set beside the bound.
- **For the paper:** "no single A3 setting tuned for the scenario closes the gap (HN1), but choosing the setting per scene in hindsight would close it at 4 places and overtake LSRA at 5" (plan §8 allowed for this).

## 5. LSRA-M, LSRA fed only measurement reports (layer-3 filtered, σ = 8.2 dB on every link)

| Cell | LSRA-M J (%) | Minus LSRA (95 %) |
|---|---|---|
| 4 places, A2 | 4.44 | +0.85 [+0.17, +1.52] |
| 4 places, A1 | 9.47 | +0.71 [+0.03, +1.39] |
| 5 places, A2 | 3.30 | +0.69 [+0.34, +1.11] |
| 6 places, A2 | 3.06 | +0.74 [+0.31, +1.20] |

- Using standard measurement reports instead of the geometry costs LSRA about 0.7–0.85 pp.
- At 4 places under A2 it still beats the tuned A3 by **3.87 pp [+2.44, +5.23]**.
- At 6 places, PH-OR is 1.3 pp better than LSRA-M (interval across zero).

## 6. Today (baseline under A1) against the proposal (LSRA under A2): the split, 4 places (95 %)

| Baseline | T | P1, planner alone | Ab, admission alone | I, needs both |
|---|---|---|---|---|
| A3-T | +8.41 [+4.59, +13.04] | +3.24 [+1.85, +4.55] | +3.69 [+0.08, +8.04] | +1.47 [+0.74, +2.27] |
| A3-T-NF | +6.04 [+2.16, +10.87] | +0.88 [−0.08, +1.93] | 0.00 | +5.17 [+1.40, +9.95] |
| PH-O | +8.86 [+4.10, +14.22] | +3.69 [+0.88, +7.01] | −0.04 [−0.48, +0.34] | +5.20 [+1.42, +10.02] |
| PH-CV | +9.44 [+4.72, +14.85] | +4.28 [+1.40, +7.66] | +0.13 [−0.00, +0.37] | +5.03 [+1.29, +9.70] |
| PH-OR | +13.02 [+7.30, +19.58] | +7.86 [+3.27, +13.22] | 0.00 | +5.17 [+1.49, +9.89] |

**Notes:**
- Coordinated admission does nothing for the predictive rules (Ab ≈ 0). Against them, about 5.2 pp of LSRA's lead needs both the planner and coordinated admission, as with R2 in the stored check.
- A3-T's tuned setting differs between A1 and A2, so its Ab includes the re-tuning.
- The rows for 5 and 6 places are in `BASE_RESULT.txt`.

## 7. How the baselines behave (4 places, A2)

| Arm | Handovers per vehicle-minute | Quick returns per vehicle-minute | Requests refused |
|---|---|---|---|
| LSRA (stored) | 1.71 | not recorded | 0 % |
| A3-T | **43.8** | **17.6** | 62.5 % |
| A3-T-NF | 0.52 | 0 | 96.9 % |
| PH-O | 1.13 | 0.02 | 97.8 % |
| PH-OR | 1.40 | 0.01 | 0 % |
| LSRA-M | 4.17 | 0.23 | 0 % |

- **Tuned A3:** tuned for the worst vehicle's failure rate, it hands over 25 times as often as LSRA, and about 40 % of its handovers are quick returns. A ping-pong-heavy setting is the price of its best J.
- **Without coordinated admission** (A0), Na et al.'s rule overloads the stations: J is 44.5 % for PH-O.

## 8. Guesses recorded before the run: 7 of 9 came true

| Guess | Outcome |
|---|---|
| HN1 held | true |
| HN2 not held | true, but for A3-T-NF, not PH-OR as guessed |
| A3-T above 6 % (A2, 4 places) | true: 8.31 % |
| A3-O at 4 places above LSRA's 3.59 % | **false**: 3.02 % |
| A3-O within 1 pp of LSRA at 5 places | true: −0.80 |
| LSRA-M worse than LSRA by 0.5–2 pp | true: +0.85 |
| LSRA-M beats A3-T | true |
| PH-O makes fewer handovers than every A3 arm | **false**: A3-T-NF makes fewer |
| Most of PH-O's requests are refused | true: 97.8 % |

## 9. What goes into the paper (as decided in plan §8)

- **HN1 held:** "with coordinated admission at full load, LSRA beats an A3 tuned for this scenario, A3 with load balancing, and Na et al.'s predictive handover even with a perfect forecast".
- **HN2:** not held against the noise-free tuned A3. Report it with the split and the lock-up explanation.
- **A3-O:** "even the best A3 setting chosen per scene in hindsight closes the gap at 4 places and overtakes LSRA at 5"; give its optimism and, if Daya approves, the cross-fit estimate.
- **LSRA-M:** standard measurement reports cost about 0.7–0.85 pp, and LSRA-M still beats the tuned A3 at full load.
- **Limits to state:**
  - the edge-of-grid choices;
  - the selection optimism of A3-O;
  - A3 tuned per load and admission model on design scenes;
  - Na et al.'s Transformer not retrained;
  - Δ_init = 0 dB is our choice;
  - LSRA-M's single σ;
  - the quick-return count is timed to the start of the return.
