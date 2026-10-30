| Arm | J, A2 (%) | Arm − LSRA, A2 (HN1) | J, A1 (%) | Arm − LSRA, A1 (HN2) | Handovers per vehicle-minute, A2 | Requests refused, A2 |
|----------------|----------|----------------------|----------|----------------------|----------|----------|
| Tuned A3 (A3-T) | 8.31 | +4.72 [+2.76, +6.58] | 12.00 | +3.24 [+1.35, +5.03] | 43.8 | 62.5 % |
| Tuned A3, noise-free signal (A3-T-NF) | 9.64 | +6.04 [+1.16, +12.75] | 9.64 | +0.88 [−0.39, +2.37] | 0.5 | 96.9 % |
| Load-balancing A3 (A3-LB) | 8.31 | +4.72 [+2.64, +6.65] | 12.00 | +3.24 [+1.28, +4.98] | 43.8 | 62.5 % |
| Na et al., perfect forecast (PH-O) | 12.49 | +8.90 [+2.98, +16.64] | 12.45 | +3.69 [+0.11, +8.37] | 1.1 | 97.8 % |
| Na et al., straight-line forecast (PH-CV) | 12.90 | +9.31 [+3.32, +16.71] | 13.04 | +4.28 [+0.54, +8.80] | 1.4 | 97.5 % |
| Na et al., perfect forecast and room check (PH-OR) | 16.62 | +13.02 [+5.52, +21.91] | 16.62 | +7.86 [+2.09, +15.01] | 1.4 | 0.0 % |
| LSRA fed measurement reports only (LSRA-M)\* | 4.44 | +0.85 [+0.17, +1.52]\* | 9.47 | +0.71 [+0.03, +1.39]\* | 4.2 | 0.0 % |
| LSRA | 3.59 | — | 8.76 | — | 1.7 | 0.0 % |

: **Table 7.** The new baselines at full load (4 places, 1.0 s): worst-vehicle RLF J and difference to LSRA under the same admission, pp, with 99.17 % scene-bootstrap intervals (Bonferroni over six).

::: {custom-style="TableNote"}
HN1 (A2) held; HN2 (A1) not held, not shown against A3-T-NF. \*LSRA-M is a variant of LSRA, not a baseline: its difference to LSRA has a 95 % interval and is not part of HN1 or HN2. A3-LB chose a load-balancing gain of 0 at 4 places, so its episodes equal A3-T's; its interval differs only through the bootstrap. LSRA's handover rate is the stored one (Section 8.4).
:::


| Places, admission | LSRA J (%) | A3-T | A3-T-NF | A3-LB | PH-O | PH-CV | PH-OR |
|----------|--------|--------------|--------------|--------------|--------------|--------------|--------------|
| 5 places, A1 | 3.97 | +1.15 [−1.46, +3.03] | +2.92 [+0.93, +5.50] | +1.15 [−1.42, +3.09] | +2.10 [−0.01, +4.72] | +1.76 [−0.12, +4.15] | +0.83 [−2.89, +4.71] |
| 5 places, A2 | 2.61 | +2.49 [+0.49, +5.69] | +4.27 [+1.31, +7.99] | +1.56 [+0.52, +3.17] | +3.24 [+0.17, +7.18] | +2.93 [+0.04, +6.75] | +2.20 [−0.94, +6.26] |
| 6 places, A1 | 2.32 | +1.61 [+0.65, +2.50] | +2.95 [+0.80, +5.73] | +1.61 [+0.62, +2.50] | +1.18 [−0.99, +3.95] | +0.97 [−1.13, +3.57] | −0.59 [−2.23, +1.35] |
| 6 places, A2 | 2.32 | +0.88 [+0.08, +1.74] | +2.95 [+0.81, +5.77] | +0.55 [−0.22, +1.25] | +0.85 [−1.04, +3.06] | +0.85 [−1.16, +3.31] | −0.59 [−2.24, +1.38] |

: **Table 8.** The new baselines with spare places (1.0 s): baseline minus LSRA under the same admission, pp, with 95 % scene-bootstrap intervals (pre-registered secondary results).


| Places | Baseline | Baseline under A1 → LSRA under A2 (%) | T, total | P1, planner alone | Ab, admission alone | I, needs both |
|----|------------|------------|--------------|--------------|--------------|--------------|
| 4 | R2 | 9.11 → 3.59 | +5.52 [+1.76, +10.25] | +0.35 [−0.03, +0.74] | +0.89 [−0.04, +2.52] | +4.28 [+0.99, +8.52] |
| 4 | R1i | 9.08 → 3.59 | +5.48 [+1.79, +10.19] | +0.32 [−0.18, +0.82] | 0.00 (exactly) | +5.17 [+1.48, +9.84] |
| 4 | R3 | 10.82 → 3.59 | +7.22 [+3.29, +12.00] | +2.06 [+0.53, +3.53] | +4.12 [+0.20, +9.17] | +1.05 [+0.29, +1.83] |
| 4 | A3, Set 5 | 11.75 → 3.59 | +8.16 [+4.31, +12.88] | +2.99 [+1.61, +4.24] | +3.44 [−0.16, +7.93] | +1.73 [+1.03, +2.49] |
| 4 | Tuned A3 (A3-T) | 12.00 → 3.59 | +8.41 [+4.59, +13.04] | +3.24 [+1.85, +4.55] | +3.69 [+0.08, +8.04] | +1.47 [+0.74, +2.27] |
| 4 | Na et al. (PH-O) | 12.45 → 3.59 | +8.86 [+4.10, +14.22] | +3.69 [+0.88, +7.01] | −0.04 [−0.48, +0.34] | +5.20 [+1.42, +10.02] |
| 5 | R2 | 4.46 → 2.61 | +1.85 [+0.09, +4.87] | +0.49 [−1.63, +2.90] | 0.00 (exactly) | +1.36 [+0.00, +3.65] |
| 5 | R3 | 6.11 → 2.61 | +3.50 [+1.65, +5.90] | +2.13 [+0.47, +3.63] | +1.01 [−0.14, +3.05] | +0.35 [−0.33, +1.38] |
| 5 | A3, Set 5 | 5.10 → 2.61 | +2.49 [+1.38, +3.48] | +1.12 [−1.61, +3.07] | −0.32 [−0.56, −0.09] | +1.68 [+0.28, +3.98] |
| 6 | R2 | 2.45 → 2.32 | +0.12 [−0.03, +0.36] | +0.12 [−0.03, +0.38] | 0.00 (exactly) | 0.00 (exactly) |
| 6 | R3 | 3.95 → 2.32 | +1.62 [+0.34, +2.97] | +1.62 [+0.32, +2.93] | +0.03 [−0.35, +0.47] | −0.03 [−0.48, +0.35] |
| 6 | A3, Set 5 | 3.94 → 2.32 | +1.61 [+0.53, +2.58] | +1.61 [+0.54, +2.57] | −0.22 [−0.38, −0.07] | +0.22 [+0.07, +0.38] |

: **Table 9.** Where the gain over today's practice comes from (1.0 s): T = P1 + Ab + I, pp, with 95 % scene-bootstrap intervals.

::: {custom-style="TableNote"}
T is the baseline under per-request admission (A1) minus LSRA under coordinated admission (A2); P1 is the baseline minus LSRA, both under A1; Ab is the baseline under A1 minus the same baseline under A2; I = T − P1 − Ab is the part that needs both. Set 5 is the best A3 setting under A2 at every load. The tuned A3 was tuned separately for A1 and A2, so its Ab includes the re-tuning. All baselines and delays are in the result files.
:::


| Model | What changes | Value and source | RW1 | RW2 | RW3 |
|------|------------------------------|--------------------|------|------|------|
| V0 | nothing: the model of Sections 3–8 | — | held | held | held |
| S1 | shadowing correlated along each vehicle's path, independently per station; the A3 rules measure it | correlation distance 3.8 m, measured at 142 GHz [Ju2021] | held | held | held |
| S2 | as S1 | 10 m on clear and 13 m on blocked links, the 3GPP street-canyon values [TR38901] | held | held | held |
| S3 | as S1 | 50 m, a stress case: the 3GPP urban-macro value without line of sight [TR38901], not our road | held | held | held |
| B1 | knife-edge loss also for a roof just below the path (−0.78 < v ≤ 0); such grazing links also take the blocked-link shadowing spread | the full knife-edge curve [P526] | held | held | held |
| B2 | a blocker counts over 1.3 times its length if taller than 2.5 m and 0.75 times if lower | measured at 28 GHz [Rodriguez2024] | held | held | held |
| B3 | the losses of all crossing blockers summed in dB, capped at 60 dB | blockage model B [TR38901] | held | held | held |
| G10 | serving-beam gain 10 dBi instead of 15 dBi, known to every planner | assumption | held | held | held |
| G20 | serving-beam gain 20 dBi, known to every planner | assumption | held | held | held |
| DI | co-channel interference through the gain 15 − min(12(θ/30°)², 15) dBi, θ the angle between the two vehicles seen from the station, instead of 0 dBi; the planners still assume 0 dBi | parabolic form of the 3GPP pattern [TR38901]; width and depth assumed | held | held | held |
| MA | the serving beam is re-pointed every 0.5 s, at landing and when service resumes, with a loss of min(12(Δθ/30°)², 30) dB in between, Δθ the change of angle | assumption | held | held | held |
| F1 | fading in every slot on every link: Rician with a Rice factor of 9 dB if clear, Rayleigh if blocked; the A3 rules' filtered measurement does not see it | assumption | held | held | held |
| H1 | the vehicle is unserved in the slot in which it lands | 0.1 s; measured interruptions 43–134 ms [Goh2025; Vicario2026] | held | held | held |
| H2a | a handover fails if the source link reaches radio-link failure while it is in flight, or if the SINR at the target is below −13.8 dB at landing; the vehicle is then unserved for 0.2 s and re-established at the strongest station with a place | [TR36839; TS38300; Park2020] | held | held | held |
| H2b | as H2a, with −8 dB and 0.5 s | [TR36839; Park2020] | not held (R1i, R2) | held | held |
| ALL | S2, B2, DI and H2a together | — | not held (R2) | held | held |

: **Table 10.** The changed models of the sensitivity check, each set against the model of Sections 3–8 (V0), and the registered tests that held in each (4 places, 1.0 s).

::: {custom-style="TableNote"}
RW1: LSRA with coordinated admission beats R1i, R2, R3, A3 Set 5, the tuned A3 and PH-O under the same admission (99.17 % intervals, Bonferroni over six). RW2: coordinated admission is worth something to LSRA (95 %). RW3: LSRA with coordinated admission beats R2 and the tuned A3 under per-request admission (97.5 %, Bonferroni over two). No planner was re-tuned; Section 7 states what the planners knew of each change. In the result files, G10, G20, DI and MA are named A1a, A1b, A2 and A3.
:::


| Model | LSRA J (%) | R1i | R2 | R3 | A3 Set 5 = A3-T | PH-O | RW1 |
|------|------|--------------|--------------|--------------|--------------|--------------|------|
| V0 | 3.59 | +5.48 [+0.83, +12.00] | +4.64 [+0.77, +10.54] | +3.11 [+1.04, +5.09] | +4.72 [+2.60, +6.68] | +8.90 [+2.84, +16.64] | held |
| S1 | 4.40 | +5.84 [+1.08, +12.96] | +4.95 [+0.95, +11.08] | +2.79 [+0.81, +4.61] | +8.08 [+4.45, +12.89] | +9.11 [+2.95, +16.55] | held |
| S2 | 4.62 | +6.00 [+1.19, +12.81] | +4.97 [+0.87, +11.29] | +2.71 [+0.76, +4.57] | +9.05 [+5.43, +13.30] | +9.12 [+2.76, +16.75] | held |
| S3 | 4.80 | +5.95 [+1.20, +12.60] | +4.92 [+0.77, +11.11] | +2.58 [+0.58, +4.50] | +10.12 [+6.16, +15.08] | +9.07 [+2.97, +16.84] | held |
| B1 | 3.62 | +5.52 [+0.93, +12.47] | +4.67 [+0.79, +10.82] | +2.96 [+0.86, +4.79] | +4.73 [+2.68, +6.63] | +8.96 [+2.84, +16.75] | held |
| B2 | 3.96 | +5.56 [+0.72, +12.53] | +4.38 [+0.48, +10.66] | +3.40 [+1.08, +5.51] | +4.81 [+3.02, +6.63] | +8.78 [+2.61, +16.33] | held |
| B3 | 3.61 | +5.48 [+0.96, +12.16] | +4.64 [+0.78, +10.87] | +3.09 [+1.03, +4.99] | +4.73 [+2.67, +6.66] | +8.92 [+2.95, +16.12] | held |
| G10 | 4.99 | +6.39 [+1.38, +13.44] | +3.97 [+1.14, +8.01] | +2.30 [+0.30, +4.07] | +4.25 [+2.22, +6.19] | +8.76 [+2.35, +16.58] | held |
| G20 | 2.58 | +4.62 [+0.67, +10.69] | +3.71 [+0.50, +9.25] | +3.41 [+1.44, +5.24] | +4.91 [+2.96, +6.78] | +8.40 [+2.91, +15.49] | held |
| DI | 5.52 | +7.28 [+2.34, +14.31] | +4.36 [+1.74, +8.34] | +2.87 [+1.17, +4.70] | +4.79 [+2.98, +6.67] | +9.49 [+3.19, +17.33] | held |
| MA | 3.59 | +5.48 [+0.83, +11.93] | +4.64 [+0.75, +10.75] | +3.11 [+0.98, +4.93] | +4.72 [+2.62, +6.66] | +8.90 [+2.75, +16.42] | held |
| F1 | 3.79 | +5.90 [+1.37, +12.69] | +4.40 [+1.15, +9.65] | +2.98 [+0.94, +4.86] | +4.69 [+2.66, +6.67] | +8.82 [+2.70, +16.22] | held |
| H1 | 3.89 | +5.94 [+1.35, +12.39] | +5.34 [+1.38, +11.42] | +3.06 [+0.89, +5.05] | +3.32 [+1.25, +5.15] | +8.36 [+2.26, +16.01] | held |
| H2a | 3.13 | +5.07 [+0.24, +11.94] | +4.18 [+0.06, +10.34] | +2.56 [+1.20, +3.96] | +4.06 [+2.73, +5.44] | +9.06 [+2.95, +16.75] | held |
| H2b | 4.69 | +4.55 [−0.24, +11.40] | +3.80 [−0.23, +9.87] | +3.08 [+1.19, +4.93] | +6.23 [+4.42, +8.00] | +7.85 [+1.74, +15.82] | not held (R1i, R2) |
| ALL | 8.96 | +6.17 [+1.17, +13.15] | +3.73 [−0.16, +9.31] | +2.02 [+0.19, +3.87] | +6.38 [+2.76, +10.97] | +10.52 [+4.13, +18.45] | not held (R2) |

: **Table 11.** RW1 in each model: the baseline minus LSRA, both under coordinated admission (4 places, 1.0 s), pp, with 99.17 % scene-bootstrap intervals (Bonferroni over six).

::: {custom-style="TableNote"}
Under A2 at 4 places the tuned A3 chose the Set 5 setting, so its episodes equal Set 5's in every model; the table shows them once, with Set 5's interval (the tuned A3's, from its own bootstrap, is in the result files and leads to the same verdicts). V0 is the model of Sections 3–8, computed from the stored episodes.
:::


| Model | LSRA, A1 (%) | RW2: LSRA A1 − LSRA A2 (95 %) | R2, A1 (%) | RW3: R2 A1 − LSRA A2 (97.5 %) | A3-T, A1 (%) | RW3: A3-T A1 − LSRA A2 (97.5 %) |
|------|------|--------------|------|--------------|------|--------------|
| V0 | 8.76 | +5.17 [+1.42, +9.96] | 9.11 | +5.52 [+1.41, +10.91] | 12.00 | +8.41 [+4.19, +13.78] |
| S1 | 9.98 | +5.58 [+1.73, +10.48] | 10.15 | +5.75 [+1.51, +11.38] | 14.05 | +9.65 [+5.23, +15.24] |
| S2 | 10.27 | +5.66 [+1.78, +10.47] | 10.45 | +5.83 [+1.44, +11.52] | 15.38 | +10.76 [+6.09, +16.24] |
| S3 | 10.49 | +5.69 [+1.84, +10.55] | 10.56 | +5.76 [+1.56, +11.66] | 17.08 | +12.28 [+7.61, +17.74] |
| B1 | 8.77 | +5.15 [+1.41, +9.72] | 9.17 | +5.55 [+1.47, +11.09] | 11.94 | +8.32 [+4.02, +13.83] |
| B2 | 9.16 | +5.20 [+1.38, +10.02] | 9.43 | +5.47 [+1.13, +11.28] | 12.06 | +8.10 [+3.78, +13.65] |
| B3 | 8.77 | +5.17 [+1.40, +9.78] | 9.13 | +5.52 [+1.40, +10.94] | 12.02 | +8.42 [+4.19, +13.76] |
| G10 | 10.62 | +5.62 [+1.66, +10.69] | 11.21 | +6.22 [+1.70, +12.08] | 13.27 | +8.28 [+3.69, +14.25] |
| G20 | 6.72 | +4.14 [+0.90, +8.33] | 7.15 | +4.57 [+1.07, +9.38] | 10.46 | +7.88 [+4.23, +12.61] |
| DI | 11.29 | +5.78 [+1.71, +10.85] | 12.86 | +7.35 [+2.83, +13.06] | 14.42 | +8.91 [+4.41, +14.75] |
| MA | 8.76 | +5.17 [+1.44, +9.88] | 9.11 | +5.52 [+1.32, +11.06] | 12.00 | +8.41 [+4.14, +13.80] |
| F1 | 8.93 | +5.15 [+1.44, +9.89] | 9.60 | +5.82 [+1.74, +11.23] | 12.13 | +8.35 [+4.09, +13.63] |
| H1 | 8.90 | +5.01 [+1.32, +9.75] | 9.99 | +6.10 [+1.93, +11.49] | 11.54 | +7.65 [+3.36, +13.07] |
| H2a | 8.04 | +4.91 [+1.01, +9.71] | 8.20 | +5.07 [+0.76, +10.74] | 11.22 | +8.09 [+3.86, +13.72] |
| H2b | 8.96 | +4.26 [+0.47, +9.06] | 9.28 | +4.59 [+0.33, +10.49] | 13.43 | +8.73 [+4.48, +14.38] |
| ALL | 15.26 | +6.30 [+2.27, +11.18] | 15.29 | +6.32 [+1.78, +12.05] | 19.39 | +10.43 [+5.64, +16.28] |

: **Table 12.** RW2 and RW3 in each model (4 places, 1.0 s): levels in %, differences in pp with scene-bootstrap intervals. RW2 and RW3 held in all 16 models.


| Setting | What changes | LSRA J (%) | LSRA A1 − LSRA (95 %) | R2 A1 − LSRA (95 %) | Cost: LSRA − ideal LSRA A2 (95 %) |
|------|------------------|------|--------------|--------------|--------------|
| C1a | Landings spread by up to ±0.5 s | 4.77 | +3.99 [+0.24, +8.73] | +4.34 [+0.67, +9.11] | +1.18 [+0.80, +1.57] |
| C1b | Late landings only, up to +0.5 s | 4.76 | +4.00 [+0.43, +8.64] | +4.36 [+0.73, +8.99] | +1.16 [+0.77, +1.62] |
| C2a | 5 % of admitted moves not run | 3.62 | +5.14 [+1.41, +9.93] | +5.49 [+1.71, +10.26] | +0.03 [+0.00, +0.07] |
| C2b | 20 % of admitted moves not run | 3.74 | +5.02 [+1.28, +9.71] | +5.38 [+1.64, +10.09] | +0.14 [+0.07, +0.25] |
| C3a | 2 % of admitted moves fail at landing | 3.58 | +5.18 [+1.49, +9.84] | +5.53 [+1.75, +10.25] | −0.01 [−0.03, +0.00] |
| C3b | 5 % fail at landing | 3.58 | +5.18 [+1.39, +9.91] | +5.54 [+1.82, +10.14] | −0.01 [−0.04, +0.00] |
| C3c | 10 % fail at landing | 3.58 | +5.18 [+1.46, +9.99] | +5.54 [+1.84, +10.13] | −0.01 [−0.04, +0.00] |
| C4a | Coordination 0.1 s slower | 3.83 | +4.93 [+1.18, +9.59] | +5.29 [+1.54, +9.97] | +0.23 [+0.08, +0.39] |
| C4b | Coordination 0.2 s slower | 4.11 | +4.65 [+0.99, +9.25] | +5.00 [+1.28, +9.69] | +0.52 [+0.26, +0.82] |
| C4c | Coordination 0.5 s slower | 5.10 | +3.66 [+0.04, +8.18] | +4.01 [+0.40, +8.33] | +1.51 [+0.79, +2.32] |

: **Table 13.** Imperfect coordination: LSRA under coordinated admission with each imperfection (4 places, 1.0 s), against LSRA and R2 under per-request admission and against ideal coordination, in pp.

::: {custom-style="TableNote"}
In the two middle columns, positive values favour LSRA with imperfect coordinated admission; the last column is positive when the imperfection costs LSRA (negative: LSRA did slightly better than with ideal coordination). References: LSRA under A1 at 8.76 %, R2 under A1 at 9.11 % and ideal LSRA under A2 at 3.59 %. Break-even (the first setting of a sequence whose interval does not lie above 0): none in C1, C2, C3 or C4, against either reference.
:::

