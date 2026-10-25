# Tables T1–T5 (multi-vehicle paper)

Made by `make_tables_ma.py` from the code and the finished result files only. Sources: `models/admission_check/ADMIT_RESULT.json` (md5 803e1017a39ef1dcd4a6e9ea10ade9ad); `models/load_check/LOAD_RESULT.json` (md5 0fb5845ca6b234261d7ca78363c5e730); `models/load_check_c6/LOAD6_RESULT.json` (md5 8e1860b938f69560610da9619e7bbc00); `models/method_heldout/HELDOUT_RESULT.json` (md5 9254d3e8964a4e49cd018d283d0837ef).

## T1. Notation

| Symbol | Meaning |
|---|---|
| Δ | slot length, 0.1 s; t, τ are slot indices |
| M, G, C | vehicles per scene (16), stations (4), places per station (4, 5 or 6) |
| A(t) | vehicles present in slot t |
| σᵢ(t), κᵢ(t), θᵢ(t) | serving station, handover timer, committed target of vehicle i |
| L | handover delay in slots: a request made at t lands in slot t + L + 1 |
| ξ, ξ_max | delay-difference check: a handover takes L + ξ slots, ξ uniform on {−ξ_max, …, ξ_max} (ξ_max = 1, 2) |
| F(t), P(t) | free vehicles (κ = 0) and vehicles in flight (κ > 0) |
| λᵢ(t) | landing station: θᵢ(t) if in flight, else σᵢ(t) |
| n_g(t), n̄_g(t) | vehicles at station g now; landing load (vehicles in flight counted where they land) |
| S, n̄ˢ_g(t) | request set of one slot; landing load after S |
| landing-feasible | n̄ˢ_g(t) ≤ max(C, n̄_g(t)) at every station |
| γ_out, N | outage threshold (−13.8 dB SINR); consecutive failed slots for a radio-link failure (3) |
| ψᵢ(τ), J | radio-link failure of vehicle i in slot τ; worst-vehicle RLF rate, maxᵢ of the mean of ψᵢ |
| π̂_{i,g}(τ | t) | forecast probability that vehicle i fails on station g in slot τ |
| r̄_{i,g}(t), H | risk after landing: mean of π̂ over the H = L + 1 slots after landing |
| β | cost of one handover in LSRA (0.05) |
| q_g(t) | places left for free vehicles at g: C minus the vehicles in flight landing at g |
| A0, A1, A2 | admission at landing only; per request at preparation (today); coordinated, as a set |

## T2. Parameters (read from the code)

| Parameter | Value | From |
|---|---|---|
| Carrier frequency | 140 GHz | channel_model_nyu.FREQ |
| Bandwidth | 380.16 MHz | channel_model.BW (66 RBs x 12 x 480 kHz) |
| Noise figure; noise power | 10 dB; -78.2 dBm | channel_model.NF_DB, N0_DBM |
| Path loss | CI model, FSPL(1 m) = 75.36 dB, exponent 1.9 | channel_model_nyu (142 GHz UMi, LOS) |
| Blockage loss | 3D boxes, ITU-R P.526 knife edge, cap 40 dB, blocked if >= 1 dB | blockage3d |
| Shadow fading (std) | 2.7 dB clear, 8.2 dB blocked | channel_model_nyu.SIG_LOS, SIG_NLOS |
| Heights | vehicle antenna 1.5 m, station 6.0 m | blockage3d.H_UE_DEFAULT, H_GNB_DEFAULT |
| Stations | 4, at the corners, 5 m outside the traffic extent | blockage3d.scene_gnb (margin 5 m) |
| Transmit power (every arm) | 30 dBm | paper2_env.PT_MAX |
| Serving beam gain; side lobe | 15 dBi; 0 dBi | paper2_env.G_SERV, G_SIDE (assumption) |
| Background interferers | 4 static vehicles per station at 30 dBm | paper2ma_env.K_BG, paper2_env.INTERF_PT_DBM |
| Sub-bands | 3, assigned round robin, fixed | paper2ma_env.N_SUBBANDS |
| Outage threshold γ_out | -13.8 dB SINR (= -92 dBm minus the noise power) | paper2_env.SINR_OUT_DB |
| Radio-link failure | N310 = 3 consecutive failed slots | paper2_env.N310 |
| Slot | 0.1 s (DAIR-V2X frame rate) | pool timestamps |
| Vehicles, places | 16 vehicles per scene; C = 4 (full load), C = 5 (load 0.8) and C = 6 (load 0.67) | run scripts |
| Handover delay | 0.2 s, 0.5 s, 1.0 s, 1.5 s (L = 2, 5, 10, 15 slots) | method_heldout.HLS |
| LSRA | β = 0.05, H = L + 1, forecast held from slot t | FREEZE.json, lsra.py |
| Scenes, episodes | 45 design and 60 held-out DAIR-V2X intersection scenes; 480 episodes per cell | pools, SEEDS_heldout60.csv |
| Statistics | scene bootstrap, 10,000 resamples; 98.75 % for four-baseline families | remeasure_ma.BOOT |

## T3. Arms and admission models

| Arm | What it does | Role |
|---|---|---|
| R0 | A vehicle below γ_out hands over to the less crowded of its two strongest other stations (current load). | rule, no capacity check |
| R1 | R0, but a station at or above C (current load) is not a candidate; with no candidate the vehicle stays. | rule with capacity check |
| R1i | R1 counting vehicles on their way (landing load), updated after each decision in the slot. | rule; R1 with the landing count |
| R2 | Joint assignment of exactly the vehicles in outage (R0's movers) to the places left; others stay. | restricted joint assignment |
| R3 | Joint assignment of every present vehicle to the G·C places, each slot, on the current geometry and current stations. | full joint assignment |
| R3F | R3 scoring each station by its mean received power at the landing slot, from the true future geometry. | reference; uses the future |
| LSRA | Landing-state risk assignment: free vehicles only, quotas q_g, cost r̄ + β per move, one assignment per slot. | proposed method |
| LSRA, perfect forecast | LSRA with the true future geometry in the risk window. | reference; uses the future |
| LSRA without the landing state | LSRA planning on the current stations with every vehicle movable. | ablation |
| LSRA with the worst-vehicle-first stage | LSRA with a min-max stage before the min-cost stage (dropped at the design stage). | ablation |

| Admission model | Rule |
|---|---|
| A0 | Every request starts; a station serves its C strongest vehicles at landing, the rest are unserved. |
| A1 | Per request at preparation (today's rule, most generous reading): strongest first, admitted if the landing load at the target is below C; departures credited; repeated passes. |
| A2 | Coordinated: the largest landing-feasible subset of the slot's requests is admitted as a set (the 6G requirement proposed here). |

## T4. Worst-vehicle RLF at 1.0 s (%), 95 % scene intervals

60 held-out scenes, 480 episodes per cell. Registered tests use 98.75 % intervals (Fig. 5). Each interval is its own scene bootstrap, so identical runs, such as LSRA under A0 and A2, can show slightly different intervals.

| Places, admission | R1 | R1i | R2 | R3 | LSRA | Registered test: LSRA beats R1, R1i, R2, R3 |
|---|---|---|---|---|---|---|
| 4 places, A0 | 9.19 [4.88, 14.44] | 9.08 [4.80, 14.45] | 8.19 [4.42, 12.87] | 8.56 [6.83, 10.38] | 3.59 [2.34, 4.99] | HM1: held |
| 4 places, A1 | 9.26 [4.86, 14.41] | 9.08 [4.75, 14.40] | 9.11 [4.85, 14.37] | 10.82 [6.59, 16.03] | 8.76 [4.45, 14.02] | HR1: not held (R1, R1i, R2 not shown) |
| 4 places, A2 | 9.26 [4.84, 14.62] | 9.08 [4.76, 14.21] | 8.23 [4.37, 12.99] | 6.70 [5.30, 8.15] | 3.59 [2.36, 4.99] | HR2: held |
| 5 places, A0 | 4.65 [2.06, 8.32] | 4.50 [1.95, 8.17] | 4.75 [2.18, 8.46] | 6.29 [4.92, 7.73] | 2.61 [1.48, 3.98] | — |
| 5 places, A1 | 4.61 [2.08, 8.28] | 4.50 [1.93, 8.23] | 4.46 [1.97, 8.07] | 6.11 [4.14, 8.67] | 3.97 [1.78, 6.80] | HL1: not held (R1, R1i, R2, R3 not shown) |
| 5 places, A2 | 4.61 [2.02, 8.28] | 4.50 [1.95, 8.10] | 4.46 [1.93, 8.07] | 5.09 [3.82, 6.47] | 2.61 [1.47, 4.04] | HL2: held |
| 6 places, A0 | 2.33 [1.14, 3.77] | 2.34 [1.18, 3.77] | 2.51 [1.30, 3.95] | 4.99 [3.38, 6.73] | 2.32 [1.15, 3.71] | — |
| 6 places, A1 | 2.33 [1.14, 3.77] | 2.34 [1.15, 3.76] | 2.45 [1.25, 3.87] | 3.95 [2.54, 5.50] | 2.32 [1.17, 3.68] | HL3: not held (R1, R1i, R2 not shown) |
| 6 places, A2 | 2.33 [1.14, 3.81] | 2.34 [1.13, 3.79] | 2.45 [1.25, 3.90] | 3.92 [2.47, 5.50] | 2.32 [1.15, 3.72] | HL4: not held (R1, R1i, R2 not shown) |

## T5. Ablations and room for forecasting, 1.0 s (worst-vehicle RLF %, 95 % scene interval; difference to LSRA)

Each interval is its own scene bootstrap (see T4).

| Arm | 4 places, A0 | 4 places, A2 | 5 places, A0 | 5 places, A2 | 6 places, A0 | 6 places, A2 |
|---|---|---|---|---|---|---|
| LSRA | 3.59 [2.34, 4.99] | 3.59 [2.36, 4.99] | 2.61 [1.48, 3.98] | 2.61 [1.47, 4.04] | 2.32 [1.15, 3.71] | 2.32 [1.15, 3.72] |
| LSRA without the landing state | 6.65 [4.52, 8.95]; +3.06 vs LSRA | 4.44 [3.00, 6.00]; +0.84 vs LSRA | 2.82 [1.59, 4.24]; +0.21 vs LSRA | 2.63 [1.48, 4.01]; +0.02 vs LSRA | 2.52 [1.28, 3.98]; +0.19 vs LSRA | 2.43 [1.27, 3.85]; +0.11 vs LSRA |
| LSRA with the worst-vehicle-first stage | 5.05 [3.66, 6.69]; +1.45 vs LSRA | 5.05 [3.65, 6.73]; +1.45 vs LSRA | 3.69 [2.66, 4.77]; +1.08 vs LSRA | 3.69 [2.67, 4.77]; +1.08 vs LSRA | 2.93 [1.92, 4.08]; +0.60 vs LSRA | 2.93 [1.91, 4.06]; +0.60 vs LSRA |
| LSRA, perfect forecast | 1.36 [0.69, 2.11]; -2.23 vs LSRA | 1.36 [0.71, 2.13]; -2.23 vs LSRA | 0.94 [0.39, 1.59]; -1.67 vs LSRA | 0.94 [0.39, 1.57]; -1.67 vs LSRA | 0.45 [0.10, 0.95]; -1.88 vs LSRA | 0.45 [0.10, 0.94]; -1.88 vs LSRA |
| R3 | 8.56 [6.83, 10.38]; +4.97 vs LSRA | 6.70 [5.30, 8.15]; +3.11 vs LSRA | 6.29 [4.92, 7.73]; +3.68 vs LSRA | 5.09 [3.82, 6.47]; +2.48 vs LSRA | 4.99 [3.38, 6.73]; +2.67 vs LSRA | 3.92 [2.47, 5.50]; +1.60 vs LSRA |
| R3F | 8.45 [6.87, 10.15]; +4.86 vs LSRA | 6.55 [5.17, 8.02]; +2.96 vs LSRA | 5.50 [4.24, 6.83]; +2.89 vs LSRA | 4.22 [3.09, 5.41]; +1.61 vs LSRA | 3.79 [2.51, 5.22]; +1.47 vs LSRA | 2.86 [1.84, 4.06]; +0.54 vs LSRA |

Pre-registered secondary test HR3 (95 % interval; 4 places, A2: LSRA without the landing state minus LSRA): +0.84 pp [+0.23, +1.62], held.

_make_tables_ma.py md5 b9a6916e231f1d1c83c07c77f9f803a4; python 3.11.9, numpy 2.4.4, pandas 3.0.3._
