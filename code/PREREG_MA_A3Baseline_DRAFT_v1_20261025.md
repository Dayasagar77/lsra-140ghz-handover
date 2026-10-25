# PRE-REGISTRATION (DRAFT v1) — the 3GPP A3 baseline: the five TR 36.839 settings against LSRA

**Date:** 25 Oct 2026, about 4:40 PM IST. An independent static review of the code and this plan (a separate Claude agent, reading only, 4:20–4:40 PM) found no bug that changes results or crashes; its one integrity point (check records must carry the code's md5) and its wording points are applied.
**Status:** DRAFT. It needs Daya's "Freeze and run now". No A3 episode has been run on any scene. Before this draft is sent, only checks that use no A3 episode run on the laptop: the rule on hand-made cases, the dry run of the analysis on relabelled stored arms, and gate G0 (§6). Their results are added to §6 as records; nothing else in this file changes after them.
**Decision it serves:** on 25 Oct, Daya asked to add the 3GPP A3 comparison ("yes kindly add it too") and chose: all five 3GPP sets; registered tests under both A2 and A1; the tests plus all delays and loads; no separate literature check.
**Builds on:**
- the held-out test, the admission check (4 places), the first load check (5 places) and the third-level load check (6 places): their frozen pre-registrations and RESULT documents;
- Theorem 2 (general form) in the theory note.

**Where:** the laptop, Windows Python. Nothing in the cloud.

## 0. Why this check (plain words)

- **What networks do today.** A handover is triggered by 3GPP event A3: a neighbour must be stronger than the serving station by an offset, on filtered measurements, for a time-to-trigger (TTT) [TS 38.331, 5.5.4.4 and 5.5.3.2].
- **What the paper compares now.** R0–R3 decide from outage, load or a joint assignment. None of them is the standard trigger, so a reader, and the earlier reviewers, will ask how LSRA compares with what networks do today.
- **No tuning choice is ours.** TR 36.839 gives five standard parameter sets for mobility simulations (Table 5.3.2.1), from slow and cautious (Set 1) to fast and eager (Set 5). All five are tested.

## 1. What is added: five arms, nothing else

| Arm | TTT | A3 offset | L3 filter K | Consecutive slots | Filter weight per 0.1 s slot |
|---|---|---|---|---|---|
| A3S1 | 480 ms | 3 dB | 4 | 5 | 0.2929 |
| A3S2 | 160 ms | 3 dB | 4 | 2 | 0.2929 |
| A3S3 | 160 ms | 2 dB | 1 | 2 | 0.6011 |
| A3S4 | 80 ms | 1 dB | 1 | 1 | 0.6011 |
| A3S5 | 40 ms | −1 dB | 0 | 1 | 1 (no filter) |

- **Source of the sets:** 3GPP TR 36.839 V11.1.0, Table 5.3.2.1 (the laptop copy `36839-b10.doc`, md5 d3dd70ec87fd1cd408c454d573fe98b1; extract `TR36839_V11.1.0_Table5.3.2.1_extract_20261006.txt`). The L1-to-L3 period is 200 ms in every set. Table 5.2.4.1 of the same TR lists the same TTT, offset and K values and notes that the filter's time characteristics are kept by scaling K when the sample period is shorter than 200 ms.
- **Everything else is as in the earlier checks:**
  - the environment class `paper2ma_env_admit.AdmissionEnv`, with capacity-aware attachment and admission at landing as the backstop;
  - the 60 held-out scenes, the frozen seeds and the harness;
  - the admission models A0, A1 and A2; 4, 5 and 6 places; delays of 0.2, 0.5, 1.0 and 1.5 s.
- **LSRA is not run again.** Each A3 episode is paired, by seed, with LSRA's stored episode of the same cell, from the frozen runs. G0 shows that this runner reproduces those stored episodes exactly.

## 2. The rule (code: `a3_rule.py`)

Every 0.1 s slot, for every vehicle present:
1. **Measure.** The received power at every station in this slot: transmit power + serving-beam gain − close-in path loss − blockage loss − the slot's shadowing draw times its standard deviation (8.2 dB if blocked, 2.7 dB otherwise). This is the quantity the simulator itself uses for a serving link. No new random numbers are drawn.
2. **Filter** (layer 3): F ← (1 − a)F + a·M in dBm, with a = 1/2^(K/4) per 200 ms sample, so a_slot = 1 − (1 − a)^(1/2) per 0.1 s slot. The first measurement starts the filter. A vehicle that leaves and comes back starts a new filter.
3. **In flight:** no request, and its counts are cleared (as for every arm).
4. **Count:** for every neighbour n of the serving station p, the consecutive slots with F_n > F_p + offset. A slot that fails resets the count, and the counts restart when the serving station changes.
5. **Time to trigger:** the condition must hold at every slot start inside the TTT window, that is in ⌊TTT / 0.1 s⌋ + 1 consecutive slots (5, 2, 2, 1, 1). The request is made at the last of them.
6. **Request** a handover to the strongest triggered neighbour. A refused request may be repeated in the next slot, as for every arm. Full power and fixed sub-bands, as for every arm.

**Modelling choices, and which side they favour:**
- **No separate measurement error is added.** TR 36.839 suggests ±2 dB at 90 % (Table 5.2.4.1). Each measurement already carries the slot's independent shadowing draw (below), which is larger, so the net effect on A3 is unclear, most of all for the unfiltered Set 5.
- **The request comes 40–80 ms before the TTT ends** (the slot grid). This favours A3 slightly.
- **Hysteresis and all cell and frequency offsets are 0.** TR 36.839 gives only the A3 offset.
- **Beam-aligned measurement:** the serving-beam gain towards every station, as in the simulator's own attachment and admission ranking.
- **Each slot's measurement includes that slot's independent shadowing draw,** the same draw that sets the link. An unfiltered set (Set 5) therefore reacts to every fluctuation. This follows from the channel model's independent draws, a limit the paper already states.
- **A refused request is asked again in the next slot** if it is still triggered. No second candidate is tried in the same slot.
- **A3 ignores load,** as the standard trigger does. Capacity enters only through admission (A0, A1, A2), as for every arm. Load balancing through cell offsets is not modelled.

## 3. Data

- **Scenes and seeds:** the 60 held-out scenes (`dair_geometry_pool_heldout60_20261020.csv`, md5 3a68d055…) and the frozen seed list `SEEDS_heldout60.csv`. That gives 480 episodes per cell.
- **Cells:** 3 admission models × 3 numbers of places × 4 delays = 36 cells, each with the five A3 arms.
- **Reuse:** this is the seventh use of these scenes, after the held-out test, the admission check, the first load check, the delay-difference check, the third-level load check and the link-budget check. The five settings are fixed by TR 36.839, and nothing is tuned on these scenes.

## 4. Primary (1.0 s, 4 places): two registered tests, both reported, whatever they show

- **HA1 (A2, coordinated admission).**
  - For each set k in {1, …, 5}: mean(A3Sk − LSRA) > 0 in per-episode worst-vehicle RLF.
  - Each comparison needs its 99 % scene interval (10,000 resamples, Bonferroni over five) above 0.
  - HA1 needs all five.
- **HA2 (A1, per-request admission).** The same.
- No claim of the form "holds under at least one model". With these two, the paper reports 14 registered tests.

## 5. Secondary (95 % scene intervals; reported, not tested)

- **Today's practice against LSRA with coordinated admission:** each A3 set under A1 minus LSRA under A2 (4 places, 1.0 s).
- **Every A3 set minus LSRA under the same admission,** in all 36 cells.
- **Tables:** levels; handovers per vehicle-step; requests per vehicle-step and the share refused (A1 and A2); the causes of the worst vehicle at 1.0 s.
- **The best A3 set per cell** (descriptive). It is picked after the run, so it favours A3.

## 6. Checks

**Before this draft is sent (no A3 episode is used):**
- **U1–U9, the rule on hand-made cases** (no scenes). They cover the slot counts and filter weights; the trigger slot for a steady and a stepped neighbour; a one-slot dip, to +1 dB, that the K = 4 filter of Set 2 rides out, and a deeper one, to −6 dB, that restarts Set 1's count; Set 5's negative offset; the strongest of two triggered neighbours; no request in flight and a fresh count after landing; a vehicle that leaves and comes back; slots out of order (refused) and a new episode at t = 0.
- **Dry run of the analysis.** The stored episodes of R0, R1, R1i, R2 and R3 are relabelled as A3S1–A3S5, in all 36 cells, and the registered analysis runs on them. It must reproduce the earlier checks' results to within 10⁻⁹ (the relabelled files are written and read again): every level, handover rate, cause share, request rate, refused share and difference from LSRA, and the HR1 and HR2 means. Output in `models/_dry_a3/` (not a result).
- **G0.** This runner reproduces the stored episodes exactly: R1 and METHOD, first 40 seeds, 1.0 s, every admission model at every number of places (18 runs), every stored column.

**After Daya's OK, before the freeze:**
- **G1.** The A3 measurement equals the simulator's own received power (its `_desired_dbm` minus the slot's shadowing term), for every slot, vehicle and station of 5 episodes (A2, 4 places), to within 10⁻⁹ dB.
- **G2.** The five A3 arms are deterministic (10 episodes, run twice, identical), every arm requests handovers, started handovers = requests − refusals, no landing excess is created, and there are no closed-loop violations (A1, 4 places, 1.0 s).
- **G3.** Every A3 arm runs in every one of the 36 cells (first seed only; only pass or fail is recorded).
- **If any check fails,** I stop and report.

**Must hold in the run.** If any fails, the results are void: stop, find the cause, report, and get a new OK before any re-run.
- **M1:** no A3 arm creates landing excess under A1 or A2.
- **M3:** 0 closed-loop violations under A1.
- **M4:** handovers started = requests − refusals in every A1 and A2 episode.
- **M5:** every A3 cell pairs with LSRA's stored cell: the same 480 seeds and the same scene in every pair.

## 7. Decisions fixed now

| Outcome | What the paper says |
|---|---|
| HA1 held | With coordinated admission at full load, LSRA beats the standard trigger at all five TR 36.839 settings, as well as the four baselines (Section 8.2). The abstract and the Conclusion name the A3 comparison in one clause, within the 200-word limit. |
| HA1 not held | The paper says plainly at which settings LSRA is not shown to beat the standard trigger under coordinated admission, in Section 8.2, the abstract and the Conclusion. It makes no claim of beating today's handover. |
| HA2 held | Section 8.1: under per-request admission LSRA beats every A3 setting, although it is not shown to beat the simple rules (HR1). |
| HA2 not held | Section 8.1: under per-request admission, A3 joins the methods that LSRA is not shown to beat, as Theorem 2 leads one to expect. |
| any must-hold check fails | The results are void. Find the cause and report; a new OK is needed before any re-run. |

- **Today's-practice contrast** (A3 under A1 minus LSRA under A2) is reported in Section 8.2 whatever it shows.
- **No tuning** of LSRA, and no A3 setting other than the five.
- **Where the results go:** Table 3 (the arms), Section 7 (design and the two tests), Sections 8.1 and 8.2, and Table 4 or a new short table. Every number is taken from `A3_RESULT.json`.

**Recorded guesses** (coded before the run; not hypotheses):
- HA1 held; HA2 not held;
- the best A3 set under A2, 4 places, 1.0 s, between 6 and 10 %;
- the best A3 set under A1, 4 places, 1.0 s, between 8.5 and 11 %;
- Set 5 makes the most handovers and Set 1 the fewest (A0, 4 places, 1.0 s);
- LSRA's lead over the best A3 set under A2 is smaller at 6 places than at 4.

**Why these guesses.**
- At full load under A2, the load-blind rules R0, R1i and R2 are at 8.2–9.1 % and R3 at 6.7 %, against 3.59 % for LSRA. A3 is also blind to load, so it should land in the same range.
- Under A1, every method was at 8.8–10.8 % (Theorem 2: only closed loops remain feasible at full load).
- At 6 places under A2, R0, which has no capacity check, was at 2.29 % against 2.32 % for LSRA. A3 may match LSRA there too.
- My guesses about per-request admission were wrong before. These are recorded so that any surprise is visible.

## 8. Threats and limits

- **A3 here is the plain trigger.** Real networks add mobility robustness and load balancing (cell offsets), conditional handover and RLF recovery. None is modelled.
- **TR 36.839 defines the sets for LTE heterogeneous networks at 2 GHz.** We use them as the standard reference sets, not as settings tuned for 140 GHz.
- **Independent shadowing draws** make single measurements noisy. The filtered sets smooth this, and the unfiltered Set 5 does not.
- **A1 is the most generous per-request reading, and A2 an ideal coordinator.**
- **Simulator only; one machine; deterministic code.**
- **This is the seventh use of the test scenes.**
- **Prior art:** no separate literature check, by Daya's choice on 25 Oct. The run adds a standard baseline, not a new method or claim.

## 9. Code and compute

- **New code:** `a3_rule.py` and `a3_check.py` (md5s recorded in §10 at the freeze), with modes `unit`, `dry`, `gate0`, `gate`, `freeze`, `test --adm A0|A1|A2 --cap 4|5|6 --hl H` and `analyse`.
  - It reuses `load_check_c6.py` (0fd5c893…), `admission_check.py` (1b16d561…) and `paper2ma_env_admit.py` (5a644059…), all unchanged.
- **Every check record** (UNIT, DRY, GATE0, GATE) stores the md5 of the two new files, and DRY also stores the md5 of the 36 stored episode files. The freeze refuses if any of them differs from the files at the freeze.
- **Checked before every step:** `models/method_heldout/FREEZE.json`, and from the freeze on `models/a3_check/A3_FREEZE.json`. The latter records the md5 of:
  - the two new files, the reused runners and the simulator files;
  - this pre-registration (frozen copy) and the four check records (UNIT, DRY, GATE0, GATE);
  - the four earlier freeze files;
  - all 36 stored episode files that are paired with the A3 runs.
- **Compute (laptop):**
  - checks before the OK: about 6 minutes; G1–G3: about 2 minutes;
  - run: 36 cells as 6 jobs of 6 cells, the 4-place cells at 1.0 s first, about 45–60 minutes;
  - analysis: once, about 3–4 minutes.
- **Outputs (`models/a3_check/`):** `UNIT.json`, `DRY.json`, `GATE0.json`, `GATE.json`, `A3_FREEZE.json`, `A3_EPISODES_C{4,5,6}_{A0,A1,A2}_hl{2,5,10,15}.csv`, `A3_RESULT.json`, `A3_RESULT.txt`, `STATUS.txt`.

## 10. Freeze record

To be filled at the freeze: Daya's approval and its time, the code md5s, the results of G1–G3 and the md5 of `A3_FREEZE.json`.
