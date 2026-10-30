# EXPLORATORY ANALYSIS (post hoc) — how optimistic is the per-scene A3 bound? A cross-fit estimate (30 Oct 2026)

**Status:** fixed before it is run (about 5:40 AM IST, 30 Oct). Daya's OK: "OK" to the proposal in the new-baselines result note.
- **Not a registered test.** It was decided after seeing the A3-O result of the new-baselines run, so the paper must label it exploratory.
- **No new episodes.** It uses only the stored files of that run and LSRA's stored episodes.

**Question.** The registered A3-O bound picks, for each test scene, the best of the 120 measured-signal A3 settings using the same 8 episodes it is then scored on. Luck therefore counts in its favour. How much of A3-O's lead over LSRA survives when the setting is chosen on some of a scene's episodes and scored on the others?

## Method (all fixed now)

**Data.** For each cell (4 and 5 places, coordinated admission A2, 1.0 s):
- the 120 files in `models/baselines_check/oracle/C{cap}_A2/`, taken in grid order (`a3_general.grid('measured')`);
- LSRA's stored episodes (`a3_check.stored_path('A2', cap, 10)`, arm METHOD).

They are paired by seed and scene. There are 60 scenes with 8 episodes each.

**Cross-fit, per scene.**
1. Take every way to split the scene's 8 episodes (in seed order) into two halves of 4. There are 35 such splits; the first episode is always in half A, so no split is counted twice.
2. For each split:
   - choose the setting with the lowest mean J on half A (ties: the earlier in grid order) and score it on the 4 episodes of half B;
   - choose on half B and score on half A.
3. Each episode's cross-fit J is the average, over the 35 splits, of the J it receives from the setting chosen on the other half.

**Quantities, per cell.**
- the in-sample bound (the registered A3-O, for reference);
- the cross-fit estimate (mean cross-fit J over the 480 episodes);
- the optimism (cross-fit minus in-sample);
- **the cross-fit minus LSRA**, paired by episode, with a 95 % scene-bootstrap interval (`remeasure_ma.scene_boot`, 10,000 resamples; seed 13000 at 4 places, 13001 at 5 places);
- the cross-fit minus the fixed tuned A3 (A3-T of the registered run, under A2), with a 95 % interval (seeds 13002, 13003).

**Reading (fixed now).**
- If the cross-fit minus LSRA is above 0 with its interval above 0, per-scene tuning without hindsight does not close the gap.
- If its interval includes 0, per-scene tuning without hindsight is level with LSRA.
- If it is below 0 with its interval below 0, honest per-scene tuning beats LSRA.

**Caveat (stated now).** Choosing on 4 episodes is noisier than choosing on 8. So the cross-fit leans slightly against A3: it is a conservative estimate of per-scene tuning, while the in-sample bound is an optimistic one. The truth lies between them.

**Code:** `Paper2/code/a3o_crossfit.py` (new file; it reads only). **Output:** `models/baselines_check/A3O_CROSSFIT.json` and `.txt`.
