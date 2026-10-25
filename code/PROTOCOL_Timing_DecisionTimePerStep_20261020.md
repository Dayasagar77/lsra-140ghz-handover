# PROTOCOL — decision time per step of LSRA (xApp time budget)

**Date:** 20 Oct 2026. **Status:** written before the run. Daya approved this step ("Figures and timing (Recommended)", 20 Oct).
**Where:** the laptop, Windows Python. Nothing in the cloud.

## 1. Question

How long does LSRA take to decide one step (handover decisions for all cars), compared with the time it has?
- One step is 0.1 s.
- A Near-RT RIC control loop runs on a time scale of about 10 ms to 1 s. The source for this range is still to be retrieved and cited before it goes in the paper.

This is a measurement, not a hypothesis test. Nothing is tuned, and no result of the paper changes.

## 2. Set-up (all frozen)

- **Scenes and seeds:** the 60 held-out scenes and the 480 frozen seeds of the held-out test.
- **Conditions:** 1.0 s delay, 4 places per station, admission at landing only (A0).
- **Environment:** the admission-check class with admission off. Gate G0 showed it reproduces the frozen episodes exactly.
- **Arms:** LSRA (METHOD) and the deployable baselines R1, R1i, R2 and R3, with β from the frozen `FREEZE.json`.
- **One thread** for numpy and scipy.
- **Warm-up:** 3 episodes per arm, untimed. Then all 480 episodes are timed.

## 3. What is measured

- **Policy time:** wall-clock time of each policy call (`time.perf_counter_ns`), one row per decision.
- **The two parts of LSRA:** the risk forecast (`lsra.window_risk`) and the assignment (`lsra.bottleneck_then_mincost`), with the number of free cars and of places on offer.
- **Reported per arm:**
  - mean, median, 90th, 99th and 99.9th percentiles and maximum;
  - the share of decisions above 10 ms and above 100 ms;
  - the same statistics split by the number of cars present (1–8, 9–12, 13–16).

## 4. Integrity check

- The timer only reads a clock.
- Each arm's 480 episodes must equal the frozen held-out episodes, column for column.
- An arm that differs is marked **VOID** and its timings are not used.

## 5. Recorded guess (not a hypothesis)

- LSRA median between 0.3 and 1.5 ms per step.
- LSRA 99th percentile below 5 ms.
- Every arm below 10 ms in more than 99.9 % of decisions.

## 6. Limits (stated now)

- **What is not timed:** the time is measured inside the simulator, in Python, on one laptop (Intel i9-12900H, Windows 11). It includes the risk forecast from geometry. It does not include E2 transport, message encoding or the RIC platform's own overhead.
- **Noise from the laptop:** other programs may run on the laptop during the run. This shows up in the tail (99.9th percentile and maximum), so the median and the 99th percentile are the main figures.
- **A slower language:** a compiled implementation would be faster, so the Python time is an upper reference, not a floor.

## 7. Code and outputs

- **Code:** `timing_lsra.py`. It checks `FREEZE.json`, `admission_check.py`, `paper2ma_env_admit.py` and this file's presence before it runs, and it runs once.
- **Outputs (`models/timing/`):**
  - `TIMING_RAW_<arm>.csv`;
  - `TIMING_RESULT.json`;
  - `TIMING_RESULT.txt`;
  - `STATUS.txt`.
