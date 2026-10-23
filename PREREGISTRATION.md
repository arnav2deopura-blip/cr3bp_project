# Pre-registration (FROZEN 2026-10-23; later changes only as dated entries under "Deviations")

Pass this file with `--prereg PREREGISTRATION.md`; its SHA-256 is stored in each report. Anything you change
after seeing results must be listed under "Deviations" at the bottom, with the date.

Freeze mechanics: there is no git here. `freeze_snapshot.py snapshot` hashes every source file; `freeze_snapshot.py seal` writes
`FREEZE.json` (date, code hash, SHA-256 of everything above "## Deviations", SHA-256 of the data CSVs). After the seal,
`run_rigor_suite_3d.py final` refuses to run if the code or the text above "## Deviations" has changed.

Frozen on: `2026-10-23`. Student: `Arnav Deopura`. Code snapshot SHA-256: `79a6455ddef80ac312af267d14d952d3facb3e9c6b6f0c7dd752b794c61968e1`.

## Data and protocol (fixed in advance)
- Systems: Earth-Moon (mu = 0.01215058535). Optional second system for the cross-mu question: Sun-Earth, generated
  with the SAME normalized tolerance (`--tol_du`).
- Tolerance: 100 km Earth-Moon = 100/384400 DU = 2.6015e-4 DU, the same normalized tolerance for Sun-Earth (38,917 km).
  Confirmed: 100 km. It is illustrative, not a mission requirement. The earlier `fidelity` run (CR3BP vs a real ephemeris) gave
  errors of 2,000-50,000 km, so every result here is about integrator error against the CR3BP, not about real trajectories.
  Changing the tolerance means regenerating the datasets.
- Labels: `--label_mode accuracy_monotone` (the only dataset label). The halving ("accuracy") label is computed alongside it
  inside `audit --dense` for H4; no separate accuracy-mode dataset is generated.
- Model (pinned, same in every command: gate, final, audit, ablate, replicate, mu): hurdle models with random-forest classifiers
  (`--n_estimators 150`) and a HistGradientBoosting QUANTILE regressor on log10(dt), quantile 0.1, `--seed 42`.
  `--regressor rf` is allowed only as a labelled sensitivity run (the CLI prints a warning). Quantile 0.1 confirmed; it was chosen
  before any tuning and has not been tuned. Because the regressor aims low on purpose, R^2 against the oracle step and selection accuracy
  are lower than for RF and are not comparable to earlier RF results.
- Certification (ML step scale, global fixed step, adaptive rtol; alpha = 0.05, delta = 0.1): for every value g on a pre-specified grid,
  one-sided binomial test of "violation rate > alpha" with p = P(Binomial(n, alpha) <= observed violations); g is certified if
  p <= delta / m (Bonferroni over the m grid values, no monotonicity assumption); the LARGEST certified g is used. A violation is an error
  above the tolerance, a collision, an escape, a non-finite result, or a run stopped by the 4,000,000-step cap. Grids: scales
  `LTT_SCALE_GRID` (18 values, 0.02-1.0), global steps `GLOBAL_DT_GRID` (50 values, 1e-4-0.49), adaptive rtol 1e-12-1e-2 in half decades.
  An adaptive (RK45/DOP853) solve that needs more than 2,000,000 right-hand-side evaluations is abandoned: it counts as a violation while
  calibrating the rtol, and in the cost comparison the orbit is skipped (counted in `n_skipped`; `adaptive_cap_hits_total` is reported).
  Without this cap a single orbit with a very close approach stalled a test comparison for over an hour.
- Splits: `gate` and `final` train on a random 70% of the development rows (`--split random`, pinned) and calibrate on rows from the
  held-out 30%, so calibration rows are drawn like the locked rows and never used for training. `--split grouped` is a stress test only.
  Other commands keep the grouped (region-shift) split, which answers a different question.
- Datasets (label pipeline version 3, SeedSequence seeding; SHA-256 recorded in each run's manifest and in `FREEZE.json`):
  - Earth-Moon, 20,000 orbits each, base seeds 12 (primary: development plus the locked set), 13 and 15 (replication):
    `data/cr3bp_3d_training_data_accuracy_monotone_seed{12,13,15}.csv`
    (sha256 15bc463aa6dd2caf8b325ff13120e94a4f52635a35228327c50b547256e0532e, 38fb401b9f3807979869a82ac97ecbea553d9e9ba79fd936fadbf1c8240660d6,
    7de9acef8b5872086fc3d3c3a2da1f5bf38751ff7e394e7b8e82d167c1899a61).
  - Sun-Earth (mu = 3.00348e-6), 5,000 orbits, base seed 14, for the cross-mu question:
    `data/cr3bp_3d_training_mu_0.000003_accuracy_monotone_seed14.csv`
    (sha256 1a39b6e40b66d5a3ca414ac300f292ddaf8ad5bcf2003d808efe1ae4fbd295fb).
- Locked test set: drawn once with `lock` (seed 2024, fraction 0.25, random) from the seed-12 CSV into `data/locked_test_set.json`
  (5,000 of 20,000 rows; about 4,060 are usable after reference-divergent rows are dropped, leaving about 12,180 usable development rows);
  evaluated by `final` only after every development decision is made. Every `final` call is logged; report the number of calls.
  A `final` that has to be repeated (for example after a bug) needs a dated reason under "Deviations".

## Hypotheses and decision rules
H1 (cost). On the locked set, the ML recommendation (candidate integrators RK4 and leapfrog, certified safety scale,
FTLE feature cost INCLUDED; model prediction cost is charged only in H1', in H1 it is reported as wall time) meets the tolerance in >= 90% of orbits AND needs fewer
right-hand-side-equivalent evaluations than each of RK45 and DOP853 in >= 50% of orbits, against BOTH the per-orbit
oracle rtol ("_spec") and the calibrated rtol ("_cal"). H1 is supported only if all evaluated comparisons pass.
Otherwise report: "ML does not beat adaptive solvers on cost."

H1' (cost, no-FTLE variant; added 2026-10-23, see Deviations). Same orbits, tolerance, calibration rows, alpha/delta and
thresholds as H1 (meets the tolerance in >= 90% of orbits AND needs fewer right-hand-side-equivalent evaluations in >= 50%),
but with (i) a model trained WITHOUT the FTLE feature (same regressor and quantile as H1), (ii) feature cost charged at
1 RHS-equivalent for the remaining state-derived features, and (iii) the model's batched inference cost charged
(`ml_total_evals_equiv_pred`, measured wall-time ratio to one compiled RHS evaluation). Judged in two parts, each on its own:
- H1'a (vs adaptive): against RK45 and DOP853, both the per-orbit oracle rtol ("_spec") and the calibrated rtol ("_cal").
  Supported only if all four comparisons pass.
- H1'b (vs global fixed step): against the best SINGLE fixed step per integrator (RK4, leapfrog), certified on the same
  calibration rows with the same alpha/delta and the same certification rule as the ML scales (see "Certification"; a collision,
  escape or step-cap run counts as a violation). Supported only if both comparisons pass.
A part (and H1 itself) is supported only if EVERY comparison in it was evaluated and passed. If any evaluated comparison fails it is
reported as not supported (with the number of comparisons that could not be evaluated, e.g. because an rtol or step could not be
certified). If none fails but at least one could not be evaluated it is reported NOT_FULLY_EVALUATED, never as supported on a subset.
The original H1 is still run and reported with its original comparisons. H1 and H1' are reported together.

H2 (safety). The certified scales (alpha = 0.05, delta = 0.1, rule under "Certification", calibrated on rows exchangeable with the
locked rows) give a violation rate <= 5% on the locked audit rows;
report the Clopper-Pearson interval, and the spread of the chosen scale over calibration repeats (`audit --ltt_repeats`).

H3 (is FTLE needed). In `ablate`, FTLE is "needed" if the paired 95% interval of the change in `rk4_r2_uncens`
or `selection_acc_no_euler` when it is removed lies entirely below zero. If it is needed, its cost is charged to the ML
side (already done in `gate`). H3 is evaluated with the pinned regressor; an RF run of `ablate` is also reported as a sensitivity check.

H4 (labels). Report the fraction of orbits with non-monotone pass/fail over the dense sweep (n >= 200) with its
interval, and the relative disagreement between the halving label and the monotone label.

H5 (error model). The conformal error-model recommender has violation rate <= 10% among covered test orbits; report
coverage and cost versus adaptive solvers as in H1.

H6 (crash/escape). Report accuracy, balanced accuracy, recall and AUC with and without the FTLE feature, and recall by
outcome type. No pass/fail threshold.

H7 (convergence). Report median fitted slopes per integrator and region against the textbook orders (Euler 1,
leapfrog 2, RK4 4). No pass/fail threshold; deviations are findings.

## Which data evaluates what
- Lock guard: a CSV that has a `locked_test_set*.json` next to it (matched by the CSV's SHA-256) is refused by every command unless
  `--locked` points at that file. `--ignore_lock` overrides this, prints a warning and is recorded in the run's `manifest.json`;
  nothing computed with it may be used for a claim about the locked set.
- Locked set (via `final`, once): H1, H1' and the locked audit for H2.
- Development rows only (every command run with `--locked data/locked_test_set.json` so locked rows are excluded): H3 (`ablate`),
  H4 and the calibration-stability part of H2 (`audit`), H5 (`errmodel`), H6 (`crash`), H7 (`convergence`), `proxy`, and `gate`.
- Replication and cross-mu: `replicate` (seeds 12, 13, 15) and `mu` (seed 12 vs the Sun-Earth seed 14); `--locked` removes the locked rows of the
  seed-12 dataset from these. `replicate` trains on a random 70% of dataset A and reports "within" (the other 30% of A) next to "replicate"
  (the same model on every row of independent dataset B); `replicate_minus_within` is the dataset-to-dataset effect. The grouped
  (region-shift) score is reported separately as `within_grouped` and is not comparable with the other two.

## Run plan (all from this folder with the base conda python; pinned arguments, defaults otherwise)
```
D12=data/cr3bp_3d_training_data_accuracy_monotone_seed12.csv   # likewise D13, D15; S14=data/cr3bp_3d_training_mu_0.000003_accuracy_monotone_seed14.csv
L=data/locked_test_set.json
python3 freeze_snapshot.py snapshot ; <fill placeholders> ; python3 freeze_snapshot.py seal
python3 run_rigor_suite_3d.py lock      --data $D12 --locked_out $L --lock_seed 2024 --lock_frac 0.25 --out_dir out
python3 run_rigor_suite_3d.py audit     --data $D12 --locked $L --prereg PREREGISTRATION.md --seeds 10 --audit_rows 100 --ltt_rows 600 --ltt_repeats 10 --tol_sens 50 --dense 200 --halo --out_dir out
python3 run_rigor_suite_3d.py ablate    --data $D12 --locked $L --prereg PREREGISTRATION.md --seeds 5 --out_dir out
python3 run_rigor_suite_3d.py ablate    --data $D12 --locked $L --prereg PREREGISTRATION.md --seeds 5 --regressor rf --out_dir out   # sensitivity
python3 run_rigor_suite_3d.py crash     --data $D12 --locked $L --prereg PREREGISTRATION.md --seeds 5 --out_dir out
python3 run_rigor_suite_3d.py proxy     --data $D12 --locked $L --prereg PREREGISTRATION.md --out_dir out
python3 run_rigor_suite_3d.py errmodel  --data $D12 --locked $L --prereg PREREGISTRATION.md --n_orbits 300 --out_dir out
python3 run_rigor_suite_3d.py convergence --data $D12 --locked $L --prereg PREREGISTRATION.md --n_orbits 300 --out_dir out
python3 run_rigor_suite_3d.py replicate --dataset s12=$D12 --dataset s13=$D13 --dataset s15=$D15 --locked $L --prereg PREREGISTRATION.md --seeds 5 --out_dir out
python3 run_rigor_suite_3d.py mu        --dataset EarthMoon=$D12 --dataset SunEarth=$S14 --locked $L --prereg PREREGISTRATION.md --seeds 5 --out_dir out
python3 run_rigor_suite_3d.py fidelity  --prereg PREREGISTRATION.md --out_dir out        # needs network + astroquery
python3 run_rigor_suite_3d.py gate      --data $D12 --locked $L --prereg PREREGISTRATION.md --out_dir out
python3 run_rigor_suite_3d.py final     --data $D12 --locked $L --prereg PREREGISTRATION.md --out_dir out    # once
```
Defaults that matter: `--seed 42`, `--n_estimators 150`, `--alpha 0.05`, `--delta 0.1`, `--n_cal 2000`, `--n_cal_adaptive 300`,
`--n_baseline 1000`, `--regressor hgb_quantile --quantile 0.1`, `--tol_km` taken from the dataset (100).

## Reported regardless of outcome
All of H1-H7, including negative results; the number of orbits skipped in each comparison; Euler as a negative control.

## Deviations (date, what, why)
- 2026-10-23 (before the freeze): replaced the fixed-sequence-from-the-smallest-value certification with Bonferroni over the grid (largest
  certified value), for the ML scales, the global fixed step and the adaptive rtol alike. Why: the earlier rule stopped at the first
  non-rejection, and step-cap runs counted as violations make the smallest scale the worst one (leapfrog 2.7% step-cap failures at scale
  0.02 in a 300-row test; 5-13% in the previous run), so nothing could be certified. Cost: each value needs p <= 0.1/m, a small loss of power.
  Also before the freeze: `gate`/`final` now use a random train/held-out split (the grouped split put the calibration rows in different
  regions than the random locked rows); `replicate` compares like with like; the original H1 verdict can no longer be "competitive" on a subset
  of its comparisons. A failed comparison now outranks missing ones in both H1 and H1'.
  Also before the freeze: adaptive solves are capped at 2,000,000 RHS evaluations (see Certification).
- 2026-10-23 (before the freeze): corrected the wording of H1 and the Data section to match the implemented cost basis. The earlier draft said
  H1 includes "prediction cost", but the code (`ml_total_evals_equiv`) only ever charged the FTLE feature; prediction cost is charged only in
  H1' (`ml_total_evals_equiv_pred`). Also removed the "accuracy (secondary)" dataset (never generated; H4 uses `audit --dense`), and filled in
  seeds, locked-set parameters, tolerance and quantile (accepted as proposed).
- 2026-10-23: added H1'a and H1'b (and the code behind them: `--no_h1_prime` off by default in `gate`/`final`,
  `calibrate_global_dt_ltt`, `measure_prediction_cost`). Why: on development data (Earth-Moon seed 12, 80 sampled orbits) the FTLE feature
  alone cost more than a whole DOP853 solve in 80/80 orbits (median ~70x), and `gate`/`final` always charge it, so H1 as written
  cannot be supported regardless of safety quality. Also seen before writing this (post hoc, development data only; the locked set
  did not exist): with an oracle step size, free features and free prediction, ML was cheaper than DOP853 at rtol 3.2e-7 in 37% / 24% / 8%
  of orbits at safety scales 1.0 / 0.6 / 0.4, and batched model inference was measured at ~730 RHS-equivalents per orbit (RK4 + leapfrog,
  hgb_quantile regressors + RF classifiers) against a median DOP853 solve of ~254 evaluations, so H1'a is expected to fail; H1'b was
  added because the adaptive solvers are not the only comparator a user of fixed-step integrators has. H1 itself is unchanged.
- 2026-10-23: pinned the regressor (hgb_quantile, quantile 0.1) as the default for gate, final, audit, ablate, replicate and mu, and added
  `--regressor/--quantile` to ablate, replicate and mu (+ `model_kw` in `rigor_3d.py`). Why: in the previous run `gate` used hgb_quantile but
  `final` used rf, so the locked evaluation did not test the model that was developed. Measured on development data (Earth-Moon seed 12, 2 seeds):
  rk4_r2_uncens 0.65-0.83 (rf) vs 0.19-0.60 (hgb_quantile); selection_acc_no_euler 0.92-0.97 vs 0.89-0.94 (majority baseline ~0.90); removing FTLE
  changed rk4_r2_uncens by -0.10 (rf) and -0.28 (hgb_quantile), same sign. `crash` is unaffected (it fits no hurdle regressor).
