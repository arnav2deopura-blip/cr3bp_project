# Exploratory follow-ups: E3c (lookahead, HB3) and the mu dial (E7, HB6 second half)

Both were pre-registered as EXPLORATORY in the dated entry of 2026-10-26 under "## Deviations" of `PREREGISTRATION_B.md`, committed before any lookahead
feature or mu-dial dataset existed (git 5d24baf). They add nothing to the sealed Holm family; there are no hypothesis tests below, only estimates with
intervals. The code they use (`lookahead_3d.py`, `mu_dial_3d.py`, `make_mu_dial_b.py`, new commands) is outside the seal, so `freeze_b.py check`
reports those files as added or changed; every confirmatory step had already run and is logged. Figures: `part_b/figures/exploratory/`.
Results: `part_b/results/exploratory/{e3c,mu_dial}/`.

## 1. E3c: how much of the trajectory must be seen, and what seeing it costs (D_explore, RK4 only)

Design as pre-registered: an RK45 solve (rtol 1e-6) over the first fraction f of the horizon gives closest approaches and timescales; the model is the q = 0.1
gradient-boosted regressor on log10 d*_RK4 with the state features plus those; the scale is certified on the 2,000 calibration rows with the sealed rule;
3,000 held-out evaluation orbits; model cost excluded (a single regressor is cheaper than the 1,128-evaluation two-integrator model; adding 1,128 makes every
share below 0). No solve hit the evaluation cap. Lookahead cost = scipy `nfev`.

| f | certified scale | violations | median abs error, dex [95 % CI] | 90th pct | integration cost | lookahead cost | cheaper than RK45-cal [95 % CI] | cheaper than DOP853-cal | cheaper than RK45-spec |
|---|---|---|---|---|---|---|---|---|---|
| 0 (state only) | 0.5 | 3.3 % | 0.32 [0.30, 0.35] | 1.03 | 1,538 | 0 | 27.0 % [25.4, 28.6] | 17.7 % | 0.5 % |
| 0.05 | 0.6 | 4.4 % | 0.28 [0.26, 0.30] | 0.99 | 1,194 | 20 | 29.5 % [27.9, 31.2] | 16.9 % | 0.1 % |
| 0.1 | 0.6 | 4.1 % | 0.25 [0.24, 0.27] | 0.90 | 956 | 38 | 29.3 % [27.7, 31.0] | 15.3 % | 0.1 % |
| 0.25 | 0.6 | 3.5 % | 0.18 [0.17, 0.19] | 0.75 | 880 | 80 | 28.9 % [27.3, 30.6] | 11.3 % | 0.0 % |
| 0.5 | 0.6 | 2.4 % | 0.12 [0.12, 0.13] | 0.45 | 576 | 158 | 23.9 % [22.4, 25.4] | 4.0 % | 0.0 % |
| 1.0 | 0.8 | 1.1 % | 0.07 [0.07, 0.08] | 0.23 | 312 | 302 | 6.9 % [6.1, 7.9] | 0.1 % | 0.0 % |

(Costs are medians over orbits that met the tolerance; the adaptive medians on the same orbits are 134 for RK45-spec and 380 for RK45-cal. Pinball loss and the
over-prediction rate are in `e3c_by_fraction.csv`; only the median error and the shares carry intervals.)

- **The error falls steadily with f and has no early plateau**: 0.32 to 0.07 dex, and the 90th percentile 1.03 to 0.23. The smallest f whose median error is within 25 % of the f = 1
  error is 1.0 in 8 of 9 groups (Hill regimes II-V and all four min-r2 bands) and 0.5 only in regime I (`e3c_by_group.csv`). What the step depends on is spread over
  the whole horizon, not concentrated in a first stretch.
- **A cheap solve gives features nearly as good as the full reference.** At f = 1 the error (0.07 dex) is close to the 0.06 dex that E3b reached with reference-derived
  features, from a 302-evaluation RK45 solve (rtol 1e-6) instead of the tight reference. The information is obtainable; its price is the point.
- **The lookahead cost cancels the saving.** The median total cost per orbit (integration + lookahead + 1 evaluation for the state features) falls from 1,539 to 635 evaluations, still 4.7x the RK45-spec solve and 1.7x
  the RK45-cal solve (the figure stacks the two medians, 614, which differs from the median of the sums). The share of orbits cheaper than RK45-cal rises slightly with a short lookahead (27.0 % to 29.5 %, intervals do not overlap) and collapses once the
  lookahead is long (6.9 % at f = 1). Against the oracle-tuned RK45-spec it is below 1 % at every f. If each orbit could pick its own best f in advance, the share
  against RK45-cal would be 35.2 % (DOP853-cal 20.0 %, RK45-spec 0.5 %); that needs an oracle and still excludes the model.
- **Reading for HB3.** To reach the accuracy of full-trajectory knowledge the policy has to see essentially the whole horizon, and at that point the lookahead alone (302) costs
  more than the oracle-tuned adaptive solve (134) and 80 % of the calibrated one (380). The break-even f* the plan asks for does not exist at the median against RK45-spec, and
  against RK45-cal the total never falls below 380. This is an exploratory result on one dataset, one model family and RK4 only; it was not replicated on D_confirm.

## 2. The mu dial (E7): the same study at five mass ratios

Design as pre-registered: seeds 31-35, 8,000 orbits per mu, sampling scaled by the Hill radius, tolerance 100 km = 2.6e-4 DU everywhere, per-mu model on 3,000 training orbits,
scales and tolerances certified on 1,000 calibration orbits, the rest (2,368-2,595 orbits) evaluated. Usable orbits 6,368-6,595 per mu; RK4 labels censored on 0.2 % (mu = 1e-6) and 0
elsewhere. Dataset hashes: `part_b/results/exploratory/mu_dial/dataset_hashes.json`.

| mu | perfect fixed step cheaper than RK45-spec [95 % CI] | median paradigm gap, dex [CI] | OLS slope of gap on log10 delta_phys [CI] | median delta_phys | integration-only ML gap, dex [CI] | violation rate [CI] | certified scale RK4 / leapfrog |
|---|---|---|---|---|---|---|---|
| 1e-6 | 37.0 % [35.1, 38.9] | +0.06 [0.05, 0.07] | 0.71 [0.68, 0.74] | 1.53 | +0.50 [0.49, 0.52] | 3.3 % [2.7, 4.1] | 0.5 / 0.5 |
| 1e-4 | 31.3 % [29.5, 33.1] | +0.10 [0.09, 0.11] | 0.86 [0.84, 0.88] | 1.59 | +0.55 [0.54, 0.57] | 3.7 % [3.0, 4.5] | 0.5 / 0.4 |
| 1e-3 | 29.4 % [27.7, 31.2] | +0.11 [0.10, 0.12] | 0.88 [0.86, 0.90] | 1.69 | +0.70 [0.68, 0.72] | 2.7 % [2.1, 3.4] | 0.4 / 0.3 |
| 0.01215 (Earth-Moon) | 23.5 % [21.8, 25.2] | +0.17 [0.16, 0.18] | 0.90 [0.88, 0.91] | 2.14 | +1.20 [1.17, 1.23] | 3.4 % [2.8, 4.2] | 0.25 / 0.3 |
| 0.05 | 24.4 % [22.7, 26.2] | +0.21 [0.19, 0.22] | 0.93 [0.92, 0.95] | 2.62 | +1.35 [1.31, 1.38] | 3.5 % [2.8, 4.4] | 0.3 / 0.25 |

Mean decomposition against RK45-spec (dex), mu = 1e-6 to 0.05: paradigm +0.10, +0.15, +0.17, +0.22, +0.30; prediction +0.25, +0.27, +0.29, +0.36, +0.48; safety scale +0.30, +0.32, +0.41, +0.59, +0.53;
overhead +0.66, +0.61, +0.53, +0.35, +0.34; total +1.35, +1.36, +1.40, +1.55, +1.66.

- **Both gaps fall as mu falls**, as the plan expected (HB6): the order of the five medians is the same for the paradigm gap and the integration-only ML gap, 0.05 > 0.012 > 1e-3 > 1e-4 > 1e-6.
  The share where a perfect fixed step wins follows it except at the top (23.5 % at Earth-Moon, 24.4 % at 0.05, intervals overlap).
- **The alternative stated in advance is what the data show for the paradigm gap**: it moves with mu through the timescale range (median delta_phys 1.53 to 2.62) and the gap-on-log-delta slope
  stays at 0.86-0.93 from 1e-4 to 0.05 (0.71 at 1e-6). That is the T1 relation again, across systems.
- **The violation rate does not move with mu** (2.7-3.7 %, intervals overlap); the certification fixes it. What moves is the price of safety and the prediction error: the certified scale falls
  from 0.5 to about 0.25-0.3 and the safety-scale term rises from +0.30 to +0.53-0.59 dex, the prediction term from +0.25 to +0.48. Between 1e-6 and 0.05 the mean paradigm term grows by +0.19 dex, prediction by +0.22 and safety scale by +0.23 (selection -0.02), so the three contribute
  comparably; the median integration-only gap grows by +0.84 dex (medians do not add up the way means do).
- **Overhead matters least where the integration is dearest**: its share falls from +0.66 to +0.34 dex as mu rises, so the model's fixed cost is a larger obstacle in the Kepler-like corner.

Limits specific to the dial: (1) the training and calibration sets are 3,000 and 1,000 orbits, so the Earth-Moon member is not comparable to D_explore (certified scales 0.25/0.3 against 0.5/0.5, because
a smaller model and fewer calibration rows both force a more conservative scale); within the dial the sizes are equal. (2) Hill-scaling only controls the neighbourhood of the secondary: at mu = 1e-6 the
median closest approach is 54 Hill radii (1.5 at mu = 0.05), so at small mu most of the sample is primary-dominated and rarely meets the secondary, which by itself lowers the paradigm gap; mu and the share of
orbits that interact with the secondary change together. (3) One seed per mu, bootstrap over orbits only (no model-refit variance). (4) The cross-mu transfer test with dimensionless features (Issue 10) was
not run, so the Part A transfer failure is still confounded. (5) No Sun-Earth system.

## 3. What these change in the Part B account

HB3 (lookahead) is now answered exploratorily: seeing the trajectory buys accuracy but not net cost. The mu half of HB6 is answered in the plan's direction, descriptively: the Kepler-like corner is easier
for both the paradigm and the model, and the ML integration still loses to adaptive at every mu. Still not done: E8 (periodic families, halo/NRHO behaviour) and E12 (2D control).

## Errata (2026-10-26)

Added after a re-read for Part C (plan Track 0). The text above is unchanged; where an item below contradicts it, this section governs. Code and outputs:
`run_part_c_3d.py e3c_paired`, `part_c/results/c0/e3c_paired/`. Same D_explore rows, same model fits, same 3,000 evaluation orbits as the E3c table.

**E1. "Intervals do not overlap" (Section 1, third bullet) is wrong.** The two intervals are [25.4, 28.6] and [27.9, 31.2]; they overlap. The paired comparison on the same 3,000
orbits is the right test: at the Part B certified scales (0.5 for f = 0, 0.6 for f = 0.05) f = 0.05 wins on 885 orbits and f = 0 on 810; 96 orbits are won only with lookahead,
21 only without; exact McNemar p = 1.2e-12. So the share difference (+2.5 points) is real *as measured*; item E2 says what it is a difference of.

**E2. The f = 0.05 row is not risk-matched with f = 0.** It has a larger certified scale (0.6 against 0.5) and a larger realized violation rate (4.4 % against 3.3 %; its upper bound, 5.2 %,
passes 5 %). A larger scale makes every integration cheaper, so part of the "gain" is extra risk. Re-evaluated on the same 3,000 orbits:

| comparison | f = 0 wins | f = 0.05 wins | difference | McNemar p |
|---|---|---|---|---|
| each at its Part B scale (0.5 / 0.6) | 810 | 885 | +2.5 points | 1.2e-12 |
| both at scale 0.5 | 810 | 798 | -0.4 points | 0.27 |
| both at scale 0.6 | 894 | 885 | -0.3 points | 0.45 |

At equal *realized* risk (violation rate and win share interpolated linearly along the scale grid 0.3-0.8, cost on a log scale), the share cheaper than RK45-cal is
24.9 % (f = 0) against 23.0 % (f = 0.05) at 3 %, 27.6 % against 26.0 % at 3.5 %, and 29.4 % against 28.1 % at 4 % (median total cost 1,741 / 1,747, 1,473 / 1,525, 1,291 / 1,347).
Against DOP853-cal the lookahead arm is further behind (e.g. 8.6 % against 15.4 % at 3 %). **Reading: a 5 %-of-horizon lookahead does not move the cost-risk frontier; at equal scale or equal risk it is neutral to
slightly worse, because its 20 evaluations are not repaid.** The earlier statement that the share "rises slightly with a short lookahead" is retracted for the risk-matched comparison; what rose was the risk the
certificate allowed (the more accurate prediction earns a coarser certified scale, 0.5 to 0.6, which is a legitimate use of the certificate but not a saving from the lookahead itself). The scale grid is coarse
(0.4, 0.5, 0.6, 0.8), so part of the 0.5-to-0.6 step may be grid resolution; the matched-scale rows above are the cleaner reading.

**E3. The f = 1 arm is an information bound, not a policy.** Its "lookahead" is a complete RK45 solve at rtol 1e-6, which is already a usable answer for the orbit (302 evaluations
at the median); a deployed policy that had paid for it would stop there. The f = 1 row shows how good the features can get, nothing about what a policy should do.

**E4. "Overhead matters least where the integration is dearest" (Section 2, fourth bullet) is arithmetic, not a finding.** The overhead term is O = log10(1 + C_over / C_int), which must fall as C_int rises.
Read the bullet as a consequence of that definition. What the dial adds is the level of C_int at each mu, not the trend of O.

**E5. The mu trends are confounded; limits (1) and (2) apply to everything in Section 2.** The median closest approach to the secondary is 54 Hill radii at mu = 1e-6 and 1.5 at mu = 0.05, so mu
changes together with how often orbits meet the secondary; and the per-mu models are trained on 3,000 orbits, where E4 says the hard cells are data-starved. The paradigm-gap curve is model-free and is the
more trustworthy of the two (it does not depend on training size); the integration-only ML-gap curve mixes mu with a small-data effect. Part C Track 4.1 (stratified by closest approach in Hill radii, matched
resampling, a model-size control) is the planned remedy. Figure panel 3 now carries its Clopper-Pearson band (figure code only; no number changed).
