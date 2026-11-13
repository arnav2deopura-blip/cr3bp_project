# The Whole Project, Explained: Pre-Run Step-Size Prediction in the Earth–Moon Three-Body Problem

**What this document is.** A complete, plain-language walk-through of the project in `extended-orbit-ML-2`: the physics, the numerical methods, the machine learning, the statistics, the four preregistered studies, every figure and table of the manuscript, every main equation, the literature it builds on, and every code file. It is written the way a computational-physics professor would explain it to a smart non-specialist, so jargon gets explained the first time it appears and then collected in a glossary at the end (Section 20).

**Where the facts come from.** Every number is taken from the project's own files: `manuscript/main.tex` (the PLOS ONE draft), `archive/study_a/RESULTS_REPORT.md` (Study A; short version in `STUDY_A_SUMMARY.md`), `part_b/RESULTS_PART_B.md`, `part_c/RESULTS_PART_C.md`, `part_d/RESULTS_PART_D.md`, the preregistrations, `part_d/NARRATIVE.md`, the post hoc checks in `posthoc/README.md` and `posthoc/results/` (with the second round summarized in `DEFENSE_STATUS.md`), the code under `orbit_ml_cr3bp/`, and the figure PNGs in `manuscript/figures/draft/`. Where I could not verify something, I say so. Section 19 lists problems I found while reading; read it before submitting anything. **Last checked against the project files on 2026-11-06.** The newest project files are from the Nov 3–4 round (git commit `1f8db60`), and this document already covered that round. What the 2026-11-06 check added is listed in Section 19.5 (notes 22–25), plus a few corrections in the text: the number of tests in Table 2 (Section 16), the seal dates (Section 3.4), one sentence in Section 15B.3 that mixed up two ratios, the open decision about Study E (Section 15A.7), the poster files (Section 15B.7) and more questions in Section 21.

**Naming.** The manuscript calls the four studies **A, B, C, D**. The code and result folders call them **Part A** (the top-level `out/`, `data/`, `PREREGISTRATION.md`), **Part B** (`part_b/`, `out_b/`, `data_b/`), **Part C** and **Part D**. They are the same thing.

---

## Contents

1. [The story on one page](#1-the-story-on-one-page)
2. [Why anyone cares: the practical problem](#2-why-anyone-cares-the-practical-problem)
3. [How the project got here: from a 2D two-body study to four 3D studies](#3-how-the-project-got-here)
4. [Physics primer: the circular restricted three-body problem](#4-physics-primer-the-circular-restricted-three-body-problem)
5. [Numerical-integration primer](#5-numerical-integration-primer)
6. [The Sundman time transformation (the key idea of Study C)](#6-the-sundman-time-transformation)
7. [Machine-learning primer: labels, features, models, overhead](#7-machine-learning-primer)
8. [Safety certification: how a failure rate is guaranteed](#8-safety-certification)
9. [Fair comparison: the three ways to tune the adaptive solver, and the cost decomposition](#9-fair-comparison)
10. [Statistics and research-integrity primer](#10-statistics-and-research-integrity-primer)
11. [The datasets](#11-the-datasets)
12. [Study A: the original hypothesis, and the negative result](#12-study-a)
13. [Study B: the diagnosis](#13-study-b)
14. [Study C: the fix](#14-study-c)
15. [Study D: replication, need for ML, wall-clock, conditional safety, transfer](#15-study-d)
15A. [After the seals: post hoc checks and the Study E plan](#15a-after-the-seals-post-hoc-checks-and-the-study-e-plan)
15B. [The second round of checks (Nov 3–4): new comparators and corrected claims](#15b-the-second-round-of-checks-2026-11-0304)
16. [Every figure and table, explained](#16-every-figure-and-table-explained)
17. [Discussion: mechanism, literature, limits, novelty, impact](#17-discussion-mechanism-literature-limits-novelty-impact)
18. [The code, file by file](#18-the-code-file-by-file)
19. [Critical notes: problems and inconsistencies I found](#19-critical-notes)
20. [Glossary](#20-glossary)
21. [Questions a judge or reviewer is likely to ask, with answers](#21-questions-a-judge-or-reviewer-is-likely-to-ask)
22. [The literature, paper by paper](#22-the-literature-paper-by-paper)

---

## 1. The story on one page

**The question.** A computer simulates a spacecraft's path by taking many small time steps. Most professionals use *adaptive* solvers that change their step size as they go: small steps near the Moon, big steps far away. *Fixed-step* solvers use one step size the whole way, which is simpler and has a cost you know in advance. The project asks: **can a model look at a spacecraft's starting position and velocity, plus how long we want to simulate, and pick one step size before the run, so that the run is cheaper than an adaptive solver and still accurate (final position within 100 km of the truth) with a certified failure rate?**

**The starting hypothesis.** Three physical quantities should govern how hard a trajectory is to simulate: how chaotic it is (the finite-time Lyapunov exponent, FTLE), how close it is to the Earth or Moon, and its energy-like conserved quantity (the Jacobi constant).

**Study A (negative result).** No. With all costs counted, the learned fixed-step policy was cheaper than an adaptive solver on **0 of 996** test trajectories. One reason was embarrassing but important: computing the FTLE requires simulating the whole trajectory anyway, and it cost about **70 times** a complete adaptive solve. Removing FTLE did not measurably hurt the model.

**Study B (diagnosis).** Why does it lose? The project split the cost gap into named pieces and found the deep reason: **a single physical step must be small enough for the fastest moment of the trip** (a close lunar flyby), so it wastes effort everywhere else. That waste grows almost one-for-one with how much the trajectory's natural pace varies (a quantity called the *timescale range*, Δτ; correlation 0.91). Still **0 of 14,161** trajectories were cheaper.

**Study C (the fix).** Change the clock. In **Sundman time**, a fictitious time variable runs slower near the bodies, so one constant step in that clock automatically becomes small physical steps near the Moon and large ones far away. The model now predicts one number, the fictitious step, using cheap analytic "when and how close is my next close approach" features (periapsis features). Result: **cheaper than an equal-risk adaptive RK45 solver on 71.4% of trajectories**, with a certified failure rate. (The same features and compiled model *without* the new clock won only 31.6%, so the clock is what turned a minority of wins into a majority. Most of the drop in *cost*, though, came from the cheaper compiled model and the periapsis features; Section 15B.)

**Study D (stress test).** The headline **replicated** on brand-new data (71.6%). A **four-number physics rule** recovers about **three quarters** of the saving, and a linear model ties the tree model, so most of the gain is physics, not fancy ML. But: it does **not** beat the stronger DOP853 solver at equal risk, it is **not** faster in real compiled wall-clock time over the full 1–20 time-unit range (ratio 0.885; it *is* 1.6× faster at short horizons), and it **fails badly on catalogued periodic orbits** it never saw in training (70% failure rate).

**After the studies (post hoc checks, Section 15A).** Extra checks run after all four seals did not change any verdict, but they made the result smaller in two ways. First, **"cheaper on 71% of trajectories" is not "cheaper in total"**: the policy wins the many short, cheap trajectories and loses the few long, expensive ones, so in total it used **19–25% more** force evaluations than equal-risk RK45. It is cheaper in total only at 1–3 TU (RK45 needs 1.48× as many evaluations there). Second, when the adaptive solvers are tuned as carefully as the policy was, the shares drop: to about **two thirds** against RK45, and **below half (44.9%)** against DOP853 on the replication set. A Study E is drafted (not yet sealed or run) to test the short-horizon claim properly on new data.

**A second round of checks (Nov 3–4, Section 15B)** asked the obvious reviewer question: *could an ordinary adaptive solver use the same tricks?* Yes, and it does better. DOP853 run in the same Sundman time needs only **41%** of the policy's evaluations. RK45 whose tolerance is *predicted* by the same small model leaves the policy cheaper on only **34%** of trajectories. The manuscript was retitled and several overstated sentences were corrected.

**The take-home.** What matters is **the choice of time variable plus cheap knowledge of the next close approach**, not a better learning algorithm. Those ideas help an adaptive solver even more than they help a fixed step, so a pre-run fixed step is at best a niche tool (short horizons, a known population, an adaptive fallback). One lead remains open: a higher-order fixed step in Sundman time looked very promising when its step was chosen with perfect knowledge (Section 15B.3).

---

## 2. Why anyone cares: the practical problem

Mission designers for lunar missions (for example NASA's Gateway station orbit, or the CAPSTONE cubesat) integrate trajectories **thousands to millions of times**:

- **Monte Carlo dispersion studies**: take a planned trajectory, perturb the starting state thousands of times to mimic navigation errors, and simulate each one to see how often the mission still works.
- **Grid searches for transfers**: try many starting conditions to find a good path.
- **Orbit-family continuation**: trace a whole family of periodic orbits and test each member's stability.

Each simulation calls the "force function" (the code that computes gravity at a point) hundreds to millions of times. If you could cut that count, and especially if you could know the cost of each run in advance (useful for packing runs onto parallel hardware such as GPUs, where all lanes must wait for the slowest), you would save real computing time.

The **standard tool** is an adaptive solver such as **RK45** or **DOP853** (Section 5.4). These estimate their own error at every step and shrink or grow the step. They are robust, but you do not know their cost in advance, and every step does extra work to estimate the error.

The **appeal of a fixed step** is simplicity, predictable cost, and the option of using special "symplectic" methods that preserve physical structure. Its weakness is that one step must serve the whole trajectory. The project tests, very carefully, whether machine learning can close that gap.

---

## 3. How the project got here

### 3.1 The 2D two-body origin (a separate, earlier project)

The project began as a planar **two-body** study ("A Physics-Informed Random Forest for Selecting Stable Timesteps in Two-Body Orbital Simulations"): one body orbiting a point mass, state $(x, y, v_x, v_y)$, gravitational parameter $GM = 1$, three integrators (Euler, RK4, leapfrog), and "stable" meaning the energy and angular momentum drift stayed below 1%. A random forest predicted a stable timestep for each integrator. That code lives in a different folder and is not part of `extended-orbit-ML-2`.

A peer review (NHSJS) raised ten major problems. The most important ones shaped everything that followed:

| Reviewer criticism | Plain meaning | How the 3D project answers it |
|---|---|---|
| Censored data mishandled | When the search gave up and returned a placeholder ($10^{-4}$), it was treated as a real answer, which distorts the model | Hurdle models, monotone labels with explicit censoring flags (Section 7.1) |
| Cost normalization missing | A bigger step is not cheaper if each step costs more work | Cost counted in force evaluations per run, including model costs (Section 7.5) |
| No held-out violation audit | Good $R^2$ does not prove the predicted step actually works | Every predicted step is actually run and checked; safety scale certified (Section 8) |
| Stability ≠ accuracy | Conserving energy does not mean the path is right (especially leapfrog) | Success redefined as final position within 100 km of a high-accuracy reference |
| Tolerance arbitrary | Results might depend on "1%" | Tolerance fixed at 100 km and its meaning stated; sensitivity reported in Study A |
| Search not validated | Halving search might miss the true answer | Dense-sweep checks of the label (Section 7.1) |
| Biased feature importance | Default random-forest importance is biased | Drop-group refits and permutation importance (Study D, D6) |
| Weak external validation | No known orbits tested | Halo orbits (Study A) and the JPL periodic-orbit catalogue (Study D, D5) |

Project notes record that the patched 2D code was rerun in October 2026 (selection accuracy 94.62% on solvable rows against 87.01% for always-leapfrog; 3.02% of 6,000 test orbits unsolvable). I did not verify those numbers here because that folder is not connected.

### 3.2 The move to 3D and the three-body problem

The two-body problem has a single natural timescale per orbit and no chaos, so it is too easy to say anything about real cislunar (Earth–Moon space) trajectories. The project moved to the **spatial (3D) circular restricted three-body problem** (Section 4), which is the standard model for orbits near the Moon and the Lagrange points, has chaos, and has close flybys that compress time.

### 3.3 Bugs fixed before the preregistered studies

Project notes record a long audit before any confirmatory run, including two critical bugs:

- **Leapfrog advanced $2\,\Delta t$ instead of $\Delta t$ per step**, which invalidated every leapfrog label and forced full regeneration of data.
- **Inconsistent tolerances across integrators** (up to a factor of 1,000), which made "pick the best integrator" meaningless.

Other fixes: the analytic Richardson halo construction was found to be wrong and replaced by numerical differential correction (`richardson.py`), FTLE overflow silently turning into zeros, FTLE zeroed at prediction time but not training time ("train/serve skew"), a binary-search indentation bug, and more. `CHANGES.md` lists "fixes 1–8" (physical collision radii, full cost accounting, monotone labels, keeping crash/escape orbits, stable calibration, independent datasets with a locked test set and preregistration, consistent cross-mass-ratio tests, run housekeeping). These are why the later studies are trustworthy: the cleanup happened *before* the sealed runs.

### 3.4 The four preregistered studies

| Study | Sealed | Question | One-line answer |
|---|---|---|---|
| A | 2026-10-23 | Can ML with FTLE, proximity and Jacobi features beat adaptive solvers on cost? | No: 0/996 |
| B | 2026-10-25 | Why not? Where does the cost go? | A single physical step pays a penalty that tracks the timescale range |
| C | 2026-10-27 | Does integrating in Sundman time with periapsis features fix it? | Per trajectory, yes against RK45 at equal risk: 71.4% (not in total) |
| D | 2026-10-28 | Does it replicate, is ML needed, is it faster in seconds, is it safe where it matters, does it transfer? | Replicates; physics does most of it; not faster overall; conditionally safe; does not transfer |

Each study used **new random seeds**, so no trajectory used to test one study was used to fit or tune another.

**About the seal dates.** The dates in the table are the UTC dates stored in the `sealed_utc` field of each `FREEZE*.json` file, and the manuscript uses the same dates. Studies A, B and D fall on the same day in New York time (the time shown on your git commits). **Study C does not.** It was sealed at 01:37 UTC on Oct 27, which is 21:37 on **Oct 26** in New York, and the git commit that records the seal (`4c3114e`) carries the Oct 26 date. Nothing is wrong, but anyone who compares the manuscript with the git log will see two different days unless you say "UTC" when you quote the dates.

---

## 4. Physics primer: the circular restricted three-body problem

### 4.1 The setup in words

- Two heavy bodies, the **Earth** (the "primary", body 1) and the **Moon** (the "secondary", body 2), orbit their common centre of mass (the **barycentre**) in perfect circles. ("Circular.")
- A third body, the **spacecraft**, has negligible mass: it is pulled by Earth and Moon but does not pull back on them. ("Restricted.")
- The spacecraft can move in all three dimensions. ("Spatial" or 3D.)

The real Moon's orbit is slightly elliptical and the Sun also pulls, so this is an idealized model. Its advantage is that it captures the essential dynamics near the Moon while staying simple enough to study cleanly.

### 4.2 Units (nondimensionalization)

To avoid huge numbers, everything is rescaled:

- **Distance unit (DU)** = Earth–Moon distance = **384,400 km**.
- **Time unit (TU)** = $1/n$ where $n$ is the Moon's mean angular speed. 1 TU ≈ **4.348 days** (3.757 × 10⁵ s). One Moon orbit takes $2\pi$ TU ≈ 27.3 days.
- **Mass ratio** $\mu = \dfrac{m_{\text{Moon}}}{m_{\text{Earth}} + m_{\text{Moon}}} = 0.01215058535$. The Moon has about 1.2% of the system's mass.
- In these units gravity's constant times the total mass is 1.

Useful conversions: the 100 km tolerance is $100/384{,}400 = 2.6015 \times 10^{-4}$ DU. Horizons of 1–20 TU are about 4–87 days.

### 4.3 The rotating frame

Instead of watching from a fixed ("inertial") viewpoint, we sit on a frame that **rotates with the Earth and Moon**, so both look frozen: the Earth sits at $(-\mu, 0, 0)$ and the Moon at $(1-\mu, 0, 0)$. The price of rotating with them is two "fictitious" forces, the same ones you feel on a merry-go-round:

- **Centrifugal force**: pushes outward from the rotation axis.
- **Coriolis force**: deflects anything moving sideways relative to its motion.

### 4.4 Equations of motion (manuscript Eq 1–2)

$$
\ddot{x} - 2\dot{y} = \frac{\partial \Omega}{\partial x}, \qquad
\ddot{y} + 2\dot{x} = \frac{\partial \Omega}{\partial y}, \qquad
\ddot{z} = \frac{\partial \Omega}{\partial z},
$$

$$
\Omega(x,y,z) = \tfrac{1}{2}(x^2+y^2) + \frac{1-\mu}{r_1} + \frac{\mu}{r_2},
$$

with $r_1 = \sqrt{(x+\mu)^2 + y^2 + z^2}$ (distance to Earth) and $r_2 = \sqrt{(x-1+\mu)^2 + y^2 + z^2}$ (distance to Moon).

Reading it term by term:

- $\ddot{x}$ is acceleration (dots mean time derivatives: one dot = velocity, two dots = acceleration).
- The $\mp 2\dot{y}$, $\pm 2\dot{x}$ terms are the **Coriolis** force.
- $\Omega$ is an **effective potential**, a "landscape" whose slope gives the force. Its three parts: $\frac12(x^2+y^2)$ is the **centrifugal** part, $\frac{1-\mu}{r_1}$ is **Earth's gravity**, $\frac{\mu}{r_2}$ is **Moon's gravity**.
- The $z$ equation has no Coriolis or centrifugal term because the rotation is about the $z$ axis.

This system is a set of **ordinary differential equations (ODEs)**: given the six numbers (position and velocity) now, they tell you how those numbers change. The computer's job, called **numerical integration**, is to march these forward in time. The code that computes the right-hand side is the **RHS function** or **force function**; one call is one **RHS evaluation**. This is the project's unit of cost.

Note the $1/r^2$ forces: when $r_2$ is small (close to the Moon), forces become huge and change rapidly. That is the root of all the numerical difficulty.

### 4.5 The Jacobi constant (manuscript Eq 3)

$$
C = 2\Omega - (\dot{x}^2 + \dot{y}^2 + \dot{z}^2)
$$

This combination stays **exactly constant** along any true trajectory (it is the rotating-frame version of energy, with a sign flip and factor of 2). It is the only conserved quantity of this problem.

Why it matters:
- **Higher $C$ = lower energy.** Since speed² $\ge 0$, motion is only possible where $2\Omega \ge C$. Regions where $2\Omega < C$ are **forbidden**; their boundaries are called **zero-velocity surfaces**.
- A numerical method that lets $C$ drift is making errors, but (crucially) **a method can conserve $C$ well and still put the spacecraft in the wrong place**. That is why the project judges accuracy by position against a reference, not by $C$ drift (a lesson from the 2D review).

### 4.6 Lagrange (libration) points and Hill regimes

There are five points in the rotating frame where all forces balance: **L1** (between Earth and Moon), **L2** (beyond the Moon), **L3** (opposite side of Earth), **L4** and **L5** (60° ahead/behind the Moon). Each has a Jacobi value:

$C_{L_1} = 3.1883,\; C_{L_2} = 3.1722,\; C_{L_3} = 3.0121,\; C_{L_4} = C_{L_5} = 2.9880.$

As energy rises ($C$ falls), "necks" open at these points and the spacecraft can travel between regions. The project sorts every starting state into one of five **Hill regimes**:

| Regime | Condition | What it means physically |
|---|---|---|
| I | $C > C_{L_1}$ | Necks closed. Trapped near the Earth or near the Moon. |
| II | $C_{L_2} < C \le C_{L_1}$ | L1 neck open: can move between Earth and Moon regions, cannot leave. |
| III | $C_{L_3} < C \le C_{L_2}$ | L2 neck open too: can leave through the far side of the Moon. |
| IV | $C_{L_4} < C \le C_{L_3}$ | L3 side opens. |
| V | $C \le C_{L_4}$ | No forbidden region anywhere. |

Regimes II–III are where trajectories thread the narrow necks near L1/L2, which is where much interesting (and chaotic) motion happens.

### 4.7 Collisions and escapes

A run "ends as a collision" if the spacecraft comes within the physical radius of the Earth (6,371 km) or the Moon (1,737.4 km), and as an "escape" if it goes beyond 10 DU from the barycentre. About **19%** of sampled starting states have a reference trajectory that collides or escapes before the horizon ("terminating" or "doomed" orbits; the code says "reference-divergent"). They were excluded from the cost tests (there is no fair "final position" to compare) but analysed separately in Study C.

### 4.8 The local timescale and the timescale range (manuscript Eq 4): the single most important concept

Near a body of mass $GM$ at distance $r$, the natural time for things to change is about $\sqrt{r^3/GM}$ (this is Kepler's third law: orbital period grows like $r^{3/2}$). The project defines the **local dynamical timescale**:

$$
\tau(t) = \min\left(\sqrt{\frac{r_1^3}{1-\mu}},\; \sqrt{\frac{r_2^3}{\mu}}\right)
$$

the shorter of the "Earth clock" and "Moon clock" at the spacecraft's current position. Small $\tau$ means things are happening fast (close to a body).

Then the **timescale range** of a whole trajectory of length $T$:

$$
\Delta_\tau = \frac{T \cdot \max_t \tau^{-1}(t)}{\int_0^T \tau^{-1}(t)\, dt}
$$

**Analogy: a road trip with one school zone.** Suppose you must drive at one constant speed for the whole trip, and it must be safe in the school zone. Then you crawl the entire way at school-zone speed. A driver who can change speed (adaptive) crawls only in the school zone. The ratio of your trip time to theirs is exactly the kind of number $\Delta_\tau$ measures:

- Numerator $T \max \tau^{-1}$: how many "beats" you'd need if every moment were as fast as the fastest moment (what a fixed step pays).
- Denominator $\int \tau^{-1} dt$: the total number of beats actually needed if you pace yourself (what an adaptive solver pays, roughly).

$\Delta_\tau = 1$ means uniformly paced. $\Delta_\tau = 12.5$ (the Fig 1 example) means a fixed step does about 12.5 times more work than necessary. $\Delta_\tau$ needs the whole trajectory, so it is only used to *explain* results, never as a model input.

### 4.9 Chaos and the finite-time Lyapunov exponent (FTLE)

**Chaos** means tiny differences in the starting state grow exponentially. Two spacecraft starting 1 metre apart might be 1,000 km apart a few weeks later. For a numerical method, this means small per-step errors also grow exponentially, so chaotic trajectories need more accuracy per step.

The **FTLE** $\lambda$ measures the growth rate over a finite time $T$:

$$
\lambda = \frac{1}{T} \ln \sqrt{\Lambda_{\max}(\Phi^{\mathsf T}\Phi)}
$$

- $\Phi$ is the **state-transition matrix (STM)**: a 6×6 table saying "if I nudge the start by a small vector $\delta$, the end moves by $\Phi\delta$." It is computed by integrating extra "variational equations" alongside the trajectory.
- $\Lambda_{\max}(\Phi^{\mathsf T}\Phi)$ is the largest stretching factor squared, so $\sqrt{\cdot}$ is the largest stretch, and $\ln(\cdot)/T$ turns it into a rate per time unit.
- $\lambda T$ (FTLE × horizon) is roughly "how many factors of $e$ an error is amplified." The project's high-chaos tail is $\lambda T \ge 10.8$, i.e. errors amplified by $e^{10.8} \approx 50{,}000$ or more.

The fatal problem for a *pre-run* predictor: **computing FTLE requires integrating the whole trajectory plus 36 extra equations.** It cannot be a cheap input to a method meant to avoid integrating.

### 4.10 Osculating two-body orbits and periapsis features (manuscript Eq 6)

If you ignore everything except one body, a spacecraft follows a **conic section** (ellipse, parabola or hyperbola): Kepler's two-body solution. At any instant, you can compute the two-body orbit that "kisses" (osculates) the current motion around the Earth, and separately around the Moon. That orbit predicts:

- the **periapsis radius** $r_p$: how close the spacecraft will get to that body;
- the **time to periapsis** $t_p$: when that closest point occurs.

The math, step by step (for body $i$ with $GM_1 = 1-\mu$ or $GM_2 = \mu$):

1. Position relative to the body $\boldsymbol{r}$; **inertial** velocity $\boldsymbol{v} = \dot{\boldsymbol{r}}_{\text{rot}} + \hat{\boldsymbol{z}}\times\boldsymbol{r}$ (undo the frame rotation; in components $(v_x - r_y,\ v_y + r_x,\ v_z)$).
2. Specific angular momentum $\boldsymbol{h} = \boldsymbol{r}\times\boldsymbol{v}$.
3. **Eccentricity vector** $\boldsymbol{e} = \boldsymbol{v}\times\boldsymbol{h}/GM - \boldsymbol{r}/|\boldsymbol{r}|$, $e = |\boldsymbol{e}|$. ($e = 0$ circle, $0<e<1$ ellipse, $e>1$ hyperbola.)
4. Periapsis radius $r_p = |\boldsymbol{h}|^2 / (GM(1+e))$.
5. For an ellipse with semi-major axis $a$ and mean motion $n = \sqrt{GM/a^3}$: the **eccentric anomaly** $E$ comes from $e\cos E = 1 - |\boldsymbol{r}|/a$ and $e\sin E = \boldsymbol{r}\cdot\boldsymbol{v}/\sqrt{GM a}$; the **mean anomaly** $M = E - e\sin E$ (Kepler's equation) measures "where along the orbit in time" you are; then $t_p = -M/n$ if $M<0$ (approaching periapsis), else $(2\pi - M)/n$ (wait for the next one).
6. Hyperbolas use the hyperbolic anomaly; an outgoing hyperbola never comes back (no future periapsis). Near-circular and near-parabolic cases are handled specially.

From these the project builds **13 periapsis features**: for each body, $\log_{10} r_p$, $\log_{10}\sqrt{r_p^3/GM}$ (the timescale at periapsis), $\min(t_p/T, 2)$, an indicator "periapsis happens before $T$", the log of the current distance (relative to the Moon's Hill radius $(\mu/3)^{1/3}$ for the Moon), and $e$; that is 6 per body, 12 total, plus a 13th: the log of the shortest periapsis timescale over both bodies.

**Why this is clever.** These features cost a few arithmetic operations (no RHS evaluation, no integration), yet they answer the question the model most needs answered: **"Will I pass close to the Moon or Earth before the end of the run, and how close?"** They are only exact for pure two-body motion, so in the real three-body problem they are approximate, and they miss some close passes (this matters in Study D).

---

## 5. Numerical-integration primer

### 5.1 What an integrator does

An integrator turns "how things change now" (the RHS) into "where things will be later" by taking steps of size $\Delta t$ (or $h$, or $dt$). Smaller steps mean more accuracy and more cost. A method of **order $p$** has error that shrinks like $\Delta t^p$: halve the step, and a 4th-order method's error drops 16-fold.

**Cost in this project** = number of RHS evaluations. A fixed-step method's cost is simply

$$\text{cost} = e \times \lceil T/\Delta t \rceil,$$

where $e$ is evaluations per step and $\lceil\cdot\rceil$ means round up. Example: RK4 ($e = 4$) over $T = 5$ TU with $\Delta t = 0.01$ costs $4 \times 500 = 2{,}000$ evaluations. Because this is known before the run, fixed-step cost is predictable.

### 5.2 The fixed-step methods

- **Euler** (1st order, 1 evaluation/step): take the current slope and walk straight. Very inaccurate. Used only as a negative control in Study A ($R^2 \approx 0.018$: unpredictable).
- **Classical RK4** (4th order, 4 evaluations/step): samples the slope at four points within the step and averages them cleverly. The workhorse of the project.
- **Leapfrog** (2nd order; Studies A and B): a **symplectic** method (next subsection). Implemented as "kick–drift–kick": half a gravity kick, then the exact motion of the free-particle-plus-rotation part, then another half kick. Because the second kick's acceleration is reused as the next step's first kick ("first same as last", FSAL), each step needs only one new evaluation. (Study A's sealed code charged 2 per step; see Section 19.)

### 5.3 Symplectic integration, Hamiltonians and canonical momenta (why leapfrog is special)

Physics can be written in **Hamiltonian** form: a function $H$ (energy) of positions $\boldsymbol{q}$ and **canonical momenta** $\boldsymbol{p}$. In the rotating frame:

$$H = \tfrac12|\boldsymbol{p}|^2 - (x p_y - y p_x) + V(\boldsymbol{q}),\quad \boldsymbol{p} = (\dot x - y,\ \dot y + x,\ \dot z),\quad V = -\frac{1-\mu}{r_1} - \frac{\mu}{r_2}.$$

Note the canonical momenta are *not* the ordinary velocities; they include the rotation. The term $-(x p_y - y p_x)$ is the rotation's contribution. Because $H$ mixes $p$ and $q$ in that term, it is "non-separable", which makes standard leapfrog not directly applicable; the project's version splits $H$ into the gravity part $V$ (a "kick") and everything else, whose exact solution is "fly straight, then rotate the frame." This is a correct symplectic splitting.

**Symplectic** means the method preserves a geometric property of Hamiltonian flow (phase-space area/volume structure). Practically, symplectic methods do not let energy drift systematically over long times. The project verified it numerically: the check $\Phi^{\mathsf T} J \Phi - J = 0$ (the defining property, with $J$ the standard symplectic matrix) held to $1.1\times10^{-10}$ on 100 states at every step size, which is the finite-difference noise floor, while RK4's defect was $5\times10^{-5}$ at $\Delta t = 0.2$. So leapfrog is truly symplectic, not merely time-symmetric.

Leapfrog was dropped in Study C because, once the safety certificate covered the whole policy (both integrators together), allowing leapfrog cost more than it saved. A Sundman-transformed leapfrog is not symplectic, so there is no symplectic version in Sundman time.

### 5.4 Adaptive solvers: RK45, DOP853, tolerances and controllers

**Embedded Runge–Kutta pairs** compute two answers of different order from the same evaluations; their difference estimates the error. If the estimate is too big, reject and retry with a smaller step; if small, accept and grow the next step.

- **RK45** = Dormand–Prince 5(4) (1980): 5th-order answer with a 4th-order error estimate. SciPy's default.
- **DOP853**: 8th order with 5th- and 3rd-order error estimates. Higher order means it can take much bigger steps for the same accuracy. A tougher competitor.
- **Tolerance**: you tell the solver an acceptable local error, `rtol` (relative) and `atol` (absolute). Tighter tolerance = more accurate = more expensive. The project set `atol = 10⁻³ × rtol`. A solve needing more than 2,000,000 evaluations was abandoned. (A post hoc check, Section 15A, found that `atol = 10⁻¹ × rtol` makes RK45 cheaper at the same risk, so the sealed setting slightly handicapped the solver.)
- **Step controller**: the rule for choosing the next step from the error estimate. SciPy uses a simple "integral" (I) controller. The **PI controller** of Gustafsson and of Hairer–Wanner (β = 0.04) also uses the previous error and is smoother; it was the primary comparator in the wall-clock test.

The key point: **tolerance is local per-step error, not final position error.** To get within 100 km at the end you have to guess an `rtol`, and the right value differs per trajectory. Section 9 explains the three ways the project chose it, which is central to a fair comparison.

### 5.5 Reference solutions ("the truth")

Every trajectory's "true" path is **DOP853 at rtol = atol = 10⁻¹³, maximum step 0.01 TU**, with dense output (a continuous interpolant so you can read the position at any time). Checks:

- **Radau IIA** (a different, implicit method) at 10⁻¹² agreed to at most **0.003 km** on 600 trajectories; collision/escape event types agreed on all 300 tested.
- Every trajectory had a second DOP853 reference (10⁻¹², max step 0.005) as a per-trajectory flag. Checked afterwards (Section 15B), the two references agreed to at most **0.05 km** on the Study C confirmation set and **0.25 km** on every Study C and D set: thousands of times below the 100 km tolerance.
- The 1% most chaotic trajectories (163 with $\lambda T \ge 10.8$) were redone at the tightest tolerance SciPy accepts (step ≤ 0.002) and with LSODA: largest disagreements **0.007 km and 0.23 km**. So the 100 km decisions are not reference artefacts.

### 5.6 Model error versus integrator error (CAPSTONE check)

The CR3BP is not the real Earth–Moon system. The project downloaded the real CAPSTONE spacecraft trajectory from JPL Horizons and propagated the CR3BP from observed states: the CR3BP drifted from reality by a median **2,066 km** within 0.25 TU and **50,046 km** at 3–5 TU. So the 100 km tolerance measures **integrator error inside the model**, not real-world accuracy. The manuscript says this clearly, which is correct and important.

### 5.7 IAS15 (a famous adaptive integrator)

**IAS15** (Rein & Spiegel 2015) is a 15th-order adaptive Gauss–Radau integrator in the REBOUND code, designed to be accurate to machine precision. In an exploratory test on 300 trajectories at the loosest setting meeting 100 km, it took a median **217 µs** per trajectory against **6.4 µs** for the compiled policy (about 29× longer in total). This is not a fair "beat IAS15" claim: IAS15 is designed for far tighter accuracy, the time includes Python overhead, and REBOUND does not report evaluation counts. It answers the reviewer objection "did you compare with IAS15?" honestly, but because IAS15 was run at the very edge of its settings it is not a fair head-to-head, so the manuscript now reports it only in S1 Appendix and draws no conclusion from it.

---

## 6. The Sundman time transformation

### 6.1 The idea in one sentence

Instead of stepping in ordinary time $t$, step in a **fictitious time $s$** that runs *slowly* near the Earth and Moon, so that one constant step in $s$ automatically becomes a small physical step where things happen fast and a large one where they don't.

**Analogy.** Imagine measuring a car trip not in seconds but in "units of attention." In the school zone, one unit of attention lasts a fraction of a second; on the empty highway it lasts a minute. If you make one decision per unit of attention, you automatically decide often in the school zone and rarely on the highway, without ever having to change your rule.

### 6.2 History (why it is called "Sundman")

- **Karl Sundman (1913)** used time transformations $dt = r\,ds$ in his famous convergent-series solution of the three-body problem, to tame collisions.
- **Levi-Civita (1920)** (2D) and **Kustaanheimo–Stiefel (1965)** (3D, "KS") combined time transformations with coordinate changes that fully remove ("regularize") the collision singularity.
- **Generalized Sundman** transformations $dt = c\, r^\alpha ds$ are used in orbit propagation (Berry & Healy 2002); time-transformed leapfrog and "algorithmic regularization" (Mikkola & Tanikawa 1999; Preto & Tremaine 1999; Wang & Nitadori 2020) apply the idea to few-body problems.

So the *idea* is a century old. The project's contribution is to use it as the target of a pre-run prediction and to test it rigorously against fair baselines.

### 6.3 The project's transformation (manuscript Eq 5)

$$
\frac{dt}{ds} = g(\boldsymbol{X}),\qquad \frac{d\boldsymbol{X}}{ds} = g(\boldsymbol{X})\,\boldsymbol{F}(\boldsymbol{X}),\qquad
g = \left(\frac{1}{\tau_1} + \frac{1}{\tau_2} + \frac{1}{\tau_c}\right)^{-1}
$$

- $\boldsymbol{X}$ is the 6-number state; $\boldsymbol{F}$ is the usual RHS.
- $\tau_1 = \sqrt{r_1^3/(1-\mu)}$, $\tau_2 = \sqrt{r_2^3/\mu}$: the Earth and Moon timescales.
- $\tau_c = 2$ TU: a **cap**, so that far from both bodies $g$ does not blow up (otherwise one fictitious step could jump an enormous physical time).
- $g$ is a **harmonic sum** ("add the rates"): it is close to the *smallest* of the three timescales but, unlike `min()`, it is smooth. Smoothness matters because RK4 keeps its 4th-order accuracy only for smooth equations (tests confirmed 4th-order convergence).
- Time itself becomes a 7th variable: the integrator solves for $(\boldsymbol{X}, t)$ together, with RK4 and constant step $\Delta s$. Each step costs **4 RHS evaluations**, and the physical step is $g\,\Delta s$.
- Close to one body, $g \approx \tau_i \propto r_i^{3/2}$: the generalized Sundman transformation with $\alpha = 3/2$.

What it is **not**: it is not a regularization (it doesn't remove the $1/r^2$ singularity, it just slows down near it), and it is not symplectic.

**The cap was chosen** from {0.5, 1, 2} on Study C pilot data; Study D's exploratory sweep found nearly identical results for $\tau_c \in \{2, 4, 8\}$ and slightly worse at 1 (the oracle Sundman step beat oracle RK45 on 47.5%, 51.6%, 53.2%, 54.0% for $\tau_c$ = 1, 2, 4, 8). So 2 sits at the knee.

### 6.4 Landing exactly at the end time

Because the physical step varies, the run won't naturally land exactly on $T$. Rule: before each step, predict the physical step $g(\boldsymbol{X}_n)\Delta s$. If $t_n + 1.5\, g\Delta s \ge T$, or if a step would overshoot $T$ (that step is thrown away but its evaluations are still counted, honestly), finish with $\lceil (T - t_n)/(g\Delta s)\rceil$ equal physical-time RK4 steps. These "landing steps" are counted in the cost.

### 6.5 Why it should work (and Fig 1 shows it)

A fixed physical step must fit the narrowest dip in $\tau(t)$. A fixed Sundman step makes the physical step *follow* $\tau(t)$, so the dip no longer dictates the whole run. Fig 1B shows exactly this: the blue Sundman physical step tracks the black timescale curve and only touches the dashed "largest passing fixed physical step" line at the moment of the flyby.

---

## 7. Machine-learning primer

### 7.1 The label: "the largest step that works"

To train a model you need a correct answer ("label") for each example. Here the label is: **the largest fixed step (or fictitious step) for which the final position is within 100 km of the reference.**

The difficulty: final error is **not always monotone** in step size. Sometimes a step of 0.1 fails, 0.09 passes, 0.08 fails again (errors can cancel by luck). So the project uses a careful **locally monotone** definition:

1. Build a geometric grid of steps with ratio $2^{1/4}$ (each step is 19% smaller than the previous), starting from the ceiling $\min(T, 2\ \text{TU})$ (for Sundman: $\Delta s \in [10^{-4}, 4]$).
2. The label is the **largest grid value that passes together with its eight finer neighbours** (two octaves, i.e. down to a quarter of its size). This rules out "lucky" isolated passes.
3. Refine with up to six points towards the next coarser value, as long as every finer refinement point still passes.
4. If the very largest grid value passes, the label is **censored**: the true answer is "at least this big" (a lower bound). Censored labels are flagged rather than treated as exact (one of the reviewers' main points).

**How often is the label non-monotone?** 1.5% (3/200) for physical RK4 in Study A; 4.0% (20/500) for the Sundman integrator in Study D, concentrated at short horizons and low chaos, not near collisions.

The labels are also **quantized** at 0.0108 dex (because of the grid), and stable: perturbing 443 starting states by 3.8 km and 1 cm/s changed the label by at most 0.14 dex (and almost always 0).

### 7.2 Features (model inputs)

| Group | Features | Available before the run? |
|---|---|---|
| State | $x, y, z, v_x, v_y, v_z$ | yes |
| Horizon | $T$ | yes |
| Jacobi | $C$, $C - C_{L_1}$ | yes (a formula) |
| Proximity | $r_1$, $r_2$, $r_1/r_2$ | yes |
| Periapsis (Study C on) | the 13 osculating-orbit features of Section 4.10 | yes (a few arithmetic operations) |
| FTLE (Study A only) | $\lambda$ over the horizon | **no**: needs the integration |

So the "12 state features" are state (6) + horizon (1) + Jacobi (2) + proximity (3). Study C's model uses 12 + 13 = 25 features.

### 7.3 The model types

- **Random forest**: many decision trees, each trained on a random subset of data, whose votes are averaged. A decision tree is a flowchart of yes/no questions ("is $r_2 < 0.05$?").
- **Hurdle model** (Study A, B): two parts per integrator: a **classifier** answering "does any step work at all?" and a **regressor** answering "if so, how big?". This avoids mixing "impossible" cases with real labels (a reviewer point).
- **Gradient-boosted trees** (Friedman 2001): trees are added one at a time, each correcting the previous trees' mistakes. scikit-learn's *histogram* version bins the features for speed. The final Study C policy uses **25 trees of depth 3, learning rate 0.2**.
- **Quantile (pinball) loss** (Koenker & Bassett 1978): normally a regressor aims at the average. With quantile loss at level $q$, it aims at the $q$-th percentile: at $q = 0.2$, the prediction is meant to be *below* the true label 80% of the time. **Why aim low on purpose?** Because too big a step fails (dangerous) while too small a step only wastes a bit of work. The target is $\log_{10}$ of the label because labels span orders of magnitude.
- **Linear quantile regression** (Study D): a straight-line formula $\log_{10}\Delta s = b + \boldsymbol{w}\cdot\boldsymbol{f}$ fitted with the same quantile loss using the HiGHS optimization solver.

### 7.4 Model and feature selection (Study C)

Model class, size and quantile were picked on a separate **selection set** by the **median total cost of trajectories that met the tolerance**, excluding any candidate with more than 6% violations. Ties within 2% went to the simpler model. This is selecting for what you actually care about (cheap and safe), not for a proxy such as $R^2$.

### 7.5 Overhead: charging the model for its own cost

A pre-run model is not free: computing features and running the trees takes time. The project converts that time into **RHS-equivalents**: wall time per call divided by the wall time of one compiled RHS evaluation, measured on an idle machine, and adds it to every trajectory's cost. Values pinned in advance:

| Model | Overhead (RHS-equivalents) |
|---|---|
| Study A without FTLE (scikit-learn) | ≈ 591 |
| Study B scikit-learn hurdle model | 1,128 |
| Study C compiled 25-tree Sundman policy | 15.04 (of which 3.6 for features) |
| Study C physical-time policy (10 trees) | 11.74 |
| Study D re-measured 25-tree model (faster machine) | 16.09 |
| Study D physics rules | 2.05–7.43 |
| Study D linear 33-feature model | 11.39 |

The drop from 1,128 to 15 came from **compiling the trees**: `compiled_model_3d.py` exports the trained trees into flat arrays and evaluates them with Numba (a just-in-time compiler for Python), and a test confirms the compiled predictions equal scikit-learn's to 10⁻¹². For scale: the median oracle-tuned RK45 solve costs about **146** evaluations, so a 1,128-evaluation model can never win.

---

## 8. Safety certification

### 8.1 The safety scale

The model's predicted step is multiplied by a **safety scale** $s \le 1$ (making it smaller and safer). The question is how to pick $s$ so the failure rate is guaranteed below a target. Too small an $s$ is safe but expensive.

### 8.2 The certificate (manuscript Eq 8)

Take $n$ **calibration trajectories** (not used for training). For a candidate scale $s_k$, actually run all $n$ with the scaled step and count violations $V(s_k)$. Test the claim "the true violation rate exceeds $\alpha$" with an exact binomial p-value:

$$
p(s_k) = \Pr\left[\text{Bin}(n,\alpha) \le V(s_k)\right]
$$

In words: *if* the true failure rate were $\alpha$ (= 5%), how likely is it to see this few failures or fewer? If that probability is tiny (≤ δ = 0.1), we are confident the true rate is below 5%.

**Fixed-sequence testing** (Studies C, D): order the grid from safest (smallest) to riskiest (largest) scale, $s_k = 0.1\cdot 2^{(k-1)/8}$ for $k = 1..27$ plus $s_{28} = 1$. Go up the list while every test passes; stop at the first failure. The chosen scale

$$
\hat{s} = \max\{s_k : p(s_j)\le\delta \text{ for all } j \le k\}.
$$

Because you stop at the first failure, you don't pay a multiple-testing penalty. This is the **Learn-then-Test (LTT)** framework of Angelopoulos, Bates, Candès, Jordan and Lei. **Guarantee:** if calibration and test trajectories come from the same distribution ("exchangeable"), the failure rate at $\hat s$ exceeds 5% with probability at most 10%. In plain terms: *"we are 90% confident the failure rate is at most 5%."* One catch: calibration and testing used only trajectories whose reference survives to the end time $T$ (no crash, no escape). A user can't know that before the run, so the guarantee is really *"at most 5% failures among trajectories that don't crash or escape."*

**Bonferroni version** (Studies A, B): test each of 18 grid values on its own at level δ/18 and keep the largest that passes. Valid without ordering, but less powerful (needs more evidence per value).

**What counts as a violation** during calibration: final error above 100 km, a collision, an escape, a non-finite number, hitting the 4,000,000-step cap, or being handed to a fallback solver that then fails. The certificate covers **the whole deployed pipeline end to end**, not just the model.

### 8.3 Marginal versus conditional guarantees (a subtle and important point)

The guarantee is **marginal**: it is about the average over the whole population. It says nothing about a sub-group. A policy can be 4% overall but 13% for trajectories that fly close to the Moon. Study D's **conditional certificate (D4)** fixes this by splitting trajectories into **strata** using only information known before the run:

- regime group: I, II–III, IV–V;
- lunar class from the osculating orbit: predicted collision, periapsis < 0.05 DU reached before $T$, periapsis 0.05–0.17 DU, or far;
- horizon band: 1–3, 3–8, 8–20 TU.

Cells with fewer than 400 calibration trajectories were merged by a fixed rule, giving **17 strata** (each with ≥ 411 calibration trajectories). Each stratum certifies its own scale at level δ/17 (a Bonferroni split of the error budget). A stratum that can't be certified would be handed to RK45 (none needed it).

### 8.4 Endpoint-only caveat

Labels and certificates judge **only the final position**. A run might wander more than 100 km off in the middle and come back. An audit of 1,000 re-run trajectories at 50 checkpoints found this happened on **2.6%** (26/1,000).

---

## 9. Fair comparison

### 9.1 The three ways to tune the adaptive solver

This is the heart of honest benchmarking. The adaptive solver needs a tolerance. Which one?

| Name | How it is set | Realistic? | Role |
|---|---|---|---|
| **Oracle-tuned** ("spec") | For each trajectory separately, the loosest rtol (to 0.02 decade) whose final error is within 100 km, found by bisection *using the true answer* | No: needs the answer | A **lower bound** on adaptive cost; the hardest possible opponent |
| **Certified** ("cal") | One rtol for all trajectories, certified on calibration rows by the same binomial rule (18 values 10⁻¹²–10⁻³) | Yes | What a careful practitioner would set |
| **Equal-risk** ("eqr") | The loosest single rtol whose calibration failure rate is no higher than the policy's own; recomputed for every policy | Yes | The **headline** comparator: same risk, compare cost |

**Why equal-risk is the fair headline.** Comparing a policy that fails 4.4% of the time against a solver that fails 1% would not be fair (the safer one should cost more). Equal-risk tunes RK45 to fail exactly as often, then asks who is cheaper. Its weakness (shown in D4) is that "equal on average" can still be very unequal in sub-groups.

A trajectory counts as a "win" for the policy only if the policy **also met the tolerance** on it.

**Share versus total.** The preregistered statistic is the **share** of trajectories on which the policy wins. That answers "if I draw one trajectory, how likely am I to save?", but not "how much does a whole batch cost?". A batch's cost is dominated by its few longest, most expensive trajectories. Section 15A adds the **ratio of totals** (all the solver's evaluations divided by all the policy's), which answers the second question.

**Is the comparison symmetric?** The policy's safety scale was certified by fixed-sequence testing on a fine grid, while the solvers' certified tolerance was chosen by the weaker Bonferroni rule on a coarse grid (Section 8.2). Section 15A re-certifies the solvers by the policy's own rule.

### 9.2 "Dex"

Cost ratios are given in **dex**, a log-10 unit: a difference of 1 dex is a factor of 10; 0.1 dex is a factor of $10^{0.1} \approx 1.26$ (26%); 0.3 dex ≈ factor 2. Logs are used because costs vary over orders of magnitude, and because logs turn multiplied factors into added terms (next subsection).

### 9.3 The cost-gap decomposition (manuscript Eq 7): where every bit of cost goes

For the physical-time policy, the log of (policy cost / adaptive cost) splits **exactly** into named terms:

$$
\log_{10}\frac{C_{\text{tot}}}{C_{\text{ad}}} = \underbrace{P}_{\text{paradigm}} + \underbrace{S}_{\text{selection}} + \underbrace{Q}_{\text{prediction}} + \underbrace{K}_{\text{safety scale}} + \underbrace{O}_{\text{overhead}} + \underbrace{R}_{\text{rounding}}
$$

| Term | Formula | Plain meaning | Analogy (buying a train ticket) |
|---|---|---|---|
| $P$ paradigm | $\log_{10}(C^*_{\text{fix}}/C_{\text{ad}})$, $C^*_{\text{fix}} = \min_k e_k\lceil T/d^*_k\rceil$ | Even with **perfect knowledge** of the best fixed step, how much worse is "any single fixed step" than adaptive? | The cheapest flat-rate ticket vs pay-per-ride |
| $S$ selection | $\log_{10}\frac{e_k/d^*_k}{e_{k^*}/d^*_{k^*}}$ | Cost of choosing the wrong integrator (RK4 vs leapfrog) | Picked the wrong ticket type |
| $Q$ prediction | $\log_{10}(d^*_k/\hat d_k)$ | Cost of predicting a step smaller than the best one (positive) or the danger of a bigger one (negative) | Guessed your trips wrong |
| $K$ safety | $\log_{10}(1/s_k)$ | Cost of shrinking the step for safety | Insurance premium |
| $O$ overhead | $\log_{10}(1 + (C_{\text{feat}}+C_{\text{pred}})/C_{\text{int}})$ | Cost of computing features and running the model | Fee for the advisor who picked your ticket |
| $R$ rounding | the rest | From rounding the number of steps up | Coins |

Here $d^*_k$ is the label (best step) for integrator $k$, $\hat d_k$ the predicted step, $e_k$ evaluations per step, $k$ the chosen integrator and $k^*$ the cheapest one at its label. For the Sundman policy, $S$ disappears (one integrator) and $P$ becomes $P_s = \log_{10}(C^*_s/C_{\text{ad}})$ with $C^*_s$ the cost at the oracle fictitious step. **Means** of the terms add up to the mean total; medians do not (logs of medians don't add), which is why the figures show means.

The decomposition is what makes this project diagnostic rather than just "it lost": it tells you **which** term to attack. Study C attacked the two biggest (paradigm and prediction) and the overhead.

---

## 10. Statistics and research-integrity primer

### 10.1 Preregistration and seals

**Preregistration** (Nosek et al. 2018) means writing down the hypotheses, tests, data and decision rules **before** seeing the test data, so you can't (even unconsciously) pick the analysis that flatters your method. Each study wrote a preregistration including the outcome *expected* from a dry run on development data.

**Seals**: before each confirmatory run, a script (`freeze_snapshot.py`, `freeze_b.py`, `freeze_c.py`, `freeze_d.py`) recorded **SHA-256 hashes** of the preregistration text, every source file, the frozen models and the datasets. A SHA-256 hash is a 64-character fingerprint: change one byte of a file and the fingerprint changes completely. The confirmatory runner refuses to start if any hash changed, and logs each step (it runs **once**; a rerun needs a written reason that is also recorded as a dated deviation). Seals (UTC dates; see the note in Section 3.4): Oct 23, 25, 27 and 28, 2026.

Honest limits stated in the manuscript: these are **internal** preregistrations, not externally registered reports. The git repository was never copied to a public server, and commit dates are set by the author's own computer, so **an outsider cannot verify when each seal was made**; the hashes only prove the files haven't changed since. The Study D seal is the weakest: the Study D code itself (23 files) was not committed at the seal, only hashed. The fix for the future is to register Study E publicly (e.g. on OSF, which timestamps it on its own server) before running anything.

**Locked set**: a portion of data set aside and evaluated exactly once at the very end, after everything else is frozen. Each study has one.

### 10.2 Hypothesis tests used

- **One-sided test**: the direction ("cheaper", "less than 10%") is fixed in advance.
- **Holm procedure** (Holm 1979): when you run five tests, the chance that at least one is a false positive is higher than 5%. Holm adjusts p-values step-wise so the **familywise** chance stays at 5%. Each of Studies B, C, D had a family of five tests.
- **Exact binomial test**: for "is this share above 50%?" (counts of wins).
- **Clopper–Pearson interval** (1934): an exact 95% confidence interval for a proportion. Example: 8,707/12,192 = 71.4%, CI 70.6–72.2%.
- **Bootstrap** (Efron 1979): resample trajectories with replacement 4,000 times and recompute the statistic, to get its uncertainty without formulas. p-values have a floor of about $2.5\times10^{-4}$ (1/4,001): if not a single resample goes the "wrong" way, the test can't say how small the true p-value is, only that it is at most this. Table 2 now marks such values.
- **Permutation test**: shuffle labels many times (10,000) to see how often a statistic as extreme arises by chance.
- **Spearman rank correlation** $\rho$ (1904): correlation of ranks; 1 = perfectly increasing relationship, 0 = none. Robust to outliers and non-linear shapes.
- **Mann–Whitney test** (1947): are values in group A typically bigger than in group B?
- **Two-proportion z-test** and **chi-square homogeneity test**: are rates different between groups?
- **Intersection–union test**: when a hypothesis has several parts that must *all* hold, its p-value is the largest of the parts' p-values.
- **McNemar test**: compares two methods on the same items by looking only at the items where they disagree (used in the Study B errata, and in Section 15B to check that "equal-risk" RK45 really fails as often as the policy).
- **AUC** (area under the ROC curve): how well a score separates two groups. 0.5 = coin flip, 1.0 = perfect. "FTLE separates violations with AUC 0.93" means a random violating point has a higher FTLE than a random non-violating point 93% of the time.
- **$R^2$**: fraction of variation explained by a model (1 = perfect).
- **MSE**: mean squared error.

### 10.3 Unit of analysis

The trajectory (in D5, the periodic-orbit member, resampled within family). Terminating trajectories (the reference crashes or escapes) are excluded from cost tests and analysed separately; untrusted periodic-orbit references are excluded and listed.

### 10.4 Software

Python 3.11.16 with NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.0, pandas 3.0.5, Numba 0.67.0, Matplotlib 3.11.1 (Study A ran under Python 3.13.13 with NumPy 2.5.1). IAS15 via REBOUND 5.2.2.

---

## 11. The datasets

### 11.1 How a starting state is drawn

A fixed **mixture**:

- **55% near a Lagrange point** (one of the five, chosen at random): each position component offset uniformly within ±0.1 DU, each velocity within ±0.15 DU/TU.
- **20% wide box**: $x\in[-0.5,1.5]$, $y\in[-0.8,0.8]$, $z\in[-0.3,0.3]$ DU, velocity within ±0.5 DU/TU.
- **25% near the Moon**: distance log-uniform between 1.2 lunar radii and 0.15 DU, random direction, velocity within ±0.6 DU/TU. (Added because the older sampling almost never came closer than 0.037 DU to the Moon, far from realistic near-rectilinear halo orbits.)

Horizon $T$: **log-uniform 1–20 TU** (log-uniform means each factor of 2 is equally likely, so short and long horizons are both well represented).

Important: this population is **synthetic**. Every share (e.g. "71.4% of trajectories") describes this mixture, not a real mission catalogue.

### 11.2 Table 1 (the datasets), explained

| Study | Dataset | Seed | Candidates | Usable | Role |
|---|---|---|---|---|---|
| A | development + locked | 12 | 20,000 | – | fitting, calibration (2,000); 5,000 locked rows (4,065 usable) for H1–H2 |
| A | replication | 13, 15 | 20,000 each | – | secondary replication |
| B | exploration | 21 | 20,000 | 16,170 | policy fitting (11,170 rows) |
| B | confirmation | 22 | 20,000 | 16,161 | tests T1–T5: 2,000 calibration, 14,161 evaluation |
| B | lock | 23 | 5,000 | 4,034 | gated policy (descriptive) |
| C | training | 41 | 100,000 | 81,111 | fitting |
| C | enriched training | 141 | 20,000 | 13,500 | extra near-Moon states (weights 0.29/0.11/0.60), reweighted back to the natural mix |
| C | selection | 42 | 20,000 | 16,190 | model, quantile, guard choice |
| C | confirmation | 43 | 20,000 | 16,192 | tests C1–C5: 4,000 calibration, 12,192 evaluation |
| C | lock | 44 | 5,000 | 4,020 | one final evaluation |
| D | selection | 52 | 10,000 | 8,101 | rule selection, dry runs |
| D | confirmation | 53 | 20,000 | 16,239 | D1, D2 (4,000 cal, 12,239 eval); D4 calibration |
| D | lock | 54 | 5,000 | 4,074 | one final evaluation |
| D | replication | 55 | 20,000 | 16,253 | D3 (4,000 cal, 12,253 eval); D4 evaluation |
| D | JPL periodic orbits | – | 1,071 members | 942 | D5 (1,647 runs) |

"Candidates" = sampled starting states; "usable" = the reference reaches the horizon (about 81%). In total about **325,000** Earth–Moon starting states. Using a new seed per study means the test data are genuinely new; a model cannot have memorized them.

**Augmenting rare cases (enriched training) and reweighting**: near-Moon cases are hard and rare, so Study C oversampled them for training and then gave each oversampled row a smaller weight so the model still learns the natural population. This is "importance weighting."

### 11.3 Where the data live

`data/` (Study A, seeds 12–15), `data_b/` (seeds 21–23, the μ-dial sets 31–35 and the horizon sweeps), `data_c/` (41–44, 141, pilots), `data_d/` (52–55 and the raw JPL responses in `data_d/jpl/`). The large CSVs are not in git; their hashes are in `FREEZE*.json` and `part_*/DATA_HASHES_*.json`. `archive/` holds the pre-seal trial datasets (seeds 1–4, 99) and their runs.

---

## 12. Study A

**Folder:** top level (`PREREGISTRATION.md`, `FREEZE.json`, `CODE_SNAPSHOT.json`, `snapshots/`, `data/`, `out/`, `RESULTS_REPORT.md`, `final_eval_log.jsonl`). **Runner:** `run_rigor_suite_3d.py`. **Sealed:** 2026-10-23.

### 12.1 Design

- Integrators offered: RK4 and leapfrog (Euler kept only as a negative control).
- Model: hurdle models per integrator: a **random-forest classifier (150 trees)** ("can any step pass?") and a **gradient-boosted quantile regressor at $q = 0.1$** on $\log_{10}\Delta t$.
- Features: state, horizon, Jacobi, proximity, **and FTLE**.
- The chosen integrator is the one with the larger predicted step per evaluation ($\hat d_k / e_k$), i.e. cost-normalized selection.
- Safety scale certified per integrator with the Bonferroni LTT rule ($\alpha = 0.05$, $\delta = 0.1$).
- Comparators: RK45 and DOP853, each oracle-tuned ("spec") and certified ("cal").

### 12.2 Hypotheses and results

| ID | Hypothesis (plain words) | Result | Verdict |
|---|---|---|---|
| H1 | With FTLE cost charged: within tolerance on ≥ 90% **and** cheaper than each adaptive solver on ≥ 50% | Within tolerance 959/996 (96.3%), cheaper **0/996** against each of the four comparators | Not supported |
| H1′a | Same without FTLE but charging inference cost (added after development data were seen, recorded as a deviation) | Within 960/999 (96.1%), cheaper **0/999** | Not supported |
| H1′b | Without FTLE, against the best *single certified fixed step* (post hoc, a favourable comparator) | ML cheaper than global fixed RK4 on 89.2% and global leapfrog on 98.1% | Supported, but against a weak baseline |
| H2 | Certified scales keep violations ≤ 5% | RK4 17/500 (3.4%, CI 2.0–5.4%); leapfrog 19/498 (3.8%, CI 2.3–5.9%). Unscaled: RK4 11.2%, leapfrog 14.5%, Euler 27.1% | Point estimates met; intervals do not exclude > 5% |
| H3 | Removing FTLE lowers RK4 $R^2$ (whole 95% interval below 0) | $\Delta R^2 = -0.25$ (−0.54 to 0.04); selection accuracy change +0.008 (−0.009 to 0.024) | Not met: FTLE not shown to be needed |
| H4 | Report non-monotone label rate | 1.5% of 200 (0.3–4.3%) | Reported |
| H5 | Conformal error-model recommender ≤ 10% violations | 17.4% (15/86 covered orbits) | Not met |
| H6 | Crash/escape prediction with vs without FTLE | AUC 0.993 with FTLE vs 0.780 without, but FTLE "has seen" the crash (it is integrated over the same horizon) | Reported, with caveat |
| H7 | Convergence slopes match textbook orders | Euler 1.000, leapfrog 2.000, RK4 4.017 | Reported: integrators correct |

### 12.3 Why A lost, in numbers

- **FTLE is unaffordable.** On 80 development trajectories the FTLE computation cost **more than a whole DOP853 solve on every one**, by a median factor of about **70**.
- **Even without FTLE, the model was too expensive**: about **591** RHS-equivalents of inference against median oracle-tuned adaptive costs of **146 (RK45)** and **122 (DOP853)**.
- **The fixed step itself was expensive**: mean ML policy cost about 15,468 evaluations per orbit against about 1,077 (RK45) and 898 (DOP853) at the oracle tolerance (from `RESULTS_REPORT.md`).

### 12.4 Other Study A findings worth knowing

- **Halo orbits** (validated by differential correction, not the analytic Richardson formula): RK4 and leapfrog met the tolerance on all 40 test runs each (RK4 median error ~10⁻⁶ km).
- **Earth–Moon vs Sun–Earth**: within Sun–Earth, integrator selection reached 91.8% against 61.1% for always-RK4; transfer between systems did not beat always-RK4 in either direction (61.8% vs 61.1%; 81.9% vs 86.1%). Features contain no μ, so poor transfer was expected.
- **CAPSTONE fidelity**: the model error (2,000–50,000 km) dwarfs the 100 km integrator tolerance (Section 5.6).
- **Integrator selection** beat always-RK4 only slightly on the locked set (89.4% vs 86.8%).
- **Worst-case errors** in the scaled audit reached 848,682 km (RK4): the safety layer controls how *often* errors exceed tolerance, not how *large* the worst error is.

### 12.5 What A taught

The original hypothesis ("FTLE, proximity and Jacobi govern difficulty, so a model using them can predict a safe step") fails as a *design*: the most informative quantity is the one you can't afford. The cost claim fails even without it. The question became **why**.

---

## 13. Study B

**Folder:** `part_b/` (preregistration, results, log, figures), `out_b/`, `data_b/`. **Runners:** `run_failure_study_3d.py`, `run_failure_experiments_3d.py`, `run_confirm_b_3d.py`. **Sealed:** 2026-10-25.

### 13.1 Design

Same kind of physical-time policy but **no FTLE** (only free features), every cost charged (1,128 RHS-equivalents for the scikit-learn model), scales and tolerances re-certified on 2,000 confirmation rows, **14,161 evaluation trajectories**. Then the **cost-gap decomposition** (Section 9.3) on every trajectory.

### 13.2 The headline

The policy was cheaper than an adaptive solver on **0 of 14,161** trajectories against **all six** comparator settings (RK45 and DOP853 × oracle, certified, equal-risk).

Much of that zero is the price of the slow model: its 1,128 RHS-equivalents are more than three typical adaptive solves. Charged nothing for the model (post hoc), the same policy would have won 27% against certified and 23% against equal-risk RK45, though almost none against the oracle-tuned solvers (0.3% and 3.6%), and it would still have used about 11–14× as many evaluations in total.

### 13.3 The decomposition (Fig 3A)

Mean over the 13,540 trajectories that met the tolerance, against oracle-tuned RK45:

| Term | Mean (dex) |
|---|---|
| Paradigm $P$ | +0.21 |
| Selection $S$ | +0.03 |
| Prediction $Q$ | +0.38 |
| Safety scale $K$ | +0.30 |
| Overhead $O$ | +0.50 |
| Rounding $R$ | −0.01 |
| **Total** | **+1.41** (a factor of about 26) |

Which term dominates per trajectory: overhead 39.7%, prediction 32.1%, paradigm 20.6%, safety 2.8%, selection 0.5% (violations 3.9%).

### 13.4 The five confirmatory tests

| Test | Hypothesis | Result | Holm p | Verdict |
|---|---|---|---|---|
| T1 paradigm | Even the *oracle* fixed step beats oracle-tuned RK45 on < 50%, and its gap rises with $\Delta_\tau$ | 3,658/14,160 (25.8%); Spearman $\rho = 0.909$; slope 0.904 dex per dex | < 0.001 | Supported |
| T2 information | Label spread under tiny perturbations (3.8 km) rises with $\lambda T$ | $\rho = 0.41$ ($n = 443$) | < 0.001 | Supported in sign only |
| T3 calibration | Per-regime ("Mondrian") scales cut cost at equal certified risk | **+0.073 dex costlier** (0.072–0.075) | 1.000 | Not supported |
| T4 where | Gap and violations larger near the Moon and in regimes II–III | Violations 394/2,552 (15.4%) within 0.05 DU of the Moon vs 152/11,609 (1.3%) elsewhere; gap +0.48 vs +0.07 dex | < 0.001 | Supported |
| T5 horizon | Gaps rise with horizon on the same starting states | Mean per-state $\rho$ 0.908 and 0.727; median paradigm gap −0.20 dex at 1 TU to +0.34 at 20 TU | < 0.001 | Supported |

### 13.5 What each result means

- **T1 is the key discovery.** A slope of **0.904 dex per dex** between $\log\Delta_\tau$ and the paradigm gap means: if the timescale range doubles, the fixed-step penalty almost exactly doubles. That is what the road-trip argument (Section 4.8) predicts with slope 1. **The fixed physical step is wrong in kind near a body.**
- **But how strong the paradigm wall is depends on the comparator.** Against oracle-tuned RK45 a perfect fixed step wins only 25.8%. Against *one certified tolerance* for everyone, the same perfect fixed step wins **67.9%** (RK45) and **55.8%** (DOP853). So "a single step can't win" holds against an oracle opponent, not a realistic one; against realistic opponents the losses come from **prediction, safety and overhead**.
- **Violations are "step-overs."** Of the 447 violations not due to a classifier error, **446** used a scaled step *larger than the label* (median 2×). Only **1 of 13,524** runs whose step was at or below the label violated. The ratio (step / minimum timescale) separates violations with AUC 0.85. Failures come from stepping over a close pass.
- **T2 caveat.** The label is stable (spread ≤ 0.14 dex, almost always 0) while the model's median error is 0.32 dex. So the model's error comes from **missing information in the features**, not from the starting state being fundamentally unpredictable. Supporting this, an exploratory test (E3b) that gave the model trajectory-derived features (closest approach, timescales, transit counts) cut the median error from **0.294 to 0.057 dex**, but those features need the integration.
- **T3: splitting the calibration costs more.** Per-regime scales had to split the error budget δ across groups, so each group needed more evidence, giving smaller scales overall.
- **T5:** the same starting states relabelled at 1, 2, 5, 10, 20 TU: the longer you integrate, the worse the fixed step does. Violations rose from 1.1% to 9.2%.

### 13.6 Exploratory work in Study B (not in the Holm family)

- **E3c lookahead**: run a cheap RK45 over the first fraction $f$ of the horizon to get features. Error falls steadily with $f$ (0.32 → 0.07 dex) with no early plateau: the information is spread over the whole trajectory. At $f = 1$ the lookahead alone (302 evaluations) costs more than the oracle-tuned solve (134). An erratum (2026-10-26) corrected a claim that two intervals didn't overlap (they did) and noted the small-$f$ gain was partly extra risk.
- **E5 calibration alternatives**: global scale, per-regime, dispersion bins, abstention to adaptive for the 20% most uncertain, label-conformal scaling. None made the policy cheaper than adaptive; abstention cut mean cost 26%.
- **E9 runtime guard**: abort if step/τ exceeds 0.3: catches 57% of violations with 7.9% false alarms.
- **E10 physics rule** $\Delta t = c\,\tau(\text{IC})\,T^{-b}$: 12× costlier than the ML policy at similar risk, so the physical-time model learned more than a Kepler timescale.
- **E11 gated policy**: use ML only in cells where it wins; with realistic overhead no cell qualified.
- **E11b cost predictor for batching**: a model predicting the *adaptive* solver's cost from the starting state ranks it with Spearman 0.964; sorting a lock-step batch by predicted cost raised modelled lane efficiency from 0.080 to 0.536 at batch size 32. A genuinely useful side result (but modelled, not measured on a GPU).
- **E7 μ dial**: the study at five mass ratios (10⁻⁶ to 0.05) with sampling scaled by the Hill radius. Gaps fell as μ fell; the violation rate did not; trends are confounded (erratum).
- **E1 reference validity**: Radau agreement to 0.003 km (Section 5.5).

### 13.7 What B taught

Two things to attack: the **paradigm** term (needs a different time variable) and the **prediction** term (needs a cheap summary of the next close approach). Plus: make the model **cheap** (overhead).

---

## 14. Study C

**Folder:** `part_c/`, `out_c/`, `data_c/`. **Runners:** `run_part_c_3d.py`, `run_confirm_c_3d.py`. **Sealed:** 2026-10-27.

### 14.1 Three changes, each aimed at one term

| Term attacked | Change |
|---|---|
| Paradigm | Integrate in **Sundman time** (Section 6), predict one fictitious step $\Delta s$ |
| Prediction | Add the **13 analytic periapsis features** (Section 4.10) |
| Overhead | **Compiled** 25-tree model (15.04 RHS-equivalents instead of 1,128) |
| Safety scale | **End-to-end fixed-sequence** certification on a fine grid, quantile tuned ($q = 0.2$) |

Controls: a **physical-time policy** with the same features (10 trees, RK4 only, abstaining to certified RK45 when the lunar osculating orbit returns within the horizon with a short periapsis timescale) and **one global certified fictitious step** with no model.

### 14.2 Results of the five tests

| Test | Hypothesis | Result | Holm p | Verdict |
|---|---|---|---|---|
| C1 | Periapsis features lower median step error | 0.319 → 0.282 dex; difference 0.037 (0.027–0.048) | < 0.001 | Supported (small) |
| C2 | Physical-time policy cheaper than equal-risk RK45 on > 50% | 31.6% | 1.000 | Not supported |
| C3 | Oracle Sundman step beats oracle RK45 on > 50%, and gap no longer tracks $\Delta_\tau$ | 51.1% (50.2–52.0); $\rho$ 0.225 vs 0.911 | 0.012 | Supported (narrow) |
| **C4** | **Sundman policy cheaper than equal-risk RK45 on > 50%** | **8,707/12,192 = 71.4% (70.6–72.2)** | **< 0.001** | **Supported** |
| C5 | Model beats one global certified step | saves 0.306 dex (0.302–0.310); cheaper on 89.9% | < 0.001 | Supported |

### 14.3 What each change bought (Fig 3B)

Mean decomposition terms (dex) against oracle-tuned RK45:

| Term | Study B | Study C physical time | Study C Sundman time |
|---|---|---|---|
| Paradigm | +0.21 | +0.12 | **−0.005** |
| Selection | +0.03 | +0.001 | – |
| Prediction | +0.38 | +0.34 | **+0.10** |
| Safety scale | +0.30 | +0.06 | +0.06 |
| Overhead | +0.50 | +0.05 | +0.07 |
| Rounding | −0.01 | −0.01 | −0.01 |
| **Total** | **+1.41** | **+0.55** | **+0.21** |

**Careful: this table mixes subsets.** Each column averages over the trajectories *that policy* got right, and the physical-time policy handed its hardest near-Moon cases to RK45, so its +0.12 comes from an easier set. Measured over all trajectories, the paradigm term is +0.25 in physical time (in both B and C) and 0.00 in Sundman time, and the totals are +1.39 (B) → +0.47 (C physical) → +0.20 (C Sundman). So the cheaper compiled model, periapsis features and better certificate removed about 0.9 dex, and the new clock about 0.3 dex (Section 15B.2).

Reading: the paradigm term **vanished** in Sundman time; prediction became much easier because the Sundman label varies far less (standard deviation **0.33 dex** for $\Delta s$ vs **0.91 dex** for the physical step). The gap against the hardest opponent shrank from a factor of 26 to a factor of 1.6. Against realistic opponents it became negative (cheaper): −0.21 dex vs certified RK45, −0.13 dex vs equal-risk RK45.

### 14.4 Shares cheaper (Fig 5A, Table 3)

At a realized violation rate of **535/12,192 = 4.39%** (certified scale 0.872):

| Comparator | Share cheaper |
|---|---|
| RK45 equal-risk | 71.4% |
| RK45 certified | 82.4% (optimistic: these solvers failed only 1.3% of the time, far safer than the policy; Section 15B.4) |
| DOP853 equal-risk | 50.6% |
| DOP853 certified | 58.6% |
| RK45 oracle | 8.7% |
| DOP853 oracle | 6.1% |

Median costs on trajectories that met the tolerance: Sundman policy **235** (220 integration + 15 overhead); RK45 oracle 146, certified 398, equal-risk 332; DOP853 equal-risk 230.

**Post hoc, in total.** Summed over all 12,192 trajectories, the policy used 1,434 evaluations per trajectory on average against 1,201 for equal-risk RK45, so RK45's total was **0.84** of the policy's. The policy wins most trajectories but loses on total cost (Section 15A).

### 14.5 Where it wins and loses (Fig 5B)

At **1–3 TU** it wins > 93% in every regime. At **8–20 TU** it wins only 2.8% (regime I), 6.4% (II), 9.5% (III); 52% (IV) and 40% (V). Long chaotic horizons are expensive and an adaptive controller is well matched to them.

### 14.6 Safety where it lands

Marginal 4.4%, but **13.5%** (299/2,222) for trajectories that passed within 0.05 DU of the Moon vs **0.4%** beyond 0.8 DU. Violations were usually mild (median error 1.7× tolerance; 6.6% exceeded 10×). Endpoint-only audit: 2.6% of runs were wrong mid-run but right at the end.

### 14.7 Terminating ("doomed") orbits (the 19% excluded)

On the 3,808 terminating trajectories (the reference crashes or escapes before $T$; the code calls them "reference-divergent"), the Sundman policy ended in the **same event type** (collision with Earth, collision with Moon, or escape) on **3,805 (99.9%)** and **never passed through a body**, because the transformation shrinks the step near the primaries. But it got the *time* of the event right (by the plan's strict rule) on only **46.7%**, because a fixed step can't place an event more precisely than its own length. A fixed physical step was much worse there. A free "crash flag" from the osculating orbit (periapsis inside a body before $T$) had recall 81% and precision 96%.

### 14.8 Lock set

On 4,020 locked trajectories: 4.28% violations, cheaper than equal-risk RK45 on 70.0% and DOP853 on 49.4%. Consistent.

### 14.9 Honest caveats Study C itself lists

C3(a) is a narrow pass that depends on $\tau_c$ and did not replicate in Study D (50.6%); the physical-time policy's certificate upper bound reached 5.17%; the Sundman label is non-monotone on ~4.5% of orbits; costs are RHS counts, not wall-clock (tested in D2).

---

## 15. Study D

**Folder:** `part_d/`, `out_d/`, `data_d/`. **Runners:** `run_part_d_3d.py`, `run_confirm_d_3d.py`, `run_explore_d_3d.py`, `gen_part_d.sh`, `fetch_jpl_d.py`, `ias15_baseline_d.py`. **Sealed:** 2026-10-28. Study D fitted **no** new step-size model; it froze Study C's.

### 15.1 Five confirmatory tests

| Test | Question | Result | Holm p | Verdict |
|---|---|---|---|---|
| D1 | Is ML needed? Model vs the best physics rule | model cheaper by **0.0715 dex** (~18%), on 69.5% of 11,565 paired orbits | < 0.001 | Supported |
| D2 | Faster in compiled wall-clock than compiled DP5(4)-PI at equal risk? | throughput ratio **0.885** (0.862–0.908) | 1.000 | Not supported |
| D3 | Does C4 replicate on new data? | **8,768/12,253 = 71.6%** (70.8–72.4), violations 2.69% | < 0.001 | Supported |
| D4 | Conditional safety: every stratum < 7% and still cheaper on > 50% | worst stratum **3.1%**; cheaper on **66.4%** | < 0.001 | Supported |
| D5 | Transfer to JPL periodic orbits: family-balanced violations < 10% | **70.0%** (67.8–72.3) | 1.000 | Not supported |

All five outcomes matched the expectations written in advance from dry runs.

### 15.2 D1 in detail: physics does most of the work (Fig 6A)

| Arm | Coefficients | Overhead | Extra cost vs 25 trees (dex) |
|---|---|---|---|
| Linear model, 33 engineered features | 34 | 11.39 | **−0.017** (cheaper, at higher risk 4.7% vs 2.6%) |
| Rule $T$, regime, $r_p$ (**test arm**) | 4 | 7.32 | +0.072 |
| Rule $T$, regime, $\tau_p$ | 4 | 7.43 | +0.077 |
| Rule $T$, regime, $\tau_p$, $C-C_{L_1}$ | 5 | 7.28 | +0.078 |
| Rule $T$, $\tau_p$ | 3 | 7.26 | +0.087 |
| Rule $T$, regime | 3 | 2.07 | +0.090 |
| Rule $T$ only | 2 | 2.05 | +0.163 |
| One global Sundman step | 1 | – | +0.274 (replication set) |

**Reading.** Going from one global step (+0.274) to the best 4-number rule (+0.072) captures (0.274 − 0.072)/0.274 ≈ **74%** of the model's saving: "a four-coefficient rule recovers about three quarters." (Those two gaps come from different datasets. A post hoc rerun of the global step on the same confirmation set gave a gap of 0.269 dex, so the rule recovers **73%** (95% CI 73–74%) on one dataset: the "three quarters" holds.) The linear model **ties** the trees (the −0.017 is below the 0.038 dex resolution of the scale grid and comes with higher risk). So the trees are not the secret; the **periapsis features** are. But the margin still matters against solvers: the model is cheaper than equal-risk RK45 on 70.2% and the best rule on only 53.0%; against equal-risk DOP853, 49.3% vs 34.5%.

### 15.3 D2 in detail: wall-clock (Fig 8)

Both the policy and three adaptive solvers were compiled with Numba over **identical** RHS arithmetic and the same checks, timed on one thread of an Apple M5:

- Overall ratio vs DP5(4)-PI: **0.885** (the policy is *slower* in total), though faster on **78.2%** of individual trajectories (median per-trajectory ratio 1.48).
- By horizon: **1.61** at 1–3 TU, **1.02** at 3–8 TU, **0.75** at 8–20 TU. The few long trajectories dominate total time.
- vs DOP853: 0.642.
- Evaluation counts and seconds agreed trajectory by trajectory (Spearman 0.99); cost per evaluation at parity (22.4 vs 23.1 ns).
- **A cautionary tale**: the first implementation (reusing the Study C kernel) gave **0.595**, because the Sundman step computed extra square roots in every stage and its landing steps allocated memory (26.6 vs 18.7 ns per evaluation). **Evaluation counts alone had hidden this.**
- Batches of 10³–10⁵ on 1 or 10 threads: ratio 0.82–0.91. Modelled lock-step lane efficiency was similar for both (no advantage for fixed steps unless sorted).
- Deviation note: the load-average guard was relaxed from 1.5 to 1.6 because the load hovered at 1.50 with the CPU 94.5% idle; recorded as a dated deviation.

### 15.4 D3: replication

New calibration gave scale 0.800 (vs 0.872); violations 2.69%; shares: RK45 equal-risk 71.6%, certified 79.1%, DOP853 equal-risk 51.3%, certified 54.1%, oracle RK45 5.5%, oracle DOP853 3.6%. **One piece did not replicate**: C3(a), the oracle Sundman step beating oracle RK45, came out 50.6% (p = 0.110), not above 50%; its mechanism (ρ 0.221 vs 0.905) did replicate.

### 15.5 D4: conditional safety (Fig 7, S1 Table)

With **one** scale, near-Moon passes violated 13.1% and the worst free stratum (IV–V, far, 8–20 TU) 10.4%. With **per-stratum** scales, **every one of 17 strata ≤ 3.1%** (largest upper bound 4.9%), overall 2.4%, still cheaper than equal-risk RK45 on 66.4%. Where trajectories actually passed close to the Moon, the improvement was partial (5.1% below 0.05 DU) because the osculating orbit sees only some close passes.

**The comparator is also only safe on average**: equal-risk RK45 violated 24.1% and 30.4% in two regime-I strata where the stratified policy violated 1.5% and 1.9%. Exploratory **D4b** (fairer): RK45 certified *per stratum* (1.5% overall violations): the policy is cheaper on **56.2%**; DOP853 certified per stratum: only **23.5%**. Post hoc, with the solvers certified on a grid as fine as the policy's, these fell to **42.5%** and **20.7%**: certified per stratum by the same rule, both solvers are cheaper than the stratified policy on most trajectories.

### 15.6 D5: periodic orbits (Fig 9, S2 Table)

From the **JPL Three-Body Periodic Orbits API**: 21 streams in 11 families (halo L1/L2, Lyapunov L1/L2, vertical L1/L2, axial L1/L2, distant retrograde orbit (DRO), butterfly, resonant), 50 members per stream; members dropped if they didn't close to 10⁻⁸ in one period, hit a body, or the reference stopped (129 of 1,071). Horizons of 1, 2, 5, 10, 20 periods within 1–20 TU; runs with disagreeing references (>5 km) excluded (102 of 1,749). JPL's mass ratio differs from the project's by 2.1×10⁻⁸ (negligible).

Results: family-balanced violation **70.0%**; vertical and axial families 96–100%; Lyapunov 78–84%; halo, DRO, butterfly, resonant 33–47%. **Why**: the predicted step is **0.3–0.5 dex too long** on vertical, axial and Lyapunov families (prediction failure); on halos etc. it is about right, and failures come from **unstable** members (the policy has no instability feature). On one vertical L2 member, Sundman RK4 converged at 4th order as the step shrank: **"the integrator was right and the step was wrong."** Recalibrating the scale (0.183) cut violations to 3.5% but made it never cheaper than RK45. The comparators also struggled (RK45 26–66% at equal risk) because these orbits amplify errors strongly (stability index up to >100).

**Why the population matters:** random starting states rarely land on these special, highly structured orbits, so the model never learned them. This is classic **distribution shift**.

### 15.7 Exploratory analyses

- **D6 feature importance (Fig 6B)**: refit with each group removed. For the Sundman target, removing periapsis raised test MSE by 0.0048 and the horizon by 0.0086; **proximity and Jacobi contributed nothing detectable**. For the physical target periapsis was 9× as important. Adding FTLE helped only a little (0.0023 of 0.007) and costs an integration. **The original hypothesis is not supported as a statement about what a pre-run predictor needs.** Drop-group refits were preferred over permutation importance because correlated features fool permutation (Strobl et al. 2008).
- **D7 chaos maps (Fig 10)**: on planar surfaces of section (y = 0, z = ż = 0, ẏ > 0) at C = 3.19, 3.10, 2.95, FTLE separated violations with **AUC 0.93, 0.92, 0.93**; the four lowest FTLE deciles were almost violation-free; the top decile violated 84–97%. FTLE explains failures *after the fact*.
- **D10 Sundman design**: τ_c sweep (Section 6.3) and non-monotone label analysis (4.0%).
- **D12 is the reference the problem?** For the 163 most chaotic trajectories, two stricter references agreed to ≤ 0.23 km; the policy violated **25.2%** there vs 2.6% overall. The error is the policy's.
- **Leapfrog symplecticity** check and **IAS15** comparison (Sections 5.3, 5.7).

### 15.8 Lock set

4,074 trajectories: ML 2.45% violations, cheaper than equal-risk RK45 70.3%, DOP853 48.9%; best rule 4.32%, 52.3%, 34.2%.

---

## 15A. After the seals: post hoc checks and the Study E plan

**What "post hoc" means here.** *Post hoc* (Latin: "after this") means an analysis decided on **after** the results were seen. Such analyses can't prove anything the way a preregistered test can, because you could keep looking until something flattering turns up. They can still reveal problems. On 2026-10-30, a review plan (`CR3BP_Rigor_and_Bug_Plan.md`, kept outside the repository) asked for a set of checks on the sealed Studies A–D. The rules were:

- **No sealed verdict or number is changed.** Every new number is labelled "post hoc" and shown *next to* the sealed number it qualifies, never in place of it.
- **The definitions were committed before any computation.** `posthoc/README.md` (checks P1–P9) was committed to git before any of them ran, so the definitions could not be tuned to the answers.
- **Every rerun first reproduces the sealed result exactly.** Only then does it compute anything new, which proves the new code computes the same thing as the old.
- **Negative results stay in.** Anything that wasn't preregistered and might still be useful goes into a new, separately preregistered **Study E** on fresh data.
- **Hybrid fixed/adaptive switching** was ruled out of scope.

The code is `run_posthoc.py` (one command per check), `orbit_ml_cr3bp/posthoc_metrics.py` and `orbit_ml_cr3bp/symmetric_cert.py`. Results are in `posthoc/results/` and in the manuscript's new S4 Table and its subsection "Post hoc checks of the comparators".

### 15A.1 Share versus total cost (P1): the most important new finding

**The idea.** Suppose you run 100 trajectories. On 70 short ones the policy saves 50 evaluations each (3,500 saved). On 30 long ones it loses 200 each (6,000 lost). It "wins on 70%" but costs **more** in total. The preregistered statistic (share of trajectories) can't see this. The **ratio of totals** can:

$$\text{ratio of totals} = \frac{\sum \text{solver evaluations}}{\sum \text{policy evaluations}}\qquad(\text{above 1: the policy is cheaper in total})$$

The policy's sum includes its overhead and its failed runs. Uncertainty comes from a bootstrap over trajectories (4,000 resamples).

| Comparator | C confirmation | D replication | C lock | D lock |
|---|---|---|---|---|
| RK45 equal-risk | **0.84** (0.81–0.86) | **0.80** (0.78–0.82) | 0.85 | 0.85 |
| RK45 certified | 1.01 (0.98–1.04) | 0.91 (0.88–0.93) | 1.03 | 0.97 |
| DOP853 equal-risk | 0.69 | 0.67 | 0.70 | 0.69 |
| DOP853 certified | 0.78 | 0.70 | 0.80 | 0.75 |

So against equal-risk RK45, the very comparison where the policy wins 71% of trajectories, RK45 used only 80–84% of the policy's evaluations in total. In other words, the policy used **19–25% more**. On average per trajectory: 1,434 against 1,201 (Study C), and 1,643 against 1,318 (replication).

**Where it wins in total: short horizons only.** Replication set, equal-risk RK45 total ÷ policy total, by horizon:

| Horizon | Ratio of totals | Meaning |
|---|---|---|
| 1–3 TU | **1.48** (1.44–1.52) | RK45 needs 48% more: the policy is clearly cheaper |
| 3–8 TU | 0.97 (0.94–1.00) | about even |
| 8–20 TU | 0.69 (0.67–0.71) | the policy needs ~45% more; this band holds 70% of the policy's total evaluations |

At 1–3 TU the policy even beats equal-risk DOP853 in total (1.26). This matches the wall-clock result of D2 (1.61 at 1–3 TU, 0.75 at 8–20 TU, Section 15.3): **the horizon dependence was already there in evaluation counts**, and the share had hidden it.

Table 3 also gained a "median total cost over **all** trajectories" row (251, 267, 267, 272), alongside the sealed median over trajectories that met the tolerance.

### 15A.2 How much the share depends on the sampling mixture (P2)

All shares describe the project's synthetic mix of starting states (Section 11.1). The check re-weights the trajectories as if they had been drawn in other proportions. Both alternative mixes were declared in advance:

| Population (replication set) | Share cheaper than equal-risk RK45 | Ratio of totals |
|---|---|---|
| As sampled | 71.6% | 0.80 |
| Equal weight for all 7 sampling families (L1–L5, wide box, near Moon) | 71.5% | 0.81 |
| **Cislunar**: near-Moon, L1 and L2 states only, a third each | **57.8%** (56.3–59.1%) | 0.75 |

By family, the policy does best around L3, L4 and L5 (85–87%, ratio of totals at least 1.17), which are far from the Moon, and worst near the Moon (55%, ratio of totals 0.70) and around L1/L2 (55–64%). **The near-Moon and L1/L2 regions, where most cislunar missions fly, are where the policy's advantage is smallest.**

### 15A.3 Is the comparison fair to the solvers? Symmetric certification (P7)

**The asymmetry.** The policy's safety scale was certified the powerful way: fixed-sequence testing at the full error budget δ, on a grid with steps of 0.038 dex. The solvers' "certified" tolerance used the weaker Bonferroni rule, which splits δ over 18 tests, on a coarse grid 0.53 dex apart (a factor 3.4 between neighbours). So the solvers were forced to pick a tolerance tighter, and so costlier, than they needed. **Analogy:** two runners must each pick a safe pace. One may choose to the nearest second; the other only to the nearest minute, and must play extra safe.

**The fix.** Certify RK45 and DOP853 by the policy's own rule (fixed sequence, full δ) on a fine grid: rtol = 10^(−10+k/8), i.e. 0.125 dex apart. Because a solver's cost changes only about 1/5 as fast as its tolerance, that is a *finer* cost resolution than the policy's grid. The certified RK45 tolerance loosened from 2.0×10⁻⁷ to 5.6×10⁻⁷. The solvers' realized failure rates rose to 4.0–4.8% (the policy's: 2.7–4.4%), so this is a fair match, not a free lunch for the solvers.

| Policy cheaper than… | C confirmation: sealed → symmetric | D replication: sealed → symmetric |
|---|---|---|
| RK45 certified | 82.4% → **71.6%** | 79.1% → **67.4%** |
| DOP853 certified | 58.6% → **51.8%** | 54.1% → **44.9%** |
| Ratio of totals, RK45 certified | 1.01 → 0.84 | 0.91 → 0.75 |

On the replication set the policy is **no longer cheaper than certified DOP853 on most trajectories**. The same holds for the per-stratum comparison D4b (Section 15.5: 56.2% → 42.5% against RK45, 23.5% → 20.7% against DOP853). The compiled solvers of D2 give the same picture in evaluation counts: ratio of totals against DP5(4)-PI 0.84 at equal risk and 0.81 certified symmetrically.

The headline equal-risk comparison (71.4%, 71.6%) is not affected by this. Equal-risk already matches the solver to the policy's own failure rate, using a fine search.

### 15A.4 Other comparator settings (P8)

- **Absolute tolerance.** The solvers used `atol = 10⁻³ × rtol` (Section 5.4). Four settings were tried: 10⁻¹, 10⁻³ and 10⁻⁶ × rtol, plus a "position-only" variant. Whichever was cheapest at equal risk on the *calibration* trajectories would be used, a choice made before looking at the evaluation trajectories. The winner was **10⁻¹ × rtol**. Against that better-tuned RK45 the policy is cheaper on **64.2%** of replication trajectories (instead of 71.6%), with ratio of totals 0.75 (0.73–0.77). (This was quoted in the abstract until Nov 3; it is now in the body only, to make room for the Section 15B results.)
- **Stricter confidence (δ = 0.05 instead of 0.1).** The certified scales did not change (0.872 and 0.800), so the result does not hinge on the 90% confidence level.

### 15A.5 Checks that came out fine

- **"Three quarters" on one dataset (P4):** 73% (95% CI 73–74%); see Section 15.2.
- **Random seed of the model (P6):** refitting the same 25-tree model with five other random seeds gives shares of 68.4–70.2% against equal-risk RK45 (the frozen fit: 70.2%) and ratios of totals of 0.81–0.83. The result isn't luck of one fit.
- **Overhead per column (P3):** Table 3 charged 15.04 RHS-equivalents in three columns and 16.09 in one. Charging the same value everywhere moves shares by at most 0.3 points, and ratios of totals by at most 0.001.
- **The empty Fig 4 cell (P5) is physical.** Regime I means the L1 "neck" is closed (Section 4.6): a spacecraft is trapped either in a small bubble around the Moon or in the large region around the Earth. Of 2,675 usable regime-I trajectories, every one in the lunar bubble stayed within **0.047 DU** of the Moon. Every one in the Earth region stayed beyond **0.169 DU**. Nothing can sit in the 0.05–0.17 DU band. (The one trajectory at 0.1691 DU was a calibration trajectory, not in the plotted set.)

### 15A.6 Code and reproducibility changes (Phase 4)

None of these touches a sealed result; each is recorded in `CHANGES.md`, and the seal checks now list them as files changed since the seals (expected and documented).

- **Collision check between steps (P9).** Collision is detected only at step endpoints, so a long physical step could in principle jump "through" the Moon. `collision_segment.py` checks the straight segment between two steps against each body's sphere. It is *not* wired into any sealed code, which would change sealed results. It is ready for Study E. (Sundman-time steps shrink near the bodies, so the main policy isn't affected; the physical-time arms could be.)
- **Data pipeline v6.** The same as v5, but it skips the FTLE integration (never a model input after Study A). A test checks that every label is identical to v5's.
- **One rule for "usable" rows.** `usable_rows` now uses the reference outcome directly. A test shows it selects exactly the same rows on all 27 existing datasets.
- **Shared constants.** The solvers' settings (`atol` ratio, the 2,000,000-evaluation cap, the 18-value rtol grid) now live once in `config.py` instead of being copied in two files; the values are bit-identical.
- **Reproducibility.** `environment.yml` and `environment.lock.txt` pin the exact package builds. `reproduce.sh` rebuilds the environment, tests, figures, manuscript and post hoc analyses with one command each.
- **Housekeeping.** The old pre-seal analysis scripts moved to `archive/study_a_tools/`. The 9,106-line Study A report moved to `archive/study_a/`, with a short `STUDY_A_SUMMARY.md` at the root.
- **Tests.** Two new suites, `tests/test_posthoc.py` (13 tests) and `tests/test_hardening.py` (5 tests). One of them checks that the symmetric certification of a solver gives the same answer as the policy's certification code when fed the same numbers. All six suites passed at that point. A seventh suite, `tests/test_defense.py` (12 tests), was added on Nov 3 (Section 18.4), and `DEFENSE_STATUS.md` records that its 12 tests pass. Counting the test functions in the files on 2026-11-06 gives 90 across the seven suites (9 + 5 + 30 + 16 + 13 + 12 + 5).

### 15A.7 Study E: the next preregistered study (draft only)

The post hoc checks leave one claim standing: **the policy is cheaper in total at short horizons (1–3 TU).** But that claim was *found* by looking at Study D's data, so it is a hypothesis, not a result. Study E (`part_e/PREREGISTRATION_E.md`) is designed to test it properly on new seeds (60–64). Its status is **DRAFT: not sealed, nothing run.**

| Test | Question in plain words |
|---|---|
| E1 (primary) | At 1–3 TU, is the policy cheaper **in total** than compiled DP5(4)-PI at equal risk, in both evaluation counts and compiled seconds? |
| E2 | Does that still hold at a 10 km and at a 1,000 km tolerance (the "tolerance dial" never run in Study D)? |
| E3 | Can a certificate on the error *along the whole path* (not just the endpoint) keep failures ≤ 5% while E1 still holds? |
| E4 | Including the "doomed" orbits that crash or escape, is the policy no worse than the solver in success rate and total cost? |
| E5 | Can an "am I out of my depth?" gate detect periodic orbits unlike the training data and hand them to the solver? |
| E6 | The timing protocol for E1: idle machine, two sessions. A second kind of CPU was dropped on 2026-11-13 (no x86-64 machine available), so the timing results will hold for one Mac only. |
| E7 (added 2026-11-13) | Over 1–20 TU, can a *learned* 8th-order fixed step in Sundman time beat the strongest adaptive solvers (DOP853 and DP5(4) in Sundman time, DP5(4) in physical time) in total cost and compiled seconds? |

Built-in fixes from the post hoc checks: adaptive solvers run **in Sundman time** are added as comparators (Section 15B.3); the document must be **registered on OSF before any run**, so its date can be verified by outsiders; the solvers are certified **symmetrically** and use `atol = 0.1 × rtol`; the main statistic is the **ratio of totals**, not the share; collisions are checked **between** steps; and the seal must refuse to run on an uncommitted (dirty) git tree, which Study D's seal did not enforce. Section 7 of the draft lists what must still be built before the dry run and seal (several weeks of work). Since Nov 3 that list has eight items; the eighth is compiled versions of the adaptive solvers in Sundman time. On 2026-11-13 it grew to ten, and a step-by-step plan was written (`part_e/STUDY_E_PLAN.md`).

**E7 (added 2026-11-13, still before any seal).** E7 follows the one open lead of Section 15B.3. It gets its own test sets (seeds 65–66, 1–20 TU), because the 8th-order step's oracle advantage did not depend on the horizon. The policy is trained on the Study C training orbits, relabelled with the 8th-order step, and its model is chosen on E_select. Two exploratory pilots on Study D data came first (P24–P25 in `posthoc/README.md`, run on 2026-11-13; no Study E data used).

- **P24 (oracle step).** On the 300 trajectories of P23, the perfectly chosen 8th-order step was cheaper than DOP853 in Sundman time, itself perfectly tuned, on 97% of them. In total it needed 1.28× less work. This is the comparison P23 never made.
- **P25 (learned step).** A learned 8th-order step, with the same 25 inputs and the same small model as the RK4 policy, needed **2.57× less work in total than the RK4 policy**. At equal risk it beat RK45 and DOP853 in physical time (1.95, 1.63) and RK45 in Sundman time (1.74). Against **DOP853 in Sundman time it tied**: ratio of totals 1.016, 95% interval 0.994–1.040, cheaper on 89% of trajectories.
- **A catch.** "Equal risk" here means equal overall. The policy's failures sat at short horizons (3.8% at 1–3 TU vs about 0.1% for the solvers), the solvers' at long ones (about 10.6% at 8–20 TU vs 4.9%). So the by-horizon numbers are not equal-risk comparisons. Whether Study E calibrates per horizon band is a decision for before the seal.

In plain words: moving the learned step from RK4 to an 8th-order method closes the gap to the best adaptive solver, from "needs 2.4× more work" to "about the same". These are one exploratory run on one dataset, not a test; E7 is where it gets tested.

**An open decision, and what the numbers say about it.** The draft proposes testing E1 against the adaptive solvers in Sundman time as well as in physical time, with the rule that a test made of several parts passes only if every part passes (the *intersection–union* rule, Section 10.2). It is marked "proposed, decide before the seal". Look at Section 15B.3 before deciding. At 1–3 TU the ratio of totals against Sundman-time DOP853 was 0.85, which is below 1: that solver needed less work than the policy. If that comparator is included, the data we already have point towards E1 failing. That would be an honest result, but it changes what Study E can claim, so the choice has to be made before the seal and not after seeing the answer. (The last four sentences are my reading of the numbers, not something the draft says.)

---

## 15B. The second round of checks (2026-11-03/04)

**Why.** A reviewer-style read of the whole project (`DEFENSE_PLAN.md`) found that the weak points were mostly **sentences that claimed more than the data** plus **one missing comparison**. All of it was then run or fixed (`DEFENSE_STATUS.md`). The rules were the same as in Section 15A: definitions written into `posthoc/README.md` (P10–P23) before running, sealed numbers untouched, every new number labelled post hoc or exploratory.

### 15B.1 Sentences that were corrected

| Old claim | Why it was wrong | Now |
|---|---|---|
| "A fixed step in physical time **never** beat adaptive integration" | True only for Study B's slow, uncompiled model. Charged nothing for the model, Study B's policy would have won 27% against certified RK45 (though it still used about 11× as many evaluations in total). Study C's compiled physical-time policy won 31.6%. | "0/14,161 with an uncompiled model, 31.6% with a compiled one, and its failures were severe (median 12× the tolerance)" |
| Title: Sundman time "recovers most of the cost gap" | Most of the cost drop came from other changes (Section 15B.2), and the policy loses in total | New title: "*…loses to adaptive integration…; in Sundman time it wins on most trajectories but not in total cost*" |
| FTLE "changed neither prediction quality" | Study A's test was too small to tell: its best guess was a large loss. Adding FTLE cuts a third of the remaining error in Study D | "FTLE carries real information but costs an integration" |
| Certificate "below 5%" | Only for trajectories that don't crash or escape (Section 8.2) | Said next to every use |
| Per-stratum safety reported without its cost | Adaptive solvers certified per stratum the same way win on most trajectories (policy 42.5% / 20.7%) | Added to abstract and conclusions |

The manuscript also gained a real **S1 Appendix** (8 pages: the dated deviations of every study, reference checks, label details, edge cases, the strata rule, IAS15 and more), p-values at their resampling floor are marked in Table 2, and "reference-divergent" became "terminating" to avoid confusion with "the two references disagree".

### 15B.2 Who gets credit for the improvement? (the waterfall, P11)

Averaged over **all** trajectories (failures included), the cost gap against oracle-tuned RK45 went:

| Step | Mean gap (dex) | What changed |
|---|---|---|
| Study B policy | +1.39 | – |
| Study C, physical time | +0.47 | compiled small model, periapsis features, end-to-end certificate |
| Study C, Sundman time | +0.20 | the new clock |

So about **0.9 dex** came from the other changes and **0.27 dex** from Sundman time. In *shares*, though, the clock is decisive: 0% → 31.6% → 71.4% against equal-risk RK45. Both statements are true; they measure different things.

### 15B.3 Could an adaptive solver use the same tricks? (P20–P23, exploratory)

These are new runs on the Study D replication set, not preregistered tests.

| Comparator (tuned to the policy's risk) | Policy cheaper on | Solver's total ÷ policy's total |
|---|---|---|
| RK45, physical time (the headline) | 71.6% | 0.80 |
| RK45 **in Sundman time** | 67.6% | 0.72 |
| DOP853 **in Sundman time** | **45.7%** | **0.41** |
| RK45 with a **learned tolerance** (same 25-input model, same overhead) | **34.0%** | 0.89 |
| LSODA (a multistep solver) | 70.3% | 0.85 |

**In words:** the time transformation makes adaptive solvers cheaper too, and DOP853 in Sundman time needs well under half the policy's work overall (ratio of totals 0.41). The short-horizon advantage does not carry over to it either: at 1–3 TU the policy keeps a total-cost edge over Sundman-time RK45 (1.31) and over RK45 with a learned tolerance (1.16), but **not** over Sundman-time DOP853 (0.85, below 1). And if you give RK45 the same machine-learning trick, predicting its *tolerance* instead of a fixed step, RK45 wins two thirds of the trajectories at the same failure rate. **The learned component is useful; a fixed step is the wrong place to put it.**

**One open lead.** Is the negative result just about RK4? With the step chosen using perfect knowledge (an "oracle"), an **8th-order fixed step in Sundman time** was cheaper than the oracle-tuned RK45 on **297 of 300** trajectories and DOP853 on **294 of 300**, while the oracle RK4 step managed 127. That is a best case, not a working method: a learned step loses about as much again to prediction errors, the safety scale and overhead. It is the most promising thing to test in Study E.

Three details of this result (from `posthoc/results/rk8s_*`). (1) The horizon matters for RK4 but not for the 8th-order step. Against oracle-tuned RK45, the oracle RK4 step was cheaper on 73% of the 126 trajectories at 1–3 TU, 30% of 87 at 3–8 TU and only 10% of 87 at 8–20 TU, while the oracle 8th-order step was cheaper on 98%, 100% and 100%. (2) The 8th-order step cost a median of 0.27 dex less than the oracle RK4 step (a factor of about 1.9). (3) On 40 of the 300 trajectories the 8th-order search reached the top of its range (the largest fictitious step tried, 4), so the best step there is at least that large. The plan notes that this makes the 8th-order cost an upper bound, which works against the 8th-order method and not for it. The result was spot-checked against a fresh reference.

### 15B.4 Checks that held up

- **Equal risk really is equal on test data**: policy 4.39% failures vs RK45 4.11%; the McNemar test finds no difference (p = 0.24). The preregistered *certified* solvers were much safer (1.3–1.9%), which is why the 82.4% share against them was optimistic; Table 3 now shows the solvers' failure rates and puts the fairer "symmetric" rows first.
- **Multiple testing**: correcting over all 15 tests of Studies B–D at once changes no verdict (the closest, C3, goes to p = 0.030).
- **The timescale law is not a horizon effect**: controlling for the horizon, the rank correlation stays 0.90 (and drops to 0.02 in Sundman time).
- **D4 would pass at 5%** as well as at the preregistered 7% (p = 0.021).
- **Overhead**: it is only about 1% of the policy's total; the 1–3 TU advantage survives doubling it (1.40) but not ×10 (0.99).
- **Bootstrap intervals** barely change with the more careful BCa method (≤ 0.004).

### 15B.5 Checks that weakened something

- The policy's failures are **milder** than RK45's on three of the four evaluation sets (median 1.7× vs 2.5× the tolerance), **but not on the Study D lock set** (2.5× vs 2.1×).
- **Mixture dial**: the share falls steadily as near-Moon starts become more common, from 75% (none) to 55% (all); the ratio of totals never rises above parity. The 55/20/25 sampling mix was fixed in the very first commit, before the Study A seal, but it isn't written in the Study A preregistration.
- **Periodic orbits and "out of distribution"**: measured as distance to the nearest training examples, the Fig 10 grids and most periodic families are far from the training data, but the **axial and halo orbits are inside it and the axial ones still fail 96–100%**. A simple "am I far from my training data?" gate (Study E's E5) would not catch them.

### 15B.6 A rule to know before any science fair

The Society for Science guidance on generative AI (Oct 2025) says AI must **never** write the first draft of the research plan, abstract, paper, poster or conclusions; AI-written **code** is allowed with explicit citation and a log of the prompts. The manuscript, poster text and this document were drafted with AI. For a fair, the student has to write the abstract, poster and paper in their own words, and keep the AI transcripts as the prompt log.

### 15B.7 The poster files: only one is corrected

The project folder holds several copies of the poster, and they do not all say the same thing. I searched each one for the old wording.

| File | Corrected? |
|---|---|
| `poster/cr3bp_poster_v4.js` (changed Nov 4) | **Yes.** Its claim text now says "0 of 14,161 uncompiled, 31.6% compiled" and that changing the clock "won on 71% of trajectories but not in total". |
| `poster/cr3bp_poster_v3.js`, `cr3bp_poster_48x36.js` (top level), `affinity_bridge/poster.js` | **No.** Each still has the old title wording "recovers most of the cost gap". v3 and `cr3bp_poster_48x36.js` also still say "0 of 14,161" without the word "uncompiled". |
| `CR3BP_Poster.af` (Oct 31) and `poster/cr3bp_poster.af` (Nov 1) | **No.** These Affinity files were saved before v4 existed, and `DEFENSE_STATUS.md` says they were not touched. The v4 script has to be re-run in Affinity. |

None of the poster files is tracked by git (since 2026-11-13 they are listed in `.gitignore`, so that the Study E seal can require a clean tree), so there is no history to go back to if one is overwritten. Before printing, make sure the file you print was produced from v4. Remember the rule in Section 15B.6: the final poster words have to be the student's own.

---

## 16. Every figure and table, explained

Figures are made by `manuscript/scripts/figNN.py` (shared style in `style.py`), written as 600-dpi TIFFs for submission (`manuscript/figures/FigN.tif`) and PNG drafts (`manuscript/figures/draft/`), with the data behind each panel in `manuscript/figures/data/*.csv`.

### Fig 1. A close lunar pass compresses the local timescale by orders of magnitude

- **(A)** One trajectory near the Moon in the rotating $x$–$y$ plane, with L1 and L2 marked; the inset shows the whole Earth–Moon frame. Horizon 3.51 TU, closest lunar approach 0.018 DU (~6,900 km), $\Delta_\tau = 12.5$.
- **(B)** Along the same trajectory, on a log scale: the local timescale $\tau(t)$ (black), the physical step of the Sundman integrator at its largest passing fictitious step (blue), and the largest fixed physical RK4 step that passes (dashed).
- **What to see.** $\tau$ dips sharply around $t \approx 2.3$ TU (the flyby). The blue Sundman step follows that dip, so it is large most of the time and small only at the flyby. The fixed step (dashed) must sit at the bottom of the dip **for the whole run**.
- **How it was chosen**: by a fixed rule (the median-$\Delta_\tau$ trajectory among 141 candidates with one flyby below 0.05 DU, horizon 3–8 TU, uncensored labels), not cherry-picked. This is good practice.

### Fig 2. The fixed physical step's penalty tracks the timescale range; the Sundman step's does not

- **y-axis**: the paradigm gap = log₁₀(cost at the oracle fixed step / cost of oracle-tuned RK45). Above 0 = fixed step costs more.
- **(A)** Study B, 14,160 trajectories, hexagonal-bin density with medians and interquartile ranges in 12 bins. A near-straight rising band: $\rho = 0.91$.
- **(B)** Study C, same bins: grey squares (fixed physical step) keep rising ($\rho = 0.91$); blue circles (fixed Sundman step) stay flat near 0 ($\rho = 0.22$).
- **What it proves**: T1 (the mechanism) and C3(b) (its removal). This is the single most important figure of the project.

### Fig 3. Where the cost goes, before and after the time transformation

- Bar chart of the mean decomposition terms (Section 9.3) against oracle-tuned RK45, with bootstrap intervals.
- **(A)** Study B: paradigm +0.21, selection +0.03, prediction +0.38, safety +0.30, overhead +0.50, rounding −0.01, total **+1.41**.
- **(B)** Study C Sundman: paradigm −0.005, selection none, prediction +0.10, safety +0.06, overhead +0.07, rounding −0.01, total **+0.21**.
- The safety term has no error bar because one scale applies to all trajectories.

### Fig 4. The penalty and the failures concentrate at close lunar passes

- Heat maps over Hill regime (rows I–V) × closest lunar approach along the reference (columns <0.05, 0.05–0.17, 0.17–0.8, >0.8 DU), Study B.
- **(A)** median paradigm gap: largest (+0.66 to +1.10 dex) in the <0.05 column for regimes II–V; near zero or negative far from the Moon in regimes III–V.
- **(B)** violation rate: up to **48.3%** (regime IV, <0.05 DU) and **38.1%** (V, <0.05); 0.0–1.2% far from the Moon in most regimes.
- Note: the regime I × 0.05–0.17 DU cell is empty. A post hoc check showed this is physics, not a bug: in regime I a trajectory is trapped either close to the Moon (always within 0.047 DU) or in the Earth region (always beyond 0.169 DU), so nothing can fall in that band (Section 15A.5). The caption now says so.

### Fig 5. Where the Sundman-time policy is cheaper

- **(A)** Share of trajectories where each policy met the tolerance *and* cost less than RK45 and DOP853 at the oracle, certified and equal-risk tolerances. Grey hatched = physical-time policy; light blue dotted = one global Sundman step; dark blue = Sundman policy; white diamonds = the same policy on the Study D replication set. Dashed line at 50%. The Sundman policy clears 50% against certified and equal-risk RK45 and (barely) against DOP853 at certified and equal-risk tolerances; nothing clears the oracle.
- **(B)** Share cheaper than equal-risk RK45 by regime × horizon band, with counts. Near-yellow (> 89%) at 1–3 TU; dark blue (3–9%) at 8–20 TU in regimes I–III.

### Fig 6. Most of the saving comes from the time transformation and analytic periapsis information

- **(A)** Extra cost of each Study D arm relative to the 25-tree policy (Section 15.2). The linear model sits slightly left of 0; rules sit at +0.07 to +0.16; the global step at +0.274.
- **(B)** Exploratory D6: increase in test MSE when a feature group is removed and the model refitted. Blue = Sundman target, grey hatched = physical target. Periapsis and horizon matter; proximity and Jacobi don't (for Sundman); FTLE adds a little but is offline-only.

### Fig 7. A per-stratum certificate moves risk out of the tails

- **(A)** Violation rate in each of the 17 strata: filled blue = stratified policy (all ≤ ~3%), open light blue = one scale for all (up to ~10%), orange triangles = equal-risk RK45 (near 0 in most strata, but **24% and 30%** in two regime-I strata). Dashed line = 7% test level.
- **(B)** The same by the closest lunar approach actually made (not known before the run): the stratified policy still has ~5% below 0.17 DU.

### Fig 8. Faster in compiled wall-clock only at short horizons

- **(A)** Throughput ratio (comparator time / policy time; > 1 favours the policy) by horizon band, against DP5(4)-PI (primary), DP5(4) with SciPy's controller, and DOP853. Bars above 1 at 1–3 TU, around 1 at 3–8 TU, below 1 at 8–20 TU and overall.
- **(B)** Batch throughput ratio vs batch size (10³–10⁵) on 1 and 10 threads: all below 1.

### Fig 9. On periodic-orbit families outside the training population, the predicted step is too long

- **(A)** Violation rate per family: the policy (blue, with intervals) is near 100% for axial and vertical families, ~80% for Lyapunov, 33–47% for the rest; RK45 and DOP853 markers are lower but often far above the 10% line too.
- **(B)** Median log₁₀(predicted / oracle step): +0.31 to +0.53 for vertical, axial and Lyapunov (too long by a factor 2–3.4); near 0 for halo, DRO, butterfly, resonant.

### Fig 10. Failures lie on high-FTLE ridges

- **(A)** FTLE (colour) over a 110 × 110 grid of the surface of section at $C = 3.10$, with policy violations as orange circles: the violations cluster in the near-Earth region (around $x \approx 0$) and along bright FTLE ridges.
- **(B)** Violation rate by FTLE decile at three energies: ~0% in deciles 1–4, rising steeply to 84–97% in decile 10. AUC 0.92–0.93.
- **What a surface of section is**: instead of drawing full 3D paths, record each crossing of a plane ($y = 0$, moving in $+y$) as a dot in ($x$, $\dot x$). Regular motion leaves smooth curves; chaos leaves scattered dots. Here it is used as a grid of starting conditions.

### Table 1. Datasets

Explained in Section 11.2.

### Table 2. Preregistered confirmatory tests

All 19 tests (A-H1, A-H1′a, A-H2, A-H3, B-T1–T5, C-C1–C5, D-D1–D5) with hypothesis, result, Holm p and verdict. Summary: Study A 0 of 3 primary claims supported (H2 consistent with its guarantee); Study B 4 of 5; Study C 4 of 5; Study D 3 of 5. The seven results that were not supported or not met (A-H1, A-H1′a, A-H3, B-T3, C-C2, D-D2, D-D5) are reported with the same prominence as the positive ones, which is exactly right. Since Nov 3 the table also marks p-values that sit at the resampling floor (they are bounds) and notes that C3's threshold part did not replicate.

### Table 3. The Sundman policy across four independent evaluations

| | C confirmation | D replication | C lock | D lock |
|---|---|---|---|---|
| N | 12,192 | 12,253 | 4,020 | 4,074 |
| Violations | 4.4% | 2.7% | 4.3% | 2.5% |
| Cheaper than RK45 equal-risk | 71.4% | 71.6% | 70.0% | 70.3% |
| Cheaper than DOP853 equal-risk | 50.6% | 51.3% | 49.4% | 48.9% |
| Median total cost | 235 | 255 | 251 | 260 |
| Certified scale | 0.872 | 0.800 | 0.872 | 0.800 |

**Four independent evaluations agree within a few points.** This consistency is the strongest evidence in the project that the headline is real and not a fluke of one dataset.

The manuscript's Table 3 now also carries **post hoc rows**, each labelled as such and placed next to the sealed rows:

| Post hoc row | C confirmation | D replication | C lock | D lock |
|---|---|---|---|---|
| Cheaper than RK45, certified symmetrically | 71.6% | 67.4% | – | – |
| Cheaper than DOP853, certified symmetrically | 51.8% | 44.9% | – | – |
| Median total cost, all trajectories | 251 | 267 | 267 | 272 |
| Ratio of totals, RK45 equal-risk | 0.84 | 0.80 | 0.85 | 0.85 |
| Ratio of totals, RK45 certified | 1.01 | 0.91 | 1.03 | 0.97 |
| Ratio of totals, DOP853 equal-risk | 0.69 | 0.67 | 0.70 | 0.69 |
| Ratio of totals, DOP853 certified | 0.78 | 0.70 | 0.80 | 0.75 |

These agree across the four evaluations too. What they consistently say is that **the policy wins most trajectories but loses on total cost** (Section 15A). A footnote also notes that the columns charged slightly different overheads (15.04 vs 16.09), and that this changes nothing material. Since Nov 3 the table also shows how often **the solvers themselves failed** at each tolerance (equal-risk RK45 about as often as the policy; the preregistered "certified" solvers far less, 1.1–1.9%), and lists the fairer symmetric rows before the preregistered certified ones.

---

## 17. Discussion: mechanism, literature, limits, novelty, impact

### 17.1 The mechanism, in one paragraph

A fixed physical step must be small enough for the fastest moment, so its cost scales with $T\max\tau^{-1}$. An adaptive solver's cost scales with $\int\tau^{-1}dt$. Their ratio is $\Delta_\tau$, and the measured slope of almost exactly 1 between $\log\Delta_\tau$ and the penalty is what this predicts. In Sundman time the step is proportional to $\tau$ automatically, so the only remaining question is "how fine must the fictitious step be?", a quantity with much less spread (0.33 vs 0.91 dex). The periapsis features supply the one piece of the future that matters most: whether and how closely the trajectory will pass a body before the end. Residual failures are what a two-body picture can't see: strong three-body effects on chaotic long-horizon trajectories, and close passes the osculating orbit does not predict. (The manuscript correctly labels this as an interpretation that the studies support but did not directly test.)

### 17.2 How it fits the literature

- **Learned step control during the run** (Dellnitz et al. 2023; Saz Ulibarrena & Portegies Zwart 2025) sees the trajectory and can correct itself; a pre-run choice cannot. The starting state fully determines the future, so what a pre-run choice lacks is not information but a **cheap summary** of what is coming; the periapsis features are such a summary for the next close approach.
- **Physics step criteria** (Pham, Rein & Spiegel 2024) and **time-transformed integrators** (Mikkola & Tanikawa; Preto & Tremaine; Wang & Nitadori) encode the same physics the rules and features rediscover. A 4-coefficient rule recovering ~3/4 of the saving is consistent with that.
- **Weak-baseline critique** (McGreivy & Hakim 2024: most of 76 "ML beats solver" papers used weak baselines). This project's headline is against RK45 at equal risk, which is fair but **not the strongest** baseline: it does not beat DOP853 at equal risk, either solver at an oracle tolerance, or compiled wall-clock over the full range. The project says this itself, which is its credibility. The post hoc checks (Section 15A) showed the baseline also needed **more tuning** than it got. Certified by the policy's own rule, or given a better absolute tolerance, the solvers cut the policy's advantage to about two thirds of trajectories, and DOP853 certified that way won on most of them. The manuscript now says so in its Discussion. The strongest baselines turned out to be ones no study had preregistered: adaptive solvers in Sundman time, and RK45 with a learned tolerance (Section 15B.3).

### 17.3 Limitations, in order of likely impact

1. **Distribution shift**: unsafe on structured orbits outside training (D5); risk concentrated in the high-$\lambda T$ tail (25%).
2. **Scope**: one system, one tolerance (100 km), one horizon range (1–20 TU). How conclusions move with tolerance is unknown (the planned D8 tolerance dial was not run).
3. **Synthetic population**: all shares are specific to the sampling mixture. A near-Moon + L1 + L2 mix alone gives 58% instead of 72% (post hoc).
4. **Endpoint-only error**: 2.6% of runs wrong mid-run.
5. **Excluded terminating trajectories** (19%): event type matched on 99.9% but event time (strict rule) on only 46.7%, cost there not compared, and the certificate only covers trajectories that survive.
6. **One machine, one compiler, no GPU** for timing, in one session started with a relaxed load guard.
7. **Model vs reality**: CR3BP vs ephemeris differs by 2,000–50,000 km.
8. **Not beaten**: oracle-tuned solvers and DOP853 at equal risk; and, post hoc, **not cheaper in total evaluations** over 1–20 TU (only at 1–3 TU). Also, adaptive solvers certified per stratum by the policy's own rule were cheaper than the stratified policy on most trajectories (the policy won 42.5% against RK45 and 20.7% against DOP853).
9. **Beaten by adaptive solvers given the same ideas** (exploratory, Section 15B.3): DOP853 in Sundman time and RK45 with a learned tolerance. These comparisons were run once, on one dataset, and were not preregistered.
10. **Seals not externally verifiable** (Section 10.1).
11. **Collisions checked only at step endpoints**: a long physical step could cross the Moon unnoticed. This can't happen to the Sundman policy (its steps shrink near the bodies), but it could hide collisions of the physical-time arms.

### 17.4 What is new (novelty), stated honestly

- **The question itself in this setting**: a *pre-run* choice of one step for the **spatial** CR3BP, with **all** costs charged (features and inference), a **finite-sample certified failure rate**, and **risk-matched adaptive comparators**. The manuscript says "we are not aware of" such a study; the project's literature review supports that, but "not aware of" is the right strength of claim.
- **A diagnostic decomposition** of the cost gap into paradigm / selection / prediction / safety / overhead, which shows *which* term binds and turns a negative result into a design.
- **The quantitative law** that the paradigm penalty scales with the timescale range ($\rho = 0.91$, slope 0.90), and its **removal** by Sundman time ($\rho$ 0.22), replicated.
- **Analytic periapsis features** as a free "lookahead" for a pre-run predictor.
- **Combining Learn-then-Test certification** (including per-stratum conditional certificates) with step-size selection in orbital mechanics. The project's literature review found no prior application of conformal / risk-control certification to timestep selection in orbital mechanics.
- **Methodological rigour unusual for the field**: four sealed preregistrations, independent seeds, locked sets, negative results reported in full, replication, compiled wall-clock check that exposed an evaluation-count illusion.

What is **not** new: the Sundman transformation (1913) and periapsis/Kepler mathematics are classical; gradient-boosted trees are standard. The novelty is in the **question, the diagnosis and the rigour**, not the ingredients.

### 17.5 Impact (realistic)

- **Practical**: a possible tool for **short-horizon (≤ ~3–8 TU; post hoc, cheaper in total only at 1–3 TU) Monte Carlo dispersion studies** from a known population, with per-stratum certificates and an adaptive fallback. The manuscript states this as a hypothesis for future testing, not a proven recommendation. That is the right framing, and since Nov 3 it carries a further caution: even at 1–3 TU, DOP853 in Sundman time was cheaper in total, so a user should compare against that first.
- **Scientific**: shows *why* pre-run step prediction fails in physical time (a structural, not a learning, limitation) and what fixes it. That knowledge transfers to other multiscale ODE problems with close encounters.
- **Methodological**: a worked template for how to benchmark an ML-for-numerics claim fairly (count all costs, risk-match, certify, preregister, replicate, check wall-clock, test out of distribution).
- **For a science fair**: the strongest selling point is not "ML beats solvers" (it doesn't, once the solvers get the same ideas) but "a rigorous, preregistered investigation that found the real mechanism and a partial fix, and honestly mapped where it fails."

### 17.6 Future work the manuscript proposes

Test tolerance and horizon dependence; train on or abstain for periodic-orbit families; replace the Sundman function with a regularizing Kustaanheimo–Stiefel transformation near the Moon; move to an ephemeris model and other mass ratios; certify a path error instead of an endpoint error; measure batched GPU execution. Since Nov 3 the list starts with a learned step for a **higher-order fixed-step method in Sundman time**, compared against adaptive solvers in the same time variable (Section 15B.3). Several of these (tolerance dependence, path error, abstention on periodic orbits, the short-horizon total-cost claim) are now written into the draft **Study E** preregistration (Section 15A.7).

---

## 18. The code, file by file

About 22,000 lines of Python. The reusable library is the package `orbit_ml_cr3bp/`; the top-level scripts are command-line entry points that call it. Every command writes a **timestamped run folder with a `manifest.json`** (inputs, code fingerprint, package versions), so any result can be traced back to the exact code and data.

### 18.1 How a study flows through the code (the pipeline)

1. **Generate data** – `generate_3d_data.py` (or `gen_part_d.sh`) → `data_3d.py` samples starting states, `physics_3d.py` computes the reference trajectory, `search_3d.py` finds the label for each integrator, `diagnostics_3d.py` adds explanatory columns (timescale range, closest approaches), `sundman_3d.py` gives Sundman labels → one CSV row per trajectory plus a `_metadata.json`.
2. **Fit and freeze a policy** – `policy_b_3d.py` / `policy_c_3d.py` (fit models, choose scale), `osculating_3d.py` (periapsis features), `compiled_model_3d.py` (compile the trees), `rules_d.py` (physics rules) → frozen `.joblib` files in `part_*/frozen/`.
3. **Seal** – `freeze_*.py seal` hashes code, preregistration, frozen models and data.
4. **Confirmatory run** – `run_confirm_*_3d.py run <step>` (refuses to run if a hash changed; each step once) → policy ledgers (one row per trajectory with every cost and outcome).
5. **Tests** – `confirm_*_3d.py` functions compute the preregistered statistics and Holm adjustment → `confirm_tests_confirm.json`.
6. **Figures and reports** – `make_figures_*.py`, `manuscript/scripts/figNN.py`, the `RESULTS_*.md` files.

### 18.2 The package `orbit_ml_cr3bp/`

#### Core physics

| File | What it does, simply |
|---|---|
| `config.py` | All constants in one place: μ, Earth/Moon radii, DU = 384,400 km, 100 km tolerance, Lagrange-point positions, reference tolerances (10⁻¹³, max step 0.01), the 2 TU step ceiling, sampling weights (0.55/0.20/0.25), near-Moon sampling range, μ-dial systems, Sun–Earth parameters. Comments explain which values are legacy (Part A) and which are current. Since 2026-10-30 it also holds the adaptive solvers' shared settings (`ADAPTIVE_ATOL_RATIO` = 10⁻³, `MAX_ADAPTIVE_NFEV` = 2,000,000, the 18-value `RTOL_GRID_B`), which were previously copied in two files. |
| `physics_3d.py` | The heart. Numba-compiled CR3BP right-hand side; Euler, RK4 and symplectic leapfrog steps (kick–drift–kick in canonical momenta); fixed-step "endpoint" integrators with collision/escape/non-finite checks; variational equations and FTLE (variational and divergence versions); Jacobi constant; the DOP853 reference trajectory with dense output; Lagrange point locations; the state sampler (`generate_3d_orbit`); the table of evaluations per step (`EVALS_PER_STEP_3D`: Euler 1, RK4 4, leapfrog 1, rk4s 4). |
| `sundman_3d.py` | RK4 in Sundman time ("rk4s"): the $g(\boldsymbol{X})$ function with the harmonic sum and $\tau_c$ cap, the 7-variable augmented system, the landing rule, and the evaluation counter. |
| `osculating_3d.py` | The 13 analytic periapsis features (Section 4.10): eccentricity vector, periapsis radius, time to periapsis via Kepler's equation, hyperbolic and near-parabolic cases. Compiled with Numba; costs no RHS evaluation. |
| `richardson.py` | Halo-orbit initial conditions by **numerical differential correction** (Newton iteration on the state-transition matrix until the orbit closes). The docstring explains why the earlier analytic Richardson (1980) implementation was abandoned (an unused coefficient, a missing amplitude constraint, mixed normalizations). The function name `richardson_halo_initial_conditions` is kept for compatibility. |
| `diagnostics_3d.py` | Trajectory-based explanatory quantities (never model inputs): $\tau(t)$, $\Delta_\tau$ (physical and empirical), crossings, closest approaches, reference cross-check, Hill regime. |
| `guard_3d.py` | Runtime "step-over" guard: aborts a fixed-step run when $\Delta t/\tau$ exceeds a threshold and hands over to an adaptive solve; the fallback's cost and failures are counted. With the threshold at infinity it reproduces the unguarded kernels bit for bit (tested). |

#### Labels and data

| File | What it does |
|---|---|
| `search_3d.py` | Finds the label: the largest passing step. Contains the legacy Jacobi-drift search, the accuracy search, and the **locally monotone** search (`find_dt_for_integrator_accuracy_monotone_3d`) described in Section 7.1; the step ceiling rule (legacy vs physical). |
| `data_3d.py` | Feature extraction, sampling-mode dispatch, labelling (Jacobi or accuracy mode; Sundman labels), and the parallel dataset generator. Keeps reference-divergent orbits with their outcome instead of dropping them. Pipeline **v6** (for Study E) is v5 without the FTLE integration; the FTLE column is left empty (NaN) and its status column says "not computed" (code −99). |
| `sweeps_b_3d.py` | Horizon sweep: the same starting states relabelled at several horizons (Study B, T5). |
| `mu_dial_3d.py` | The μ-dial experiment: sampling scaled by the Hill radius so different mass ratios are comparable; dimensionless features. |
| `lookahead_3d.py` | E3c: features from a cheap partial RK45 over the first fraction of the horizon, and their cost. |

#### Study A-era modelling and audits

| File | What it does |
|---|---|
| `modeling_3d.py` | Hurdle models (random-forest classifier + RF or gradient-boosted quantile regressor), cost-normalized integrator choice, grouped spatial train/test split, evaluation metrics, older diagnostics carried over from 2D (chaotic hold-out, ablation, dataset-size study, speed comparison, FTLE correlation, RKF45 comparison, violation audit, safety-factor fitting). |
| `audit_3d.py` | Position-error audits and the original **LTT certification** (`calibrate_scale_ltt`, Bonferroni or fixed-sequence), Clopper–Pearson intervals, dense sweeps, tolerance sensitivity, halo validation, calibration stability. |
| `fair_baseline_3d.py` | Matched-accuracy comparison against SciPy RK45/DOP853: per-orbit oracle rtol by bisection, certified rtol, equal-risk rtol, prediction-cost measurement. The docstring warns that SciPy-vs-compiled wall times are not fair (the reason D2 was done properly later). |
| `rigor_3d.py` | Multi-seed evaluation with intervals, cross-μ transfer (Earth–Moon ↔ Sun–Earth), replication across datasets, feature ablation. |
| `error_model_3d.py` | Alternative approach (H5): learn the error as a function of step, then *choose* the step from a conformal upper bound; also the convergence-order study (H7). |
| `proxy_analysis_3d.py` | How well Jacobi drift predicts position error (it often gives false assurance). |
| `outcome_3d.py` | Crash/escape classifier (H6), with the caveat that FTLE "has seen" the crash. |
| `validation_3d.py` | Tolerance sensitivity, dense-sweep validation, accuracy vs reference, halo validation. |
| `fidelity_3d.py` | CR3BP vs real ephemeris (CAPSTONE): frame conversion and drift by lag. |
| `ensemble_regime_3d.py` | Lock-step batch cost *model* (not a GPU measurement): how much lane time is wasted when lanes in a batch finish at different times. |
| `uncertainty.py`, `conformal_uncertainty.py` | Tree-ensemble spread and conformal intervals for the RK4 step (Study A era). |
| `baselines.py` | Simple physics baselines (Keplerian timescale etc.) for the stable step. |
| `plotting_3d.py` | Study A-era figures (classifier accuracy, regressor performance, Poincaré sections, heatmaps…). Not used for the manuscript figures. |
| `protocol_3d.py` | Research-integrity plumbing: SHA-256 of files, code fingerprint, dataset metadata checks, independent-dataset checks, the **locked test set** (lock, load, refuse to touch without `--locked`), freeze status, run manifests. |

#### Study B

| File | What it does |
|---|---|
| `policy_b_3d.py` | The Study B policy: fit the state-only models, certify the scale efficiently ("lazy" scan that gives exactly the same answer as the full chain), certify adaptive rtols and the equal-risk rtol, per-orbit oracle rtol search, run everything and build the per-orbit **ledger**. `usable_rows` now selects rows by `reference_outcome == "ok"` (same rows as before on every dataset, tested). |
| `failure_study_3d.py` | The **cost-gap decomposition** (P, S, Q, K, O, R; exact in log space), pre-specified cells (regime × closest approach), failure taxonomy, risk ledger, bootstrap intervals, the "lever surface" used in Study C planning, and the Sundman version of the decomposition. |
| `experiments_b_3d.py` | Helpers for exploratory experiments: pinball loss, learning curves, reliability, kNN out-of-distribution score, the physics-rule baseline, AUC, guard tables. |
| `confirm_b_3d.py` | Statistics for T1–T5 and Holm adjustment. Pure functions (read nothing from disk) so they could be dry-run before the seal. |

#### Study C

| File | What it does |
|---|---|
| `policy_c_3d.py` | The Study C policies: quantile regressors, the leapfrog gate (tested, then dropped), **end-to-end fixed-sequence certification** of one scale for the whole deployed pipeline, abstention flags, importance weights for the enriched training set, fold splitting, evaluation and total-cost summaries. |
| `compiled_model_3d.py` | Exports scikit-learn gradient-boosted trees to flat arrays and evaluates them in Numba; computes features in compiled code; **measures overhead** in RHS-equivalents. Also the compiled linear-model policies used in D1. A test asserts the compiled predictions equal scikit-learn's to 10⁻¹². |
| `doomed_3d.py` | Runs the policy on the reference-divergent ("doomed") orbits and scores outcome agreement; the free physical crash flag. |
| `endpoint_audit_3d.py` | Re-runs trajectories with 50 checkpoints to measure mid-run errors (the 2.6% result). |
| `confirm_c_3d.py` | Statistics for C1–C5. |

#### Study D

| File | What it does |
|---|---|
| `rules_d.py` | The physics rules (features $\log T$, Hill regime, $\tau_p$, $r_p$, $C - C_{L_1}$), linear quantile fitting, compiled evaluation and overhead measurement; the 33-feature linear model. |
| `deploy_d.py` | Lean deployment of any Sundman step policy (model, rule or constant): certify one scale or one per stratum with Study C's exact rule, run, and score against comparators; cached reference positions. A test checks it reproduces a Study C ledger orbit for orbit. |
| `strata_d.py` | The 17 free strata (lunar class, regime group, horizon band), the fixed merge rule, per-stratum certification at δ/G, and the stratum table. |
| `periodic_d.py` | D5: loads JPL families, selects members, checks closure, builds runs at 1–20 periods, double references, runs the frozen policy and comparators. Handles JPL's slightly different units. |
| `compiled_solvers_d.py` | D2: compiled Dormand–Prince 5(4) (with SciPy's controller, which reproduces SciPy's evaluation counts exactly, and with the PI controller), compiled DOP853 from SciPy's own coefficients, the compiled Sundman policy (a "fast" version with one reciprocal per body and no memory allocation), parallel batch versions, certification of compiled rtols, and timing. |
| `exploratory_d.py` | D6 grouped importance, the symplecticity check, D7 chaos slice helpers, dense sweeps for the landing analysis. |
| `confirm_d_3d.py` | Statistics for D1–D5. |

#### Post hoc checks and Study E groundwork (added 2026-10-30)

| File | What it does |
|---|---|
| `posthoc_metrics.py` | The post hoc statistics (Section 15A): share, ratio of totals and mean log ratio with bootstrap intervals; breakdowns by horizon, regime and family; re-weighting to the declared sampling mixtures; the "share of the saving recovered" used for "three quarters". Pure functions, unit-tested. |
| `symmetric_cert.py` | Certifies an adaptive solver's tolerance with the policy's own rule (fixed sequence, full δ) on the fine grid 10⁻¹⁰…10⁻³ (0.125 dex apart); the absolute-tolerance variants; the equal-risk search; running SciPy solvers over a set of rows. |
| `collision_segment.py` | Segment–sphere test: does the straight line between two integrator steps pass through the Earth or Moon? Not used by any sealed code; meant for Study E. |
| `defense_metrics.py` (Nov 3) | Statistics of the second round (Section 15B): the McNemar test, failure severity, partial rank correlation, BCa intervals, cost concentration, the near-Moon mixture dial, resampling-floor flags. Pure functions, unit-tested. |
| `defense_runs.py` (Nov 3) | The new comparators: SciPy RK45 / DOP853 in Sundman time (stopping exactly when physical time reaches $T$), RK45 with one tolerance per trajectory, and a fixed-step 8th-order Runge–Kutta in Sundman time with its oracle-step search. |

### 18.3 Top-level scripts

| File | Role |
|---|---|
| `generate_3d_data.py` | Command line for generating a labelled dataset (number of orbits, label mode, pipeline version, seed, workers, tolerance, $\tau_c$, sampling weights). |
| `gen_part_d.sh` | Generates the four Study D datasets (seeds 52–55). |
| `run_rigor_suite_3d.py` | **Study A runner**: `gate` (development rehearsal), `final` (the one locked evaluation), `audit`, `ablate`, `replicate`, `mu`, `crash`, `proxy`, `errmodel`, `convergence`, `fidelity`, `lock`. |
| `freeze_snapshot.py` | Study A seal (`CODE_SNAPSHOT.json`, `snapshots/code_<hash>.tar.gz`, `FREEZE.json`). |
| `run_failure_study_3d.py` | Study B `policy` command (fit, certify, run, ledger) plus analysis commands (`decompose`, `e1`, `e3a`, …). |
| `run_failure_experiments_3d.py` | Study B exploratory experiments E4, E5, E9, E10 (and more). |
| `run_confirm_b_3d.py`, `freeze_b.py` | Study B confirmatory runner and seal. |
| `make_sweep_b.py`, `make_mu_dial_b.py` | Build the horizon-sweep and μ-dial datasets. |
| `make_figures_b.py` | Study B figures (waterfall, paradigm, where-map, step-over, information floor, horizon dial). |
| `run_part_c_3d.py` | Study C commands: `lever` (where could ML win for a given overhead?), `e3c_paired` (the errata re-analysis), `policy_c`, `freeze_policy`, `apply_policy`, selection, pilot, doomed, audit… |
| `run_confirm_c_3d.py`, `freeze_c.py` | Study C confirmatory runner and seal. |
| `make_figures_c.py` | Study C figures (bars, paradigm, wins, cells). |
| `run_part_d_3d.py` | Study D commands: `refs`, `overheads`, `select_rules`, `freeze_rule`, `freeze_lin33`, and the D1–D5 computations. |
| `run_confirm_d_3d.py`, `freeze_d.py` | Study D confirmatory runner and seal. |
| `run_explore_d_3d.py` | Study D exploratory commands: `d6`, `d7`, `d10`, `d12`, `d4b`, `symplectic`, `ias15_prep`, `ias15_combine`. |
| `ias15_baseline_d.py` | Runs IAS15 in REBOUND (inertial frame) on prepared trajectories; uses a different Python that has REBOUND. |
| `fetch_jpl_d.py` | Downloads the JPL periodic-orbit families and saves the raw responses with SHA-256 hashes. |
| `make_figures_d.py` | Study D figures (d1, d2, d4, d5, d7). |
| `validate_on_capstone.py` | Pulls CAPSTONE's real trajectory from JPL Horizons and converts it to CR3BP units (fidelity check). |
| `run_fair_baseline_3d.py` | Small CLI around `fair_baseline_3d.py` (Study A era). |
| `smoke_test_3d.py` | A quick smoke test, dated Oct 13, before the sealed studies. |
| `archive/study_a_tools/` | Older tools dated Oct 13, moved here on 2026-10-30 with their git history kept: `run_3d_analysis_from_data.py` (the original analysis script), `cost_analysis.py` (a training-cost break-even calculator), `generalization_tests.py` (region-transfer tests), `phase_space_analysis.py` (phase-space FTLE grids). No Study B–D code uses them; a README says how to rerun them. |
| `run_posthoc.py` | The post hoc checks of Section 15A, one command each: `aggregates`, `fig4_check`, `three_quarters`, `symmetric`, `atol`, `d4b_fine`, `compiled`, `seeds`, and since Nov 3 the 14 commands of Section 15B (`risk`, `waterfall`, `overhead`, `concentration`, `timescale`, `mixture_curve`, `multiplicity`, `bca`, `refs`, `ood`, `sundman_adaptive`, `learned_rtol`, `multistep`, `rk8s`). Each writes a timestamped folder in `posthoc/results/` and first checks it reproduces the sealed numbers. |
| `reproduce.sh` | One-command rebuilds: `env` (pinned conda environment from `environment.lock.txt`), `tests`, `b`/`c`/`d` (study figures), `manuscript` (Fig 1–10 and the PDF), `posthoc`, `check` (seal checks). Never reruns a confirmatory test. |

### 18.4 Tests (`tests/`)

| File | What it protects |
|---|---|
| `test_phase0.py` | Study B preconditions: FSAL leapfrog is bit-identical to the plain version, the evaluation table, Hill thresholds, step-ceiling modes, a circular orbit has $\Delta_\tau = 1$ (uniformly paced), legacy labels reproduce, convergence orders 1/2/4. |
| `test_failure_study.py` | The decomposition identity is exact (synthetic data), the cell-merge rule, the fast "lazy" scale search equals the full chain, the oracle-rtol search matches the original. |
| `test_part_c.py` | 30 tests, e.g.: fixed-sequence certification's false-certification rate is ≤ δ; osculating periapsis matches a two-body integration; a receding hyperbola has no future periapsis; compiled trees equal scikit-learn; Sundman RK4 with constant $g$ is classical RK4 and converges at 4th order in $\Delta s$; the Sundman step shrinks near a body; the guard at infinite threshold reproduces unguarded runs; doomed-orbit agreement rules; endpoint-audit trajectories match the endpoint kernels. |
| `test_part_d.py` | Compiled DP5(4) with SciPy's controller reproduces `solve_ivp` exactly; the DOP853 port matches SciPy's evaluation counts on almost every orbit; the PI controller is accurate; a rule fit recovers a planted linear law; the deploy runner matches Study C's integrator; strata cover every row; periodic-orbit references are stable. |
| `test_posthoc.py` | 13 tests: the post hoc statistics match hand calculations and the sealed share; mixture weights and bootstrap behave; the solvers' new certification uses the same rule as the policy's on a grid at least as fine (and the sealed grid fails this); the equal-risk search matches the sealed one. |
| `test_defense.py` | 12 tests (Nov 3): the new statistics match hand calculations; the Sundman-time adaptive solve ends exactly at $T$ and agrees with a tight physical solve; the 8th-order fixed step converges at 8th order; the RK4 helper reproduces the sealed kernel. |
| `test_hardening.py` | 5 tests: a collision between step endpoints is caught; the new `usable_rows` rule picks the same rows on every dataset; a near-parabolic periapsis edge case; pipeline v6 labels equal v5's; the solver settings are defined in one place only. |

### 18.5 The manuscript folder

`main.tex` (PLOS ONE template), `references.bib` (DOIs checked against Crossref on 2026-10-29 per its header), `plos2025.bst`, `main.pdf`/`main_draft.pdf`, `scripts/fig01–fig10.py` + `style.py`, `figures/` (TIFF), `figures/draft/` (PNG), `figures/data/` (CSV behind every panel), `figures/README.md` (build instructions). Added 2026-10-30: `scripts/si_tables.py`, which writes the supporting tables S1–S4 as CSV files into `si/` from the result files (nothing typed by hand; S3 includes bootstrap intervals on the rule coefficients, S4 the post hoc summaries), and `CITATION_AUDIT.md` (Section 19, note 21). Added Nov 3: `si/S1_Appendix.tex` and `.pdf` (the secondary methods and every study's dated deviations) and 23 more S4 CSV files for Section 15B.

### 18.6 Records and logs

| File | Purpose |
|---|---|
| `PREREGISTRATION.md`, `part_b/PREREGISTRATION_B.md`, `part_c/PREREGISTRATION_C.md`, `part_d/PREREGISTRATION_D.md` | The sealed plans, each with a dated "Deviations" section. |
| `FREEZE.json`, `FREEZE_B.json`, `FREEZE_B.v1.json`, `FREEZE_C.json`, `FREEZE_D.json`, `CODE_SNAPSHOT.json` | The seals (hashes). `FREEZE_B.v1.json` is the first Study B seal, replaced *before* any confirmatory step because of a runner-folder bug. |
| `confirm_log_b/c/d.jsonl`, `final_eval_log.jsonl` | One line per confirmatory step, proving each ran once. |
| `archive/study_a/RESULTS_REPORT.md` (Study A, ~440 KB; moved from the root), `STUDY_A_SUMMARY.md` (its executive summary), `part_*/RESULTS_PART_*.md`, `part_b/RESULTS_E3C_MU_DIAL.md`, `part_d/NARRATIVE.md` | Results write-ups. |
| `posthoc/README.md`, `posthoc/results/` | The post hoc definitions (committed before any of them ran) and their run folders. |
| `part_e/PREREGISTRATION_E.md` | The Study E draft (not sealed). |
| `poster/`, `affinity_bridge/`, `cr3bp_poster_48x36.js`, `CR3BP_Poster.af` | Poster scripts, figures and Affinity files (not tracked by git). Only `poster/cr3bp_poster_v4.js` carries the corrected claims (Section 15B.7). |
| `DEFENSE_PLAN.md`, `DEFENSE_STATUS.md` | The Nov 3 review of every claim and the record of what was done about each item (Section 15B). |
| `environment.yml`, `environment.lock.txt` | The software environment, with exact package builds pinned. |
| `part_*/PART_*_LOG.md` | Day-by-day decision logs. |
| `CHANGES.md` | Every code change with dates. |
| `LITERATURE_REVIEW.md` | Literature review with verification status. |
| `MANUSCRIPT_PLAN.md` | PLOS ONE requirements, claims and wording rules, section plan, figure specs. |
| `part_d/references.bib` | Citation audit for Study D (19 entries, each tagged with what was opened). |
| `part_*/frozen/`, `part_*/selection/`, `part_*/results/` | Frozen models/rules, selection-stage outputs, copies of the sealed results. |
| `out/`, `out_b/`, `out_c/`, `out_d/` | Full run folders (untracked by git). |
| `archive/` | Pre-seal trial data and runs (Oct 15); since Oct 30 also the Study A report and the old Study A tools. |

---

## 19. Critical notes

You asked me to be critical. These are the problems and soft spots I found while reading. None overturns the main conclusions, but several should be fixed before submission or judging.

**Status as of 2026-11-06 (the project files last changed on 2026-11-04).** Many of these notes have since been dealt with (on Oct 30 and in the Nov 3–4 round, Section 15B). Each note ends with a **Status** line where something changed. Notes without one are unchanged.

### 19.1 Things that look wrong or inconsistent

1. **The abstract's "0 of 14,161" sentence is ambiguous, arguably misleading.** It says "A fixed physical step was cheaper than an adaptive solver on 0 of 14,161 trajectories." That is the *predicted, safety-scaled, overhead-charged policy*. A *perfect* fixed physical step was cheaper than oracle-tuned RK45 on 25.8% and than certified RK45 on 67.9% (Study B). A reader will take the abstract sentence as a statement about fixed steps in general. Suggest: "A learned fixed physical step, with its overhead and safety margin counted, was cheaper … on 0 of 14,161." **Status: fixed**; the abstract now uses this wording.
2. **Leapfrog cost in Study A is misstated.** The manuscript says "Studies A and B offered RK4 and leapfrog, with $e_{\text{RK4}} = 4$ and $e_{\text{leapfrog}} = 1$," and the Integrators paragraph says leapfrog needs one RHS evaluation per step. The sealed Study A snapshot (`snapshots/code_79a6455ddef8.tar.gz`, `physics_3d.py` line 1018) has `"leapfrog": 2`, and the current code comment confirms "Part A … charged 2." It does not change Study A's 0/996 conclusion (charging less would only help the policy, which still lost), but the sentence is factually wrong for Study A. **Status: fixed**; the manuscript now says leapfrog was charged 2 per step in Study A and 1 in Study B.
3. **Supporting information was referenced but did not exist yet.** S1 Appendix, S1–S3 Tables and S1 Data are cited throughout (non-monotone labels, periapsis edge cases, landing rule, per-stratum and per-family tables, dated deviations). I found no such files in the folder. The manuscript is not submittable until they exist. **Status: partly fixed.** S1–S4 Tables now exist as CSV files in `manuscript/si/`, generated by `scripts/si_tables.py`; S4 is new and holds the post hoc summaries. Since Nov 3 the **S1 Appendix exists** (`manuscript/si/S1_Appendix.pdf`); **S1 Data** (the deposit of ledgers and figure data) is still missing.
4. **Placeholders remain**: author affiliation, corresponding email, repository DOI. **Status: still open**, and so is the Zenodo deposit the DOI would point to. Also open since Nov 3: the **mentor acknowledgment and the contributions statement**, which only the author can write (a marked placeholder is in `main.tex`).
5. **Fig 4, regime I × closest approach 0.05–0.17 DU is empty** (n = 0) while the neighbouring cells have hundreds. It may be physical (regime-I near-Moon orbits that start beyond 0.05 DU may fall closer or crash and be excluded), but it is worth a 5-minute check, because a reviewer will ask. **Status: checked; it is physical** (Section 15A.5), and the caption now explains it.
6. **The Claude Project description is out of date** and would mislead anyone (including a judge or a new collaborator) who reads it: it describes random-forest hurdle models as the method, a 3-evaluation leapfrog, Richardson validation, FTLE as a default feature, and calls the project "well-positioned for … publication" without mentioning that the headline is a negative result plus a partial fix. Update it to the four-study story. **Status: still open**; it lives outside the repository, so only you can change it.

### 19.2 Claims that need careful wording when presenting

7. **Always say which comparator.** "71% cheaper" is against **RK45 at equal marginal risk**. Against DOP853 at equal risk it is a coin flip (50.6%, 51.3%), against DOP853 certified per stratum it loses (23.5%), against oracle-tuned solvers it loses badly (6–9%), and in total compiled wall-clock it is slower (0.885). Leading with "ML beats adaptive solvers" would be false; the project's own wording rule ("cheaper than RK45 at equal marginal risk … never faster than adaptive solvers") is correct. **Since the post hoc checks, add two more cautions.** "Cheaper on 71% of trajectories" must never be said as "71% cheaper" or "saves cost overall": in total it used 19–25% *more* evaluations, and it is cheaper in total only at 1–3 TU. And against solvers certified as carefully as the policy, the shares are about 67–72% (RK45) and 45–52% (DOP853). **Since Nov 3, one more:** against adaptive solvers given the same ideas (Sundman time, or a learned tolerance), the policy loses (Section 15B.3).
8. **"ML is needed" (D1) is only half true.** The trees beat the best rule by 18%, but a **linear model ties the trees**. The honest claim is "the periapsis information is needed; tree models are not."
9. **C3(a) failed to replicate** (50.6%, p = 0.11). Don't cite C3 as a clean replicated success; cite its mechanism part (ρ 0.22 vs 0.91), which did replicate.
10. **The original hypothesis did not survive** as a design principle (D6: proximity and Jacobi add nothing for the Sundman target; FTLE is unaffordable). Be upfront that the project's success came from abandoning it.
11. **The certificate is marginal by default.** Without per-stratum certification, near-Moon trajectories fail at ~13%. If anyone quotes "certified below 5%," they should add "on average over the sampled population."
12. **Shares describe a synthetic population.** A real mission's mix of regimes and horizons could give a very different share. **Status: quantified.** A near-Moon + L1 + L2 mix gives 57.8% instead of 71.6% (Section 15A.2).

### 19.3 Methodological soft spots (already acknowledged in the manuscript, but worth knowing)

13. The headline equal-risk comparator is matched to the policy's risk on 4,000 calibration rows; a tolerance that is both safer and cheaper might exist. **Status: checked post hoc.** A better absolute tolerance (0.1 × rtol) does make RK45 cheaper at equal risk (share 64.2%), and the certified comparators were handicapped by a coarser grid and a weaker rule (Section 15A.3–15A.4).
14. Overheads are wall-time ratios on one machine; RHS-equivalents depend on how fast the RHS is (18.3 vs 25–28 ns changed the same model from 15.04 to 16.09).
15. The preregistrations are internal, not external registered reports; the Study D seal did not require a clean git tree. The hashes make later changes detectable, but because the repository was never public, nobody outside can confirm *when* the seals were made (Section 10.1). **Status: stated in the manuscript and S1 Appendix;** Study E must be registered on OSF first.
16. One tolerance (100 km), one system, horizons 1–20 TU only; the tolerance dial (D8) and path-error label (D9) were not run. (Both are now planned as E2 and E3 of the draft Study E.)
17. Study A's H1′ was added after development data had been seen (properly recorded as a deviation).

### 19.4 Process and presentation notes

18. **AI-assistance disclosure.** The manuscript states that the analysis code was written with an AI coding agent and the text drafted with an AI model under the author's direction. That is honest and PLOS ONE has an AI policy, but check that the model names are exactly right, and check the **current ISEF / regional fair rules** on AI assistance and how it must be disclosed; I'm not certain what this year's rules require, and judges will likely ask what the student personally did. The student should be able to explain every equation and design choice in this document without notes. **Status: rules checked, action still needed.** The Society for Science's generative-AI table (Oct 2025) forbids AI-written first drafts of the abstract, paper, poster and conclusions, and allows AI-written code with citation and a prompt log (Section 15B.6). The model names in the manuscript still need the author's confirmation.
19. **Housekeeping.** Old pre-seal tools (Section 18.3) and the very large `RESULTS_REPORT.md` could be archived or summarized; the top level mixes four studies' scripts. Not urgent, but it would help a reviewer reproduce things. **Status: done.** The old tools and the Study A report are in `archive/`, with `STUDY_A_SUMMARY.md` at the root; `reproduce.sh` and a pinned environment were added (Section 15A.6).
20. **Citation mismatch for HiGHS.** The manuscript says the rules were fitted with the "HiGHS interior point solver~\cite{huangfu2018}", but `huangfu2018` is "Parallelizing the dual revised simplex method" (a simplex paper, not interior point). Either cite the HiGHS software/interior-point reference or say "HiGHS solver". Also verify which HiGHS method SciPy actually used. **Status: fixed.** SciPy's `highs-ipm` runs HiGHS's interior-point solver **IPX**, now cited separately as Schork & Gondzio (2020), with Huangfu & Hall kept for HiGHS itself.
21. **Literature review caveats.** `LITERATURE_REVIEW.md` flags several items as unverified (authors not captured, 2026 items abstract-only). The manuscript's `references.bib` claims DOIs were checked; make sure no unverified item from the review slipped into the manuscript. **Status: checked** (`manuscript/CITATION_AUDIT.md`). None of the flagged items is cited, except Dellnitz et al. (2023), whose authors and journal were then verified. Three other entries were checked against publisher or index records. The remaining 39 standard references were not re-resolved one by one; do that once before submission.

### 19.5 Added at the 2026-11-06 check

22. **Study C's seal date differs between the manuscript and the git log.** The manuscript gives UTC dates (Oct 27 for Study C, in the sentence at line 312 of `main.tex`). The git commit that records the Study C seal is dated Oct 26 in New York time. Both are correct (Section 3.4). Say "UTC" when you quote the dates, and consider adding "UTC" to that manuscript sentence. **Status: open.**
23. **Old poster files are still in the folder** (Section 15B.7). Only `poster/cr3bp_poster_v4.js` is corrected, and the Affinity files have not been rebuilt from it. **Status: open.**
24. **The copy of `DEFENSE_PLAN.md` attached to the Claude Project is the version from before the fixes.** It is dated Nov 3, lists the twelve P0 items as work still to do, and says S1 Appendix "does not exist yet". Most of those items are now fixed (Section 15B). The current record is `DEFENSE_STATUS.md` in the repository folder, which is not attached to the Project. Do not use the Project copy as a to-do list. **Status: open**; replacing it is a one-step job.
25. **Count the negative results correctly.** Table 2 of the manuscript has 19 tests. Seven were not supported or not met: A-H1, A-H1′a, A-H3, B-T3, C-C2, D-D2 and D-D5. (A-H2 is "consistent with the guarantee", and C3 is supported only narrowly.) An earlier version of this document said 18 tests and five negative results. **Status: fixed here.**

---

## 20. Glossary

| Term | Meaning |
|---|---|
| Adaptive solver | Integrator that changes its step during the run based on an error estimate (RK45, DOP853, IAS15). |
| AUC | Area under the ROC curve; probability that a random positive scores higher than a random negative (0.5 chance, 1 perfect). |
| Barycentre | Centre of mass of Earth and Moon. |
| Bonferroni | Splitting an error budget δ equally over m tests (δ/m each). |
| Bootstrap | Resampling the data with replacement to estimate uncertainty. |
| Calibration set | Data held out from training, used only to choose the safety scale or tolerance. |
| Canonical momenta | Hamiltonian momenta; in the rotating frame $p = (\dot x - y, \dot y + x, \dot z)$. |
| Censored label | A label that is only a bound (the search hit its limit). |
| Certified tolerance / scale | Chosen by the binomial LTT rule so the failure rate is ≤ α with confidence 1 − δ. |
| Clopper–Pearson | Exact confidence interval for a proportion. |
| Collision radius | Earth 6,371 km; Moon 1,737.4 km. |
| Conditional vs marginal guarantee | Holds in each subgroup vs only on average. |
| Coriolis force | Fictitious sideways force in a rotating frame. |
| CR3BP | Circular restricted three-body problem. |
| Dense output | Continuous interpolation of a solver's solution between steps. |
| Dex | log₁₀ unit; 1 dex = ×10, 0.1 dex ≈ ×1.26. |
| DOP853 | 8th-order adaptive Runge–Kutta with embedded 5th/3rd-order error estimates. |
| DP5(4), RK45 | Dormand–Prince 5(4) adaptive Runge–Kutta. |
| DRO | Distant retrograde orbit: a stable orbit around the Moon going "backwards." |
| DU, TU | Distance unit (384,400 km), time unit (≈ 4.348 days). |
| Equal-risk tolerance | The loosest adaptive tolerance whose failure rate equals the policy's. |
| Exchangeable | Calibration and test data are drawn the same way, so statistics transfer. |
| Fictitious time | The Sundman time variable $s$, with $dt = g\,ds$. |
| Fixed-sequence testing | Test hypotheses in a fixed order, stop at the first failure; no multiplicity penalty. |
| FSAL | First same as last: reusing the last evaluation of a step as the first of the next. |
| FTLE | Finite-time Lyapunov exponent: rate of exponential separation of nearby trajectories. |
| Gradient boosting | Adding trees sequentially, each correcting the previous errors. |
| Halo orbit | Periodic 3D orbit around L1 or L2 (used by many missions). |
| Hamiltonian | Energy function $H(q,p)$ generating the dynamics. |
| Hill radius | Region where the Moon's gravity dominates: $(\mu/3)^{1/3} \approx 0.159$ DU. |
| Hill regime | One of five energy classes I–V defined by the Jacobi constant. |
| Holm procedure | Step-down correction controlling familywise error over several tests. |
| Horizon $T$ | How long the trajectory is simulated (1–20 TU). |
| Hurdle model | Classifier ("is it feasible?") + regressor ("how much?"). |
| IAS15 | 15th-order adaptive integrator in REBOUND. |
| Intersection–union test | A test of a claim with several parts that must all hold. Its p-value is the largest of the parts' p-values, so it passes only if every part passes. |
| Jacobi constant $C$ | The conserved quantity of the CR3BP. |
| JPL Horizons / Periodic Orbits API | NASA JPL services for real ephemerides and catalogued periodic orbits. |
| Label | The "right answer" for training: largest passing step. |
| Lagrange (libration) points L1–L5 | Equilibrium points in the rotating frame. |
| Learn-then-Test (LTT) | Framework for certifying risk of a tuned predictor via hypothesis testing. |
| Leapfrog | 2nd-order symplectic kick–drift–kick integrator. |
| Learned tolerance | Using a model to predict an adaptive solver's tolerance for each trajectory, instead of predicting a fixed step (Section 15B.3). |
| Locked set | Data evaluated only once, at the very end. |
| Lyapunov orbit | Planar periodic orbit around L1 or L2. |
| McNemar test | Paired test of two methods on the same items that looks only at the items where exactly one of them failed. |
| Mean / eccentric anomaly | Angles locating a body along a Kepler orbit in time/geometry. |
| Monotone label | Largest step that passes along with all finer neighbours over two octaves. |
| Numba | Just-in-time compiler that turns Python numerical code into fast machine code. |
| Oracle-tuned | Tuned per trajectory using the true answer (not available in practice). |
| Order of a method | How fast error shrinks with step: $\propto \Delta t^p$. |
| Osculating orbit | The two-body Kepler orbit that matches the current position and velocity. |
| Overhead | Cost of features and model inference, in RHS-equivalents. |
| Oracle step | The best step for a trajectory, found using the true answer; a best case no real method can reach. |
| Paradigm gap $P$ | Penalty of the best possible single step vs adaptive. |
| Periapsis | Closest point of an orbit to the body. |
| Post hoc | Decided after the results were seen; can reveal problems but cannot confirm a claim the way a preregistered test can. |
| PI controller | Step-size controller using current and previous error. |
| Partial correlation | Correlation between two quantities after removing what a third one (here the horizon) explains in both. |
| Pinball (quantile) loss | Loss that makes a model predict a chosen percentile. |
| Preregistration | Public-in-advance (here: sealed internal) plan of hypotheses and tests. |
| Random forest | Average of many randomized decision trees. |
| Ratio of totals | Total solver evaluations ÷ total policy evaluations over a whole set (above 1: the policy is cheaper in total). Unlike the share, it is dominated by the expensive trajectories. |
| Reference solution | The high-accuracy "truth" (DOP853 at 10⁻¹³). |
| Terminating ("doomed", in code "reference-divergent") | Trajectory whose reference collides or escapes before $T$. |
| Regularization (of collisions) | Coordinate + time change that removes the $1/r$ singularity (Levi-Civita, KS). |
| RHS evaluation | One call of the force function; the cost unit. |
| RHS-equivalent | Other costs (model, features) converted into RHS-evaluation units by timing. |
| RK4 | Classical 4th-order Runge–Kutta, 4 evaluations per step. |
| Rotating (synodic) frame | Frame rotating with Earth and Moon. |
| Safety scale $s$ | Multiplier ≤ 1 on the predicted step. |
| Sampling mixture | The proportions in which starting states are drawn (55% near Lagrange points, 20% wide box, 25% near the Moon); every share depends on it. |
| Segment–sphere test | Checks whether the straight line between two integrator steps passes through a body, to catch collisions a long step would jump over. |
| Share cheaper | Fraction of trajectories on which the policy met the tolerance and cost less than the solver (the preregistered statistic). |
| SHA-256 | Cryptographic fingerprint of a file. |
| Spearman ρ | Rank correlation. |
| State-transition matrix Φ | Linear map from initial perturbations to final perturbations. |
| Stratum / strata | Subgroups defined before the run for conditional certification. |
| Sundman transformation | $dt = g(\boldsymbol{X})\,ds$: a clock that slows near the bodies. |
| Surface of section | Plane where crossings of trajectories are recorded as points. |
| Symmetric certification | Certifying the solvers' tolerance with exactly the same rule and grid fineness as the policy's scale, so neither side gets a more favourable tuning. |
| Symplectic | Preserves the geometric (phase-space) structure of Hamiltonian flow. |
| Throughput ratio | Total comparator time / total policy time (> 1 favours the policy). |
| Timescale $\tau$, range $\Delta_\tau$ | Local orbital time $\min(\sqrt{r_1^3/(1-\mu)}, \sqrt{r_2^3/\mu})$; ratio of fixed-step to paced work. |
| Tolerance (rtol, atol) | Adaptive solver's local error target. |
| Violation | Final error > 100 km, collision, escape, non-finite value or step cap. |
| Zero-velocity surface | Boundary of the energetically forbidden region. |

---

## 21. Questions a judge or reviewer is likely to ask

**"So does your ML beat standard solvers?"** Against RK45 tuned to the same failure rate, yes, on about 71% of trajectories in two independent datasets, in force evaluations. Against the stronger DOP853 it's a tie; against solvers tuned with knowledge of the answer, no; and in total compiled wall-clock time over 1–20 TU, no (it's faster only at short horizons). And winning most trajectories isn't winning in total: summed over all trajectories it used about 20% more evaluations than RK45, because it loses the long, expensive ones. It is cheaper in total only at 1–3 TU. Against solvers tuned as carefully as the policy (post hoc), the share drops to about two thirds against RK45 and below half against DOP853. And when the solvers get the same ideas (run in Sundman time, or with a learned tolerance), they beat it.

**"Isn't 'cheaper on 71% of trajectories' the same as 'saves money'?"** No. Think of a shop that is cheaper than its rival on 71% of items but far more expensive on the big-ticket ones: your total bill can still be higher. The study reports both numbers now, and the total is what a batch of simulations actually costs.

**"Did you change your results after seeing them?"** No. The sealed verdicts and numbers are unchanged. Later checks are labelled post hoc, their definitions were committed before they ran, and their numbers sit next to the sealed ones, including the ones that make the result look worse. The claim they leave standing (short horizons) goes to a new preregistered Study E on fresh data.

**"Couldn't an adaptive solver use Sundman time too?"** Yes, and that was tested (exploratory). DOP853 in Sundman time needs only 41% of the policy's evaluations. Giving RK45 a tolerance predicted by the same small model also beats the policy on most trajectories at the same failure rate. The lesson: the time variable and the periapsis information are the useful discoveries, and they help adaptive solvers most.

**"Is the negative result just about RK4?"** Possibly. With a perfectly chosen step, an 8th-order fixed step in Sundman time beat the oracle-tuned solvers on 98–99% of 300 trajectories. Whether a *learned* step can keep enough of that margin is the open question for Study E.

**"What did you do yourself, and what did the AI and your mentor do?"** This needs an honest, specific answer that matches the written contributions statement exactly; prepare it in your own words (Section 15B.6).

**"Why not just use an adaptive solver?"** For most purposes you should. The value of a pre-run step is a known cost per run, which helps batch scheduling; the study shows when that can be had cheaply (short horizons, known population) and when it can't.

**"What did the ML actually learn?"** Mostly physics. A four-number rule using the horizon, energy regime and the predicted periapsis radius gets three quarters of the saving; a linear model ties the trees.

**"Why did your original hypothesis fail?"** FTLE is the most informative quantity, but computing it requires integrating the trajectory, which is exactly what we're trying to avoid (it cost ~70 adaptive solves). After the time transformation, proximity and Jacobi add nothing; the horizon and periapsis information carry it.

**"How do you know it's safe?"** A binomial hypothesis test on held-out calibration trajectories certifies, with 90% confidence, that the average failure rate is ≤ 5% among trajectories that don't crash or escape before the end. We showed the average hides 13% failure near the Moon, and fixed it with per-stratum certificates (every stratum ≤ 3.1% on new data).

**"How do you know you didn't tune until it worked?"** Every study was preregistered and sealed with file hashes before the confirmatory data were touched; each confirmatory step ran once; negative results (seven of the 19 tests) are reported; the headline replicated on fresh data.

**"Is 100 km meaningful?"** It is an illustrative integrator-error tolerance inside the model. The model itself differs from the real Earth–Moon system by thousands of km, as the CAPSTONE comparison shows.

**"Why does it fail on halo/vertical/axial orbits?"** They are rare, highly structured states the random training sample barely contains, so the model predicts steps 2–3× too long. (Axial and halo orbits even look ordinary to the model, inside its training range, and still fail, so a simple "too unfamiliar" check would not catch them.) It is a distribution-shift failure; the integrator itself converges correctly when the step is small enough.

**"What's novel?"** The rigorous test of pre-run step selection in the spatial CR3BP with all costs charged and certified risk; the decomposition that locates the loss; the timescale-range law and its removal by Sundman time; analytic periapsis features as a free lookahead; and the methodology.

**"Your method lost. Why is this a good project?"** Because it answers a question that had not been tested fairly: every cost is charged (including the model's own), the failure rate is certified, and the comparators are tuned to the same risk. It also explains why a fixed physical step loses (its penalty follows how much the trajectory's pace varies, Spearman ρ = 0.91, slope 0.90), tests a fix on new data (71%, replicated), and then reports where the fix stops: total cost, DOP853, periodic orbits, and adaptive solvers that get the same ideas.

**"Did you pick the measure that makes the ML look good?"** The share of trajectories was chosen and justified before the data were seen. Total cost was added later (post hoc). It showed that the policy loses over 1–20 TU, and that result is in the abstract.

**"Is 'equal risk' really equal?"** On test data, yes within noise. Study C confirmation: policy 4.39% violations against 4.11% for equal-risk RK45 (McNemar test p = 0.24). Study D replication: 2.69% against 2.89% (p = 0.33). The policy's failures were also milder on three of the four sets (median 1.7× the tolerance against 2.5×), but not on the Study D lock set, where they were worse (2.50× against 2.13×).

**"Isn't 'timescale range against cost gap' circular?"** The timescale range comes from the path and the gap from the costs, so the link is not automatic. If accuracy rather than pace limited the step, the slope would not be close to 1. The slope is 0.90, and the rank correlation stays at 0.90 after taking out the horizon (post hoc). In Sundman time it falls to 0.02.

**"What does Sundman time do, in one sentence?"** It makes the clock run slowly near a body, so a constant step in the new clock is a small real step close to the Earth or Moon and a large one far away; it does not remove the singularity (Section 6).

**"Why should I trust code written by an AI?"** Because of what was checked: the compiled solvers reproduce SciPy's step counts, the integrators converge at their textbook orders, the cost decomposition is exact, and the seals show the code did not change after the dry runs (Section 18.4). The honest gap: an independent check written by the student by hand has not been done, and `DEFENSE_STATUS.md` leaves it undone because it is only worth something if the student writes it. `DEFENSE_PLAN.md` describes it as a short script that re-runs the Sundman RK4 step, the evaluation counter and the violation check on 100 ledger rows and matches the stored costs and errors (the column names are given there). It would defend the code and the student's understanding at once.

**"How does this connect to your earlier two-body paper?"** I could not check this against the 2D paper, so treat the answer as a draft. The 2D study judged how often its model picked the right integrator and a stable step (cost-normalized selection accuracy about 97.7% in the version that was reviewed). As far as I can tell from the project notes it did not compare cost with adaptive solvers, which is the question of this project. A two-body orbit also has a much simpler pace (none that varies at all if it is circular), whereas a three-body flyby squeezes the timescale by orders of magnitude; that second point is my reasoning from the timescale-range idea, not something either study tested. Check both statements against the 2D paper before using them.

---

## 22. The literature, paper by paper

Grouped by role. "What it gives this project" is my summary.

### Dynamics and mission context
- **Szebehely (1967), *Theory of Orbits*.** The classic CR3BP textbook: equations of motion, Jacobi constant, zero-velocity surfaces. Source of Eq 1–3.
- **Koon, Lo, Marsden & Ross (2011), *Dynamical Systems, the Three-Body Problem and Space Mission Design*.** Modern treatment of Hill regions, necks at L1/L2 and transport; source of the Hill-regime classification.
- **Zimovan-Spreen, Howell & Davis (2020).** Near-rectilinear halo orbits (the Gateway orbit family) and their stability; motivates why cislunar mission analysis needs many integrations.
- **Giorgini et al. (1996), JPL Horizons; JPL Three-Body Periodic Orbits API.** Data sources for the CAPSTONE fidelity check and the D5 periodic-orbit families.

### Time transformations and regularization
- **Sundman (1913).** Time transformation $dt \propto r\,ds$ in his three-body solution; the namesake of Study C's method.
- **Levi-Civita (1920); Kustaanheimo & Stiefel (1965).** Regularization in 2D and 3D: coordinates + time change that remove the collision singularity. Cited to say what the project's transformation does *not* do, and as future work.
- **Berry & Healy (2002).** Generalized Sundman $dt = c r^\alpha ds$ for high-eccentricity orbit propagation; the project's $g$ is the $\alpha = 3/2$ case near a body.
- **Mikkola & Tanikawa (1999); Preto & Tremaine (1999); Wang & Nitadori (2020).** Time-transformed leapfrog and algorithmic regularization keeping symplectic or time-symmetric structure; shows the idea is established in few-body codes.

### Integrators and step control
- **Dormand & Prince (1980).** The RK45 pair.
- **Hairer, Nørsett & Wanner (1993), *Solving ODEs I*.** DOP853, step controllers; the standard reference.
- **Gustafsson, Lundh & Söderlind (1988).** PI step-size control (the D2 primary comparator's controller).
- **Rein & Liu (2012), REBOUND; Rein & Spiegel (2015), IAS15.** The N-body code and its high-order adaptive integrator (exploratory baseline).
- **Pham, Rein & Spiegel (2024).** A timestep criterion from acceleration and its derivatives that recovers orbital and pericentre timescales; the "physics knows the step" idea the project's rules echo.
- **Boekholt & Portegies Zwart (2015).** Reliability of chaotic N-body simulations: conservation is not accuracy; use converged references. Justifies position-error labels against a tight reference.

### Chaos indicators
- **Shadden, Lekien & Marsden (2005); Gawlik et al. (2009).** FTLE and Lagrangian coherent structures (ridges of FTLE as transport barriers), including in the restricted three-body problem. Background for FTLE and the D7 ridge result.

### ML for numerical integration
- **Dellnitz et al. (2023).** Reinforcement learning chooses steps during integration (quadrature and ODEs).
- **Saz Ulibarrena & Portegies Zwart (2025).** Reinforcement learning for adaptive time-stepping in the chaotic gravitational three-body problem; the closest analogue (but in-run, general three-body).
- **Breen et al. (2020), "Newton versus the machine."** A neural network replaces the integrator in a restricted slice of the three-body problem; fails near close encounters.
- **McGreivy & Hakim (2024).** Of 76 papers claiming ML beats numerical PDE solvers, most used weak baselines or selective reporting. The project is designed to pass this critique.

### Statistics and ML methods
- **Angelopoulos, Bates, Candès, Jordan & Lei (2025), Learn then Test.** The risk-control certification framework behind Eq 8.
- **Nosek et al. (2018), "The preregistration revolution."** Why preregister.
- **Friedman (2001).** Gradient boosting.
- **Koenker & Bassett (1978).** Quantile regression (pinball loss).
- **Strobl et al. (2008).** Permutation importance is biased with correlated features; justifies drop-group refits in D6.
- **Clopper & Pearson (1934); Holm (1979); Efron (1979); Spearman (1904); Mann & Whitney (1947).** Exact proportion intervals, multiple-testing correction, bootstrap, rank correlation, rank-sum test.
- **Huangfu & Hall (2018), "Parallelizing the dual revised simplex method."** The paper behind the HiGHS optimization solver used for linear quantile regression. It is a simplex paper, so it is now cited for HiGHS itself (see note 20 in Section 19).
- **Schork & Gondzio (2020), "Implementation of an interior point method with basis preconditioning."** *Math. Program. Comput.* 12:603–635. The IPX interior-point solver inside HiGHS, which is what actually fitted the rules; added to the references on 2026-10-30.

### Software
- **Harris et al. (2020) NumPy; Virtanen et al. (2020) SciPy; Pedregosa et al. (2011) scikit-learn; McKinney (2010) pandas; Lam et al. (2015) Numba; Hunter (2007) Matplotlib.**

### Further reading flagged in `LITERATURE_REVIEW.md` (not all cited in the manuscript)
Richardson (1980) halo expansions (and why differentially corrected halos should be used instead); Tao (2016) and Soria-Carro & Akella (2026) on explicit symplectic integrators for non-separable Hamiltonians; Lu, Hernandez & Rein (2024) TRACE close-encounter switching; Flamant et al. (2020) and Chen & Tao (2021) on neural CR3BP surrogates and their limits; Roberts et al. (2017) on block cross-validation. Several entries there are marked *unverified*; check before citing.

---

*End of document. Every number above comes from the project's own files. It was first written on 2026-10-29, updated on 2026-10-30 (post hoc checks, code changes, Study E draft) and on 2026-11-05 (second round of checks), and checked again against the files on 2026-11-06, when the corrections in Section 19.5 were made. If results are re-run or the manuscript changes, this document will need updating.*
