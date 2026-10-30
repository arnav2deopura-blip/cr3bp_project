# Literature Review: ML for Integrator and Step-Size Selection in the 3D CR3BP

Compiled 2026-10-29 from a web literature search. **Verification status matters:** items marked *(unverified)* had incomplete author lists, venues, or were seen only in abstracts/listings. Check the publisher page before citing. 2026 items were verified from abstracts or listings only, not full text.

## Bottom line

- No published study found shows a learned, before-the-run choice of integrator and timestep beating a well-tuned adaptive solver (DOP853, IAS15) on cost at equal accuracy. The preregistered negative result is consistent with the literature.
- Closest competing work: learned *in-run* step control (Saz Ulibarrena & Portegies Zwart 2025; Dellnitz et al.).
- Gap: pre-simulation, per-initial-condition difficulty prediction for the *3D* CR3BP, tied to FTLE, distance to the primaries and Jacobi constant, with calibrated/conformal safety guarantees.

## Categories

1. [CR3BP Dynamics, Chaos Indicators and Regularization](#1-cr3bp-dynamics-chaos-indicators-and-regularization)
2. [Numerical Integrators and Reliability of Chaotic Integration](#2-numerical-integrators-and-reliability-of-chaotic-integration)
3. [ML for Orbital Dynamics: Step-Size Selection and Surrogates](#3-ml-for-orbital-dynamics-step-size-selection-and-surrogates)
4. [ML Methodology and Benchmarking vs. Classical Solvers](#4-ml-methodology-and-benchmarking-vs-classical-solvers)

Then: [Refinements](#refinements-suggested-by-the-literature) · [Gaps](#gaps-this-project-could-fill) · [Unverified leads](#unverified-leads)

---

## 1. CR3BP Dynamics, Chaos Indicators and Regularization

### Halo orbits and Richardson validation

- **Richardson, D. L. (1980). "Analytic construction of periodic orbits about the collinear points."** *Celestial Mechanics* 22(3), 241–253. DOI not verified (the DOI 10.1007/BF01229509 belongs to the companion note below; search Springer for this one).
  Third-order analytic halo solution about L1/L2/L3: your external-validation reference. Being only third order, its initial conditions drift off the true periodic orbit within about a period, so validate against *differentially corrected* halos, not raw Richardson states.
- **Richardson, D. L. (1980). "A note on a Lagrangian formulation for motion about the collinear points."** *Celestial Mechanics* 22, 231–236. https://link.springer.com/article/10.1007/BF01229509
  Background formulation for the third-order expansion.
- **Richardson, D. L. (1980). "Halo orbit formulation for the ISEE-3 mission."** *J. Guidance and Control* 3(6), 543–548. *(citation seen in reference lists, not fetched)*
  Applied companion paper.
- **Ceccaroni, M., Celletti, A., Pucacco, G. "Halo orbits around the collinear points of the restricted three-body problem."** arXiv:1510.04960 (open access). https://arxiv.org/abs/1510.04960
  Modern perturbative treatment of halo bifurcations; warns the normal-form method is poorly suited to some L3/small-μ cases. Useful for explaining where analytic expansions like Richardson's lose accuracy (large amplitude).
- **Gil, A. F., Litteri, W., Rodriguez-Fernandez, V., Camacho, D., Vasile, M. (2024). "Generative Design of Periodic Orbits in the Restricted Three-Body Problem."** arXiv:2408.03691. https://arxiv.org/abs/2408.03691
  VAE trained on 44,112 JPL Earth–Moon periodic orbits (40 families). Generated orbits had mean dynamical error ~34 per node vs ~1e-13 for true orbits; only 46/100 converged under multiple shooting. A quantified ML reliability limit in the CR3BP and a source of periodic-orbit test cases.
- **Litteri, W., Gil, A. F., Vasile, M., Rodriguez-Fernandez, V., Camacho, D. (2026). "Generation of periodic orbits in the restricted three-body problem with a variational autoencoder."** *Celest. Mech. Dyn. Astron.* 138(3), open access. https://link.springer.com/article/10.1007/s10569-026-10299-x
  Journal version of the above; generated orbits still need numerical correction (exact numbers unverified).

### FTLE, Lagrangian coherent structures and chaos indicators

- **Short, C. R., Howell, K. C., Tricoche, X. M. (2011). "Lagrangian Coherent Structures in the Restricted Three-Body Problem."** AAS 11-250. https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Conferences/2011_AAS_ShoHowTri.pdf
  FTLE fields and LCS ridges in the CR3BP, linked to Lyapunov-orbit manifolds. FTLE ridges mark separatrices where timestep labels should change sharply; check whether your RF errors concentrate there.
- **Short, C. R., Howell, K. C. (2014). "Lagrangian coherent structures in various map representations for application to multi-body gravitational regimes."** *Acta Astronautica.* https://www.sciencedirect.com/science/article/abs/pii/S0094576513003408
  FTLE/LCS in the CR3BP, a restricted four-body problem and ephemeris models. Relevant to a real-system extension.
- **"Understanding flow around planetary moons via finite-time Lyapunov exponent maps." (2024).** *Celest. Mech. Dyn. Astron.* *(authors not captured)* https://link.springer.com/article/10.1007/s10569-024-10213-3
  Modern FTLE-map application in the CR3BP; useful for citing FTLE methodology.
- **"Variational Chaos Indicators: Application to the Restricted Three-Body Problem." (2014).** Proc. IAU Symposium 310. *(authors not captured; Tomsk State University group)* https://www.cambridge.org/core/services/aop-cambridge-core/content/view/97C93346FC654F76663362F287706ED5/S1743921314007777a.pdf
  Compares FLI, OFLI, MEGNO, SALI and APLE in the planar CR3BP. Helps justify which indicator to add next to FTLE (MEGNO/SALI converge faster and have clear thresholds).
- **"The three-body problem" (review).** arXiv:1508.02312. https://arxiv.org/abs/1508.02312
  Review covering FLI and MEGNO as cheaper alternatives to full Lyapunov exponents.

### Regularization near close encounters

- **Paez, R. I., Guzzo, M. (2020). "A study of temporary captures and collisions in the Circular Restricted Three-Body Problem with normalizations of the Levi-Civita Hamiltonian."** *Int. J. Non-Linear Mechanics.* DOI 10.1016/j.ijnonlinmec.2020.103417; open access: https://arxiv.org/abs/1904.04146
  Cartesian normal forms at L1/L2 have a convergence radius limited by distance to the secondary; Levi-Civita regularization removes that limit. Physically confirms why r2 dominates numerical difficulty and motivates a regularized-coordinate baseline. Planar, single-μ (Sun–Jupiter) only.
- *(unverified leads)* arXiv:1809.01257, "Integrability of close encounters in the spatial restricted three-body problem" (extends regularization to the spatial CR3BP); arXiv:2206.07022 (KS regularization with fast Lyapunov indicators for close encounters in the elliptic problem). Confirm authors before citing.

---

## 2. Numerical Integrators and Reliability of Chaotic Integration

### Symplectic and geometric integration

- **Tao, M. (2016). "Explicit symplectic approximation of nonseparable Hamiltonians: Algorithm and long time performance."** *Phys. Rev. E* 94, 043303. https://arxiv.org/abs/1609.02212
  Explicit, arbitrary-even-order integrators symplectic in an extended phase space. General reference for symplectic integration of the rotating-frame Hamiltonian; compare with your canonical-momentum Strang splitting (check whether yours is truly symplectic or only symmetric).
- **Soria-Carro, A., Akella, M. R. (2026). "Explicit Symplectic Integrators for Cislunar Regimes."** *J. Guidance, Control, and Dynamics* 49(3). https://doi.org/10.2514/1.G009228 (paywalled)
  Explicit symplectic integrators for non-separable Hamiltonians including CR3BP and ER3BP, with speed-ups and better first-integral preservation. Most directly relevant integrator paper; frame your leapfrog results against it.
- **"Highly Efficient Numerical Integrator for the Circular Restricted Three-Body Problem."** *(authors and venue not verified)* https://researchgate.net/publication/363149422
  Explicit symmetric variational-symplectic integrator for the Coriolis/centrifugal structure. Treat as secondary until the venue is verified.
- **Bowman group (Univ. of Alberta). "An Exactly Conservative Integrator for the n-Body Problem."** *(author list not verified)* https://www.math.ualberta.ca/~bowman/publications/nbody.pdf
  Conserves the Hamiltonian (Jacobi constant) to machine precision for the restricted problem. Illustrates that exact Jacobi conservation can coexist with large phase error: your stability-vs-accuracy point.

### Adaptive step-size control and close-encounter handling

- **Rein, H., Spiegel, D. S. (2015). "IAS15: a fast, adaptive, high-order integrator for gravitational dynamics, accurate to machine precision over a billion orbits."** *MNRAS* 446(2), 1424–1437. https://arxiv.org/abs/1409.4779
  15th-order Gauss–Radau integrator with automatic step-size control; machine precision at ~100 steps per orbit. Add as a baseline: any ML selector claiming cost savings must beat it.
- **Pham, D., Rein, H., Spiegel, D. S. (2024). "A new timestep criterion for N-body simulations."** *Open Journal of Astrophysics* 7. https://arxiv.org/abs/2401.02849
  Timestep criterion from acceleration, jerk and snap that recovers orbital-period and pericenter timescales for any eccentricity. A physics-derived "safe timestep" predictor: use it as a non-ML baseline feature. If it explains your labels as well as RF does, that is an important finding.
- **Lu, T., Hernandez, D. M., Rein, H. (2024). "TRACE: a Time-Reversible Algorithm for Close Encounters."** arXiv:2405.03800. *(author list partly verified)* https://arxiv.org/abs/2405.03800
  Switches from a reversible integrator to IAS15/BS when the Pham et al. pericenter criterion fires. The hand-built version of "detect difficult regions, switch method," which your ML tries to learn.
- **Rein, H., Tamayo, D. (2015). "WHFast: a fast and unbiased implementation of a symplectic Wisdom–Holman integrator."** *MNRAS* 452(1), 376–388. *(title paraphrased from the REBOUND docs page)* https://rebound.hanno-rein.de/
  Context for symplectic vs. adaptive trade-offs.

### Reliability of chaotic integrations: energy error vs. trajectory error

- **Boekholt, T., Portegies Zwart, S. (2015). "On the reliability of N-body simulations."** *Comput. Astrophys. Cosmol.* 2:2. https://arxiv.org/abs/1411.6671
  Introduces Brutus (arbitrary-precision Bulirsch–Stoer); shows individual double-precision three-body solutions diverge while ensemble statistics agree. Core justification for position-error labels against a converged reference rather than energy error, and for convergence-testing your DOP853 reference in chaotic regions.
- **Portegies Zwart, S., Boekholt, T. (2014). "On the minimal accuracy required for simulating self-gravitating systems by means of direct N-body methods."** arXiv:1402.6713 *(journal ApJ Letters not verified)* https://arxiv.org/abs/1402.6713
  Conjectures ensembles are statistically faithful if energy is conserved to better than 1/10. The opposite philosophy to per-trajectory validity; state which one your tolerance targets.
- **Boekholt, T., Portegies Zwart, S., Valtonen, M. (2020). "Gargantuan chaotic gravitational three-body systems and their irreversibility to the Planck length."** *MNRAS.* https://scholarlypublications.universiteitleiden.nl/access/item:3141995/download
  Shows chaos amplifying tiny errors to macroscopic scales. Supports the idea that beyond some FTLE × horizon threshold no fixed-step strategy is safe in double precision; your hurdle classifier should learn that boundary.
- **Portegies Zwart, S. (2026). "Validating direct solvers for Newton's gravitational N-body problem, and the systematic comparison between IEEE floating point and Posits."** arXiv:2608.17032. https://arxiv.org/html/2608.17032
  Uses Brutus as ground truth for benchmark problems; a recent template for reference-solution validation. *(2026 preprint, abstract-level verification)*

---

## 3. ML for Orbital Dynamics: Step-Size Selection and Surrogates

### Learned step-size and integrator selection (most directly comparable work)

- **Saz Ulibarrena, V., Portegies Zwart, S. (2025). "Reinforcement learning for adaptive time-stepping in the chaotic gravitational three-body problem."** *Commun. Nonlinear Sci. Numer. Simul.* 145, 108723 (open access). https://arxiv.org/abs/2502.12809
  DQN agent chooses the time-step parameter *during* integration to balance accuracy and cost. **Closest analogue to this project**: general three-body (not CR3BP), in-run (not pre-simulation), single integrator family. Warns random validation sets give unreliable metrics in chaotic problems.
- **Saz Ulibarrena, V., Portegies Zwart, S., et al. (2026). "ReLaTS: a Reinforcement Learning-based method for dynamically determining the coupling Time Step in multi-scale simulations of self-gravitating systems."** arXiv:2606.20832. https://arxiv.org/abs/2606.20832
  Follow-up extending RL timestep control to Bridge-coupled multiscale codes; notes Cartesian-state inputs are a weak choice, which supports physics-informed features.
- **Dellnitz, M., Hüllermeier, E., et al. "Efficient time stepping for numerical integration using reinforcement learning."** arXiv:2104.03562. *(full author list and journal publication not verified)* https://arxiv.org/abs/2104.03562
  Main task is numerical quadrature, extended to differential equations; hierarchical base learners plus a meta-learner that picks among them; needs explicit knowledge of the function. Claims to beat adaptive schemes (RK45 rejections are wasteful on chaotic systems) in narrow problem classes. **The strongest prior claim against your negative result**; check whether its comparison is at matched global accuracy (the claim is from the abstract only).
- **"Meta-Learning Runge-Kutta."** ICLR 2020 submission (OpenReview), anonymous. https://openreview.net/pdf?id=rkesVkHtDr
  Frames step-size control as meta-learning; studies which inputs are useful. Not verified as peer-reviewed; cite as a preprint.
- **"Using Machine Learning to Design Time Step Size Controllers for Stable Time Integrators." (2026).** *J. Scientific Computing.* *(authors not captured)* https://link.springer.com/article/10.1007/s10915-026-03496-1
  Bayesian optimization tunes classical PI-type controller parameters instead of replacing the controller. A hybrid route: learn controller parameters per region rather than a fixed dt.
- **Guo, Y., Dietrich, F., Bertalan, T., Doncevic, D. T., Dahmen, M., Kevrekidis, I. G., Li, Q. (2022). "Personalized Algorithm Generation: A Case Study in Learning ODE Integrators."** *SIAM J. Sci. Comput.* DOI 10.1137/21M1418629; arXiv:2105.01303. https://arxiv.org/abs/2105.01303
  Learns integrator coefficients tailored to a problem class; related algorithm selection/adaptation framing.
- **"Adaptive time step control using reinforcement learning for stiff chemical kinetic systems."** APS DFD 2023, abstract J40.00005 (abstract only). https://meetings.aps.org/Meeting/DFD23/Session/J40.5
  Same idea in stiff chemistry; abstract-level evidence only.

### Surrogates and emulators for three-body dynamics and their limits

- **Breen, P. G., Foley, C. N., Boekholt, T., Portegies Zwart, S. (2020). "Newton versus the machine: solving the chaotic three-body problem using deep neural networks."** *MNRAS* 494(2), 2465–2470. https://arxiv.org/abs/1910.07291
  Network trained on Brutus solutions; accurate over a bounded interval at fixed cost, up to 10^8 times faster than a state-of-the-art solver. Restricted to a 2D, equal-mass, zero-initial-velocity slice; fails near close encounters. The canonical positive claim; your work probes the decision layer instead of the surrogate.
- **Saz Ulibarrena, V., Horn, P., Portegies Zwart, S., Sellentin, E., Koren, B., Cai, M. X. (2024). "A hybrid approach for solving the gravitational N-body problem with Artificial Neural Networks."** *J. Comput. Phys.* 496 (open access per ScienceDirect listing).
  Hybrid integrator using NNs only where safe, falling back to numerical integration. Close to your "safe region" framing.
- **Flamant, C., Protopapas, P., Sondak, D. (2020). "Solving Differential Equations Using Neural Network Solution Bundles."** arXiv:2006.14372. https://arxiv.org/abs/2006.14372
  Planar CR3BP (μ=0.01) surrogate working only in a small initial-condition box; 178.6 GPU-hours, local minima, exponentially growing error bound. A quantified CR3BP failure mode.
- **Chen, R., Tao, M. (2021). "Data-driven Prediction of General Hamiltonian Dynamics via Learning Exactly-Symplectic Maps."** ICML. https://arxiv.org/abs/2103.05632
  Uses planar CR3BP as its non-separable benchmark; notes symplectic recurrent networks lose symplecticity there. Restricted to the non-chaotic regime.
- **Trani, A. A., Leigh, N. W. C., Boekholt, T. C. N., Portegies Zwart, S. (2024). "Isles of regularity in a sea of chaos amid the gravitational three-body problem."** *Astron. Astrophys.* 689, A24.
  Regular islands embedded in chaotic phase space. Supports grouped spatial splits: difficulty is structured on fine scales.
- **Nehma, G., Tiwari, M., Lingam, M. "Deep Learning Based Dynamics Identification and Linearization of Orbital Problems using Koopman Theory."** arXiv:2403.08965. https://arxiv.org/abs/2403.08965
  Koopman linearization of the CR3BP near L1; limitations not verified.
- **Chen, Y., Cui, M., Qian, Y.-J., Cheng, Y.-F., Chen, W.-X. (2026). "Physics-informed neural network with hard constraints based on dual-step training strategy for periodic orbits in the CR3BP."** *Nonlinear Dynamics* 114, art. 826. DOI 10.1007/s11071-026-12640-3. *(accuracy figures not verified)*
- **"Physics-Informed Neural Networks for Discovering Periodic Orbits in the Gravitational Three-Body Problem." (2026).** arXiv:2607.23501. *(authors not captured; context only)* https://arxiv.org/abs/2607.23501
- **Liao, S., Li, X., Yang, Y. (2021). "Three-body problem — from Newton to supercomputer plus machine learning."** arXiv:2106.11010 (*New Astronomy* 2022 per citing references). https://arxiv.org/abs/2106.11010
  ANN classifies periodic/unstable/non-periodic three-body orbits, with some misclassifications. General three-body, not CR3BP.

---

## 4. ML Methodology and Benchmarking vs. Classical Solvers

### Benchmarking and negative results

- **McGreivy, N., Hakim, A. (2024). "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations."** *Nature Machine Intelligence* 6(10), 1256–1269. DOI 10.1038/s42256-024-00897-5; open access: https://www.osti.gov/servlets/purl/2475825
  Of 76 articles claiming to beat a numerical method, 60 (79%) used a weak baseline; outcome-reporting and publication biases were widespread. Your preregistered negative result is what the authors call for. Follow their rules: compare at equal accuracy and against the strongest available method.
- **Koehler, F., Thuerey, N. (2025). "Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data."** arXiv:2510.23111. https://arxiv.org/pdf/2510.23111
  Counterpoint: networks trained on low-fidelity solver data can beat those solvers against a higher-fidelity reference. Cite for balance.

### Conformal prediction, importance, validation

- **Angelopoulos, A. N., Bates, S. (2021, v3 2022). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification."** arXiv:2107.07511. https://arxiv.org/abs/2107.07511
  Split conformal, conformalized quantile regression, risk control, distribution shift. Marginal coverage does not guarantee coverage in the hardest (close-approach/high-FTLE) regions; use group-conditional (Mondrian) conformal or conformal risk control on the violation rate.
- **Strobl, C., Boulesteix, A.-L., Zeileis, A., Hothorn, T. (2007). "Bias in random forest variable importance measures: Illustrations, sources and a solution."** *BMC Bioinformatics* 8:25. https://pubmed.ncbi.nlm.nih.gov/17254353/
  RF importance is biased when predictors differ in scale or number of categories.
- **Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., Zeileis, A. (2008). "Conditional variable importance for random forests."** *BMC Bioinformatics* 9:307. https://link.springer.com/article/10.1186/1471-2105-9-307
  Permutation importance inflates correlated predictors. Directly relevant: r1, r2, r1/r2, C and C − C_L1 are strongly correlated. Use conditional/grouped permutation or drop-column ablation.
- **Roberts, D. R., et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure."** *Ecography* 40(8), 913–929. https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881 (OA PDF: https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)
  Block CV is "nearly universally more appropriate" than random CV for predicting new regions, but blocking can force extrapolation. Justifies grouped spatial splits; report the interpolation/extrapolation regime.

---

## Refinements Suggested by the Literature

1. **Reframe the question** from "can ML beat adaptive solvers?" (no, consistent with McGreivy & Hakim) to "what physical quantities predict integrator failure, and where are the safe regions for cheap fixed-step methods?" Story: preregistered negative result plus a quantified failure map.
2. **Add the strongest baselines:** IAS15 via REBOUND (rotating-frame Coriolis terms need custom setup) and the Pham–Rein–Spiegel criterion as a zero-ML dt predictor. If RF does not beat the physics criterion, report that.
3. **Add a regularized baseline** (Levi-Civita planar, KS for 3D) near the primaries.
4. **Treat labels as interval-censored:** binary search brackets true dt between last pass and first fail. Verify monotonicity; report non-monotone fractions from dense sweeps.
5. **Make conformal guarantees local:** Mondrian (region/FTLE-bin) conformal or conformal risk control; report coverage per bin.
6. **Fix importance analysis:** conditional or grouped permutation importance for correlated features, alongside ablation.
7. **Validate the reference:** convergence-test DOP853 (tolerance halving or high-precision subset) in chaotic regions; label non-converged ICs "unpredictable" rather than assigning a dt.
8. **Choose chaos indicators deliberately:** add MEGNO or SALI next to FTLE; test whether FTLE ridges line up with label discontinuities.
9. **Halo validation:** use differentially corrected halos seeded by Richardson; treat Richardson error as a confounder at large amplitude.
10. **Real-system extension:** Earth–Moon cislunar orbits (JPL periodic-orbit catalog gives labeled families). Near-Earth asteroids need an ephemeris model, which breaks the Jacobi-constant feature.

## Gaps This Project Could Fill

- No study found gives a **pre-simulation, per-initial-condition map of integrator safety in the spatial CR3BP** across multiple mass ratios, linking FTLE, r1/r2 and C − C_L1 to maximum safe timestep for Euler/RK4/symplectic methods.
- No study found applies **conformal prediction or calibrated safety margins** to timestep selection in orbital mechanics.
- **Preregistered negative results** are rare in ML for numerical methods; a transparent ML-vs-IAS15/DOP853 cost comparison at matched accuracy answers McGreivy & Hakim's call.

## Unverified Leads

Not confirmed in the search; check before citing: Mullahy (1986) on hurdle models for count data; Candès, Lei & Ren on conformalized survival analysis (conformal lower bounds under censoring, fits interval-censored dt labels); Nosek et al. (2018), "The preregistration revolution," *PNAS* 115(11), 2600–2606; Quinlan & Tremaine (1992) on shadowing of N-body integrations; Hairer, Lubich & Wanner, *Geometric Numerical Integration*.
