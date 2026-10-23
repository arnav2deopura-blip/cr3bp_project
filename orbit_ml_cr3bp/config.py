from __future__ import annotations
import os
from multiprocessing import cpu_count

# CR3BP PHYSICS PARAMETERS
MU = 0.01215058535
MASS_RATIOS_FOR_SENSITIVITY = [0.001, 0.01215058535, 0.5]
COLLISION_RADIUS = 0.01  # legacy value; physical per-body radii now come from system_params()/collision_radii()
EJECTION_RADIUS = 10.0

# Per-integrator Jacobi-drift tolerance used by the timestep search and by
# every downstream validation/audit function. Single source of truth --
# search_3d.py, modeling_3d.py, and validation_3d.py all import this
# instead of keeping their own copies.
INTEGRATOR_TOLERANCES_3D = {"euler": 1e-5, "leapfrog": 1e-5, "rk4": 1e-5}

SAFETY_FACTOR_CANDIDATES_3D = (1.0, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)

# FTLE PARAMETERS
FTLE_INTEGRATION_TIME = 10.0
FTLE_PERTURBATION = 1e-6
FTLE_ENABLED = True
FTLE_METHOD = "variational"   # "variational" or "divergence"
FTLE_MAX_STEP = 0.01
FTLE_STEP_SAFETY = 0.05
FTLE_DT_FLOOR = 1e-5
FTLE_MAX_STEPS = 2_000_000

# COMPUTATIONAL PARAMETERS
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, cpu_count() // 2)))
CHUNK_SIZE = 32
SEED_COUNT = 20
NUM_TRAINING_ORBITS = 20000
NUM_BENCHMARK_ORBITS = 2000

# 2D FEATURE COLUMNS (kept for compatibility)
BASIC_FEATURE_COLUMNS = ["x", "y", "vx", "vy"]
PHYSICS_FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS + [
    "jacobi",
    "r1",
    "r2",
    "r1_r2_ratio",
    "ftle",
    "c_minus_c_l1",
]
TARGET_COLUMNS = ["dt_euler", "dt_rk4", "dt_leapfrog"]

# 3D FEATURE COLUMNS
# Added "sim_time" to capture integration duration (critical for stability)
BASIC_FEATURE_COLUMNS_3D = ["x", "y", "z", "vx", "vy", "vz", "sim_time"]
PHYSICS_FEATURE_COLUMNS_3D = BASIC_FEATURE_COLUMNS_3D + [
    "jacobi_3d",
    "r1_3d",
    "r2_3d",
    "r1_r2_ratio_3d",
    "ftle_3d",
    "c_minus_c_l1_3d",
]
TARGET_COLUMNS_3D = ["dt_euler", "dt_rk4", "dt_leapfrog"]   # same names as 2D

# DATA GENERATION PARAMETERS
L1_X = 0.8369
L2_X = 1.1557
L3_X = -1.0051
L4_X = 0.4878
L4_Y = 0.8660
L5_X = 0.4878
L5_Y = -0.8660
MAX_LAG_OFFSET = 0.10

# SIMULATION TIME RANGE (in normalized time units)
# For 3D CR3BP, primary orbital period = 2π, so use 1 to 100 time units.
# This is the range used by the ORIGINAL Jacobi-conservation-based search
# (label_mode="jacobi" in data_3d.py). Left unchanged for reproducibility of
# existing results; the accuracy-based search below uses its own, shorter
# range instead.
SIM_TIME_LOG_MIN = 0.0          # 10^0 = 1
SIM_TIME_LOG_MAX = 2.0          # 10^2 = 100

# FILE NAMES
TRAINING_DATASET_NAME = "cr3bp_training_data.csv"
BENCHMARK_DATASET_NAME = "cr3bp_benchmark_data.csv"
MODEL_NAME = "cr3bp_integrator_recommender.pkl"
TRAINING_DATASET_3D_NAME = "cr3bp_3d_training_data.csv"
MODEL_3D_NAME = "cr3bp_3d_integrator_recommender.pkl"

# BASELINE COMPARISON PARAMETERS
BASELINE_FACTOR = 0.01

# PLOTTING PARAMETERS
POINCARE_Y_THRESHOLD = 0.0
POINCARE_VY_MIN = 0.0
GRID_RESOLUTION = 40

# =============================================================================
# ACCURACY-BASED (DOP853-REFERENCE) STABILITY SEARCH
# =============================================================================
# Everything below is used only when data_3d.py's generation functions are
# called with label_mode="accuracy" instead of the original "jacobi". See
# search_3d.find_dt_for_integrator_accuracy_3d for the search itself and its
# module-level docstring for the full rationale. None of this is read by the
# label_mode="jacobi" path, so it does not change any existing result.

# Nondimensionalization length unit (CR3BP "DU"), used only to convert a
# physical position-error tolerance (km) into the CR3BP-normalized units the
# rest of the pipeline works in, and to convert results back to km for
# reporting. This is the NOMINAL Earth-Moon distance; the true Earth-Moon
# distance varies ~356,500-406,700 km over the real (eccentric, precessing)
# lunar orbit -- CR3BP itself assumes a fixed circular orbit at this nominal
# value, so this constant is only as good as that modeling assumption.
# VERIFY/cite the specific value you use if this goes in the paper.
DU_KM = 384400.0

# Physical body radii used as the "collision" (impact) radii, and the length
# unit (DU, km) of each supported CR3BP system. Primary = body 1 (at -mu),
# secondary = body 2 (at 1-mu). Earth-Moon: DU is the nominal Earth-Moon
# distance above. Sun-Earth: DU is 1 au. Radii are mean/nominal values.
R_EARTH_KM = 6371.0
R_MOON_KM = 1737.4
R_SUN_KM = 695700.0
AU_KM = 149597870.7
MU_SUN_EARTH = 3.00348e-6
SYSTEMS = {
    "earth_moon": {"mu": MU, "du_km": DU_KM, "r1_km": R_EARTH_KM, "r2_km": R_MOON_KM},
    "sun_earth": {"mu": MU_SUN_EARTH, "du_km": AU_KM, "r1_km": R_SUN_KM, "r2_km": R_EARTH_KM},
}


def system_params(mu):
    """Length unit and body radii for a mass ratio. Unknown mu falls back to
    the Earth-Moon DU and the legacy COLLISION_RADIUS for both bodies
    (generic=True) -- those numbers are NOT physical for that system."""
    for name, p in SYSTEMS.items():
        if abs(mu - p["mu"]) <= 1e-3 * p["mu"]:
            return {**p, "name": name, "generic": False}
    legacy_km = COLLISION_RADIUS * DU_KM
    return {"name": "generic", "mu": mu, "du_km": DU_KM, "r1_km": legacy_km, "r2_km": legacy_km, "generic": True}


def collision_radii(mu):
    p = system_params(mu)
    return p["r1_km"] / p["du_km"], p["r2_km"] / p["du_km"]


def du_km_for_mu(mu):
    return system_params(mu)["du_km"]

# Default physical position-error tolerance for the accuracy-based search.
# This is an ILLUSTRATIVE default for exercising the pipeline, not a cited
# operational/mission navigation requirement -- pick (and justify, citing a
# real mission's requirement if you use one) a value appropriate to whatever
# you are validating against before reporting results from this.
POSITION_TOL_KM = 100.0
POSITION_TOL_DU_EARTH_MOON = POSITION_TOL_KM / DU_KM

# Mission-scale horizon cap for the accuracy-based search (1-20 TU), instead
# of the original search's 1-100 TU range. 1 TU in the Earth-Moon CR3BP is
# roughly 4.34 days; treat that conversion as approximate unless you verify
# it yourself.
ACCURACY_SIM_TIME_LOG_MIN = 0.0                 # 10^0 = 1 TU
ACCURACY_SIM_TIME_LOG_MAX = 1.301029995663981   # log10(20) -> 20 TU

# DOP853 reference-trajectory tolerances (physics_3d.reference_trajectory_dense_3d).
# Tighter than INTEGRATOR_TOLERANCES_3D's role in the old search -- this is a
# ground-truth trajectory, not a stability threshold.
REFERENCE_RTOL = 1e-11
REFERENCE_ATOL = 1e-11
REFERENCE_MAX_STEP = 0.01

# Sampling-mode mixture used by data_3d.generate_one_3d_training_orbit when
# label_mode="accuracy". Replaces the old "mixed" mode's fixed 80/20
# lagrange/wide split so that near-Moon states (see "near_moon" mode in
# physics_3d.generate_3d_orbit) are represented in training at all -- under
# the OLD "mixed" sampling, essentially no generated orbit came within
# r2_3d < 0.037 of the secondary body (checked against the existing
# cr3bp_3d_training_data.csv), which is far outside the r2 range of a
# realistic near-rectilinear halo orbit (NRHO). Must sum to 1.0.
SAMPLING_WEIGHTS_3D = {"lagrange": 0.55, "wide": 0.20, "near_moon": 0.25}

# r2 (distance-to-secondary) range for "near_moon" sampling, in DU. Lower
# bound is NEAR_MOON_R2_MIN_FACTOR times the secondary's physical radius
# (see SYSTEMS), so states closer than ~20% above the surface are never
# sampled; r2 is drawn log-uniformly so close passes are represented. Upper
# bound (~57,700 km) is chosen to comfortably span typical NRHO-type perilune/
# apolune distances -- this is an approximate design choice, not a value
# taken from a specific mission's ephemeris; if you have real near-Moon
# reference states (e.g. from validate_on_capstone.py), check this range
# against their actual r2_3d and adjust if needed.
NEAR_MOON_R2_MIN_FACTOR = 1.2
NEAR_MOON_R2_MIN = NEAR_MOON_R2_MIN_FACTOR * R_MOON_KM / DU_KM
NEAR_MOON_R2_MAX = 0.15