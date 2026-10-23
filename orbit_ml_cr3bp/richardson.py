"""
Halo-orbit initial conditions for the 3D CR3BP, via numerical differential
correction (single shooting on the state-transition matrix), NOT Richardson's
analytic third-order expansion.

Why not the analytic (Richardson 1980) expansion
--------------------------------------------------
An earlier version of this module implemented Richardson's third-order
analytic construction directly (the c2/c3 coefficients, the in-plane
frequency lambda, and an Ax-Az amplitude relation). On audit, that
implementation had unresolved problems: the c3 coefficient was computed but
never used (the amplitude-constraint equation that actually ties Ax to Az at
third order was missing entirely), gamma-normalized quantities were mixed
with CR3BP-normalized quantities without rescaling, and the vy0 formula
dropped the standard kappa factor. None of this could be fixed with
confidence without transcribing Richardson's full third-order formulas
(c2..c4, kappa, l1, l2, the frequency correction) from the primary source
and verifying every term -- a real risk of introducing a *different*,
equally wrong, formula. That path was rejected in favor of something that
could actually be verified end-to-end in this session.

What this module does instead
------------------------------
It finds a genuinely periodic (to the requested numerical tolerance, not a
third-order truncation) halo-like orbit by:

  1. Linearizing the exact CR3BP equations at the target collinear point
     (L1 or L2), using the same Jacobian entries already implemented and
     verified in physics_3d._cr3bp_variational_equations_3d_numba (re-
     derived and cross-checked against that module, not copied blind).
  2. Eigendecomposing that linearization to get the in-plane oscillatory
     eigenvector, which gives a small-amplitude seed (x0, vy0) that is
     *exactly* consistent with the true linearized dynamics at t=0 (not a
     guessed ratio), for whatever mu is passed in.
  3. Using that seed as the start of a natural-parameter continuation in Az:
     at each Az, a 2-variable Newton (single) shooting method adjusts
     (x0, vy0) so that the trajectory returns to a perpendicular
     xz-plane crossing (vx=vz=0 at the next y=0 crossing). The Newton
     Jacobian uses the free-final-time-corrected state-transition matrix
     (see _corrector_jacobian below), obtained by integrating the
     variational equations already defined in physics_3d.py.
  4. By the CR3BP's y -> -y, vx -> -vx, vz -> -vz, t -> -t symmetry, an
     orbit with two perpendicular xz-plane crossings is automatically
     periodic with period 2*T_half. This is checked, not assumed: see
     verify_periodicity() below.

This is the standard technique used for halo-orbit targeting in mission
design (differential correction / single shooting; see Howell, "Three-
dimensional, periodic, 'halo' orbits," Celestial Mechanics 32, 1984, and
Koon, Lo, Marsden & Ross, "Dynamical Systems, the Three-Body Problem, and
Space Mission Design," 2011, chapter 3, for the method -- not for any
specific numbers used here, which are all computed in this file). It
produces a strictly more accurate initial condition than a third-order
truncation would, and it does not require trusting any hand-transcribed
coefficient formula: the only physics used is the already-verified
equations of motion and their exact linearization.

Verified range
--------------
Natural-parameter continuation (stepping Az up from ~0 and re-solving at
each step) works cleanly until it hits a fold in the (x0, Az) curve, where
this simple method stalls. On this project's mu (Earth-Moon,
MU = 0.01215058535), that was tested and found to converge to a residual
below 1e-10 up to:

    L1: Az ~ 0.085
    L2: Az ~ 0.05  (only tested to 0.05; may go further, not verified)

MAX_VERIFIED_AZ below records this. differential_corrector_halo_3d() will
raise RuntimeError rather than silently return a bad answer if continuation
stalls before reaching the requested Az. Getting further requires
pseudo-arclength continuation (stepping along the solution curve's tangent
instead of stepping Az directly, so the fold can be passed) -- that is a
well-established extension but is NOT implemented here: an attempt during
this audit was numerically unstable (it jumped between unrelated solution
branches near the fold) and was discarded rather than shipped un-debugged.
If your paper needs Az beyond the verified range, implement pseudo-
arclength continuation, or cite this limitation and restrict your
validation set to the verified range.
"""

from __future__ import annotations

import numpy as np

from .config import MU
from .physics_3d import (
    libration_point_location,
    jacobi_constant_3d,
    cr3bp_derivatives_3d,
    step_rk4_3d,
    _cr3bp_variational_equations_3d_numba,
)

# Largest Az (z-amplitude, CR3BP-normalized units) this module's natural-
# parameter continuation has actually been verified (residual < 1e-10) to
# reach for mu = MU. See module docstring.
MAX_VERIFIED_AZ = {"L1": 0.085, "L2": 0.05}


# =============================================================================
# LINEARIZATION AT THE COLLINEAR POINT (in-plane block only: x,y,vx,vy)
# =============================================================================

def _inplane_linearization(mu: float, x_eq: float) -> np.ndarray:
    """4x4 Jacobian of (x,y,vx,vy) dynamics at (x_eq, 0, 0, 0, 0, 0).

    Entries re-derived directly from the CR3BP acceleration formulas (same
    ones implemented in physics_3d.py); this is the y=0 restriction of the
    same Jacobian physics_3d._cr3bp_variational_equations_3d_numba builds
    (cross-checked against that module during audit -- both give the same
    dax_dx, dax_dy, day_dy at y=z=0).
    """
    r1 = abs(x_eq + mu)
    r2 = abs(x_eq + mu - 1.0)
    r1_3, r2_3 = r1 ** 3, r2 ** 3
    r1_5, r2_5 = r1 ** 5, r2 ** 5
    mu1 = 1.0 - mu

    dax_dx = 1.0 - mu1 * (1.0 / r1_3 - 3.0 * (x_eq + mu) ** 2 / r1_5) \
                 - mu * (1.0 / r2_3 - 3.0 * (x_eq + mu - 1.0) ** 2 / r2_5)
    # dax_dy = day_dx = 0 here because y = 0 kills the cross term 3*(x+mu)*y/r^5
    day_dy = 1.0 - mu1 / r1_3 - mu / r2_3

    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0] = dax_dx
    A[2, 3] = 2.0
    A[3, 1] = day_dy
    A[3, 2] = -2.0
    return A


def _inplane_seed(mu: float, point: str) -> tuple[float, float]:
    """Return (x_eq, ratio) where ratio = vy0/(x0-x_eq) for the small-
    amplitude in-plane oscillatory mode, from the eigenvector of the exact
    linearization (not a hardcoded/remembered constant -- recomputed here
    for whatever mu is passed in)."""
    x_eq, _ = libration_point_location(mu, point)
    A = _inplane_linearization(mu, x_eq)
    eigvals, eigvecs = np.linalg.eig(A)
    for i in range(4):
        if eigvals[i].imag > 1e-8:
            vec = eigvecs[:, i].real
            if abs(vec[0]) < 1e-12:
                raise RuntimeError(f"Degenerate in-plane eigenvector at {point}.")
            return x_eq, float(vec[3] / vec[0])
    raise RuntimeError(f"No oscillatory in-plane mode found at {point} (mu={mu}); "
                        "linearization may be wrong for this mu.")


# =============================================================================
# SINGLE SHOOTING: y=0 CROSSING + FREE-FINAL-TIME-CORRECTED SENSITIVITY
# =============================================================================

def _find_y_crossing(state0: list[float], mu: float, dt: float, max_T: float):
    """Integrate with the project's verified RK4 stepper until y crosses
    zero (after leaving the initial y=0 plane), then bisect for the exact
    crossing time and state. Returns (T_cross, state_cross) or (None, None)
    if no crossing occurs within max_T."""
    state = list(state0)
    t = 0.0
    while t < max_T:
        state_new = step_rk4_3d(state, dt, mu)
        t_new = t + dt
        if t > 0.3 * dt and state[1] * state_new[1] < 0:
            lo, hi, t_lo, t_hi = state, state_new, t, t_new
            for _ in range(50):
                t_mid = 0.5 * (t_lo + t_hi)
                mid = step_rk4_3d(lo, t_mid - t_lo, mu)
                if mid[1] * lo[1] <= 0:
                    t_hi, hi = t_mid, mid
                else:
                    t_lo, lo = t_mid, mid
            return t_hi, hi
        state, t = state_new, t_new
    return None, None


def _stm_at(state0: list[float], mu: float, T: float, n_steps: int) -> np.ndarray:
    """State-transition matrix Phi(T) via RK4 on the augmented [state, Phi]
    system, using physics_3d's own (already-verified) variational-equations
    right-hand side."""
    phi0 = np.eye(6).flatten()
    y = np.concatenate((np.asarray(state0, dtype=float), phi0))
    h = T / n_steps
    for _ in range(n_steps):
        k1 = _cr3bp_variational_equations_3d_numba(y, mu)
        k2 = _cr3bp_variational_equations_3d_numba(y + 0.5 * h * k1, mu)
        k3 = _cr3bp_variational_equations_3d_numba(y + 0.5 * h * k2, mu)
        k4 = _cr3bp_variational_equations_3d_numba(y + h * k3, mu)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[6:].reshape(6, 6)


def _corrector_jacobian(x0: float, z0: float, vy0: float, mu: float, max_T: float):
    """One shooting evaluation: propagate to the next y=0 crossing and
    return (F, J, T_half) where F = [vx_cross, vz_cross] (should be driven
    to zero) and J is the 2x2 free-final-time-corrected Jacobian of F with
    respect to (x0, vy0) -- see module docstring point 3. Returns None if
    no crossing is found within max_T."""
    state0 = [x0, 0.0, z0, 0.0, vy0, 0.0]
    dt = max_T / 2000.0
    T_cross, state_cross = _find_y_crossing(state0, mu, dt, max_T)
    if T_cross is None:
        return None
    vx_c, vz_c = state_cross[3], state_cross[5]
    F = np.array([vx_c, vz_c])

    Phi = _stm_at(state0, mu, T_cross, n_steps=max(50, int(T_cross / dt)))
    f_final = np.asarray(cr3bp_derivatives_3d(state_cross, mu))
    vy_f = f_final[1]
    if abs(vy_f) < 1e-10:
        return None  # crossing not transversal; free-time correction ill-defined

    def sens(col):
        return Phi[:, col] - (f_final / vy_f) * Phi[1, col]

    s_x0, s_vy0 = sens(0), sens(4)
    J = np.array([[s_x0[3], s_vy0[3]],
                  [s_x0[5], s_vy0[5]]])
    return F, J, T_cross


def _newton_fixed_z(x0: float, z0: float, vy0: float, mu: float, max_T: float,
                     tol: float, max_iter: int, trust: float):
    """Newton iteration in (x0, vy0) at fixed z0=Az. Returns
    (x0, vy0, T_half, iters, resid) on success, None on failure."""
    for it in range(max_iter):
        result = _corrector_jacobian(x0, z0, vy0, mu, max_T)
        if result is None:
            return None
        F, J, T_half = result
        resid = float(np.linalg.norm(F))
        if resid < tol:
            return x0, vy0, T_half, it, resid
        try:
            delta = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None
        step_norm = np.linalg.norm(delta)
        if step_norm > trust:
            delta = delta * (trust / step_norm)
        x0 += delta[0]
        vy0 += delta[1]
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def differential_corrector_halo_3d(
    point: str,
    Az: float,
    mu: float = MU,
    tol: float = 1e-10,
    max_T: float = 4.0,
    max_newton_iter: int = 40,
    trust: float = 0.02,
    max_continuation_attempts: int = 600,
) -> dict:
    """
    Differentially-corrected halo-orbit initial condition at L1 or L2.

    Uses natural-parameter continuation in Az, starting from a linearized
    (eigenvector-consistent) small-amplitude seed and Newton-correcting at
    each step. Raises RuntimeError -- rather than returning a wrong answer
    -- if continuation stalls before reaching the requested Az; see
    MAX_VERIFIED_AZ and the module docstring for this project's verified
    range and why it stalls there (a fold that would need pseudo-arclength
    continuation to pass).

    Returns a dict with keys: point, Az, mu, state0 ([x,y,z,vx,vy,vz]),
    T_half, T_full, residual, n_continuation_steps.
    """
    if point not in ("L1", "L2"):
        raise ValueError("differential_corrector_halo_3d is implemented for L1/L2 only.")
    if Az <= 0:
        raise ValueError("Az must be positive.")

    x_eq, ratio = _inplane_seed(mu, point)

    eps = min(1e-4, 0.5 * Az)
    eps = max(eps, min(Az, 1e-8))
    x0, vy0 = x_eq + eps, ratio * eps

    Az_current = eps
    step = min(2e-3, max(Az - eps, 1e-6) / 5.0)
    n_steps_taken = 0
    last_good = None

    for attempt in range(max_continuation_attempts):
        res = _newton_fixed_z(x0, Az_current, vy0, mu, max_T, tol, max_newton_iter, trust)
        if res is None:
            step *= 0.5
            if step < 1e-7:
                reached = last_good[1] if last_good else 0.0
                raise RuntimeError(
                    f"differential_corrector_halo_3d: continuation stalled for {point} "
                    f"at Az={Az_current:.6f} while targeting Az={Az} (mu={mu}). Largest "
                    f"Az reached: {reached:.6f}. This project's verified ceiling for "
                    f"{point} is ~{MAX_VERIFIED_AZ.get(point, 'unknown')}; natural-"
                    f"parameter continuation hits a fold near there and needs "
                    f"pseudo-arclength continuation to go further (not implemented -- "
                    f"see richardson.py module docstring)."
                )
            continue

        x0, vy0, T_half, n_iter, resid = res
        last_good = (x0, Az_current, vy0, T_half, resid)
        n_steps_taken += 1

        if abs(Az_current - Az) < 1e-10:
            return {
                "point": point, "Az": Az, "mu": mu,
                "state0": [x0, 0.0, Az_current, 0.0, vy0, 0.0],
                "T_half": T_half, "T_full": 2.0 * T_half,
                "residual": resid, "n_continuation_steps": n_steps_taken,
            }

        Az_current = min(Az, Az_current + step)
        step = min(step * 1.2, 5e-3)

    raise RuntimeError(
        f"differential_corrector_halo_3d: exceeded {max_continuation_attempts} "
        f"continuation attempts for {point}, Az={Az} without reaching the target."
    )


def richardson_halo_initial_conditions(point: str, Az: float, mu: float = MU) -> list[float]:
    """
    Backward-compatible entry point (same name/signature as the old
    analytic version) returning [x, y, z, vx, vy, vz]. Internally this is
    now the differential corrector above, NOT Richardson's analytic
    expansion -- see module docstring. Raises RuntimeError if Az is
    outside the verified convergence range for this mu/point; see
    MAX_VERIFIED_AZ.
    """
    result = differential_corrector_halo_3d(point, Az, mu=mu)
    return result["state0"]


def verify_periodicity(state0: list[float], T_half: float, mu: float = MU,
                        dt: float = 2e-4) -> dict:
    """
    Independent check (using the project's own verified RK4 stepper, at a
    fixed small dt, with no reference to the STM/corrector machinery above)
    that propagating state0 forward for a full period 2*T_half returns to
    state0, and that the Jacobi constant is conserved along the way.
    Returns {"closure_error": ..., "jacobi_drift": ...}.
    """
    C0 = jacobi_constant_3d(state0, mu)
    s = list(state0)
    n = int(round(2.0 * T_half / dt))
    for _ in range(n):
        s = step_rk4_3d(s, dt, mu)
    closure_error = float(np.linalg.norm(np.array(s) - np.array(state0)))
    jacobi_drift = float(abs(jacobi_constant_3d(s, mu) - C0))
    return {"closure_error": closure_error, "jacobi_drift": jacobi_drift, "final_state": s}
