import math
import numpy as np

from orbit_ml_cr3bp.config import MU, DU_KM, collision_radii
from orbit_ml_cr3bp import physics_3d as P


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    return bool(cond)


def main():
    ok = True
    state = [0.4878 + 0.03, 0.866 + 0.01, 0.01, 0.01, -0.02, 0.005]
    T = 2.0
    sol, reached = P.reference_trajectory_dense_3d(state, T, MU)
    ref = sol.sol(T)[:3]

    def err(step, dt):
        success, prev, curr, t_prev, t_act, reason = P.integrate_endpoint_3d(step, state, dt, T, MU)
        return success, t_act, np.linalg.norm(np.asarray(curr[:3]) - ref) * DU_KM

    ok &= check("reference reaches horizon", reached)
    for name, step, order in (("euler", P.step_euler_3d, 1.0), ("leapfrog", P.step_leapfrog_symplectic_3d, 2.0),
                              ("rk4", P.step_rk4_3d, 4.0)):
        s1, t1, e1 = err(step, 0.05)
        s2, t2, e2 = err(step, 0.025)
        ok &= check(f"{name}: lands exactly on T", s1 and s2 and abs(t1 - T) < 1e-12 and abs(t2 - T) < 1e-12)
        slope = math.log2(e1 / e2)
        ok &= check(f"{name}: convergence slope near {order}", abs(slope - order) < 0.5, f"slope={slope:.3f}")
    _, _, ea = err(P.step_rk4_3d, 0.1)
    _, _, eb = err(P.step_rk4_3d, 0.0999)
    ok &= check("rk4: error smooth across dt=0.1 / 0.0999", abs(math.log10(ea) - math.log10(eb)) < 0.1,
                f"{ea:.4g} vs {eb:.4g} km")

    d = P.compute_ftle_3d_variational_details(state, MU, 2.0)
    ok &= check("ftle finite on a regular orbit", np.isfinite(d["ftle"]) and d["status"] == 0, str(d))
    r1c, r2c = collision_radii(MU)
    crash = [1.0 - MU + 0.006, 0.0, 0.0, 0.0, 0.3, 0.0]
    d2 = P.compute_ftle_3d_variational_details(crash, MU, 1.0)
    ok &= check("ftle truncated (finite, flagged) when the trajectory hits the Moon",
                np.isfinite(d2["ftle"]) and d2["status"] in (1, 4), str(d2))
    ok &= check("physical radii", abs(r1c - 6371.0 / DU_KM) < 1e-9 and abs(r2c - 1737.4 / DU_KM) < 1e-9)
    ok &= check("NRHO-like perilune (3240 km from Moon center) is outside the Moon's collision radius",
                3240.0 / DU_KM > r2c)

    ok1, drift, t_end, reason = P.max_jacobi_drift_3d(P.step_rk4_3d, state, 0.01, 1.0, MU)
    ok &= check("jacobi drift kernel runs", ok1 and np.isfinite(drift), f"drift={drift:.3e}")
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
