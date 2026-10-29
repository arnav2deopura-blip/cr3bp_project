"""
IAS15 (REBOUND) baseline for Part D, Tier 3 (exploratory). Run with the interpreter that has REBOUND (not the project environment):

  python3 ias15_baseline_d.py --inp out_d/explore/ias15_input.csv --out out_d/explore/ias15_output.csv

REBOUND 5.2.2 (installed in the base interpreter; the project environment does not have it). The CR3BP in the inertial frame is an N-body problem: two massive primaries (masses 1 - mu and mu, separation 1, angular velocity 1, G = 1) on a circular orbit about the barycenter and a
massless test particle. A rotating-frame state (x, y, z, vx, vy, vz) becomes an inertial state at t = 0 (the frames coincide) with velocity v + omega x r = (vx - y, vy + x, vz); the final
position is rotated back by the angle T and compared with the DOP853 reference of the project. For every orbit the loosest epsilon (to 0.02 decades) whose final error is within the tolerance is
found by bisection (the per-orbit oracle, the 'spec' of Parts B/C), and the steps and the wall time of that run are recorded. REBOUND does not expose a force-evaluation counter, so the cost
is steps and wall time only.
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import rebound

MU = 0.01215058535
DU_KM = 384400.0


def run(state, T, eps, mu=MU):
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.integrator = "ias15"
    sim.integrator.epsilon = eps                                       # REBOUND 5: the default adaptive mode is the Pham-Rein-Spiegel (2024) criterion
    sim.add(m=1.0 - mu, x=-mu, y=0.0, vx=0.0, vy=-mu)                 # primary: radius mu from the barycenter, speed mu (omega = 1)
    sim.add(m=mu, x=1.0 - mu, y=0.0, vx=0.0, vy=1.0 - mu)
    x, y, z, vx, vy, vz = state
    sim.add(m=0.0, x=x, y=y, z=z, vx=vx - y, vy=vy + x, vz=vz)
    t0 = time.perf_counter()
    try:
        sim.integrate(T, exact_finish_time=1)
    except Exception:
        return None
    wall = time.perf_counter() - t0
    p = sim.particles[2]
    c, s = math.cos(T), math.sin(T)
    xr, yr = p.x * c + p.y * s, -p.x * s + p.y * c
    return np.array([xr, yr, p.z]), sim.steps_done, wall


def oracle_eps(state, T, ref, tol_km, lo=-12.0, hi=-1.0, width=0.05):
    def ok(le):
        r = run(state, T, 10.0 ** le)
        if r is None:
            return None
        return (np.linalg.norm(r[0] - ref) * DU_KM <= tol_km), r
    best = None
    good = None
    x = -9.0
    r = ok(x)
    if r is not None and r[0]:
        good, best = x, r[1]
        while x + 1.0 <= hi + 1e-9:
            x += 1.0
            r = ok(x)
            if r is None or not r[0]:
                bad = x
                break
            good, best = x, r[1]
        else:
            return good, best
    else:
        bad = x
        while x - 1.0 >= lo - 1e-9:
            x -= 1.0
            r = ok(x)
            if r is not None and r[0]:
                good, best = x, r[1]
                break
            bad = x
        if good is None:
            return None, None
    while bad - good > width:
        mid = 0.5 * (good + bad)
        r = ok(mid)
        if r is not None and r[0]:
            good, best = mid, r[1]
        else:
            bad = mid
    return good, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol_km", type=float, default=100.0)
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    rows = []
    for i, r in df.iterrows():
        st = r[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)
        ref = r[["ref_x", "ref_y", "ref_z"]].to_numpy(dtype=float)
        le, res = oracle_eps(st, float(r["sim_time"]), ref, a.tol_km)
        rec = {"row_id": int(r["row_id"]), "ias15_log10_eps": le}
        if res is not None:
            pos, steps, wall = res
            rec.update({"ias15_steps": steps, "ias15_wall_s": wall, "ias15_err_km": float(np.linalg.norm(pos - ref) * DU_KM)})
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(i + 1, "of", len(df), flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
