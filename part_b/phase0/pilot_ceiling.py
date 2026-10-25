import sys, time, json
sys.path.insert(0, "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2")
import numpy as np, pandas as pd
from multiprocessing import Pool
from orbit_ml_cr3bp.config import MU, du_km_for_mu
from orbit_ml_cr3bp.physics_3d import reference_trajectory_dense_3d
from orbit_ml_cr3bp.search_3d import find_dt_for_integrator_accuracy_monotone_3d as mono, _dt_upper_bound_3d

CSV = "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2/data/cr3bp_3d_training_data_accuracy_monotone_seed12.csv"
S = ["x","y","z","vx","vy","vz"]

def work(args):
    i, state, T, old_dt = args
    t0 = time.time()
    sol, ok = reference_trajectory_dense_3d(state, T, MU)
    t_ref = time.time() - t0
    out = {"i": i, "T": T, "t_ref": t_ref}
    if not ok:
        out["div"] = True; return out
    old_ceiling = max(_dt_upper_bound_3d(state, MU), 0.005)
    out["old_ceiling"] = old_ceiling
    for name, dtm in [("c2", min(T, 2.0)), ("c1", min(T, 1.0))]:
        for k in ["rk4", "leapfrog"]:
            t0 = time.time()
            r = mono(k, state, T, MU, dt_max=dtm, reference=(sol, ok), return_details=True)
            out[f"{k}_{name}_dt"] = r["dt"]; out[f"{k}_{name}_cens"] = r["censored"]; out[f"{k}_{name}_t"] = time.time() - t0
    # reference check at 1e-13 and half max_step
    t0 = time.time()
    sol13, ok13 = reference_trajectory_dense_3d(state, T, MU, rtol=1e-13, atol=1e-13)
    out["t_ref13"] = time.time() - t0
    if ok13:
        d = np.linalg.norm(sol.sol(T)[:3] - sol13.sol(T)[:3]) * du_km_for_mu(MU)
        out["ref_diff_km"] = d
    else:
        out["ref13_div"] = True
    sol_h, ok_h = reference_trajectory_dense_3d(state, T, MU, rtol=1e-13, atol=1e-13, max_step=0.005)
    if ok_h and ok13:
        out["ref_diff_km_halfstep"] = np.linalg.norm(sol_h.sol(T)[:3] - sol13.sol(T)[:3]) * du_km_for_mu(MU)
    return out

if __name__ == "__main__":
    d = pd.read_csv(CSV)
    u = d[~d.ftle_nan & ~d.reference_divergent.fillna(False)].reset_index()
    rng = np.random.RandomState(7)
    pick = rng.choice(len(u), size=int(sys.argv[1]), replace=False)
    jobs = [(int(u.loc[j,"index"]), list(u.loc[j,S].astype(float)), float(u.loc[j,"sim_time"]), float(u.loc[j,"dt_rk4"])) for j in pick]
    t0 = time.time()
    with Pool(8) as p:
        res = p.map(work, jobs, chunksize=2)
    print("wall", time.time()-t0)
    df = pd.DataFrame(res)
    df = df.merge(u[["index","dt_rk4","rk4_censored","dt_leapfrog","leapfrog_censored","orbit_label","r2_3d","jacobi_3d"]].rename(columns={"index":"i"}), on="i")
    df.to_csv("pilot_ceiling.csv", index=False)
    print(df.describe().T[["count","mean","50%","max"]])
    for name in ["c2","c1"]:
        for k in ["rk4","leapfrog"]:
            print(name, k, "censored frac", df[f"{k}_{name}_cens"].mean())
