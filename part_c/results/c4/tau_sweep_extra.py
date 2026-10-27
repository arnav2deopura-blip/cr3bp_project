import sys, glob
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from multiprocessing import Pool
from run_part_c_3d import _tau_label_job
from orbit_ml_cr3bp import policy_b_3d as pb
if __name__ == "__main__":
    d = sorted(glob.glob("out_c/sundman_pilot/sundman_pilot_*"))[-1]
    M = pd.read_csv(d + "/pilot_table.csv", low_memory=False)
    st = M[pb.STATE_COLS].to_numpy(dtype=float); T = M["sim_time"].to_numpy(dtype=float)
    out = {}
    for tc in [1.0, 2.0, 4.0, 8.0, 16.0]:
        with Pool(8) as pool:
            rs = pd.DataFrame(pool.map(_tau_label_job, [(int(M["row_id"].iloc[i]), st[i], float(T[i]), 0.01215058535, 100.0, tc) for i in range(len(M))], chunksize=8))
        x = M[["row_id", "RK45_spec_nfev", "DOP853_spec_nfev", "RK45_cal_nfev", "C_fix_star", "log_delta_phys"]].merge(rs[["row_id", "rk4s_evals", "rk4s_censored"]], on="row_id")
        h = x["RK45_spec_nfev"].notna() & x["rk4s_evals"].notna()
        g = x[h]
        from scipy.stats import spearmanr
        Ps = np.log10(g.rk4s_evals / g.RK45_spec_nfev)
        print(tc, {"n": int(h.sum()), "vs_RK45_spec": round(float((g.rk4s_evals < g.RK45_spec_nfev).mean()), 4), "vs_DOP853_spec": round(float((g.rk4s_evals < g.DOP853_spec_nfev).mean()), 4),
                   "vs_RK45_cal": round(float((g.rk4s_evals < g.RK45_cal_nfev).mean()), 4), "median_evals": float(g.rk4s_evals.median()), "median_Ps": round(float(Ps.median()), 3),
                   "rho": round(float(spearmanr(g.log_delta_phys, Ps).statistic), 3), "censored": round(float(x.rk4s_censored.mean()), 4)}, flush=True)
