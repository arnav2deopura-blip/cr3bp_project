import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from orbit_ml_cr3bp import policy_b_3d as pb, failure_study_3d as fs, policy_c_3d as pc
from orbit_ml_cr3bp.osculating_3d import add_osc_features
from orbit_ml_cr3bp import experiments_b_3d as xb
from run_failure_study_3d import load_partb, split_rows
d, _ = load_partb("data_b/cr3bp_3d_training_data_accuracy_monotone_seed21.csv")
U = add_osc_features(fs.add_derived(pb.usable_rows(d).reset_index(drop=True)))
tr_i, cal_i, ev_i = split_rows(U, 2000, 3000, 42)
tr, te = U.iloc[tr_i], U.iloc[ev_i]
te = te[te.rk4_success == 1]; tr = tr[tr.rk4_success == 1]
feats = pc.FEATS_OSC
y, yt = np.log10(tr.dt_rk4), np.log10(te.dt_rk4)
off_tr, off_te = tr.osc_logtau_hat.to_numpy(), te.osc_logtau_hat.to_numpy()
print("offset only (dt = c * tau_hat): median |resid| after fitting the 10 % quantile of y - off:", end=" ")
c = np.quantile(y - off_tr, 0.1); print(round(float(np.median(np.abs(off_te + c - yt))), 3), "c=", round(float(c), 3))
for nit, md in ((10, 3), (25, 3), (50, 4), (300, None)):
    for tgt in ("raw", "structured"):
        lr = 0.3 if nit <= 10 else 0.2 if nit <= 25 else 0.15 if nit <= 50 else 0.05
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, max_iter=nit, max_depth=md, learning_rate=lr, random_state=0)
        if tgt == "raw":
            reg.fit(tr[feats], y); yh = reg.predict(te[feats])
        else:
            reg.fit(tr[feats], y - off_tr); yh = reg.predict(te[feats]) + off_te
        print(f"HGB {nit}x{md} {tgt:10s} median |resid| {np.median(np.abs(yh - yt)):.3f}  p90 {np.quantile(np.abs(yh - yt), 0.9):.3f}  over {np.mean(yh > yt):.3f}")
