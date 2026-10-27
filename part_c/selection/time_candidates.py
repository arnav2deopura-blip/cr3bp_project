"""Idle-machine single-call cost (RHS-equivalents) of every S1 candidate model class (structure fitted on D_train + enriched)."""
import sys, json, time, os
sys.path.insert(0, ".")
import numpy as np
from run_part_c_3d import load_c_frame, _fit_for_config
from orbit_ml_cr3bp import compiled_model_3d as cm, policy_c_3d as pc

if __name__ == "__main__":
    TR = "data_c/train/cr3bp_3d_training_data_accuracy_monotone_seed41.csv"
    EN = "data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv"
    Utr, w = load_c_frame([TR, EN], enriched={EN})
    Usel, _ = load_c_frame(["data_c/select/cr3bp_3d_training_data_accuracy_monotone_seed42.csv"])
    st = Usel[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=float)[:2000]
    T = Usel["sim_time"].to_numpy(dtype=float)[:2000]
    cfgs = json.load(open("part_c/selection/S1_sundman.json")) + json.load(open("part_c/selection/S1_phys.json"))
    out = {}
    print("load average at start", os.getloadavg(), flush=True)
    for cfg in cfgs:
        feats = pc.FEATS_OSC if cfg.get("features", "osc") == "osc" else pc.pb.FEATS_STATE
        kind, models = _fit_for_config(cfg, Utr, w, feats)
        if kind == "linear":
            k = models["candidates"][0]
            c = cm.measure_linear_single_cost(models["linear"][k], st, T, repeats=7)
        else:
            pol = cm.CompiledHGBPolicy(models, single=True)
            c = cm.measure_single_call_cost(pol, st, T, repeats=7)
            c["size"] = pol.size()
        out[cfg["name"]] = {k: v for k, v in c.items()}
        print(f"{cfg['name']:24s} {c['rhs_equiv']:7.2f}  range {[round(x, 2) for x in c['rhs_equiv_range']]}  rhs_ns {c['rhs_ns_per_call']:.1f}  policy_ns {c['policy_ns_per_call']:.0f}"
              + (f"  feat {c['features_rhs_equiv']:.2f}  python-dispatch {c['python_dispatch_rhs_equiv']:.2f}" if "features_rhs_equiv" in c else ""), flush=True)
    print("load average at end", os.getloadavg())
    json.dump({"load_start": None, "costs": out}, open("part_c/selection/S1_overheads_idle.json", "w"), indent=1, default=float)
