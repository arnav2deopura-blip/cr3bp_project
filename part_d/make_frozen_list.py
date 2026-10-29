"""Write part_d/FROZEN_FILES.json (what the seal hashes) and part_d/DATA_HASHES_D.json (SHA-256 of the Part D datasets and the raw JPL responses)."""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from orbit_ml_cr3bp import protocol_3d as proto  # noqa: E402

SETS = {"select": 52, "confirm": 53, "lock": 54, "replication": 55}
data = [f"data_d/{k}/cr3bp_3d_training_data_accuracy_monotone_seed{s}.csv" for k, s in SETS.items()]
train = ["data_c/train/cr3bp_3d_training_data_accuracy_monotone_seed41.csv", "data_c/train_enriched/cr3bp_3d_training_data_accuracy_monotone_seed141.csv"]
jpl = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "data_d", "jpl", "*.json")))
part_c = [f"part_c/frozen/{n}/policy_frozen.{e}" for n in ("phys", "sund", "global", "partb_refit") for e in ("joblib",)] + \
         [f"part_c/frozen/{n}/policy_frozen_spec.json" for n in ("phys", "sund", "global", "partb_refit")]
part_d = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "part_d", "frozen", "*", "policy_frozen*")))
sel = ["part_d/selection/overheads_idle.json", "part_d/selection/rule_selection.json"]
dry = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "part_d", "results", "dry", "*")))
d5_dry = sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "out_d", "dry", "d5", "d5_*", "d5_runs.csv.gz")))[-1:]      # the dry sample the D5 confirmation step recalibrates from
files = train + data + jpl + part_c + part_d + sel + dry + d5_dry
json.dump(files, open(os.path.join(ROOT, "part_d", "FROZEN_FILES.json"), "w"), indent=1)
hashes = {"datasets": {p: proto.sha256_file(os.path.join(ROOT, p)) for p in data}, "jpl": {p: proto.sha256_file(os.path.join(ROOT, p)) for p in jpl},
          "jpl_fetch_log": json.load(open(os.path.join(ROOT, "data_d", "jpl", "jpl_hashes.json")))}
json.dump(hashes, open(os.path.join(ROOT, "part_d", "DATA_HASHES_D.json"), "w"), indent=1)
print(len(files), "frozen files")
