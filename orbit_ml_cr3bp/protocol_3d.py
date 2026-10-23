from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LOCK_FORMAT = "cr3bp_locked_v1"
LOCK_GLOB = "locked_test_set*.json"
MIN_PIPELINE_VERSION = 3
ACCURACY_LABEL_MODES = ("accuracy", "accuracy_monotone")


def json_safe(o):
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, pd.DataFrame):
        return json_safe(o.to_dict(orient="records"))
    return str(o)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_path(csv_path):
    base, _ext = os.path.splitext(str(csv_path))
    return base + "_metadata.json"


def read_metadata(csv_path):
    p = metadata_path(csv_path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def check_dataset_pipeline(csv_path):
    meta = read_metadata(csv_path)
    if meta is None:
        print(f"WARNING: no metadata file next to {csv_path}; cannot verify label_pipeline_version or seeding.")
        return None
    if meta.get("label_mode") not in ACCURACY_LABEL_MODES:
        raise ValueError(f"{csv_path} has label_mode={meta.get('label_mode')!r}; the rigor suite needs "
                         f"an accuracy-labeled dataset (accuracy or accuracy_monotone).")
    v = meta.get("label_pipeline_version")
    if v is None or v < MIN_PIPELINE_VERSION:
        print(f"WARNING: {csv_path} has label_pipeline_version={v} (< {MIN_PIPELINE_VERSION}). Results on it are "
              f"not valid for new claims; regenerate.")
    return meta


def _legacy_seeding(meta):
    return meta.get("seeding") != "seedsequence"


def check_independent_datasets(paths):
    paths = list(paths)
    seen = {}
    metas = []
    for p in paths:
        h = sha256_file(p)
        if h in seen:
            raise ValueError(f"{p} and {seen[h]} are the same file; datasets must be independent.")
        seen[h] = p
        metas.append((p, read_metadata(p)))
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            (pa, ma), (pb, mb) = metas[i], metas[j]
            if ma is None or mb is None:
                print(f"WARNING: cannot verify independence of {pa} and {pb} (missing metadata).")
                continue
            mu_a, mu_b = ma.get("mu"), mb.get("mu")
            if mu_a is None or mu_b is None:
                print(f"WARNING: cannot verify independence of {pa} and {pb} (no mu in metadata).")
                continue
            if abs(mu_a - mu_b) > 1e-12:
                continue
            sa, sb = ma.get("base_seed"), mb.get("base_seed")
            if sa is None or sb is None:
                print(f"WARNING: cannot verify independence of {pa} and {pb} (no base_seed in metadata).")
                continue
            la, lb = _legacy_seeding(ma), _legacy_seeding(mb)
            if not la and not lb and sa == sb:
                raise ValueError(f"{pa} and {pb} share base_seed={sa}: identical orbits.")
            if la and lb:
                n = max(int(ma.get("num_orbits", 0)), int(mb.get("num_orbits", 0)))
                if abs(sa - sb) < n:
                    raise ValueError(f"{pa} and {pb} use legacy seeding (base_seed + i) with seeds {sa} and {sb}: "
                                     f"their orbit streams overlap. Regenerate with the current generator.")


def find_lock_for(csv_path):
    """Path of a lock file in the CSV's own directory (name locked_test_set*.json) whose recorded csv_sha256
    matches this CSV, else None. Used to make development commands refuse to load a locked CSV without --locked."""
    d = os.path.dirname(os.path.abspath(str(csv_path)))
    cands = sorted(glob.glob(os.path.join(d, LOCK_GLOB)))
    if not cands:
        return None
    h = sha256_file(csv_path)
    for p in cands:
        try:
            with open(p) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("format") == LOCK_FORMAT and rec.get("csv_sha256") == h:
            return p
    return None


def lock_test_set(data, out_path, csv_path, seed=2024, test_frac=0.25):
    if "row_id" not in data.columns:
        raise ValueError("data needs a row_id column (load_accuracy_data adds it)")
    existing = find_lock_for(csv_path)
    if existing:
        raise FileExistsError(f"{csv_path} already has a locked test set ({existing}). A CSV is locked once; archive "
                              f"that file (and its final-evaluation log) before drawing another.")
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} already exists. A locked set is drawn once; archive the old file "
                              f"(and its final-evaluation log) before drawing a new one.")
    n = len(data)
    n_lock = int(round(test_frac * n))
    if not 0 < n_lock < n:
        raise ValueError(f"test_frac={test_frac} gives {n_lock} locked rows of {n}")
    rng = np.random.RandomState(int(seed))
    ids = np.sort(rng.choice(data["row_id"].to_numpy(dtype=int), size=n_lock, replace=False))
    rec = {
        "format": LOCK_FORMAT,
        "csv": os.path.basename(str(csv_path)),
        "csv_sha256": sha256_file(csv_path),
        "n_rows_in_split_pool": int(n),
        "n_locked": int(n_lock),
        "seed": int(seed),
        "test_frac": float(test_frac),
        "split": "random",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_ids": [int(i) for i in ids],
    }
    d = os.path.dirname(str(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f)
    return rec


def load_locked(path, csv_path=None):
    with open(path) as f:
        rec = json.load(f)
    if rec.get("format") != LOCK_FORMAT:
        raise RuntimeError(f"{path} is not in the {LOCK_FORMAT} format (written by a different version of "
                           f"protocol_3d.py). Archive it and re-run `lock`.")
    if csv_path is not None and sha256_file(csv_path) != rec["csv_sha256"]:
        raise ValueError(f"{path} was drawn from a different CSV than {csv_path}")
    return rec


def split_locked(data, locked):
    ids = np.asarray(locked["row_ids"], dtype=int)
    mask = data["row_id"].isin(ids).to_numpy()
    return data[~mask].reset_index(drop=True), data[mask].reset_index(drop=True)


def log_final_evaluation(log_path, record):
    n_prev = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            n_prev = sum(1 for line in f if line.strip())
    entry = {"evaluation_number": n_prev + 1, "utc": datetime.now(timezone.utc).isoformat(), **json_safe(record)}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return n_prev + 1


FROZEN_SNAPSHOT = "CODE_SNAPSHOT.json"
FREEZE_RECORD = "FREEZE.json"
DEVIATIONS_MARKER = "\n## Deviations"
_SKIP_DIRS = {"__pycache__", "archive", "data", "out", "snapshots", ".git"}


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_files(root=None):
    """Every *.py under the project root, except caches, data, outputs and the archive."""
    root = root or project_root()
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        out += [os.path.join(base, f) for f in sorted(files) if f.endswith(".py")]
    return out


def code_fingerprint(root=None):
    """No git here, so the code identity is a SHA-256 over (relative path, file SHA-256) of every source file."""
    root = root or project_root()
    files = {os.path.relpath(p, root): sha256_file(p) for p in code_files(root)}
    blob = "".join(f"{k}\0{v}\n" for k, v in sorted(files.items()))
    return {"combined_sha256": hashlib.sha256(blob.encode()).hexdigest(), "files": files}


def _diff_files(frozen_files, current_files):
    return {
        "changed": sorted(k for k in current_files if k in frozen_files and current_files[k] != frozen_files[k]),
        "added": sorted(k for k in current_files if k not in frozen_files),
        "removed": sorted(k for k in frozen_files if k not in current_files),
    }


def compare_to_frozen(root=None):
    """None if no CODE_SNAPSHOT.json exists; else whether the current source tree still matches it."""
    root = root or project_root()
    path = os.path.join(root, FROZEN_SNAPSHOT)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        frozen = json.load(f)
    cur = code_fingerprint(root)
    d = _diff_files(frozen.get("files", {}), cur["files"])
    return {"frozen_combined_sha256": frozen.get("combined_sha256"), "current_combined_sha256": cur["combined_sha256"],
            "matches": frozen.get("combined_sha256") == cur["combined_sha256"], **d}


def prereg_body_sha256(path):
    """SHA-256 of the pre-registration text above the '## Deviations' heading, so dated deviations can be appended
    after the freeze without changing it."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return hashlib.sha256(text.split(DEVIATIONS_MARKER, 1)[0].encode("utf-8")).hexdigest()


def freeze_status(root=None, prereg_path=None):
    """None if nothing has been sealed (no FREEZE.json). Otherwise {'problems': [...], ...}: empty problems means the
    code and the pre-registration body are exactly what was sealed."""
    root = root or project_root()
    path = os.path.join(root, FREEZE_RECORD)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    problems = []
    cur = code_fingerprint(root)
    if cur["combined_sha256"] != rec.get("code_combined_sha256"):
        snap = os.path.join(root, FROZEN_SNAPSHOT)
        frozen_files = json.load(open(snap)).get("files", {}) if os.path.exists(snap) else {}
        d = _diff_files(frozen_files, cur["files"])
        problems.append("code differs from the sealed snapshot (changed: %s; added: %s; removed: %s)"
                        % (d["changed"] or "-", d["added"] or "-", d["removed"] or "-"))
    if prereg_path is None:
        problems.append("no --prereg given, so the pre-registration body cannot be checked against the seal")
    elif not os.path.exists(prereg_path):
        problems.append(f"--prereg {prereg_path} not found")
    elif prereg_body_sha256(prereg_path) != rec.get("prereg_body_sha256"):
        problems.append("the pre-registration text above '## Deviations' differs from the sealed version")
    return {"sealed_utc": rec.get("sealed_utc"), "code_combined_sha256": rec.get("code_combined_sha256"),
            "prereg_body_sha256": rec.get("prereg_body_sha256"), "problems": problems}


def _git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=here, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5,
                                    cwd=here, check=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def _package_versions():
    out = {"python": sys.version}
    for name in ("numpy", "pandas", "scipy", "sklearn", "numba", "joblib"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out


def start_run(root, cmd, args, data_paths=None, use_run_dir=True):
    os.makedirs(root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if use_run_dir:
        out_dir = os.path.join(root, f"{cmd}_{stamp}")
        k = 1
        while os.path.exists(out_dir):
            out_dir = os.path.join(root, f"{cmd}_{stamp}_{k}")
            k += 1
        os.makedirs(out_dir)
        manifest_path = os.path.join(out_dir, "manifest.json")
    else:
        out_dir = root
        manifest_path = os.path.join(root, f"manifest_{cmd}.json")
    inputs = []
    for p in data_paths or []:
        if p and os.path.exists(p):
            inputs.append({"path": str(p), "sha256": sha256_file(p), "metadata": read_metadata(p)})
        else:
            inputs.append({"path": str(p), "sha256": None, "metadata": None})
    manifest = {
        "command": cmd,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": json_safe(vars(args)),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git": _git_info(),
        "code": {"combined_sha256": code_fingerprint()["combined_sha256"],
                 "vs_frozen_snapshot": compare_to_frozen()},
        "inputs": inputs,
    }
    with open(manifest_path, "w") as f:
        json.dump(json_safe(manifest), f, indent=2)
    return out_dir