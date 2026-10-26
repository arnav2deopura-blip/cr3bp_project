"""Generate the mu-dial datasets (exploratory, pre-registered 2026-10-26): seeds 31-35, one per mu, 8,000 orbits each.
  python make_mu_dial_b.py --out_dir data_b/mu_dial [--n 8000] [--only 0.05]"""
import argparse
import hashlib
import json
import os
import time

from orbit_ml_cr3bp import mu_dial_3d as md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data_b/mu_dial")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--only", type=float, default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    hashes = {}
    for i, mu in enumerate(md.DIAL_MUS):
        if a.only is not None and abs(mu - a.only) > 1e-12 * max(1.0, a.only):
            continue
        seed = 31 + i
        out = os.path.join(a.out_dir, f"mu_{mu:g}_seed{seed}.csv")
        t0 = time.time()
        df = md.generate(mu, a.n, seed, out, n_jobs=a.n_jobs)
        u = df[~df["ftle_nan"] & ~df["reference_divergent"]]
        hashes[os.path.basename(out)] = {"mu": mu, "seed": seed, "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
                                         "rows": int(len(df)), "usable": int(len(u)), "rk4_censored": float(u["rk4_censored"].astype(bool).mean()),
                                         "wall_s": round(time.time() - t0)}
        print(os.path.basename(out), hashes[os.path.basename(out)], flush=True)
        json.dump(hashes, open(os.path.join(a.out_dir, "hashes.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
