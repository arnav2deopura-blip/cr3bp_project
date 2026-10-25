import sys, time, json
sys.path.insert(0, "/Users/arnavdeopura/School/Everything Portfolio/Research/extended-orbit-ML-2")
import numpy as np, pandas as pd
from multiprocessing import Pool
from orbit_ml_cr3bp.audit_3d import dense_sweep_log_3d, summarize_dense_sweep_log

CSV = sys.argv[1]
d = pd.read_csv(CSV)
d = d[~d.reference_divergent].reset_index(drop=True)
n = min(int(sys.argv[2]), len(d))
rows = np.random.RandomState(42).choice(len(d), size=n, replace=False)

def work(chunk):
    return dense_sweep_log_3d(d, rows=chunk)

if __name__ == "__main__":
    t0 = time.time()
    chunks = np.array_split(rows, 40)
    with Pool(8) as p:
        parts = p.map(work, chunks)
    df = pd.concat(parts, ignore_index=True)
    print("wall", round(time.time() - t0, 1), "records", len(df))
    df.to_csv(sys.argv[3], index=False)
    s = summarize_dense_sweep_log(df)
    for k, v in s.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
