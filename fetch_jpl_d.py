"""
Fetch the Earth-Moon periodic-orbit families of the JPL Three-Body Periodic Orbits API (Part D, D5) and save every response RAW with its SHA-256.

  python fetch_jpl_d.py [--out data_d/jpl]

Only https://ssd-api.jpl.nasa.gov/periodic_orbits.api is contacted (GET, the documented parameters sys / family / libr / branch). The responses are
data: they are parsed as JSON by periodic_d.py and nothing in them is executed or followed. The hashes go to part_d/DATA_HASHES_D.json under "jpl".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://ssd-api.jpl.nasa.gov/periodic_orbits.api"
# (file stem, query parameters beyond sys=earth-moon). The families the plan names; resonant branches that return no data are recorded as empty.
QUERIES = (
    [(f"halo_L{l}_{b}", {"family": "halo", "libr": l, "branch": b}) for l in (1, 2) for b in ("N", "S")]
    + [(f"lyapunov_L{l}", {"family": "lyapunov", "libr": l}) for l in (1, 2)]
    + [(f"vertical_L{l}", {"family": "vertical", "libr": l}) for l in (1, 2)]
    + [(f"axial_L{l}", {"family": "axial", "libr": l}) for l in (1, 2)]
    + [("dro", {"family": "dro"})]
    + [(f"butterfly_{b}", {"family": "butterfly", "branch": b}) for b in ("N", "S")]
    + [(f"resonant_{pq}", {"family": "resonant", "branch": pq}) for pq in ("12", "21", "13", "31", "23", "32", "34", "43")]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_d/jpl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hashes = {}
    for stem, q in QUERIES:
        url = API + "?" + urllib.parse.urlencode({"sys": "earth-moon", **q})
        path = os.path.join(a.out, stem + ".json")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
                status = r.status
        except urllib.error.HTTPError as e:
            body, status = e.read(), e.code
        with open(path, "wb") as f:
            f.write(body)
        try:
            j = json.loads(body)
            n = j.get("count", 0) if isinstance(j, dict) else 0
        except Exception:
            n = -1
        hashes[stem] = {"url": url, "http_status": status, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(), "count": n}
        print(f"{stem:18s} http {status}  {len(body):>9,d} bytes  count {n}", flush=True)
        time.sleep(0.5)
    with open(os.path.join(a.out, "jpl_hashes.json"), "w") as f:
        json.dump(hashes, f, indent=1)


if __name__ == "__main__":
    main()
