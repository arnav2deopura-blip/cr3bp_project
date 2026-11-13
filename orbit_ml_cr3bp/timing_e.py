"""
Study E, draft section 7 item 6: the wall-clock protocol E6 for E1b and E7b. Nothing here is used by a sealed kernel.

  require_idle      refuses to start (raises) when the 1-minute load average is above 1.0; no relaxation (Study D's D2 started at load 6.5)
  machine_info      platform, CPU, core count, Python / numba / numpy versions, load average
  time_sessions     two sessions (by default) over the same orbits; in each session every orbit is timed for every arm, the arm order shuffled
                    per orbit (seeded), with compiled_solvers_d.time_per_orbit (minimum over 3 repeats of elapsed / reps). The idle check runs
                    again before each session and every 1,000 orbits; a session that finds the machine busy stops and is reported as such.
  throughput        the E1b / E7b statistic from the per-orbit times: Σ comparator time / Σ policy time, from the mean of the sessions' per-orbit
                    times (each session's own ratio is reported too), with the D2 orbit bootstrap (confirm_d_3d.d2_throughput)

Every arm is timed with the endpoint collision check only (no chord test, no checkpoints): the timing compares the integrators, and the same
checks are applied to every arm. One machine only (decision D4): the results hold for this machine, and the record says which one it is.
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np

__all__ = ["MAX_LOAD", "require_idle", "machine_info", "time_sessions", "throughput"]

MAX_LOAD = 1.0


class BusyMachine(RuntimeError):
    pass


def require_idle(max_load=MAX_LOAD):
    l1 = os.getloadavg()[0]
    if l1 > max_load:
        raise BusyMachine(f"1-minute load average {l1:.2f} > {max_load}: close other programs and wait (E6 does not relax this)")
    return l1


def machine_info():
    import numba
    import subprocess
    cpu = platform.processor()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu, "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numba": numba.__version__, "numpy": np.__version__, "load_average": list(os.getloadavg())}


def time_sessions(arms, n_orbits, sessions=2, seed=0, log_path=None, max_load=MAX_LOAD, check_every=1000):
    """arms: {name: fn(i, reps) -> checksum} (numba loops over orbit i). Returns {"times": {session: {arm: (n_orbits,) ns}}, "sessions": [...]}.
    Raises BusyMachine before a session if the machine is busy; a session interrupted by load is marked "busy" and its times are NaN after the stop."""
    from .compiled_solvers_d import time_per_orbit
    names = list(arms)
    for nm in names:                                       # compile every loop before any timing
        arms[nm](0, 1)
    out, meta = {}, []
    for ses in range(sessions):
        l0 = require_idle(max_load)
        rng = np.random.RandomState(seed + ses)
        T = {nm: np.full(n_orbits, np.nan) for nm in names}
        status, t0 = "complete", time.time()
        for i in range(n_orbits):
            if i and i % check_every == 0 and os.getloadavg()[0] > max_load:
                status = f"busy at orbit {i}"
                break
            for nm in rng.permutation(names):
                T[nm][i] = time_per_orbit(lambda reps, nm=nm, i=i: arms[nm](i, reps))
        rec = {"session": ses, "status": status, "load_start": l0, "load_end": os.getloadavg()[0], "seconds": time.time() - t0, "n_timed": int(np.isfinite(T[names[0]]).sum())}
        meta.append(rec)
        out[ses] = T
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec, "machine": machine_info()}) + "\n")
    return {"times": out, "sessions": meta}


def throughput(res, policy_arm, comparator_arm, n_boot=4000, seed=0):
    """E1b / E7b from time_sessions' result: the D2 statistic on the mean over complete sessions of the per-orbit times, plus each session's ratio."""
    from .confirm_d_3d import d2_throughput
    ok = [s["session"] for s in res["sessions"] if s["status"] == "complete"]
    if not ok:
        return {"evaluable": False, "reason": "no complete session"}
    tp = np.mean([res["times"][s][policy_arm] for s in ok], axis=0)
    tc = np.mean([res["times"][s][comparator_arm] for s in ok], axis=0)
    r = d2_throughput(tp, tc, n_boot=n_boot, seed=seed)
    r["per_session_ratio"] = {int(s): float(np.nansum(res["times"][s][comparator_arm]) / np.nansum(res["times"][s][policy_arm])) for s in ok}
    r["n_complete_sessions"] = len(ok)
    r["evaluable"] = True
    return r
