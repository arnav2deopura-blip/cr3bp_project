"""
Collision check between integrator nodes (posthoc/README.md P9, plan F6). The sealed kernels test r1 and r2 only at step endpoints, so a long physical
step can jump across the Moon (radius 0.0045 DU) without registering. This module tests the straight chord between consecutive nodes against each body's
sphere: the closest point of the segment p0 -> p1 to the centre c is p0 + t (p1 - p0) with t = clamp((c - p0).(p1 - p0) / |p1 - p0|^2, 0, 1).
It costs no RHS evaluation. The chord is an approximation of the curved path inside one step; it is exact for the question "did the step's
endpoints straddle the body", which is the failure mode the endpoint check misses.

Not wired into any sealed kernel (that would change sealed results); Study E uses it in the violation definition of physical-time arms.
"""
from __future__ import annotations

import numpy as np

from .config import MU, collision_radii

__all__ = ["segment_min_distance", "first_crossing", "crossed_bodies"]


def segment_min_distance(P, center):
    """Distance from `center` (3,) to each segment P[i] -> P[i+1] of the node positions P (n, 3). Returns (n - 1,) distances."""
    P = np.asarray(P, dtype=float)
    c = np.asarray(center, dtype=float)
    a, b = P[:-1], P[1:]
    d = b - a
    dd = np.einsum("ij,ij->i", d, d)
    t = np.where(dd > 0, np.einsum("ij,ij->i", c - a, d) / np.where(dd > 0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[:, None] * d
    return np.linalg.norm(q - c, axis=1)


def first_crossing(P, center, radius):
    """Index i of the first segment P[i] -> P[i+1] that comes within `radius` of `center`, or -1."""
    hit = np.flatnonzero(segment_min_distance(P, center) < radius)
    return int(hit[0]) if len(hit) else -1


def crossed_bodies(P, mu=MU, radii=None):
    """{'primary': first segment index or -1, 'secondary': ...} for node positions P (n, 3) in the rotating frame (Earth at (-mu, 0, 0), Moon at
    (1 - mu, 0, 0)). radii: (r_primary, r_secondary) in DU; default config.collision_radii(mu)."""
    rc1, rc2 = radii if radii is not None else collision_radii(mu)
    return {"primary": first_crossing(P, (-mu, 0.0, 0.0), rc1), "secondary": first_crossing(P, (1.0 - mu, 0.0, 0.0), rc2)}
