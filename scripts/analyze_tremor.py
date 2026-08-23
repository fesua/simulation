"""Localize the tremor in time and name its correlate.

The operator reports the arms shake "right before grasping the bolt", not uniformly
during transit. So the question is not "how much tremor" but "what is different at the
moments it happens". This slices the 500 Hz diagnostic log into tremor / calm samples
and prints how each candidate signal differs between them.

Candidates, and what an elevated value would mean:
  smin  - smallest singular value of J. Low => near-singular wrist, where the damped
          least-squares step turns a small Cartesian error into a large joint motion
          that flips sign between substeps.
  dq    - per-substep joint step. At the DQ_MAX clip the IK is saturating.
  res   - IK position residual. High => the servo cannot reach its reference.
  refz  - reference height, to confirm the bursts really sit at the pre-grasp descent.
  grip  - gripper command, to see whether bursts coincide with closing.
"""
import pathlib
import sys

import numpy as np

SUBSTEPS = 17


def main(paths):
    for f in paths:
        d = np.load(f)
        print(f"\n===== {pathlib.Path(f).name} =====")
        for side in ("left", "right"):
            tcp = d[f"{side}_tcp"]
            # jerk proxy at the physics rate, smoothed over one knot so a burst spans
            # a window rather than a single spike
            j = np.linalg.norm(np.diff(tcp, n=2, axis=0), axis=1)
            k = np.convolve(j, np.ones(SUBSTEPS) / SUBSTEPS, mode="same")

            n = min(len(k), *(len(d[f"{side}_{c}"]) for c in ("dq", "res", "smin", "refz", "grip")))
            k = k[:n]
            hot = k >= np.quantile(k, 0.95)
            cold = k <= np.quantile(k, 0.50)

            print(f"  -- {side}: jerk p50={np.quantile(k,.5)*1e6:.1f} "
                  f"p95={np.quantile(k,.95)*1e6:.1f} um/step^2")
            for c in ("smin", "dq", "res", "refz", "grip"):
                v = d[f"{side}_{c}"][:n]
                h, c0 = v[hot].mean(), v[cold].mean()
                ratio = h / c0 if abs(c0) > 1e-12 else float("inf")
                print(f"     {c:5s} tremor={h:9.5f}  calm={c0:9.5f}  x{ratio:6.2f}")

            if f"{side}_cmdtcp" in d:
                C = d[f"{side}_cmdtcp"][:n]
                jc = np.linalg.norm(np.diff(C, n=2, axis=0), axis=1)
                jc = np.convolve(jc, np.ones(SUBSTEPS) / SUBSTEPS, mode="same")
                dev = np.linalg.norm(tcp[:n] - C, axis=1)
                print(f"     COMMANDED jerk p95={np.quantile(jc,.95)*1e6:.1f} um/step^2 "
                      f"vs MEASURED p95={np.quantile(k,.95)*1e6:.1f}")
                print(f"     |measured-commanded| tremor={dev[hot].mean()*1000:.3f} mm "
                      f"calm={dev[cold].mean()*1000:.3f} mm")

            # where in the episode do the bursts sit?
            idx = np.flatnonzero(hot)
            if len(idx):
                t = idx / (SUBSTEPS * 30.0)
                print(f"     bursts at t = {np.percentile(t,10):.1f}..{np.percentile(t,90):.1f} s "
                      f"(median {np.median(t):.1f} s of 30 s)")


if __name__ == "__main__":
    main(sys.argv[1:])
