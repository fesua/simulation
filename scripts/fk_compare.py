"""Ladder step 2d: compare Isaac (USD) FK against the MuJoCo (MJCF) reference.

Gate: max position error must be sub-millimetre and max orientation error sub-0.1 deg
across all sampled configurations. cm_bridge ladder step 2 set the precedent (0.00 mm).

Run:  .venv/bin/python scripts/fk_compare.py
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MJ = ROOT / "outputs/fk_mujoco.json"
IS = ROOT / "outputs/fk_isaac.json"

POS_TOL_MM = 1.0
ROT_TOL_DEG = 0.1


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def main() -> int:
    mj = json.loads(MJ.read_text())
    isaac = json.loads(IS.read_text())

    frames = [f for f in mj["frames"] if f in isaac["frames"]]
    skipped = [f for f in mj["frames"] if f not in isaac["frames"]]
    if skipped:
        print(f"skipped (not a physics body in Isaac): {skipped}")

    n = min(len(mj["records"]), len(isaac["records"]))
    per_frame = {f: {"pos": [], "rot": []} for f in frames}
    worst = {"pos": (0.0, None), "rot": (0.0, None)}

    for i in range(n):
        rm, ri = mj["records"][i], isaac["records"][i]
        assert np.allclose(rm["q_deg"], ri["q_deg"]), f"config mismatch at {i}"
        for f in frames:
            pm = np.array(rm["frames"][f]["pos"])
            pi = np.array(ri["frames"][f]["pos"])
            dp = float(np.linalg.norm(pm - pi)) * 1000.0

            Rm = np.array(rm["frames"][f]["xmat"]).reshape(3, 3)
            Ri = quat_wxyz_to_mat(ri["frames"][f]["quat_wxyz"])
            dr = rot_angle_deg(Rm, Ri)

            per_frame[f]["pos"].append(dp)
            per_frame[f]["rot"].append(dr)
            if dp > worst["pos"][0]:
                worst["pos"] = (dp, (i, f))
            if dr > worst["rot"][0]:
                worst["rot"] = (dr, (i, f))

    print(f"\ncompared {n} configurations x {len(frames)} frames\n")
    print(f"{'frame':20s} {'max pos (mm)':>14s} {'mean pos (mm)':>14s} {'max rot (deg)':>14s}")
    for f in frames:
        p = np.array(per_frame[f]["pos"])
        r = np.array(per_frame[f]["rot"])
        print(f"{f:20s} {p.max():14.6f} {p.mean():14.6f} {r.max():14.6f}")

    ok = worst["pos"][0] <= POS_TOL_MM and worst["rot"][0] <= ROT_TOL_DEG
    print(f"\nworst position   : {worst['pos'][0]:.6f} mm   at config/frame {worst['pos'][1]}")
    print(f"worst orientation: {worst['rot'][0]:.6f} deg  at config/frame {worst['rot'][1]}")
    print(f"\ngate: pos <= {POS_TOL_MM} mm and rot <= {ROT_TOL_DEG} deg  ->  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
