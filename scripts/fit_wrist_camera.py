"""Refine the D405 lens bore on the Pika gripper and emit a camera pose.

The slice atlas (scripts/find_wrist_camera.py) shows a clean annulus around
(x~0, y~70) at z~120 mm in the tool frame, with its axis along +z -- i.e. the lens
looks along the approach direction, toward the grasp point. This fits that bore
precisely and reports where its front face sits.

Run:  .venv/bin/python scripts/fit_wrist_camera.py
"""

import pathlib
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
MESH = ROOT / "assets/meshes/pika_gripper_rzp90.STL"
OUT = ROOT / "outputs/wrist_camera_fit.png"

# search window around the feature seen in the atlas
X_LIM, Y_LO, Y_HI = 45.0, 40.0, 100.0


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def slice_points(tri, z, tol=0.8):
    zs = tri[:, :, 2]
    hit = (zs.min(axis=1) <= z + tol) & (zs.max(axis=1) >= z - tol)
    return tri[hit].reshape(-1, 3)


def fit_circle(xy):
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(sol[2] + cx**2 + cy**2))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return cx, cy, r, resid


def main() -> int:
    tri = read_stl(MESH)

    print("  z(mm)    n     cx      cy      r     rms")
    best, rows = None, []
    for z in np.arange(95.0, 145.0, 2.0):
        pts = slice_points(tri, z)
        sel = pts[(np.abs(pts[:, 0]) < X_LIM) & (pts[:, 1] > Y_LO) & (pts[:, 1] < Y_HI)]
        if len(sel) < 40:
            continue
        cx, cy, r, resid = fit_circle(sel[:, :2])
        rows.append((z, len(sel), cx, cy, r, resid))
        tag = ""
        if resid < 1.5 and 8.0 < r < 30.0:
            tag = "  <-- clean bore"
            if best is None or resid < best[5]:
                best = (z, len(sel), cx, cy, r, resid)
        print(f"  {z:6.1f} {len(sel):5d} {cx:7.2f} {cy:7.2f} {r:6.2f} {resid:6.2f}{tag}")

    if best is None:
        print("\nno clean circular bore found in the search window")
        return 1

    z0, _, cx, cy, r, resid = best
    # axial extent: how far the bore of this radius persists in z
    zs_in = [
        row[0]
        for row in rows
        if abs(row[2] - cx) < 3.0 and abs(row[3] - cy) < 3.0 and abs(row[4] - r) < 4.0
    ]
    z_lo, z_hi = (min(zs_in), max(zs_in)) if zs_in else (z0, z0)

    print(f"\nlens bore fit")
    print(f"  centre (tool frame) : x={cx:.2f}  y={cy:.2f} mm")
    print(f"  radius              : {r:.2f} mm  (rms {resid:.2f})")
    print(f"  axial extent        : z {z_lo:.1f} .. {z_hi:.1f} mm  (best fit at {z0:.1f})")
    print(f"  optical axis        : +z of the tool frame (toward the fingertips)")
    print(f"\nsim camera pose, attachment_site frame (metres):")
    print(f"  position = ({cx/1000:+.5f}, {cy/1000:+.5f}, {z_hi/1000:+.5f})")
    print(f"  looks along +z; fingertip plane is at z = 0.24764 m")
    print(f"  -> lens sits {247.64 - z_hi:.1f} mm behind the fingertip plane")

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    pts = slice_points(tri, z0)
    ax[0].scatter(pts[:, 0], pts[:, 1], s=2, c="#2b6cb0", linewidths=0)
    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "r-", lw=1.6, label=f"fit r={r:.1f}")
    ax[0].plot([cx], [cy], "r+", ms=12)
    ax[0].set_title(f"cross-section at z = {z0:.0f} mm")
    ax[0].set_xlabel("x  finger axis (mm)")
    ax[0].set_ylabel("y  body (mm)")
    ax[0].set_xlim(-120, 120)
    ax[0].set_ylim(-60, 120)
    ax[0].legend(loc="lower right", fontsize=9)

    v = tri.reshape(-1, 3)
    m = (np.abs(v[:, 0]) < X_LIM) & (v[:, 1] > Y_LO) & (v[:, 1] < Y_HI)
    ax[1].scatter(v[m][:, 2], v[m][:, 1], s=1, c="#444", linewidths=0)
    ax[1].axvspan(z_lo, z_hi, color="r", alpha=0.18, label="bore extent")
    ax[1].axhline(cy, color="r", lw=1.0, ls="--")
    ax[1].set_title("side profile of the boss region")
    ax[1].set_xlabel("z  approach (mm)")
    ax[1].set_ylabel("y  body (mm)")
    ax[1].legend(loc="upper right", fontsize=9)
    for a in ax:
        a.set_aspect("equal")
        a.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
