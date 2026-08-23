"""Locate the RealSense D405 lens on the Pika gripper from the mesh geometry.

Operator report: the camera sits in the round feature on the gripper body, above the
fingertips. This slices `pika_gripper_rzp90.STL` (already in the tool/attachment frame,
so results are directly usable as a sim camera pose) and looks for that circular boss.

Frame (tool / attachment_site, mm):
  x  finger opening axis, +-107.5
  y  body asymmetry, -43.9 .. +104.3
  z  approach axis, 0 = flange side, 247.6 = fingertip plane

Outputs a slice atlas plus a circle-fit table so the lens centre can be read off.

Run:  .venv/bin/python scripts/find_wrist_camera.py
"""

import pathlib
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
MESH = ROOT / "assets/meshes/pika_gripper_rzp90.STL"
OUT = ROOT / "outputs/wrist_camera_slices.png"


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    tri = raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
    nrm = raw[:, 0:12].copy().view("<f4").reshape(n, 3)
    return tri, nrm


def slice_points(tri, z, tol=1.2):
    """Vertices of triangles that straddle the plane z=const."""
    zs = tri[:, :, 2]
    hit = (zs.min(axis=1) <= z + tol) & (zs.max(axis=1) >= z - tol)
    return tri[hit].reshape(-1, 3)


def fit_circle(xy):
    """Algebraic (Kasa) circle fit; returns centre, radius, rms residual."""
    x, y = xy[:, 0], xy[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(sol[2] + cx**2 + cy**2))
    resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return np.array([cx, cy]), r, resid


def main() -> int:
    tri, nrm = read_stl(MESH)
    v = tri.reshape(-1, 3)
    print(f"{MESH.name}: {len(tri)} tris")
    print(
        f"  bbox  x[{v[:,0].min():7.1f},{v[:,0].max():7.1f}]"
        f"  y[{v[:,1].min():7.1f},{v[:,1].max():7.1f}]"
        f"  z[{v[:,2].min():7.1f},{v[:,2].max():7.1f}] (mm)"
    )

    # Faces whose normal points along -y are the "outer face" of the body paddle; a lens
    # bore shows up there as a ring of near-radial normals. Scan z for circular structure.
    zs = np.arange(10.0, 240.0, 10.0)
    rows = []
    for z in zs:
        pts = slice_points(tri, z)
        if len(pts) < 30:
            continue
        # candidate lens region: the body side (+y), away from the finger rails
        sel = pts[(np.abs(pts[:, 0]) < 45) & (pts[:, 1] > 0)]
        if len(sel) < 30:
            continue
        c, r, resid = fit_circle(sel[:, :2])
        rows.append((z, len(sel), c[0], c[1], r, resid))

    print("\n  z(mm)   n    cx      cy      r     rms   (circle fit on |x|<45, y>0)")
    for z, n, cx, cy, r, resid in rows:
        star = "  <-- circular" if resid < 3.0 and 8 < r < 40 else ""
        print(f"  {z:6.1f} {n:5d} {cx:7.2f} {cy:7.2f} {r:6.2f} {resid:6.2f}{star}")

    # Slice atlas
    zs_plot = np.arange(20.0, 250.0, 20.0)
    ncol = 4
    nrow = int(np.ceil(len(zs_plot) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    for ax, z in zip(axes.ravel(), zs_plot):
        pts = slice_points(tri, z)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.6, c="#2b6cb0", linewidths=0)
        ax.set_title(f"z = {z:.0f} mm", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-120, 120)
        ax.set_ylim(-60, 120)
        ax.axhline(0, color="#aaa", lw=0.5)
        ax.axvline(0, color="#aaa", lw=0.5)
        ax.grid(alpha=0.25, lw=0.4)
    for ax in axes.ravel()[len(zs_plot):]:
        ax.axis("off")
    fig.suptitle(
        "Pika gripper cross-sections in the tool frame (x = finger axis, y = body, z = approach)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(f"\nwrote {OUT}")

    # Side profile: y-z projection, which is where a forward-facing lens bore is visible
    fig2, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(v[:, 2], v[:, 1], s=0.25, c="#444", linewidths=0)
    ax[0].set_xlabel("z (approach, mm)")
    ax[0].set_ylabel("y (body, mm)")
    ax[0].set_title("side profile  (looking down the finger axis)")
    ax[1].scatter(v[:, 2], v[:, 0], s=0.25, c="#444", linewidths=0)
    ax[1].set_xlabel("z (approach, mm)")
    ax[1].set_ylabel("x (finger axis, mm)")
    ax[1].set_title("front profile  (looking along the body axis)")
    for a in ax:
        a.set_aspect("equal")
        a.grid(alpha=0.3, lw=0.4)
    fig2.tight_layout()
    out2 = ROOT / "outputs/wrist_camera_profile.png"
    fig2.savefig(out2, dpi=110)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
