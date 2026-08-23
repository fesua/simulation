"""Locate the RealSense D405 module on the Pika gripper.

Operator correction: the round dome at y ~ 73 is the fisheye lens (unused). The D405
is the flat, elongated feature near z ~ 120 mm spanning x -15..15, y 35..55 in the
tool frame. A D405 is a stereo module (18 mm baseline) whose colour stream comes from
the left imager, so this resolves the individual apertures rather than fitting one
circle to the whole face.

Run:  .venv/bin/python scripts/fit_d405.py
"""

import pathlib
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
MESH = ROOT / "assets/meshes/pika_gripper_rzp90.STL"

X_LIM = (-28.0, 28.0)
Y_LIM = (28.0, 62.0)
Z_SCAN = np.arange(108.0, 130.0, 1.0)


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def slice_points(tri, z, tol=0.5):
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
    v = tri.reshape(-1, 3)
    win = v[
        (v[:, 0] > X_LIM[0]) & (v[:, 0] < X_LIM[1])
        & (v[:, 1] > Y_LIM[0]) & (v[:, 1] < Y_LIM[1])
        & (v[:, 2] > 100.0)
    ]
    print(f"window vertices: {len(win)}")
    print(
        f"  x[{win[:,0].min():6.1f},{win[:,0].max():6.1f}]  "
        f"y[{win[:,1].min():6.1f},{win[:,1].max():6.1f}]  "
        f"z[{win[:,2].min():6.1f},{win[:,2].max():6.1f}] (mm)"
    )
    z_face = float(win[:, 2].max())
    print(f"  front-most material in window (lens face): z = {z_face:.2f} mm")

    print("\n  z(mm)    n   |  left aperture (x<0)        |  right aperture (x>0)")
    print("                 |  cx      cy      r    rms   |  cx      cy      r    rms")
    per_z = []
    for z in Z_SCAN:
        p = slice_points(tri, z)
        m = (p[:, 0] > X_LIM[0]) & (p[:, 0] < X_LIM[1]) & (p[:, 1] > Y_LIM[0]) & (p[:, 1] < Y_LIM[1])
        s = p[m]
        if len(s) < 20:
            continue
        out = [f"  {z:6.1f} {len(s):5d}  |"]
        fits = {}
        for tag, sel in (("L", s[s[:, 0] < 0]), ("R", s[s[:, 0] > 0])):
            if len(sel) < 12:
                out.append("      --      --     --    --  |")
                continue
            cx, cy, r, resid = fit_circle(sel[:, :2])
            fits[tag] = (cx, cy, r, resid)
            out.append(f" {cx:7.2f} {cy:7.2f} {r:6.2f} {resid:5.2f} |")
        print("".join(out))
        if len(fits) == 2 and all(f[3] < 1.2 and 2.0 < f[2] < 12.0 for f in fits.values()):
            per_z.append((z, fits))

    if per_z:
        zs = [p[0] for p in per_z]
        L = np.array([[p[1]["L"][0], p[1]["L"][1], p[1]["L"][2]] for p in per_z])
        R = np.array([[p[1]["R"][0], p[1]["R"][1], p[1]["R"][2]] for p in per_z])
        lc, rc = L.mean(axis=0), R.mean(axis=0)
        base = float(np.hypot(rc[0] - lc[0], rc[1] - lc[1]))
        mid = ((lc[0] + rc[0]) / 2, (lc[1] + rc[1]) / 2)
        print(f"\ntwo apertures resolved over z {min(zs):.0f}..{max(zs):.0f} mm")
        print(f"  left  imager : x={lc[0]:7.2f}  y={lc[1]:7.2f}  r={lc[2]:5.2f} mm")
        print(f"  right imager : x={rc[0]:7.2f}  y={rc[1]:7.2f}  r={rc[2]:5.2f} mm")
        print(f"  baseline     : {base:.2f} mm   (D405 nominal 18 mm)")
        print(f"  midpoint     : x={mid[0]:7.2f}  y={mid[1]:7.2f} mm")
        print(f"\nsim camera pose candidates, attachment_site frame (metres, axis +z):")
        print(f"  colour = LEFT imager : ({lc[0]/1000:+.5f}, {lc[1]/1000:+.5f}, {z_face/1000:+.5f})")
        print(f"  stereo midpoint      : ({mid[0]/1000:+.5f}, {mid[1]/1000:+.5f}, {z_face/1000:+.5f})")
    else:
        print("\ncould not resolve two clean apertures; see the plot")

    zs_plot = np.arange(112.0, 128.0, 2.0)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, z in zip(axes.ravel(), zs_plot):
        p = slice_points(tri, z)
        m = (p[:, 0] > X_LIM[0]) & (p[:, 0] < X_LIM[1]) & (p[:, 1] > Y_LIM[0]) & (p[:, 1] < Y_LIM[1])
        ax.scatter(p[m][:, 0], p[m][:, 1], s=6, c="#c53030", linewidths=0)
        ax.set_title(f"z = {z:.0f} mm  (n={m.sum()})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.grid(alpha=0.3, lw=0.4)
        ax.axvline(0, color="#bbb", lw=0.5)
    fig.suptitle("RealSense D405 module on the Pika gripper (tool frame, mm)", fontsize=13)
    fig.tight_layout()
    out = ROOT / "outputs/d405_slices.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
