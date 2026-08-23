"""Fine slice atlas of the Pika gripper's camera boss, to pick out the D405 apertures.

The coarse fit found an outer housing (r ~28 mm, z 105-117) and much smaller circles
near z 123-125. A D405 has several openings on its face (two IR imagers, the RGB
imager, the projector), so a single circle fit smears them together. These fine,
zoomed slices show the individual features so the optical centre can be chosen.

Run:  .venv/bin/python scripts/zoom_wrist_camera.py
"""

import pathlib
import struct

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
MESH = ROOT / "assets/meshes/pika_gripper_rzp90.STL"
OUT = ROOT / "outputs/wrist_camera_zoom.png"


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3)


def slice_points(tri, z, tol=0.6):
    zs = tri[:, :, 2]
    hit = (zs.min(axis=1) <= z + tol) & (zs.max(axis=1) >= z - tol)
    return tri[hit].reshape(-1, 3)


def main() -> int:
    tri = read_stl(MESH)
    zs = np.arange(104.0, 136.0, 2.0)
    ncol = 4
    nrow = int(np.ceil(len(zs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow))
    for ax, z in zip(axes.ravel(), zs):
        p = slice_points(tri, z)
        m = (np.abs(p[:, 0]) < 55) & (p[:, 1] > 35) & (p[:, 1] < 110)
        ax.scatter(p[m][:, 0], p[m][:, 1], s=3, c="#2b6cb0", linewidths=0)
        ax.set_title(f"z = {z:.0f} mm   (n={m.sum()})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-55, 55)
        ax.set_ylim(35, 110)
        ax.grid(alpha=0.3, lw=0.4)
        ax.axvline(0, color="#bbb", lw=0.5)
    for ax in axes.ravel()[len(zs):]:
        ax.axis("off")
    fig.suptitle(
        "Pika camera boss, fine slices  (x = finger axis, y = body; tip plane at z=247.6)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")

    # where does material actually end on the +z face of the boss?
    v = tri.reshape(-1, 3)
    boss = v[(np.abs(v[:, 0]) < 40) & (v[:, 1] > 45) & (v[:, 1] < 105)]
    print(f"\nboss region vertices: {len(boss)}")
    print(f"  z range: {boss[:,2].min():.1f} .. {boss[:,2].max():.1f} mm")
    for lo, hi in ((100, 110), (110, 118), (118, 122), (122, 126), (126, 132)):
        s = boss[(boss[:, 2] >= lo) & (boss[:, 2] < hi)]
        if len(s) == 0:
            continue
        print(
            f"  z {lo:3d}-{hi:3d}: n={len(s):5d}  "
            f"x[{s[:,0].min():6.1f},{s[:,0].max():6.1f}]  y[{s[:,1].min():6.1f},{s[:,1].max():6.1f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
