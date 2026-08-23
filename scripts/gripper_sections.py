#!/usr/bin/env python3
"""Orthographic silhouettes and true cross-sections of every gripper variant, plus overlays.

Why this exists: three separate conclusions on this project have already been wrong because a
mesh frame was ASSUMED rather than checked (the tool +90 deg bake, the lens offset used as if it
were TCP-frame, and -- an hour ago -- the STEP origin taken for the tool axis, which made the sim
look 10 mm narrower than it is). Pictures with the datum drawn on them are cheaper than another
retraction.

Frames. Each variant is re-centred on ITS OWN jaw axis, taken as the midpoint of the left and
right pinch faces, and shifted so the fingertip plane is z = 0. That makes half-gap and tip
distance directly comparable across CAD files that do not share an origin. The as-authored
coordinates are printed alongside so nothing is hidden by the normalisation.
"""
from __future__ import annotations

import pathlib
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Labels are Korean; DejaVu has no Hangul and silently renders tofu boxes.
for _f in ("Noto Sans CJK KR", "NanumSquare", "Noto Serif CJK KR"):
    if any(f.name == _f for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.collections import PolyCollection, LineCollection

OUT = pathlib.Path("/home/plaif/workspace/simulation/grippers")
SIM = pathlib.Path("/home/plaif/workspace/robotics_lab/rb_servo_server/descriptions/"
                   "meshes/robots/rb3_730e/visual/tool")
STEP = pathlib.Path("/home/plaif/workspace/pika_gripper_tip/analysis/stl/orig")
TRAVEL = 47.0          # mm, FINGER_TRAVEL_M = 0.047 in both stack_real.yaml and the sim


def read_stl(p) -> np.ndarray:
    b = pathlib.Path(p).read_bytes()
    if b[:5].lower() == b"solid" and b"facet" in b[:2000]:
        v = np.array([l.split()[1:4] for l in b.decode("utf-8", "ignore").splitlines()
                      if l.split()[:1] == ["vertex"]], float)
        return v.reshape(-1, 3, 3)
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    return a[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(float)


def pinch_x(T, side):
    """x of the face nearest the jaw axis: max for a left finger, min for a right one."""
    x = T.reshape(-1, 3)[:, 0]
    return x.max() if side == "L" else x.min()


def slice_tris(T, axis, val):
    """Segments where the triangles cross the plane axis == val."""
    segs = []
    d = T[:, :, axis] - val
    sgn = np.sign(d)
    cross = ~((sgn[:, 0] == sgn[:, 1]) & (sgn[:, 1] == sgn[:, 2]))
    for tri, dd in zip(T[cross], d[cross]):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if dd[i] == 0.0:
                pts.append(tri[i])
            if dd[i] * dd[j] < 0:
                t = dd[i] / (dd[i] - dd[j])
                pts.append(tri[i] + t * (tri[j] - tri[i]))
        if len(pts) >= 2:
            segs.append([pts[0], pts[1]])
    return np.asarray(segs) if segs else np.zeros((0, 2, 3))


PLANES = {"xy": (0, 1, 2, "x (jaw axis)", "y"),
          "xz": (0, 2, 1, "x (jaw axis)", "z (tip at 0)"),
          "yz": (1, 2, 0, "y", "z (tip at 0)")}


def draw(ax, T, plane, color, alpha=0.30, label=None, section=None):
    a, b, n, _, _ = PLANES[plane]
    ax.add_collection(PolyCollection(T[:, :, [a, b]], facecolors=color, edgecolors="none",
                                     alpha=alpha, label=label))
    if section is not None:
        s = slice_tris(T, n, section)
        if len(s):
            ax.add_collection(LineCollection(s[:, :, [a, b]], colors=color, linewidths=0.9))


def frame(ax, plane, title, xlim=None, ylim=None, zlabel=None):
    _, _, _, xl, yl = PLANES[plane]
    if zlabel:                       # the datum is not always the tip; never mislabel it
        xl = xl.replace("z (tip at 0)", zlabel); yl = yl.replace("z (tip at 0)", zlabel)
    ax.set_xlabel(f"{xl}  [mm]"); ax.set_ylabel(f"{yl}  [mm]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.4); ax.set_aspect("equal")
    ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)


def main() -> int:
    OUT.mkdir(exist_ok=True)

    # variant -> (left mesh, right mesh, optional base mesh)
    VARIANTS = {
        "sim(pika_finger)": (SIM / "pika_finger_left.STL", SIM / "pika_finger_right.STL",
                             SIM / "pika_gripper_base.STL"),
        "STEP(gripper)":    (STEP / "gripper_finger_L.stl", STEP / "gripper_finger_R.stl", None),
        "STEP(sense_assy)": (STEP / "sense_finger_assy_L.stl", STEP / "sense_finger_assy_R.stl", None),
        "STEP(sense_arm)":  (STEP / "sense_arm_L.stl", STEP / "sense_arm_R.stl", None),
        "STEP(sense_blade)": (STEP / "sense_blade_L.stl", STEP / "sense_blade_R.stl", None),
        "STEP(sense_pad)":  (STEP / "sense_pad_L.stl", STEP / "sense_pad_R.stl", None),
    }
    COLORS = dict(zip(VARIANTS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]))

    norm = {}
    print(f"{'variant':>20} {'as-authored 안쪽면 L/R':>26} {'축':>8} {'반간격':>8} {'팁 z':>9}")
    for name, (lf, rf, bf) in VARIANTS.items():
        L, R = read_stl(lf), read_stl(rf)
        xl, xr = pinch_x(L, "L"), pinch_x(R, "R")
        axis = (xl + xr) / 2.0                 # the jaw axis this CAD actually uses
        half = (xr - xl) / 2.0
        ztip = max(L.reshape(-1, 3)[:, 2].max(), R.reshape(-1, 3)[:, 2].max())
        sh = np.array([-axis, 0.0, -ztip])
        norm[name] = dict(L=L + sh, R=R + sh, base=(read_stl(bf) + sh) if bf else None,
                          half=half, axis=axis, ztip=ztip)
        print(f"{name:>20} {xl:11.1f} / {xr:<11.1f} {axis:8.1f} {half:7.1f}mm {ztip:8.1f}")

    # ---- per-variant sheets: three planes, silhouette + mid-plane section ------------
    for name, d in norm.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        for ax, pl in zip(axes, ("xz", "yz", "xy")):
            sec = {"xz": 0.0, "yz": -d["half"], "xy": -34.0}[pl]   # y=0 / left pinch face / 34mm up from tip
            for T, c in ((d["L"], COLORS[name]), (d["R"], COLORS[name])):
                draw(ax, T, pl, c, 0.28, section=sec)
            if d["base"] is not None:
                draw(ax, d["base"], pl, "0.45", 0.18, section=sec)
            frame(ax, pl, f"{name}  |  {pl.upper()}  (section at {sec:+.0f} mm)")
        fig.suptitle(f"{name}   open half-gap {d['half']:.1f} mm   "
                     f"(as authored: axis x={d['axis']:.1f}, tip z={d['ztip']:.1f})", fontsize=11)
        fig.tight_layout()
        f = OUT / f"variant_{name.replace('(', '_').replace(')', '')}.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f"  wrote {f.name}")

    # ---- overlay of every variant, normalised ---------------------------------------
    for pl in ("xz", "yz", "xy"):
        fig, ax = plt.subplots(figsize=(9, 8))
        for name, d in norm.items():
            draw(ax, d["L"], pl, COLORS[name], 0.22, label=f"{name}  ±{d['half']:.1f}mm")
            draw(ax, d["R"], pl, COLORS[name], 0.22)
        frame(ax, pl, f"ALL VARIANTS overlaid, normalised (jaw axis x=0, tip z=0)  |  {pl.upper()}")
        h = [plt.Line2D([], [], color=COLORS[n], lw=6, alpha=0.5,
                        label=f"{n}  half-gap {norm[n]['half']:.1f}mm") for n in norm]
        ax.legend(handles=h, fontsize=8, loc="lower right")
        fig.tight_layout(); f = OUT / f"overlay_all_{pl}.png"
        fig.savefig(f, dpi=140); plt.close(fig)
        print(f"  wrote {f.name}")

    # ---- the sim gripper OPEN vs CLOSED ---------------------------------------------
    d = norm["sim(pika_finger)"]
    states = {"open (grip=100)": 0.0, "closed (grip=0)": TRAVEL}
    for lab, t in states.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        for ax, pl in zip(axes, ("xz", "yz", "xy")):
            sec = {"xz": 0.0, "yz": -(d["half"] - t), "xy": -34.0}[pl]
            draw(ax, d["L"] + [t, 0, 0], pl, "#1f77b4", 0.30, section=sec)
            draw(ax, d["R"] - [t, 0, 0], pl, "#1f77b4", 0.30, section=sec)
            draw(ax, d["base"], pl, "0.45", 0.15)
            frame(ax, pl, f"sim {lab}  |  {pl.upper()}")
        fig.suptitle(f"sim gripper {lab}: half-gap {d['half'] - t:.1f} mm, "
                     f"full gap {2 * (d['half'] - t):.1f} mm   (M12 head 18.4 mm)", fontsize=11)
        fig.tight_layout()
        f = OUT / f"sim_{lab.split()[0]}.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f"  wrote {f.name}")

    for pl in ("xz", "xy"):
        fig, ax = plt.subplots(figsize=(9, 8))
        draw(ax, d["L"], pl, "#1f77b4", 0.20, label="open")
        draw(ax, d["R"], pl, "#1f77b4", 0.20)
        draw(ax, d["L"] + [TRAVEL, 0, 0], pl, "#d62728", 0.35, label="closed")
        draw(ax, d["R"] - [TRAVEL, 0, 0], pl, "#d62728", 0.35)
        draw(ax, d["base"], pl, "0.45", 0.12)
        if pl == "xz":
            for s in (+1, -1):
                ax.axvline(s * 9.2, color="k", ls="--", lw=1.0)
            ax.text(0, -20, "M12 head Ø18.4", ha="center", fontsize=9)
        frame(ax, pl, f"sim gripper OPEN vs CLOSED  |  {pl.upper()}   "
                      f"(travel {TRAVEL:.0f} mm -> full gap {2 * (d['half'] - TRAVEL):.1f} mm)")
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, alpha=0.5, label=l)
                           for c, l in (("#1f77b4", f"open  ±{d['half']:.1f}mm"),
                                        ("#d62728", f"closed ±{d['half'] - TRAVEL:.1f}mm"))],
                  fontsize=9, loc="lower right")
        fig.tight_layout(); f = OUT / f"overlay_open_vs_closed_{pl}.png"
        fig.savefig(f, dpi=140); plt.close(fig)
        print(f"  wrote {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
