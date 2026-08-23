#!/usr/bin/env python3
"""Pika GRIPPER tips vs Pika SENSE tips, straight from the STEP-derived STLs.

The two STEP files do not share an origin (their jaw axes sit 30.8 mm apart in x and their tip
planes 46.9 mm apart in z), so any comparison has to state its datum. Two are drawn:

  aligned BY TIP    -- fingertip plane to z=0, jaw axis to x=0.  Shows shape and jaw-gap
                       differences with the working end held fixed.
  aligned BY MOUNT  -- the base (proximal) plane to z=0 instead. This is the one that answers
                       "is one tip longer", because it holds the carriage interface fixed and
                       lets the tips fall where they fall.

Both are shown because they answer different questions and disagreeing about which datum is in
use is exactly how this project has produced wrong conclusions before.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/plaif/workspace/simulation/scripts")
from gripper_sections import read_stl, pinch_x, draw, frame, STEP, OUT   # noqa: E402

GRIPPER = ("pika GRIPPER tip", STEP / "gripper_finger_L.stl", STEP / "gripper_finger_R.stl", "#d62728")
SENSE = ("pika SENSE tip", STEP / "sense_finger_assy_L.stl", STEP / "sense_finger_assy_R.stl", "#2ca02c")


def load(lf, rf):
    L, R = read_stl(lf), read_stl(rf)
    xl, xr = pinch_x(L, "L"), pinch_x(R, "R")
    axis = (xl + xr) / 2.0
    V = np.concatenate([L, R]).reshape(-1, 3)
    return dict(L=L, R=R, axis=axis, half=(xr - xl) / 2.0,
                ztip=V[:, 2].max(), zbase=V[:, 2].min(), ylo=V[:, 1].min(), yhi=V[:, 1].max())


def shifted(d, datum):
    z0 = d["ztip"] if datum == "tip" else d["zbase"]
    s = np.array([-d["axis"], 0.0, -z0])
    return d["L"] + s, d["R"] + s


def dims(name, d):
    L = d["L"].reshape(-1, 3)
    xin = pinch_x(d["L"], "L")
    face = L[np.abs(L[:, 0] - xin) < 2.0]
    print(f"  {name}")
    print(f"      총 길이 (z)          {d['ztip'] - d['zbase']:7.1f} mm")
    print(f"      무는 면 z 범위        {face[:, 2].min() - d['zbase']:7.1f} .. "
          f"{face[:, 2].max() - d['zbase']:.1f} mm  (마운트 기준)")
    print(f"      무는 면 길이          {face[:, 2].max() - face[:, 2].min():7.1f} mm")
    print(f"      팁에서 무는 면 끝까지  {d['ztip'] - face[:, 2].max():7.1f} mm")
    print(f"      열림 반간격           {d['half']:7.1f} mm   (전체 {2 * d['half']:.1f})")
    print(f"      핑거 두께 (x)         {L[:, 0].max() - L[:, 0].min():7.1f} mm")
    print(f"      폭 (y)               {d['yhi'] - d['ylo']:7.1f} mm")
    print(f"      as-authored          axis x={d['axis']:.1f}, tip z={d['ztip']:.1f}, "
          f"base z={d['zbase']:.1f}")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    G, S = load(*GRIPPER[1:3]), load(*SENSE[1:3])
    print("STEP 팁 실측:")
    dims(GRIPPER[0], G); dims(SENSE[0], S)

    # --- one sheet per variant -----------------------------------------------------
    for (nm, _, _, col), d in ((GRIPPER, G), (SENSE, S)):
        Lp, Rp = shifted(d, "tip")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        for ax, pl in zip(axes, ("xz", "yz", "xy")):
            sec = {"xz": 0.0, "yz": -d["half"], "xy": -34.0}[pl]
            draw(ax, Lp, pl, col, 0.30, section=sec)
            draw(ax, Rp, pl, col, 0.30, section=sec)
            frame(ax, pl, f"{nm}  |  {pl.upper()}  (section {sec:+.0f} mm)")
        fig.suptitle(f"{nm}   길이 {d['ztip'] - d['zbase']:.1f} mm, 열림 반간격 {d['half']:.1f} mm "
                     f"(as authored: axis x={d['axis']:.1f}, tip z={d['ztip']:.1f})", fontsize=11)
        fig.tight_layout()
        f = OUT / f"tip_{'gripper' if d is G else 'sense'}.png"
        fig.savefig(f, dpi=130); plt.close(fig); print(f"  wrote {f.name}")

    # --- overlays, one per datum ---------------------------------------------------
    for datum, note in (("tip", "팁 평면 z=0 (형상·간격 비교)"),
                        ("mount", "마운트 평면 z=0 (길이 비교)")):
        fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
        for ax, pl in zip(axes, ("xz", "yz", "xy")):
            for (nm, _, _, col), d in ((GRIPPER, G), (SENSE, S)):
                Lp, Rp = shifted(d, datum)
                sec = {"xz": 0.0, "yz": -d["half"], "xy": None}[pl]
                if pl == "xy":
                    sec = (Lp.reshape(-1, 3)[:, 2].min() + Lp.reshape(-1, 3)[:, 2].max()) / 2
                draw(ax, Lp, pl, col, 0.26, section=sec)
                draw(ax, Rp, pl, col, 0.26)
            frame(ax, pl, f"{pl.upper()}   ({note})",
                  zlabel="z (mount at 0)" if datum == "mount" else None)
        h = [plt.Line2D([], [], color=c, lw=6, alpha=0.55,
                        label=f"{n}   길이 {d['ztip']-d['zbase']:.1f}mm, 반간격 {d['half']:.1f}mm")
             for (n, _, _, c), d in ((GRIPPER, G), (SENSE, S))]
        axes[0].legend(handles=h, fontsize=9, loc="lower left")
        fig.suptitle(f"pika GRIPPER tip vs pika SENSE tip — aligned by {datum.upper()}", fontsize=12)
        fig.tight_layout()
        f = OUT / f"tips_overlay_by_{datum}.png"
        fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f.name}")

    # --- the sense assembly broken into its parts ----------------------------------
    parts = [("sense_arm", "#9467bd"), ("sense_blade", "#8c564b"), ("sense_pad", "#e377c2")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    for ax, pl in zip(axes, ("xz", "yz", "xy")):
        for nm, col in parts:
            d = load(STEP / f"{nm}_L.stl", STEP / f"{nm}_R.stl")
            # keep the ASSEMBLY datum so the parts stay in their true relative places
            s = np.array([-S["axis"], 0.0, -S["ztip"]])
            draw(ax, d["L"] + s, pl, col, 0.32)
            draw(ax, d["R"] + s, pl, col, 0.32)
        frame(ax, pl, f"pika SENSE, parts in assembly position  |  {pl.upper()}")
    axes[0].legend(handles=[plt.Line2D([], [], color=c, lw=6, alpha=0.6, label=n)
                            for n, c in parts], fontsize=9, loc="lower left")
    fig.tight_layout(); f = OUT / "tip_sense_parts.png"
    fig.savefig(f, dpi=140); plt.close(fig); print(f"  wrote {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
