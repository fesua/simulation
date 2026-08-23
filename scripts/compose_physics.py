#!/usr/bin/env python3
"""Stack the per-physics clips into one labelled grid so the settings can be compared by eye.

Each clip is [oblique overview | near-top view] side by side. The grid keeps them time-aligned,
which is the whole point: the same scene and the same policy differ only in the physics setting,
so anything that differs between cells at the same instant is caused by the setting.
"""
from __future__ import annotations

import pathlib
import sys

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

SRC = pathlib.Path("/home/plaif/workspace/simulation/physics")
ORDER = ["current", "softfinger_kp1e4", "verysoft_kp1e3",
         "friction_hi_mu1.2", "friction_lo_mu0.2", "soft_and_grippy"]
COLS, SCALE = 3, 0.5


def main() -> int:
    clips, names = [], []
    for n in ORDER:
        f = SRC / f"{n}.mp4"
        if not f.exists():
            print(f"  skip {n} (no clip)")
            continue
        clips.append(iio.mimread(f, memtest=False))
        names.append(n)
    if not clips:
        print("no clips"); return 1
    n_fr = min(len(c) for c in clips)
    h, w = clips[0][0].shape[:2]
    cw, ch = int(w * SCALE), int(h * SCALE)
    rows = (len(clips) + COLS - 1) // COLS
    print(f"{len(clips)} clips, {n_fr} frames, cell {cw}x{ch}, grid {COLS}x{rows}")

    out = SRC / "physics_compare.mp4"
    with iio.get_writer(out, fps=30, quality=6) as wr:
        for k in range(n_fr):
            canvas = Image.new("RGB", (cw * COLS, ch * rows), (16, 16, 16))
            for i, (c, nm) in enumerate(zip(clips, names)):
                im = Image.fromarray(c[k][..., :3]).resize((cw, ch), Image.BILINEAR)
                d = ImageDraw.Draw(im)
                # label with a dark plate behind it so it reads over any background
                d.rectangle([0, 0, cw, 22], fill=(0, 0, 0))
                d.text((6, 5), f"{nm}   t={k/30:.1f}s", fill=(255, 220, 80))
                canvas.paste(im, ((i % COLS) * cw, (i // COLS) * ch))
            wr.append_data(np.asarray(canvas))
    print(f"wrote {out}")

    # also one montage per view, full size, for when the grid is too small to judge a grasp
    for tag, sl in (("overview", slice(0, w // 2)), ("topview", slice(w // 2, w))):
        o = SRC / f"physics_{tag}.mp4"
        with iio.get_writer(o, fps=30, quality=7) as wr:
            for k in range(n_fr):
                cells = []
                for c, nm in zip(clips, names):
                    im = Image.fromarray(c[k][:, sl][..., :3])
                    im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR)
                    d = ImageDraw.Draw(im)
                    d.rectangle([0, 0, im.width, 20], fill=(0, 0, 0))
                    d.text((6, 4), nm, fill=(255, 220, 80))
                    cells.append(np.asarray(im))
                r = [np.concatenate(cells[i:i + COLS], axis=1)
                     for i in range(0, len(cells), COLS)]
                if len(r) > 1 and r[-1].shape[1] < r[0].shape[1]:
                    pad = np.zeros((r[0].shape[0], r[0].shape[1] - r[-1].shape[1], 3), np.uint8)
                    r[-1] = np.concatenate([r[-1], pad], axis=1)
                wr.append_data(np.concatenate(r, axis=0))
        print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
