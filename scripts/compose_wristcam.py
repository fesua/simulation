"""Put the CAD-derived sim wrist views next to real wrist frames for step-4 comparison.

The real reference is a side-by-side L|R capture from the wrist RealSenses; it is split
back into halves so each sim view sits directly under its real counterpart.

Run:  .venv/bin/python scripts/compose_wristcam.py --layout aligned
"""

import argparse
import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "montage/lr_ep_f1.png"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 22)
TITLE = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 26)

CW, CH, BAR, TH = 560, 420, 34, 44


def cell(img: Image.Image, label: str) -> Image.Image:
    im = img.convert("RGB").copy()
    im.thumbnail((CW, CH))
    c = Image.new("RGB", (CW, CH + BAR), (18, 18, 20))
    c.paste(im, ((CW - im.width) // 2, BAR + (CH - im.height) // 2))
    ImageDraw.Draw(c).text((10, 6), label, font=FONT, fill=(240, 240, 240))
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="aligned")
    args = ap.parse_args()

    real = Image.open(REAL)
    half = real.width // 2
    real_l = real.crop((0, 0, half, real.height))
    real_r = real.crop((half, 0, real.width, real.height))

    cells = []
    for side, rimg in (("left", real_l), ("right", real_r)):
        p = ROOT / f"outputs/wristcam_{args.layout}_{side}.png"
        if not p.exists():
            print(f"missing {p.name}")
            return 1
        cells.append((cell(rimg, f"[REAL] {side} wrist"), cell(Image.open(p), f"[SIM] {side} wrist (D405 CAD pose)")))

    sheet = Image.new("RGB", (CW * 2, TH + (CH + BAR) * 2), (10, 10, 12))
    ImageDraw.Draw(sheet).text(
        (12, 9),
        "손목캠 — 실기 vs CAD 유도 D405 pose (left imager, 광축 +z_tool)",
        font=TITLE,
        fill=(245, 245, 245),
    )
    for i, (r, s) in enumerate(cells):
        sheet.paste(r, (i * CW, TH))
        sheet.paste(s, (i * CW, TH + CH + BAR))
    out = ROOT / f"outputs/wristcam_compare_{args.layout}.png"
    sheet.save(out)
    print(f"wrote {out}  {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
