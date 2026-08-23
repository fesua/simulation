"""Wrist-camera montage: gripper open vs closed, both arms, plus the policy-format crop.

Bottom row is exactly what the policy would receive (openpi resize_with_pad 224x224),
so the jaw state can be judged in the frame the model actually sees.

Run:  .venv/bin/python scripts/compose_grip_montage.py
"""

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/wristcam_grip_montage.png"
F = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 22)
T = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 27)

CW, CH, BAR, TH = 520, 390, 34, 46


def cell(path: pathlib.Path, label: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((CW, CH))
    c = Image.new("RGB", (CW, CH + BAR), (18, 18, 20))
    c.paste(im, ((CW - im.width) // 2, BAR + (CH - im.height) // 2))
    ImageDraw.Draw(c).text((10, 6), label, font=F, fill=(240, 240, 240))
    return c


def main() -> int:
    rows = [
        [
            ("wc_g100_left.png", "좌완 손목캠 — grip 100 (열림)"),
            ("wc_g0_left.png", "좌완 손목캠 — grip 0 (닫힘)"),
        ],
        [
            ("wc_g100_right.png", "우완 손목캠 — grip 100 (열림)"),
            ("wc_g0_right.png", "우완 손목캠 — grip 0 (닫힘)"),
        ],
        [
            ("pol_g100_left.png", "[정책 입력] 224×224 pad — 열림"),
            ("pol_g0_left.png", "[정책 입력] 224×224 pad — 닫힘"),
        ],
    ]
    sheet = Image.new("RGB", (CW * 2, TH + (CH + BAR) * len(rows)), (10, 10, 12))
    ImageDraw.Draw(sheet).text(
        (12, 10),
        "Pika 그리퍼 개폐 — 손목 D405 뷰  (finger_pos = (1 − grip/100) × 47 mm)",
        font=T,
        fill=(245, 245, 245),
    )
    for r, row in enumerate(rows):
        for c, (fn, lab) in enumerate(row):
            p = ROOT / "outputs" / fn
            if not p.exists():
                print(f"missing {fn}")
                return 1
            sheet.paste(cell(p, lab), (c * CW, TH + r * (CH + BAR)))
    sheet.save(OUT)
    print(f"wrote {OUT}  {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
