"""Compose the four tool-yaw candidates into one labelled comparison sheet.

Run:  .venv/bin/python scripts/compose_yaw_sheet.py
"""

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/tool_yaw_candidates.png"

FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 22)
TITLE_FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 28)

PW, PH, BAR = 640, 480, 36
TITLE_H = 46

CELLS = [
    ("axial_yawp0.png", "yaw 0°  (현재에서 −90°)"),
    ("axial_yawp90.png", "yaw 90°  ← 현재 URDF 값"),
    ("axial_yawp180.png", "yaw 180°  (현재에서 +90°)"),
    ("axial_yawp270.png", "yaw 270°  (현재에서 −180°)"),
]


def panel(path: pathlib.Path, label: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((PW, PH))
    canvas = Image.new("RGB", (PW, PH + BAR), (18, 18, 20))
    canvas.paste(im, ((PW - im.width) // 2, BAR + (PH - im.height) // 2))
    ImageDraw.Draw(canvas).text((10, 8), label, font=FONT, fill=(240, 240, 240))
    return canvas


def main() -> int:
    sheet = Image.new("RGB", (PW * 2, TITLE_H + (PH + BAR) * 2), (10, 10, 12))
    ImageDraw.Draw(sheet).text(
        (12, 10),
        "Pika tool visual yaw 후보 — 왼팔, reset pose, 툴 축 방향 뷰 (attachment frame)",
        font=TITLE_FONT,
        fill=(245, 245, 245),
    )
    for i, (fn, label) in enumerate(CELLS):
        p = ROOT / "outputs" / fn
        if not p.exists():
            print(f"missing: {p}")
            continue
        r, c = divmod(i, 2)
        sheet.paste(panel(p, label), (c * PW, TITLE_H + r * (PH + BAR)))
    sheet.save(OUT)
    print(f"wrote {OUT}  {sheet.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
