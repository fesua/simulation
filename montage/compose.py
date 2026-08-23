"""Compose the expected-simulation montage grid with labels + real-frame reference."""
from PIL import Image, ImageDraw, ImageFont

PW, PH = 640, 480
BAR = 34
FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 20)
TITLE_FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 26)


def panel(img_path, label):
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((PW, PH))
    canvas = Image.new("RGB", (PW, PH + BAR), (18, 18, 20))
    canvas.paste(im, ((PW - im.width) // 2, BAR + (PH - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.text((10, 7), label, font=FONT, fill=(235, 235, 235))
    return canvas


cells = [
    ("out_aligned_overview.png", "[SIM] 셀 오버뷰 — reset pose, 녹색/회색 박스(바닥 있음)"),
    ("out_aligned_top.png", "[SIM] 탑뷰 정렬 — 검정→좌 녹색박스, 회색→우 회색박스"),
    ("out_random_top.png", "[SIM] 탑뷰 무작위 혼합 (M12x25, 평가 조건)"),
    ("out_aligned_L_wrist.png", "[SIM] 좌 손목캠 예상 뷰 (정렬)"),
    ("out_random_R_wrist.png", "[SIM] 우 손목캠 예상 뷰 (무작위)"),
    ("lr_ep_f1.png", "[REAL] 실기 손목캠 L|R (2026-06-28 에피소드)"),
]

TITLE_H = 44
W = PW * 3
H = TITLE_H + (PH + BAR) * 2
out = Image.new("RGB", (W, H), (10, 10, 12))
d = ImageDraw.Draw(out)
d.text(
    (12, 9),
    "M12 볼트 pick-place 시뮬레이션 예상 몽타주 — robotics_lab 자산 기반 (MuJoCo 프리뷰, Isaac Sim 구축 목표 구도)",
    font=TITLE_FONT,
    fill=(240, 240, 240),
)
for i, (p, label) in enumerate(cells):
    r, c = divmod(i, 3)
    out.paste(panel(p, label), (c * PW, TITLE_H + r * (PH + BAR)))
out.save("expected_sim_montage.png")
print("saved", out.size)
