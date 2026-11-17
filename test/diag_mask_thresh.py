"""输出 103438 所有 81 格的 red_mask/black_mask 比例，并尝试调整阈值重跑 detect_pieces。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import analyze_image

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
a = analyze_image(path)
h, w = a.image.shape[:2]
hsv = cv2.cvtColor(a.image, cv2.COLOR_BGR2HSV)

# 当前阈值
red_mask_old = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask_old = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)

# 尝试放宽黑色阈值：V 上提到 220，S 放宽到 200
black_mask_new = (hsv[..., 0] < 30) & (hsv[..., 2] > 40) & (hsv[..., 2] < 230) & (hsv[..., 1] < 200)

# 尝试放宽红色阈值：H 范围扩大到 0~16（红偏橙）或 160~180
red_mask_new = ((hsv[..., 0] <= 16) | (hsv[..., 0] >= 160)) & (hsv[..., 1] > 40) & (hsv[..., 2] > 80)

row0, col0, cell = a.row0, a.col0, a.cell
rad = int(cell * 0.22)

print("所有 81 格的 mask 比例 (old/new)：")
print(f"{'':>6} {'rr_old':>7} {'br_old':>7} {'rr_new':>7} {'br_new':>7} {'grid':>6}")
missed = []
for r in range(10):
    for c in range(9):
        cx = int(col0 + c * cell)
        cy = int(row0 + r * cell)
        y0, y1 = max(0, cy - rad), min(h, cy + rad)
        x0, x1 = max(0, cx - rad), min(w, cx + rad)
        rr_o = red_mask_old[y0:y1, x0:x1].mean()
        br_o = black_mask_old[y0:y1, x0:x1].mean()
        rr_n = red_mask_new[y0:y1, x0:x1].mean()
        br_n = black_mask_new[y0:y1, x0:x1].mean()
        g = a.grid_raw[r][c]
        if g != '.':
            print(f"  {r},{c}   {rr_o:>7.3f} {br_o:>7.3f} {rr_n:>7.3f} {br_n:>7.3f}   {g}")
        else:
            # 看实际可能漏掉的（有 mask 但没检测到的）
            if rr_o > 0.01 or br_o > 0.01 or rr_n > 0.02 or br_n > 0.02:
                missed.append((r, c, rr_o, br_o, rr_n, br_n))
                print(f"  {r},{c}   {rr_o:>7.3f} {br_o:>7.3f} {rr_n:>7.3f} {br_n:>7.3f}   .  <-- MISS")

print("\n漏掉但有 mask 信号的格子:")
for item in missed:
    print(f"  row{item[0]} col{item[1]}  rr_old={item[2]:.3f} br_old={item[3]:.3f} rr_new={item[4]:.3f} br_new={item[5]:.3f}")

# 用新阈值重跑 classify 过程看看 grid 会不会变好
print("\n\n===== 用新阈值重跑 classify =====")
piece_mask_new = (red_mask_new | black_mask_new).astype(np.uint8)
from engine.recognize import find_grid

crop = a.image[a.crop_box[1]:a.crop_box[3], 0:a.crop_box[2]]
pieces_new = []
num, labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask_new, connectivity=8)
scale = crop.shape[1] / 720.0
area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
min_side = max(8, int(15 * scale))
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    if not (area_lo <= area <= area_hi):
        continue
    w_cc, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if w_cc < min_side or h_cc < min_side or min(w_cc, h_cc) / max(w_cc, h_cc) < 0.55:
        continue
    cx, cy = cents[i]
    r = int(15 * scale)
    y0s, y1s = max(0, int(cy) - r), min(crop.shape[0], int(cy) + r)
    x0s, x1s = max(0, int(cx) - r), min(crop.shape[1], int(cx) + r)
    rp = red_mask_new[a.crop_box[1]+y0s:a.crop_box[1]+y1s, x0s:x1s].mean()
    bp = black_mask_new[a.crop_box[1]+y0s:a.crop_box[1]+y1s, x0s:x1s].mean()
    pieces_new.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))

print(f"新阈值 detect_pieces 得到 {len(pieces_new)} 个棋子 (旧阈值是 50)")
for cx, cy, color, area in sorted(pieces_new, key=lambda p: (round(p[1]/35.5), p[0])):
    print(f"  cx={cx:6.1f} cy={cy:6.1f}  color={color}  area={area:.0f}")
