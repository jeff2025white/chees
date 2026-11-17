"""调整 black_mask 阈值，看看能不能覆盖 row9 col1/col7。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify, infer_pieces, detect_red_bottom

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

# 旧阈值
red_old = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
blk_old = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)

# 新阈值：black_mask 放宽 V 到 250, S 到 130
blk_new = (hsv[..., 0] < 30) & (hsv[..., 2] > 40) & (hsv[..., 2] < 250) & (hsv[..., 1] < 130)

# red_mask 也调整一下：H<16 或 H>160, S>50, V>80
red_new = ((hsv[..., 0] < 16) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 80)

# 先看看新 black_mask 在 row9 col1/col7 的覆盖
cell = 76.3
row0, col0 = 180.3, 55.7
for name, cx, cy in [("row9 col1", col0+1*cell, row0+9*cell), 
                      ("row9 col7", col0+7*cell, row0+9*cell),
                      ("row9 col0 (能检测)", col0+0*cell, row0+9*cell),
                      ("row0 col1 (红)", col0+1*cell, row0+0*cell),
                      ("row0 col0 (漏检?)", col0+0*cell, row0+0*cell),
                      ("棋盘底色", col0+4*cell, row0+4*cell)]:
    y0s, y1s = max(0, int(cy-30)), min(crop.shape[0], int(cy+30))
    x0s, x1s = max(0, int(cx-30)), min(crop.shape[1], int(cx+30))
    blk_patch = blk_new[y0s:y1s, x0s:x1s]
    red_patch = red_new[y0s:y1s, x0s:x1s]
    print(f"{name}: red_new={red_patch.sum()}/{red_patch.size} ({100*red_patch.mean():.1f}%)  "
          f"blk_new={blk_patch.sum()}/{blk_patch.size} ({100*blk_patch.mean():.1f}%)")

# ======== 全流程：修改后的 detect_pieces 能否恢复正确识别 ========
def detect_pieces_new(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 16) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 0] < 30) & (hsv[..., 2] > 40) & (hsv[..., 2] < 250) & (hsv[..., 1] < 130)
    piece_mask = (red_mask | black_mask).astype(np.uint8)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    scale = crop.shape[1] / 720.0
    area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
    min_side = max(8, int(15 * scale))

    pieces = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_lo <= area <= area_hi):
            continue
        w, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w < min_side or h_cc < min_side or min(w, h_cc) / max(w, h_cc) < 0.55:
            continue
        cx, cy = cents[i]
        r = int(15 * scale)
        y0r, y1r = max(0, int(cy) - r), min(crop.shape[0], int(cy) + r)
        x0r, x1r = max(0, int(cx) - r), min(crop.shape[1], int(cx) + r)
        rp = red_mask[y0r:y1r, x0r:x1r].mean()
        bp = black_mask[y0r:y1r, x0r:x1r].mean()
        pieces.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))
    return pieces

pieces = detect_pieces_new(crop)
print(f"\n新 detect_pieces: {len(pieces)} 个")
for cx, cy, color, area in sorted(pieces, key=lambda p: (round(p[1]/cell), p[0])):
    print(f"  cy={cy:6.1f} cx={cx:6.1f} {color} area={area:.0f}")

row0, col0, cell_val, score, hit = find_grid(crop, pieces)
print(f"\n网格: row0={row0:.1f} col0={col0:.1f} cell={cell_val:.1f} hit={hit}")
grid_raw, centers = classify(crop, row0, col0, cell_val)
reds = sum(row.count('@') for row in grid_raw)
blks = sum(row.count('#') for row in grid_raw)
print(f"\ngrid_raw: {reds}R+{blks}B")
for r, row in enumerate(grid_raw):
    print(f"row{r}: {''.join(row)}")

red_bottom = detect_red_bottom(grid_raw)
grid = infer_pieces(grid_raw, red_bottom=red_bottom)
print(f"\ninfer_pieces 后的 grid:")
print("\n".join("".join(row) for row in grid))
