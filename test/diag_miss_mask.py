"""最终最终方案：旧 classify + 漏检格点补漏用窄 mask。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, detect_red_bottom, infer_pieces
from engine.decide import decide_position

IMG_DIR = r'g:\chessCn\images'

def classify_final(crop, row0, col0, cell, red_bottom=False):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 旧阈值
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    # 漏检补专用 mask：亮棕色棋子底色（row0 col0, row9 col1/col7 这种）
    # 窄条件：H=12~17, S=90~130, V=150~220，排除红方偏红区域
    red_dark = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    miss_mask = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 90) & \
                (hsv[..., 1] <= 130) & (hsv[..., 2] >= 150) & (hsv[..., 2] <= 220) & ~red_dark
    
    h, w = crop.shape[:2]
    rad = max(6, int(cell * 0.22))
    grid = []
    for r in range(10):
        cy = row0 + r * cell
        line = []
        for c in range(9):
            cx = col0 + c * cell
            y0s, y1s = max(0, int(cy)-rad), min(h, int(cy)+rad)
            x0s, x1s = max(0, int(cx)-rad), min(w, int(cx)+rad)
            if y1s - y0s < 4 or x1s - x0s < 4:
                line.append(".")
                continue
            rr = red_mask[y0s:y1s, x0s:x1s].mean()
            br = black_mask[y0s:y1s, x0s:x1s].mean()
            mr = miss_mask[y0s:y1s, x0s:x1s].mean()
            is_red = (r >= 5) if red_bottom else (r <= 4)
            if rr > 0.04 and rr > br * 1.2:
                line.append("@")
            elif br > 0.05 and br > rr * 1.2:
                line.append("#")
            elif rr > br * 0.8 and rr > 0.02:
                line.append("@")
            elif br > rr * 0.8 and br > 0.02:
                line.append("#")
            elif mr > 0.35:
                # 补漏：亮棕色棋子，位置先验判红黑
                line.append("@" if is_red else "#")
            else:
                line.append(".")
        grid.append(line)
    return grid

# 先在 103438 上验证所有 81 格的 mr 比例
path = os.path.join(IMG_DIR, 'Screenshot_20261117-103438.png')
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
cell = 76.3
row0, col0 = 180.3, 55.7
rad = int(cell * 0.22)

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
red_mask_old = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask_old = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
red_dark = red_mask_old
miss_mask = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 90) & \
            (hsv[..., 1] <= 130) & (hsv[..., 2] >= 150) & (hsv[..., 2] <= 220) & ~red_dark

from engine.recognize import detect_pieces, find_grid, classify as classify_old
pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)
grid_old, _ = classify_old(crop, row0, col0, cell)

print("103438 所有 81 格:")
print(f"{'r,c':>8}  {'grid_old':>8}  {'rr':>6}  {'br':>6}  {'mr':>6}")
for r in range(10):
    for c in range(9):
        cx = col0 + c * cell
        cy = row0 + r * cell
        y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
        x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
        rr = red_mask_old[y0s:y1s, x0s:x1s].mean()
        br = black_mask_old[y0s:y1s, x0s:x1s].mean()
        mr = miss_mask[y0s:y1s, x0s:x1s].mean()
        go = grid_old[r][c]
        if mr > 0.15 or go != '.':
            print(f"{r},{c:<5}  {go:>8}  {rr:>6.3f}  {br:>6.3f}  {mr:>6.3f}")
