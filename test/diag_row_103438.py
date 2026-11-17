"""打印 103438 row0 和 row7 所有格子的 mr。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

from engine.recognize import detect_pieces, find_grid, classify as classify_old, detect_red_bottom
import os

IMG_DIR = r'g:\chessCn\images'
path = os.path.join(IMG_DIR, 'Screenshot_20261117-103438.png')
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]

pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)
grid_old, _ = classify_old(crop, row0, col0, cell)
red_bottom = detect_red_bottom(grid_old)

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
red_dark = red_mask
miss_mask = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 90) & \
            (hsv[..., 1] <= 130) & (hsv[..., 2] >= 150) & (hsv[..., 2] <= 220) & ~red_dark

rad = int(cell * 0.22)
for row in [0, 2, 3, 6, 7, 9]:
    print(f"\nrow{row} (grid_old={''.join(grid_old[row])}):")
    for c in range(9):
        cx = col0 + c * cell
        cy = row0 + row * cell
        y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
        x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
        rr = red_mask[y0s:y1s, x0s:x1s].mean()
        br = black_mask[y0s:y1s, x0s:x1s].mean()
        mr = miss_mask[y0s:y1s, x0s:x1s].mean()
        marker = grid_old[row][c]
        print(f"  col{c}: grid_old={marker} rr={rr:.3f} br={br:.3f} mr={mr:.3f}  "
              f"miss_trigger={'YES' if rr==0 and br==0 and mr>0.42 else '.':>4}")
