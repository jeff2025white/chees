"""打印 103438 所有 81 格的 v3 orange 比例。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
cell = 76.3
row0, col0 = 180.3, 55.7
rad = int(cell * 0.22)

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
red_partial = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
orange_mask = (hsv[..., 0] >= 13) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 100) & \
              (hsv[..., 2] >= 140) & (hsv[..., 2] <= 210) & ~red_partial

from engine.recognize import detect_pieces, find_grid, classify as classify_old
pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)
grid_old, _ = classify_old(crop, row0, col0, cell)

print(f"{'r,c':>8}  {'grid_old':>8}  {'orange':>8}  {'rr':>6}  {'br':>6}")
for r in range(10):
    for c in range(9):
        cx = col0 + c * cell
        cy = row0 + r * cell
        y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
        x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
        o = orange_mask[y0s:y1s, x0s:x1s].mean()
        red_mask = ((hsv[...,0] < 10) | (hsv[...,0] > 160)) & (hsv[...,1] > 80) & (hsv[...,2] > 80)
        black_mask = (hsv[...,2] > 40) & (hsv[...,2] < 100) & (hsv[...,1] < 100) & (hsv[...,0] < 30)
        rr = red_mask[y0s:y1s, x0s:x1s].mean()
        br = black_mask[y0s:y1s, x0s:x1s].mean()
        go = grid_old[r][c]
        if o > 0.15 or go != '.':
            print(f"{r},{c:<5}  {go:>8}  {o:>8.3f}  {rr:>6.3f}  {br:>6.3f}")
