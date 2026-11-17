"""用 S>THRESH 的像素比例判断有无棋子。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, infer_pieces, detect_red_bottom

IMG_DIR = r'g:\chessCn\images'

path = os.path.join(IMG_DIR, 'Screenshot_20261117-103438.png')
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)

from engine.recognize import classify as classify_old
grid_old, _ = classify_old(crop, row0, col0, cell)
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
rad = int(cell * 0.38)

# 看看不同 S 阈值下，旧 classify 判为有棋子的格子 vs 空白格子的 S>比例
for s_thr in [100, 105, 110, 115, 120]:
    piece_ratios, blank_ratios = [], []
    for r in range(10):
        for c in range(9):
            cy, cx = row0 + r*cell, col0 + c*cell
            y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
            x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
            patch = hsv[y0s:y1s, x0s:x1s]
            ratio = (patch[..., 1] >= s_thr).mean()
            if grid_old[r][c] != '.':
                piece_ratios.append(ratio)
            else:
                blank_ratios.append(ratio)
    p_med = np.median(piece_ratios) if piece_ratios else 0
    b_med = np.median(blank_ratios) if blank_ratios else 0
    p_max = np.max(piece_ratios) if piece_ratios else 0
    b_max = np.max(blank_ratios) if blank_ratios else 0
    b95 = np.percentile(blank_ratios, 95) if blank_ratios else 0
    print(f"S>={s_thr} 棋子中位={p_med:.3f} 棋子最大={p_max:.3f}  空白中位={b_med:.3f} 空白最大={b_max:.3f} 空白95%={b95:.3f}")
