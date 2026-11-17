"""试试不同 red_mask H 下限 + mr 阈值组合。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, classify as classify_old, detect_red_bottom

IMG_DIR = r'g:\chessCn\images'

def test_combo(fname, h_red_lo, mr_thr):
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    cell = 76.3
    rad = int(cell * 0.22)
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < h_red_lo) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    red_dark = red_mask
    miss_mask = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 90) & \
                (hsv[..., 1] <= 130) & (hsv[..., 2] >= 150) & (hsv[..., 2] <= 220) & ~red_dark
    
    pieces = detect_pieces(crop)
    if len(pieces) < 6:
        return None
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except:
        return None
    
    grid_old, _ = classify_old(crop, row0, col0, cell)
    red_bottom = detect_red_bottom(grid_old)
    h_crop, w_crop = crop.shape[:2]
    
    grid = []
    for r in range(10):
        line = []
        for c in range(9):
            cy = row0 + r * cell
            cx = col0 + c * cell
            y0s, y1s = max(0, int(cy)-rad), min(h_crop, int(cy)+rad)
            x0s, x1s = max(0, int(cx)-rad), min(w_crop, int(cx)+rad)
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
            elif rr == 0.000 and br == 0.000 and mr > mr_thr:
                line.append("@" if is_red else "#")
            else:
                line.append(".")
        grid.append(line)
    return grid

# 试几组组合
combos = [
    ("H<10 mr_thr=0.50", 10, 0.50),
    ("H<12 mr_thr=0.50", 12, 0.50),
    ("H<14 mr_thr=0.50", 14, 0.50),
    ("H<10 mr_thr=0.55", 10, 0.55),
    ("H<10 mr_thr=0.60", 10, 0.60),
    ("H<14 mr_thr=0.55", 14, 0.55),
]

for label, h_red_lo, mr_thr in combos:
    total_errors = 0
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.startswith('Screenshot') or not fname.endswith('.png'):
            continue
        grid = test_combo(fname, h_red_lo, mr_thr)
        if grid is None:
            total_errors += 1
            continue
        reds = sum(row.count('@') for row in grid)
        blks = sum(row.count('#') for row in grid)
        if reds + blks != 32:
            total_errors += 1
    print(f"{label}: {total_errors}/8 图非 32 子")
