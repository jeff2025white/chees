"""方案：放宽 detect_pieces 阈值，看看能抓到多少。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify

IMG_DIR = r'g:\chessCn\images'

# 旧阈值
def piece_mask_old(hsv):
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    return (red_mask | black_mask).astype(np.uint8), red_mask, black_mask

# 新阈值：放宽范围，红和黑都覆盖
def piece_mask_new(hsv):
    # 极宽：H=0~30（红橙棕）和 H=160~180（红），S>=30，V>=50
    wide = ((hsv[..., 0] <= 30) | (hsv[..., 0] >= 160)) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 50)
    return wide.astype(np.uint8), None, None

# 红黑区分函数：用 V 中位数，V >= 140 判红，否则判黑
def classify_color(hsv_patch):
    V_med = np.median(hsv_patch[..., 2])
    return 'R' if V_med >= 140 else 'B'

for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scale = crop.shape[1] / 720.0
    
    for tag, (mask_func, use_v_classify) in [("旧阈值", (piece_mask_old, False)),
                                               ("新阈值", (piece_mask_new, True))]:
        mask, red_m, blk_m = mask_func(hsv)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
        min_side = max(8, int(15 * scale))
        pieces = []
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
            if use_v_classify:
                patch = hsv[y0s:y1s, x0s:x1s]
                color = classify_color(patch)
            else:
                rp = red_m[y0s:y1s, x0s:x1s].mean()
                bp = blk_m[y0s:y1s, x0s:x1s].mean()
                color = "R" if rp > bp else "B"
            pieces.append((float(cx), float(cy), color, float(area)))
        
        # 跑 find_grid
        try:
            row0, col0, cell, score, hit = find_grid(crop, pieces)
            # 跑 classify 看 grid_raw
            grid_raw, centers = classify(crop, row0, col0, cell)
            reds = sum(row.count('@') for row in grid_raw)
            blks = sum(row.count('#') for row in grid_raw)
            print(f"{fname} [{tag}]: pieces={len(pieces)} grid={reds}R+{blks}B hit={hit}")
        except Exception as e:
            print(f"{fname} [{tag}]: ERROR {e}")
