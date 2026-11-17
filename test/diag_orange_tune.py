"""验证更窄的 orange_mask + 更高阈值。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

IMG_DIR = r'g:\chessCn\images'

def classify_with_orange(crop, row0, col0, cell, orange_thr, orange_cond):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    orange_mask = orange_cond(hsv)
    
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
            or_ = orange_mask[y0s:y1s, x0s:x1s].mean()
            if rr > 0.04 and rr > br * 1.2:
                line.append("@")
            elif br > 0.05 and br > rr * 1.2:
                line.append("#")
            elif rr > br * 0.8 and rr > 0.02:
                line.append("@")
            elif br > rr * 0.8 and br > 0.02:
                line.append("#")
            elif or_ > orange_thr:
                line.append("@" if r <= 4 else "#")
            else:
                line.append(".")
        grid.append(line)
    return grid

# 定义几种 orange_cond
def orange_v1(hsv):
    """之前的：H=11~19, S>=85, V=130~230, 排除 red"""
    m = (hsv[...,0] >= 11) & (hsv[...,0] <= 19) & (hsv[...,1] >= 85) & \
        (hsv[...,2] >= 130) & (hsv[...,2] <= 230)
    red = ((hsv[...,0] < 10) | (hsv[...,0] > 160)) & (hsv[...,1] > 80) & (hsv[...,2] > 80)
    return m & ~red

def orange_v2(hsv):
    """更窄：H=12~18, S>=90, V=130~220"""
    m = (hsv[...,0] >= 12) & (hsv[...,0] <= 18) & (hsv[...,1] >= 90) & \
        (hsv[...,2] >= 135) & (hsv[...,2] <= 220)
    red = ((hsv[...,0] < 10) | (hsv[...,0] > 160)) & (hsv[...,1] > 80) & (hsv[...,2] > 80)
    return m & ~red

def orange_v3(hsv):
    """H=13~17, S>=100, V=140~210"""
    m = (hsv[...,0] >= 13) & (hsv[...,0] <= 17) & (hsv[...,1] >= 100) & \
        (hsv[...,2] >= 140) & (hsv[...,2] <= 210)
    red = ((hsv[...,0] < 10) | (hsv[...,0] > 160)) & (hsv[...,1] > 80) & (hsv[...,2] > 80)
    return m & ~red

from engine.recognize import detect_pieces, find_grid, classify as classify_old, detect_red_bottom

print("对比不同 orange 条件 + 阈值组合 (只看 row0/row9):\n")
for orange_name, orange_fn in [("v1 宽", orange_v1), ("v2 中", orange_v2), ("v3 窄", orange_v3)]:
    for thr in [0.40, 0.55, 0.70]:
        total_ok = True
        for fname in sorted(os.listdir(IMG_DIR)):
            if not fname.startswith('Screenshot') or not fname.endswith('.png'):
                continue
            path = os.path.join(IMG_DIR, fname)
            img = cv2.imread(path)
            h, w = img.shape[:2]
            y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
            crop = img[y0:y1, 0:w]
            try:
                pieces = detect_pieces(crop)
                row0, col0, cell, score, hit = find_grid(crop, pieces)
            except:
                total_ok = False
                continue
            grid_old, _ = classify_old(crop, row0, col0, cell)
            red_bottom = detect_red_bottom(grid_old)
            grid_new = classify_with_orange(crop, row0, col0, cell, thr, orange_fn)
            reds = sum(row.count('@') for row in grid_new)
            blks = sum(row.count('#') for row in grid_new)
            # 红方底线（red_bottom 决定是 row0 还是 row9）应该有 9 子
            red_back = grid_new[9] if red_bottom else grid_new[0]
            blk_back = grid_new[0] if red_bottom else grid_new[9]
            rb_count = sum(1 for ch in red_back if ch == "@")
            bb_count = sum(1 for ch in blk_back if ch == "#")
            if rb_count != 9 or bb_count != 9:
                total_ok = False
        status = "PASS" if total_ok else "FAIL"
        print(f"orange={orange_name} thr={thr:.2f}: {status}")
