"""新 detect_pieces：只用 S>=100 抓棋子底色，排除白色字（S<5 and V>200）。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify as classify_old, infer_pieces, detect_red_bottom

IMG_DIR = r'g:\chessCn\images'

def detect_pieces_v2(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 棋子底色 S>=100（棋盘底色 S≈68，被排除），排除白色字 S<5 & V>200
    piece_mask = (hsv[..., 1] >= 100) & ~((hsv[..., 1] < 5) & (hsv[..., 2] > 200))
    # 也排除纯黑/深灰（可能是阴影）
    piece_mask &= (hsv[..., 2] >= 50)
    piece_mask = piece_mask.astype(np.uint8)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    scale = crop.shape[1] / 720.0
    area_lo, area_hi = 100 * scale * scale, 900 * scale * scale
    min_side = max(10, int(18 * scale))
    
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
        patch = hsv[y0r:y1r, x0r:x1r]
        V_med = np.median(patch[..., 2])
        # 红方底色亮（V≈244），黑方底色暗一点（V≈218），但有重叠
        # 先用简单阈值：V>=235 判红，否则判黑
        color = "R" if V_med >= 235 else "B"
        pieces.append((float(cx), float(cy), color, float(area)))
    return pieces

# 同时重写 classify：用 S>=100 比例判断有无棋子，位置先验判断红黑
def classify_v2(crop, row0, col0, cell, red_bottom=False):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    rad = max(6, int(cell * 0.4))  # 采样半径大一点覆盖整个棋子
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
            patch = hsv[y0s:y1s, x0s:x1s]
            # 棋子底色 S>=100 的比例
            s_ratio = (patch[..., 1] >= 100).mean()
            # 棋盘底色 S≈68，所以空白格子的 s_ratio 应该接近 0
            if s_ratio < 0.25:
                line.append(".")
            else:
                # 位置先验判断红黑
                # red_bottom=False → 红方在上（图像 row 0~4），黑方在下（图像 row 5~9）
                # red_bottom=True → 红方在下（图像 row 5~9），黑方在上（图像 row 0~4）
                if red_bottom:
                    is_red = r >= 5
                else:
                    is_red = r <= 4
                line.append("@" if is_red else "#")
        grid.append(line)
    return grid

# 跑所有图
print(f"{'文件':>35} {'pieces':>7} {'R+B':>6} {'rbott':>5}  row0  row9")
print("=" * 80)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    
    pieces = detect_pieces_v2(crop)
    if len(pieces) < 6:
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        # 打印 area 统计
        num, _labels, stats, cents = cv2.connectedComponentsWithStats(
            (cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[...,1]>=100).astype(np.uint8), connectivity=8)
        print(f"  piece_mask 连通域数={num-1} 面积范围={stats[1:,cv2.CC_STAT_AREA].min() if num>1 else 0}~{stats[1:,cv2.CC_STAT_AREA].max() if num>1 else 0}")
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        print(f"{fname}: find_grid ERROR {e}")
        continue
    
    # 位置先验需要用旧的 detect_red_bottom，或者先跑一版旧 classify
    from engine.recognize import classify as classify_old
    grid_old, _ = classify_old(crop, row0, col0, cell)
    red_bottom = detect_red_bottom(grid_old)
    
    grid_new = classify_v2(crop, row0, col0, cell, red_bottom=red_bottom)
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    
    print(f"{fname:>35} {len(pieces):>7} {reds+blks:>3}R{reds:>2}B{blks:>2} {'red' if red_bottom else 'black':>5}  "
          f"{''.join(grid_new[0]):>9} {''.join(grid_new[9]):>9}")
    
    if reds + blks != 32:
        for r, row in enumerate(grid_new):
            if '.' * 9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
