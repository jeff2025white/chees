"""只在理论上应该有棋子的行里补漏。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, classify as classify_old, detect_red_bottom, infer_pieces

IMG_DIR = r'g:\chessCn\images'

# 标准开局行号（红方在上）
# row0: RNBAKABNR 红方底线
# row2: ..C.C..C. 红炮
# row3: P.P.P.P.P 红兵
# row6: p.p.p.p.p 黑兵
# row7: ..c.c..c. 黑炮
# row9: rnbakabnr 黑方底线
RED_SIDE_ROWS = {0, 2, 3}   # 红方应有的行（红方在上时）
BLK_SIDE_ROWS = {6, 7, 9}   # 黑方应有的行（红方在上时）

def classify_row_aware(crop, row0, col0, cell, red_bottom=False):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    red_dark = red_mask
    # miss_mask：亮棕色漏检底色（只在应有的行里生效）
    miss_mask = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 90) & \
                (hsv[..., 1] <= 130) & (hsv[..., 2] >= 150) & (hsv[..., 2] <= 220) & ~red_dark
    
    h_crop, w_crop = crop.shape[:2]
    rad = max(6, int(cell * 0.22))
    
    # 决定哪些行是红方应有的，哪些是黑方应有的
    if red_bottom:
        red_rows = {5, 6, 7}   # 红方在下
        blk_rows = {0, 2, 3}   # 黑方在上
    else:
        red_rows = RED_SIDE_ROWS
        blk_rows = BLK_SIDE_ROWS
    
    grid = []
    for r in range(10):
        cy = row0 + r * cell
        line = []
        for c in range(9):
            cx = col0 + c * cell
            y0s, y1s = max(0, int(cy)-rad), min(h_crop, int(cy)+rad)
            x0s, x1s = max(0, int(cx)-rad), min(w_crop, int(cx)+rad)
            if y1s - y0s < 4 or x1s - x0s < 4:
                line.append(".")
                continue
            rr = red_mask[y0s:y1s, x0s:x1s].mean()
            br = black_mask[y0s:y1s, x0s:x1s].mean()
            mr = miss_mask[y0s:y1s, x0s:x1s].mean()
            
            if rr > 0.04 and rr > br * 1.2:
                is_red = r in red_rows
                line.append("@" if is_red else "#")
            elif br > 0.05 and br > rr * 1.2:
                is_red = r in red_rows
                line.append("@" if is_red else "#")
            elif rr > br * 0.8 and rr > 0.02:
                is_red = r in red_rows
                line.append("@" if is_red else "#")
            elif br > rr * 0.8 and br > 0.02:
                is_red = r in red_rows
                line.append("@" if is_red else "#")
            else:
                # 补漏：只在"理论上应有棋子"的行里用 miss_mask
                should_have_piece = r in red_rows or r in blk_rows
                if should_have_piece and mr > 0.50:
                    is_red = r in red_rows
                    line.append("@" if is_red else "#")
                else:
                    line.append(".")
        grid.append(line)
    return grid

# 跑所有图
print(f"{'文件':>35} {'R+B':>6} {'rbott':>5}  row0  row9")
print("=" * 80)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    
    pieces = detect_pieces(crop)
    if len(pieces) < 6:
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        continue
    
    grid_old, _ = classify_old(crop, row0, col0, cell)
    red_bottom = detect_red_bottom(grid_old)
    
    grid_new = classify_row_aware(crop, row0, col0, cell, red_bottom=red_bottom)
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    
    print(f"{fname:>35} {reds+blks:>3}R{reds:>2}B{blks:>2} {'red' if red_bottom else 'black':>5}  "
          f"{''.join(grid_new[0]):>9} {''.join(grid_new[9]):>9}")
    
    if reds + blks != 32:
        for r, row in enumerate(grid_new):
            if '.' * 9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
