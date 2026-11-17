"""用 v3 orange 条件 + 0.15 阈值重跑所有截图。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, detect_red_bottom, infer_pieces
from engine.decide import decide_position
from engine.notation import chinese_notation

IMG_DIR = r'g:\chessCn\images'

def classify_v3(crop, row0, col0, cell):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    # orange v3: H=13~17, S>=100, V=140~210，排除 red
    red_partial = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    orange_mask = (hsv[..., 0] >= 13) & (hsv[..., 0] <= 17) & (hsv[..., 1] >= 100) & \
                  (hsv[..., 2] >= 140) & (hsv[..., 2] <= 210) & ~red_partial
    
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
            elif or_ > 0.15:
                # orange 亮棕色底色，用位置先验判红黑
                line.append("@" if r <= 4 else "#")
            else:
                line.append(".")
        grid.append(line)
    return grid

# 跑所有图
print(f"{'文件':>35} {'R+B':>6} {'view_side':>10}  row0  row9")
print("=" * 85)
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
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        print(f"{fname}: find_grid ERROR {e}")
        continue
    
    # 用旧 classify 先拿 red_bottom
    from engine.recognize import classify as classify_old
    grid_old, _ = classify_old(crop, row0, col0, cell)
    red_bottom = detect_red_bottom(grid_old)
    
    grid_new = classify_v3(crop, row0, col0, cell)
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    
    print(f"{fname:>35} {reds+blks:>3}R{reds:>2}B{blks:>2}  {'red' if red_bottom else 'black':>10}  "
          f"{''.join(grid_new[0]):>9} {''.join(grid_new[9]):>9}")
    
    if reds + blks != 32:
        for r, row in enumerate(grid_new):
            if '.' * 9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
