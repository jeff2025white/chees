"""最终方案：放宽 red_mask H<18，classify 用位置先验判红黑。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, detect_red_bottom, infer_pieces
from engine.decide import decide_position

IMG_DIR = r'g:\chessCn\images'

def classify_final(crop, row0, col0, cell, red_bottom=False):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 放宽 red_mask：H<18 或 H>160，S>80, V>80
    red_mask = ((hsv[..., 0] < 18) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    
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
            # 只要 rr 或 br 有一个够高，就判有棋子
            has_piece = (rr > 0.04 and rr > br * 1.2) or \
                        (br > 0.05 and br > rr * 1.2) or \
                        (rr > br * 0.8 and rr > 0.02) or \
                        (br > rr * 0.8 and br > 0.02)
            if not has_piece:
                line.append(".")
                continue
            # 位置先验判红黑
            # red_bottom=False → 红方在上 (row 0~4)，黑方在下 (row 5~9)
            # red_bottom=True  → 红方在下 (row 5~9)，黑方在上 (row 0~4)
            is_red = (r >= 5) if red_bottom else (r <= 4)
            line.append("@" if is_red else "#")
        grid.append(line)
    return grid

# 同时修改 detect_pieces 的 mask
def detect_pieces_final(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 18) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    piece_mask = (red_mask | black_mask).astype(np.uint8)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    scale = crop.shape[1] / 720.0
    area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
    min_side = max(8, int(15 * scale))

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
        rp = red_mask[y0r:y1r, x0r:x1r].mean()
        bp = black_mask[y0r:y1r, x0r:x1r].mean()
        pieces.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))
    return pieces

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
    
    pieces = detect_pieces_final(crop)
    if len(pieces) < 6:
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        print(f"{fname}: find_grid ERROR {e}")
        continue
    
    # 先用旧 detect_pieces + 旧 classify 判 red_bottom
    pieces_old = detect_pieces(crop)
    try:
        row0o, col0o, cello, scoreo, hito = find_grid(crop, pieces_old)
    except:
        row0o, col0o, cello = row0, col0, cell
    from engine.recognize import classify as classify_old
    grid_old, _ = classify_old(crop, row0o, col0o, cello)
    red_bottom = detect_red_bottom(grid_old)
    
    grid_new = classify_final(crop, row0, col0, cell, red_bottom=red_bottom)
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    
    print(f"{fname:>35} {len(pieces):>7}  {reds}R+{blks}B   {'red' if red_bottom else 'black':>5}  "
          f"{''.join(grid_new[0]):>9} {''.join(grid_new[9]):>9}")
    
    if reds + blks != 32:
        for r, row in enumerate(grid_new):
            if '.' * 9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
