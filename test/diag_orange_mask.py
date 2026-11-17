"""最小修复：给 piece_mask 增加亮棕色棋子底色的 mask 通道。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify as classify_old, detect_red_bottom, infer_pieces

IMG_DIR = r'g:\chessCn\images'

def detect_pieces_fixed(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 原始两个 mask
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    # 新增：亮棕色棋子底色（漏检的 row9 col1/col7 这种）
    # H=10~20, S>=80（排除棋盘底色 S=68），V=120~240（排除白字 V=255 和暗底色）
    orange_mask = (hsv[..., 0] >= 10) & (hsv[..., 0] <= 20) & (hsv[..., 1] >= 80) & \
                  (hsv[..., 2] >= 120) & (hsv[..., 2] <= 240)
    piece_mask = (red_mask | black_mask | orange_mask).astype(np.uint8)

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
        op = orange_mask[y0r:y1r, x0r:x1r].mean()
        # 优先用 rp vs bp，orange 单独处理
        if op > max(rp, bp):
            # orange 区域，看位置先验（稍后在 classify 里也会用）
            pieces.append((float(cx), float(cy), "B", float(area)))  # 暂时当黑方
        else:
            pieces.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))
    return pieces

print(f"{'文件':>35} {'pieces':>7}  grid R+B  rbott  row0  row9")
print("=" * 85)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    
    pieces = detect_pieces_fixed(crop)
    if len(pieces) < 6:
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        print(f"{fname}: find_grid ERROR {e}")
        continue
    
    grid_raw, _ = classify_old(crop, row0, col0, cell)
    reds = sum(row.count('@') for row in grid_raw)
    blks = sum(row.count('#') for row in grid_raw)
    red_bottom = detect_red_bottom(grid_raw)
    
    print(f"{fname:>35} {len(pieces):>7}  {reds}R+{blks}B   {'red' if red_bottom else 'black':>5}  "
          f"{''.join(grid_raw[0]):>9} {''.join(grid_raw[9]):>9}")
    
    if reds + blks != 32:
        for r, row in enumerate(grid_raw):
            if '.' * 9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
