"""分通道检测：red, black, orange 三个 mask 各自独立做连通域分析，最后合并。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify as classify_old, detect_red_bottom, infer_pieces

IMG_DIR = r'g:\chessCn\images'

def _mask_pieces(hsv, piece_mask, area_lo, area_hi, min_side, label=None):
    num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_lo <= area <= area_hi):
            continue
        w, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w < min_side or h_cc < min_side or min(w, h_cc) / max(w, h_cc) < 0.55:
            continue
        cx, cy = cents[i]
        out.append((float(cx), float(cy), float(area), label))
    return out

def detect_pieces_v3(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scale = crop.shape[1] / 720.0
    area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
    min_side = max(8, int(15 * scale))
    
    # 三个独立 mask
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    # orange_mask: 专门抓亮棕色漏检黑子，条件更窄，避免重叠
    # H=11~19 (避开 red_mask 的 H<10), S>=85 (棋盘底色 S=68), V=130~230
    orange_mask = (hsv[..., 0] >= 11) & (hsv[..., 0] <= 19) & \
                  (hsv[..., 1] >= 85) & (hsv[..., 2] >= 130) & (hsv[..., 2] <= 230)
    # 也排除 orange 中被 red_mask 覆盖的区域？
    orange_mask &= ~red_mask
    
    red_pieces = _mask_pieces(hsv, red_mask, area_lo, area_hi, min_side, "R")
    black_pieces = _mask_pieces(hsv, black_mask, area_lo, area_hi, min_side, "B")
    orange_pieces = _mask_pieces(hsv, orange_mask, area_lo, area_hi, min_side, "B_orange")
    
    # 合并，去重：如果 orange_piece 位置附近已有 red/black piece，就跳过
    all_pieces = red_pieces + black_pieces
    for cx, cy, area, label in orange_pieces:
        # 找位置接近的已检测棋子
        too_close = any(((cx-ap[0])**2 + (cy-ap[1])**2) ** 0.5 < 35 for ap in all_pieces)
        if not too_close:
            all_pieces.append((cx, cy, area, label))
    
    # 返回 (cx, cy, color, area) 格式
    return [(cx, cy, label if label in ("R", "B") else "B", area) for cx, cy, area, label in all_pieces]

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
    
    pieces = detect_pieces_v3(crop)
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
