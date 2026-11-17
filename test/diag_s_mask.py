"""用 S>=80 阈值分离棋子和棋盘底色。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

IMG_DIR = r'g:\chessCn\images'

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

    # 用 S>=80, V>=60, H<=30 or H>=160 抓所有饱和度够高的"有颜色"物体
    piece_mask = ((hsv[..., 0] <= 30) | (hsv[..., 0] >= 160)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask.astype(np.uint8), connectivity=8)

    area_lo, area_hi = 60 * scale * scale, 900 * scale * scale
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
        y0s, y1s = max(0, int(cy)-r), min(crop.shape[0], int(cy)+r)
        x0s, x1s = max(0, int(cx)-r), min(crop.shape[1], int(cx)+r)
        patch = hsv[y0s:y1s, x0s:x1s]
        V_med = np.median(patch[..., 2])
        # 用 V 中位区分红黑：红方底色 V 通常比黑方更亮
        color = "R" if V_med >= 235 else "B"
        pieces.append((float(cx), float(cy), color, float(area), V_med))

    # 尝试 find_grid
    try:
        from engine.recognize import find_grid, classify
        row0, col0, cell, score, hit = find_grid(crop, pieces)
        grid_raw, centers = classify(crop, row0, col0, cell)
        reds = sum(row.count('@') for row in grid_raw)
        blks = sum(row.count('#') for row in grid_raw)
        print(f"{fname}: pieces={len(pieces)}  grid={reds}R+{blks}B  hit={hit}")
        # 统计 V_med 分布
        red_vs = [p[4] for p in pieces if p[2] == "R"]
        blk_vs = [p[4] for p in pieces if p[2] == "B"]
        if red_vs:
            print(f"  红V中位={np.median(red_vs):.1f} min={min(red_vs):.1f} max={max(red_vs):.1f}")
        if blk_vs:
            print(f"  黑V中位={np.median(blk_vs):.1f} min={min(blk_vs):.1f} max={max(blk_vs):.1f}")
    except Exception as e:
        print(f"{fname}: ERROR {e}")
        # 打印 pieces 看看
        for cx, cy, color, area, V_med in sorted(pieces, key=lambda p: p[1]):
            print(f"  cy={cy:.1f} cx={cx:.1f} {color} area={area:.0f} Vmed={V_med:.1f}")
