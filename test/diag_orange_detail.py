"""单独诊断 103438 加了 orange_mask 后 pieces 出了什么问题。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
scale = crop.shape[1] / 720.0
cell = 76.3

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

# 3 个 mask
red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
orange_mask = (hsv[..., 0] >= 10) & (hsv[..., 0] <= 20) & (hsv[..., 1] >= 80) & \
              (hsv[..., 2] >= 120) & (hsv[..., 2] <= 240)

print(f"red_mask sum = {red_mask.sum()}")
print(f"black_mask sum = {black_mask.sum()}")
print(f"orange_mask sum = {orange_mask.sum()}")

# 看看 orange_mask 在哪些关键位置覆盖
for name, cx, cy in [
    ("row9 col0 (黑车,能检)", 55.7+0*cell, 180.3+9*cell),
    ("row9 col1 (黑马,漏检)", 55.7+1*cell, 180.3+9*cell),
    ("row9 col7 (黑马,漏检)", 55.7+7*cell, 180.3+9*cell),
    ("row0 col1 (红车,能检)", 55.7+1*cell, 180.3+0*cell),
    ("row3 col0 (红兵,能检)", 55.7+0*cell, 180.3+3*cell),
    ("棋盘底色 row4 col4", 55.7+4*cell, 180.3+4*cell),
]:
    y0s, y1s = max(0, int(cy-30)), min(crop.shape[0], int(cy+30))
    x0s, x1s = max(0, int(cx-30)), min(crop.shape[1], int(cx+30))
    o = orange_mask[y0s:y1s, x0s:x1s]
    hsv_patch = hsv[y0s:y1s, x0s:x1s]
    print(f"\n{name}:")
    print(f"  orange_mask={o.sum()}/{o.size} ({100*o.mean():.1f}%)")
    print(f"  patch H=[{hsv_patch[...,0].min()},{hsv_patch[...,0].max()}] S=[{hsv_patch[...,1].min()},{hsv_patch[...,1].max()}] V=[{hsv_patch[...,2].min()},{hsv_patch[...,2].max()}]")

# orange_mask 连通域分析
piece_mask = (red_mask | black_mask | orange_mask).astype(np.uint8)
num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
min_side = max(8, int(15 * scale))

good = []
too_big, too_small, weird = [], [], []
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    w, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    circ = min(w, h_cc) / max(w, h_cc)
    cx, cy = cents[i]
    if area > area_hi:
        too_big.append((cx, cy, area, w, h_cc))
    elif area < area_lo:
        too_small.append((cx, cy, area))
    elif w < min_side or h_cc < min_side or circ < 0.55:
        weird.append((cx, cy, area, w, h_cc, circ))
    else:
        good.append((cx, cy, area, w, h_cc))

print(f"\n连通域统计: good={len(good)} too_big={len(too_big)} too_small={len(too_small)} weird={len(weird)}")
print(f"\ngood 连通域 (应该是棋子):")
for cx, cy, area, w, h_cc in sorted(good, key=lambda p: (round(p[1]/cell), p[0])):
    print(f"  cx={cx:6.1f} cy={cy:6.1f} area={area:.0f} w={w:.0f} h={h_cc:.0f}")

print(f"\ntoo_big (面积>{area_hi:.0f}):")
for cx, cy, area, w, h_cc in too_big[:10]:
    print(f"  cx={cx:6.1f} cy={cy:6.1f} area={area:.0f} w={w:.0f} h={h_cc:.0f}")

# 去掉 too_big 那些，看看 good 能不能定位网格
pieces = [(float(cx), float(cy), "R", float(area)) for cx, cy, area, w, h_cc in good]
if len(pieces) >= 6:
    try:
        row0, col0, c, score, hit = find_grid(crop, pieces)
        print(f"\nfind_grid 成功: row0={row0:.1f} col0={col0:.1f} cell={c:.1f} hit={hit}")
    except Exception as e:
        print(f"\nfind_grid 仍然失败: {e}")
        # 打印 good pieces 位置看看
        print("good pieces 按 y 分桶:")
        buckets = {}
        for cx, cy, _c, _a in pieces:
            b = round(cy / cell)
            buckets.setdefault(b, []).append(cx)
        for k in sorted(buckets.keys()):
            print(f"  bucket {k}: {len(buckets[k])} 个 ({buckets[k]})")
