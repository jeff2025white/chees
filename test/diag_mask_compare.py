"""直接对比旧 mask vs S>=80 mask 在关键位置的覆盖率。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
cell = 76.3

# 旧 mask
red_old = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
blk_old = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
piece_old = (red_old | blk_old).astype(np.uint8)

# S>=80 mask（H≤30 或 H≥160，V≥60）
piece_s = ((hsv[..., 0] <= 30) | (hsv[..., 0] >= 160)) & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 60)
piece_s = piece_s.astype(np.uint8)

print(f"piece_old sum = {piece_old.sum()}")
print(f"piece_s sum   = {piece_s.sum()}")

# 采样关键位置
targets = [
    ("棋盘底色 row4 col4", 55.7+4*cell, 180.3+4*cell),
    ("红方 row0 col1 车",  55.7+1*cell, 180.3+0*cell),
    ("红方 row3 col0 兵",  55.7+0*cell, 180.3+3*cell),
    ("黑方 row9 col0 车",  55.7+0*cell, 180.3+9*cell),
    ("黑方 row9 col1 马",  55.7+1*cell, 180.3+9*cell),
    ("黑方 row9 col7 马",  55.7+7*cell, 180.3+9*cell),
    ("底部按钮区 cy≈1050", 360, 1050),
]
print()
for name, cx, cy in targets:
    y0s, y1s = max(0, int(cy-30)), min(crop.shape[0], int(cy+30))
    x0s, x1s = max(0, int(cx-30)), min(crop.shape[1], int(cx+30))
    po = piece_old[y0s:y1s, x0s:x1s]
    ps = piece_s[y0s:y1s, x0s:x1s]
    hsv_patch = hsv[y0s:y1s, x0s:x1s]
    print(f"{name}:")
    print(f"  old: {po.sum()}/{po.size} ({100*po.mean():.1f}%)  new: {ps.sum()}/{ps.size} ({100*ps.mean():.1f}%)")
    # 新 mask 没覆盖到的原因
    not_s = hsv_patch[..., 1] < 80
    print(f"  S<80 像素: {not_s.sum()}/{not_s.size} ({100*not_s.mean():.1f}%)")
    print(f"  patch S 分布: S<30={((hsv_patch[...,1]<30).sum())}  30-80={(((hsv_patch[...,1]>=30)&(hsv_patch[...,1]<80)).sum())}  S>=80={((hsv_patch[...,1]>=80).sum())}")
    print()

# 连通域对比
for tag, mask in [("OLD", piece_old), ("S>=80", piece_s)]:
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale = 1.0
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
        pieces.append((float(cx), float(cy), area))
    print(f"[{tag}] 过滤后 {len(pieces)} 个连通域:")
    for cx, cy, area in sorted(pieces, key=lambda p: (round(p[1]/cell), p[0])):
        print(f"  cx={cx:6.1f} cy={cy:6.1f} area={area:.0f}")
    print()
