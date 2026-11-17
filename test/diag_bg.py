"""采样棋盘底色的 HSV 分布，对比棋子底色，找更好的分离阈值。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
cell = 76.3

# 空位置（row4 col4）的 patch - 棋盘底色
cx_empty, cy_empty = 55.7 + 4*cell, 180.3 + 4*cell
y0s, y1s = int(cy_empty-25), int(cy_empty+25)
x0s, x1s = int(cx_empty-25), int(cx_empty+25)
bg_patch = hsv[y0s:y1s, x0s:x1s]
print("棋盘底色 (row4 col4) HSV:")
print(f"  H=[{bg_patch[...,0].min()},{bg_patch[...,0].max()}]  S=[{bg_patch[...,1].min()},{bg_patch[...,1].max()}]  V=[{bg_patch[...,2].min()},{bg_patch[...,2].max()}]")
print(f"  H 中位={np.median(bg_patch[...,0]):.1f}  S 中位={np.median(bg_patch[...,1]):.1f}  V 中位={np.median(bg_patch[...,2]):.1f}")

# 所有棋子的 patch
targets = [
    ("红方 row0 col1 (车,能检测)", 132, 180),
    ("红方 row3 col0 (兵,能检测)", 56, 410),
    ("黑方 row9 col0 (车,能检测)", 56, 867),
    ("黑方 row9 col1 (马,漏检)", 132, 867),
    ("黑方 row9 col7 (马,漏检)", 580, 867),
    ("黑方 row6 col0 (兵,能检测)", 56, 632),
]
print("\n棋子 patch HSV:")
for name, cx, cy in targets:
    y0s, y1s = max(0, int(cy-25)), min(crop.shape[0], int(cy+25))
    x0s, x1s = max(0, int(cx-25)), min(crop.shape[1], int(cx+25))
    patch = hsv[y0s:y1s, x0s:x1s]
    print(f"  {name}:")
    print(f"    H=[{patch[...,0].min()},{patch[...,0].max()}]  S=[{patch[...,1].min()},{patch[...,1].max()}]  V=[{patch[...,2].min()},{patch[...,2].max()}]")
    print(f"    中位 H={np.median(patch[...,0]):.1f}  S={np.median(patch[...,1]):.1f}  V={np.median(patch[...,2]):.1f}")

# 现在试着找能把棋子底色和棋盘底色分开的阈值
# 棋子底色需要去掉白色字 - 也就是 patch 中心区域
# 让我试试只看棋子 patch 外围（去掉中心 40%）的 HSV
print("\n棋子 patch 外围（去掉中心字）HSV:")
for name, cx, cy in targets:
    y0s, y1s = max(0, int(cy-25)), min(crop.shape[0], int(cy+25))
    x0s, x1s = max(0, int(cx-25)), min(crop.shape[1], int(cx+25))
    patch = hsv[y0s:y1s, x0s:x1s].copy()
    # 掩膜掉中心圆（棋子上的字）
    yy, xx = np.mgrid[:patch.shape[0], :patch.shape[1]]
    cy_rel, cx_rel = patch.shape[0]/2, patch.shape[1]/2
    mask_inner = ((yy - cy_rel)**2 + (xx - cx_rel)**2) < (patch.shape[0]*0.35)**2
    outer = patch[~mask_inner]
    if len(outer):
        print(f"  {name}: 中位 H={np.median(outer[:,0]):.1f}  S={np.median(outer[:,1]):.1f}  V={np.median(outer[:,2]):.1f}")
    else:
        print(f"  {name}: (无外围)")

# 用这些中位值来分，看看棋子底色的 S 和 V 跟棋盘底色有什么区别
print("\n棋盘底色外围（同样大小）:")
print(f"  中位 H={np.median(bg_patch[:,0]):.1f}  S={np.median(bg_patch[:,1]):.1f}  V={np.median(bg_patch[:,2]):.1f}")

# 实际测试：用 S>=5 且 V>=30 试试（极宽），然后按面积过滤
print("\n\n===== 尝试：极宽 mask + 面积过滤 =====")
# 极宽 mask
piece_wide = ((hsv[..., 0] <= 30) | (hsv[..., 0] >= 160)) & (hsv[..., 1] >= 5) & (hsv[..., 2] >= 30)
print(f"piece_wide 非零像素: {piece_wide.sum()}")

# 棋盘底色掩膜 - 排除浅绿/浅棕/接近白色的底色
# V 高（>=200 或 近白）或 S 极低
bg_like = (hsv[..., 1] < 8) | ((hsv[..., 2] > 210) & (hsv[..., 1] < 25))
print(f"bg_like 非零像素: {bg_like.sum()}")

# 棋子 = 宽掩膜 AND NOT 底色
piece_only = piece_wide & ~bg_like
print(f"piece_only 非零像素: {piece_only.sum()}")

# 连通域
num, labels, stats, cents = cv2.connectedComponentsWithStats(piece_only.astype(np.uint8), connectivity=8)
scale = 1.0
area_lo, area_hi = 70 * scale * scale, 900 * scale * scale
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
    color = "R" if V_med >= 140 else "B"
    pieces.append((float(cx), float(cy), color, float(area)))

print(f"\n极宽+面积过滤后得到 {len(pieces)} 个棋子:")
for cx, cy, color, area in sorted(pieces, key=lambda p: (round(p[1]/cell), p[0])):
    print(f"  cx={cx:6.1f} cy={cy:6.1f}  {color}  area={area:.0f}")
