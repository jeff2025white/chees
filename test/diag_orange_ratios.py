"""直接打印 103438 在小采样半径下各关键格点的 orange 比例。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
cell = 76.3
row0, col0 = 180.3, 55.7

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

# 几个不同的 orange 条件
def make_orange(hsv, h_lo, h_hi, s_lo, v_lo, v_hi):
    red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    return (hsv[...,0] >= h_lo) & (hsv[...,0] <= h_hi) & \
           (hsv[...,1] >= s_lo) & (hsv[...,2] >= v_lo) & (hsv[...,2] <= v_hi) & ~red

configs = [
    ("v1 H11-19 S85 V130-230", 11, 19, 85, 130, 230),
    ("v2 H12-18 S90 V135-220", 12, 18, 90, 135, 220),
    ("v3 H13-17 S100 V140-210", 13, 17, 100, 140, 210),
    ("v4 H13-17 S110 V145-205", 13, 17, 110, 145, 205),
]

targets = [
    ("row0 col0 (漏检红车)", col0+0*cell, row0+0*cell),
    ("row9 col1 (漏检黑马)", col0+1*cell, row0+9*cell),
    ("row9 col7 (漏检黑马)", col0+7*cell, row0+9*cell),
    ("row0 col1 (能检红车)", col0+1*cell, row0+0*cell),
    ("row9 col0 (能检黑车)", col0+0*cell, row0+9*cell),
    ("row4 col4 (棋盘底色)", col0+4*cell, row0+4*cell),
]

rad = int(cell * 0.22)
for label, h_lo, h_hi, s_lo, v_lo, v_hi in configs:
    print(f"\n=== {label} ===")
    mask = make_orange(hsv, h_lo, h_hi, s_lo, v_lo, v_hi).astype(np.uint8)
    
    # 先看 mask 是不是一个大连通域
    num, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        largest_area = stats[1:, cv2.CC_STAT_AREA].max()
        print(f"  连通域数={num-1}, 最大面积={largest_area:.0f}")
    
    print(f"  {'位置':<25} {'orange比例':>10}  H中位  S中位  V中位")
    for name, cx, cy in targets:
        y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
        x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
        patch = hsv[y0s:y1s, x0s:x1s]
        ratio = mask[y0s:y1s, x0s:x1s].mean()
        Hm = np.median(patch[..., 0])
        Sm = np.median(patch[..., 1])
        Vm = np.median(patch[..., 2])
        print(f"  {name:<25} {ratio:>10.3f}  {Hm:>5.1f}  {Sm:>5.1f}  {Vm:>5.1f}")
