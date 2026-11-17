"""orange_mask 在 row9 是不是形成了大面积连通域。"""
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
red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
orange_mask = (hsv[..., 0] >= 11) & (hsv[..., 0] <= 19) & (hsv[..., 1] >= 85) & \
              (hsv[..., 2] >= 130) & (hsv[..., 2] <= 230)
orange_mask &= ~red_mask

num, labels, stats, cents = cv2.connectedComponentsWithStats(orange_mask.astype(np.uint8), connectivity=8)
print(f"orange_mask: {num-1} 个连通域")
print("按面积排序的前 20 个:")
idxs = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
for idx in idxs[:20]:
    area = stats[idx, cv2.CC_STAT_AREA]
    cx, cy = cents[idx]
    w, h_cc = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    # 检查它覆盖了哪些格点
    cols_covered, rows_covered = [], []
    for c in range(9):
        for r in range(10):
            gx, gy = col0 + c*cell, row0 + r*cell
            if labels[int(gy), int(gx)] == idx:
                cols_covered.append(c)
                rows_covered.append(r)
    cols_str = ','.join(str(c) for c in sorted(set(cols_covered)))
    rows_str = ','.join(str(r) for r in sorted(set(rows_covered)))
    print(f"  id={idx}: area={area:>6.0f} cx={cx:>7.1f} cy={cy:>7.1f} w={w:>3} h={h_cc:>3}  覆盖行={rows_str} 覆盖列={cols_str}")
