"""深入诊断 103438 截图的棋子检测问题——看原始掩膜和所有连通域。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import analyze_image

PATH = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(PATH)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
print(f"crop shape = {crop.shape}")

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)

# 合并
piece_mask = (red_mask | black_mask).astype(np.uint8)
print(f"red_mask 非零像素: {red_mask.sum()}")
print(f"black_mask 非零像素: {black_mask.sum()}")
print(f"piece_mask 非零像素: {piece_mask.sum()}")

num, labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
print(f"\n所有 {num-1} 个连通域 (含最小面积):")
scale = 1.0
area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
print(f"面积过滤阈值: [{area_lo}, {area_hi}]")

# 黑方底线 row9 col1 和 col7 的格点位置（row0=180.3, cell=76.3, col0=55.7）
row0, col0, cell = 180.3, 55.7, 76.3
# row9 y = 180.3 + 9*76.3 = 867.0
print(f"\nrow9 (图像最下方底线) 格点 y = {row0 + 9*cell:.1f}")
print(f"row9 col1 格点: ({col0 + 1*cell:.1f}, {row0 + 9*cell:.1f})")
print(f"row9 col7 格点: ({col0 + 7*cell:.1f}, {row0 + 9*cell:.1f})")

# 把所有连通域按距离这两个格点的远近排序
targets = [
    ("row9 col1", col0 + 1*cell, row0 + 9*cell),
    ("row9 col7", col0 + 7*cell, row0 + 9*cell),
    ("row0 col0", col0 + 0*cell, row0 + 0*cell),
]

rows = []
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
    w_cc, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cx, cy = cents[i]
    inside = area_lo <= area <= area_hi and min(w_cc, h_cc) / max(w_cc, h_cc) >= 0.55
    # 找离哪个目标格点最近
    best_target, best_dist = None, 1e9
    for name, tx, ty in targets:
        d = ((cx - tx)**2 + (cy - ty)**2) ** 0.5
        if d < best_dist:
            best_dist = d
            best_target = name
    rows.append((area, cx, cy, w_cc, h_cc, inside, best_target, best_dist))

rows.sort(key=lambda r: r[5])  # 先按是否通过过滤
print(f"\n{'pass':>4} {'area':>5} {'cx':>7} {'cy':>7} {'w':>4} {'h':>4} {'near':>15} {'dist':>7}")
for area, cx, cy, w_cc, h_cc, inside, target, dist in rows[:60]:
    print(f" {int(inside):>3d} {area:>5d} {cx:>7.1f} {cy:>7.1f} {w_cc:>4d} {h_cc:>4d} {target:>15} {dist:>7.1f}")

# 再把 row9 col1 和 col7 附近 ±40px 的位置画一个小图存下来方便人工看
crop_debug = crop.copy()
# 画 row9 col1 周围
for name, tx, ty in targets[:2]:
    x0, y0 = int(tx - 40), int(ty - 40)
    x1, y1 = int(tx + 40), int(ty + 40)
    cv2.rectangle(crop_debug, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(crop_debug, name, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
cv2.imwrite(r'g:\chessCn\test\debug_103438.jpg', crop_debug)
print("\n调试标记图已保存到 test/debug_103438.jpg")

# 额外：看看 HSV 图像上 row9 col1 的那个位置像素到底是什么颜色
print("\n--- row9 col1 附近 (55.7+76.3=132, 867) 的颜色采样 ---")
tx, ty = col0 + 1*cell, row0 + 9*cell
sample_hsv = hsv[int(ty-20):int(ty+20), int(tx-20):int(tx+20)]
print(f"H 范围 [{sample_hsv[...,0].min()}, {sample_hsv[...,0].max()}]")
print(f"S 范围 [{sample_hsv[...,1].min()}, {sample_hsv[...,1].max()}]")
print(f"V 范围 [{sample_hsv[...,2].min()}, {sample_hsv[...,2].max()}]")
print(f"red_mask 在区域内非零: {red_mask[int(ty-20):int(ty+20), int(tx-20):int(tx+20)].sum()}")
print(f"black_mask 在区域内非零: {black_mask[int(ty-20):int(ty+20), int(tx-20):int(tx+20)].sum()}")

print("\n--- row9 col7 附近 (55.7+7*76.3=580, 867) 的颜色采样 ---")
tx, ty = col0 + 7*cell, row0 + 9*cell
sample_hsv = hsv[int(ty-20):int(ty+20), int(tx-20):int(tx+20)]
print(f"H 范围 [{sample_hsv[...,0].min()}, {sample_hsv[...,0].max()}]")
print(f"S 范围 [{sample_hsv[...,1].min()}, {sample_hsv[...,1].max()}]")
print(f"V 范围 [{sample_hsv[...,2].min()}, {sample_hsv[...,2].max()}]")
print(f"red_mask 在区域内非零: {red_mask[int(ty-20):int(ty+20), int(tx-20):int(tx+20)].sum()}")
print(f"black_mask 在区域内非零: {black_mask[int(ty-20):int(ty+20), int(tx-20):int(tx+20)].sum()}")
