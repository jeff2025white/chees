"""直接打印 103438 的三个 mask 各自返回的连通域位置。"""
import sys, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
scale = crop.shape[1] / 720.0
area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
min_side = max(8, int(15 * scale))
cell = 76.3

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
orange_mask = (hsv[..., 0] >= 11) & (hsv[..., 0] <= 19) & \
              (hsv[..., 1] >= 85) & (hsv[..., 2] >= 130) & (hsv[..., 2] <= 230)
orange_mask &= ~red_mask

def show_pieces(name, mask):
    num, _labels, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    print(f"\n{name}: area in [{area_lo:.0f}, {area_hi:.0f}], min_side={min_side}")
    out = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h_cc = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        circ = min(w, h_cc) / max(w, h_cc)
        cx, cy = cents[i]
        ok = area_lo <= area <= area_hi and w >= min_side and h_cc >= min_side and circ >= 0.55
        mark = "  OK " if ok else "  X  "
        out.append((cx, cy, area, ok))
        if ok:
            print(f"  {mark} cx={cx:6.1f} cy={cy:6.1f} area={area:>6.0f} w={w:>3.0f} h={h_cc:>3.0f} circ={circ:.2f}")
        elif area_lo <= area <= area_hi:
            print(f"  {mark} cx={cx:6.1f} cy={cy:6.1f} area={area:>6.0f} w={w:>3.0f} h={h_cc:>3.0f} circ={circ:.2f}")
    good = [p for p in out if p[3]]
    print(f"  合计 good={len(good)}  (should be 32 chess pieces)")
    return good

show_pieces("RED", red_mask)
show_pieces("BLACK", black_mask)
show_pieces("ORANGE", orange_mask)
