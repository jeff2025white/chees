"""统计所有 8 张截图上红方/黑方棋子底色的 V 值分布，找区分界点。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

IMG_DIR = r'g:\chessCn\images'

# 直接用更宽的 mask 去抓所有棋子，然后按区域采样棋子中心的 HSV
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    # 用极宽的 mask 抓棋子——覆盖 H=0~30（红到橙棕），S=30~255，V=50~255
    # 同时排除棋盘底色（偏浅绿/偏白），棋盘底色 S 极低
    all_piece = (hsv[..., 0] <= 30) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 50)
    all_piece |= (hsv[..., 0] >= 160) & (hsv[..., 1] >= 30) & (hsv[..., 2] >= 50)
    mask = all_piece.astype(np.uint8)
    
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    red_vs, blk_vs = [], []
    scale = crop.shape[1] / 720.0
    area_lo, area_hi = 60 * scale * scale, 900 * scale * scale
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_lo <= area <= area_hi):
            continue
        cx, cy = cents[i]
        r = int(15 * scale)
        y0s, y1s = max(0, int(cy)-r), min(crop.shape[0], int(cy)+r)
        x0s, x1s = max(0, int(cx)-r), min(crop.shape[1], int(cx)+r)
        patch = hsv[y0s:y1s, x0s:x1s]
        # 用 V 的中位数代表底色（避免白字的 V=255 干扰）
        V_med = np.median(patch[..., 2])
        H_med = np.median(patch[..., 0])
        S_med = np.median(patch[..., 1])
        
        # 临时判定：用红方在上 / 黑方在下的先验（y 值小是红方）
        if cy < crop.shape[0] * 0.45:
            red_vs.append((V_med, H_med, S_med, area))
        else:
            blk_vs.append((V_med, H_med, S_med, area))
    
    print(f"\n{fname}: 红={len(red_vs)} 黑={len(blk_vs)}")
    if red_vs:
        red_arr = np.array([x[0] for x in red_vs])
        print(f"  红方 V 中位={np.median(red_arr):.1f}  min={red_arr.min():.1f}  max={red_arr.max():.1f}")
    if blk_vs:
        blk_arr = np.array([x[0] for x in blk_vs])
        print(f"  黑方 V 中位={np.median(blk_arr):.1f}  min={blk_arr.min():.1f}  max={blk_arr.max():.1f}")
    # 展示前几个
    print(f"  红方 HSV/area 样本:", [(f'H={h:.0f},S={s:.0f},V={v:.0f},a={a:.0f}') for v,h,s,a in red_vs[:3]])
    print(f"  黑方 HSV/area 样本:", [(f'H={h:.0f},S={s:.0f},V={v:.0f},a={a:.0f}') for v,h,s,a in blk_vs[:3]])
