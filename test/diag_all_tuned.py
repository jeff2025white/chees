"""验证：black_mask 用 S 80~140 能不能把棋盘底色过滤掉，同时抓住漏检棋子。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, classify, infer_pieces, detect_red_bottom
from engine.decide import decide_position

IMG_DIR = r'g:\chessCn\images'

def run_tuned(fname):
    path = rf'g:\chessCn\images\{fname}'
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scale = w / 720.0
    cell = 76.3
    
    # ======== 尝试不同阈值组合 ========
    # 组合 A: 保持原 red_mask，black_mask 改为 S=80~140, H<30, V=60~250
    # 组合 B: red_mask H<16 or H>160, S>=80, V>=80; black_mask S=80~140, H<30, V=60~250
    # 组合 C: 都用 H<=30 or H>=160, S=80~140, V=60~250，然后用 V 中位区分红黑
    
    configs = [
        ("A 原red+S80-140blk",
         lambda hsv: (
             ((hsv[...,0]<10) | (hsv[...,0]>160)) & (hsv[...,1]>80) & (hsv[...,2]>80),
             (hsv[...,0]<30) & (hsv[...,1]>=80) & (hsv[...,1]<=140) & (hsv[...,2]>=60) & (hsv[...,2]<=250)
         )),
        ("B H<16red+S80-140blk",
         lambda hsv: (
             ((hsv[...,0]<16) | (hsv[...,0]>160)) & (hsv[...,1]>=80) & (hsv[...,2]>=80),
             (hsv[...,0]<30) & (hsv[...,1]>=80) & (hsv[...,1]<=140) & (hsv[...,2]>=60) & (hsv[...,2]<=250)
         )),
        ("C 统一S80-140+V判色",
         lambda hsv: (
             ((hsv[...,0]<=30) | (hsv[...,0]>=160)) & (hsv[...,1]>=80) & (hsv[...,1]<=140) & (hsv[...,2]>=60),
             np.zeros(hsv.shape[:2], dtype=np.uint8)
         )),
    ]
    
    for tag, mask_func in configs:
        red_m, blk_m = mask_func(hsv)
        piece_mask = (red_m | blk_m).astype(np.uint8)
        
        num, labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
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
            r = int(15 * scale)
            y0r, y1r = max(0, int(cy)-r), min(crop.shape[0], int(cy)+r)
            x0r, x1r = max(0, int(cx)-r), min(crop.shape[1], int(cx)+r)
            rp = red_m[y0r:y1r, x0r:x1r].mean()
            bp = blk_m[y0r:y1r, x0r:x1r].mean()
            if tag.startswith("C"):
                patch = hsv[y0r:y1r, x0r:x1r]
                V_med = np.median(patch[..., 2])
                color = "R" if V_med >= 235 else "B"
            else:
                color = "R" if rp > bp else "B"
            pieces.append((float(cx), float(cy), color, float(area)))
        
        # 尝试 find_grid + classify
        if len(pieces) >= 6:
            try:
                row0, col0, c_val, score, hit = find_grid(crop, pieces)
                grid_raw, _ = classify(crop, row0, col0, c_val)
                reds = sum(row.count('@') for row in grid_raw)
                blks = sum(row.count('#') for row in grid_raw)
                print(f"  [{tag}] pieces={len(pieces)} grid={reds}R+{blks}B hit={hit}")
            except Exception as e:
                print(f"  [{tag}] pieces={len(pieces)} ERROR {e}")
        else:
            print(f"  [{tag}] pieces={len(pieces)} (太少)")

# 跑所有 8 张图
print("=" * 60)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    print(f"\n{fname}:")
    run_tuned(fname)
