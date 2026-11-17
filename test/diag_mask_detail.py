"""直接打印 piece_mask 在关键位置的数值，搞清楚 detect_pieces 为何能检测到。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')

path = r'g:\chessCn\images\Screenshot_20261117-103438.png'
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
piece_mask = (red_mask | black_mask).astype(np.uint8)

print(f"red_mask.sum  = {red_mask.sum()}, 非零像素位置 HSV 统计:")
nz = hsv[red_mask > 0]
if len(nz):
    print(f"  H=[{nz[:,0].min()},{nz[:,0].max()}]  S=[{nz[:,1].min()},{nz[:,1].max()}]  V=[{nz[:,2].min()},{nz[:,2].max()}]")
else:
    print("  (空！)")

print(f"black_mask.sum = {black_mask.sum()}, 非零像素位置 HSV 统计:")
nz = hsv[black_mask > 0]
if len(nz):
    print(f"  H=[{nz[:,0].min()},{nz[:,0].max()}]  S=[{nz[:,1].min()},{nz[:,1].max()}]  V=[{nz[:,2].min()},{nz[:,2].max()}]")
else:
    print("  (空！)")

print(f"piece_mask.sum = {piece_mask.sum()}")

# 在 row0 col1 格点附近（cell=76.3, row0=180.3, col0=55.7, row0 col1 = 132, 180）看看 piece_mask 数值
cell = 76.3
row0_g, col0_g = 180.3, 55.7
for target in [("row0 col1 (红车)", 132, 180), ("row9 col0 (黑车)", 56, 867), 
               ("row9 col1 (黑马)", 132, 867), ("row3 col0 (红兵)", 56, 410)]:
    name, cx, cy = target
    y0s, y1s = max(0, int(cy-30)), min(crop.shape[0], int(cy+30))
    x0s, x1s = max(0, int(cx-30)), min(crop.shape[1], int(cx+30))
    patch_red = red_mask[y0s:y1s, x0s:x1s]
    patch_blk = black_mask[y0s:y1s, x0s:x1s]
    patch_piece = piece_mask[y0s:y1s, x0s:x1s]
    patch_hsv = hsv[y0s:y1s, x0s:x1s]
    print(f"\n{name} ({cx:.0f},{cy:.0f}):")
    print(f"  red_mask 非零 = {patch_red.sum()} / {patch_red.size}  ({100*patch_red.mean():.1f}%)")
    print(f"  blk_mask 非零 = {patch_blk.sum()} / {patch_blk.size}  ({100*patch_blk.mean():.1f}%)")
    print(f"  piece_mask 非零 = {patch_piece.sum()} / {patch_piece.size}  ({100*patch_piece.mean():.1f}%)")
    print(f"  patch 内 HSV: H=[{patch_hsv[...,0].min():>3},{patch_hsv[...,0].max():>3}]  "
          f"S=[{patch_hsv[...,1].min():>3},{patch_hsv[...,1].max():>3}]  "
          f"V=[{patch_hsv[...,2].min():>3},{patch_hsv[...,2].max():>3}]")

# 打印 piece_mask 非零像素的完整 HSV 分布
print("\n\npiece_mask 非零像素的完整 HSV 分布 (bin 统计):")
nz_idx = piece_mask > 0
nz_hsv = hsv[nz_idx]
print(f"共 {len(nz_hsv)} 个非零像素")
print(f"H: {np.histogram(nz_hsv[:,0], bins=[0,5,10,15,20,30,60,90,120,160,180])[0]}")
print(f"S: {np.histogram(nz_hsv[:,1], bins=[0,20,40,60,80,100,150,200,255])[0]}")
print(f"V: {np.histogram(nz_hsv[:,2], bins=[0,40,60,80,100,150,200,230,255])[0]}")

# 看看 detect_pieces 里连通域质心的颜色分布
from engine.recognize import detect_pieces
pieces = detect_pieces(crop)
print(f"\ndetect_pieces 返回的 {len(pieces)} 个棋子按 cy 分桶:")
buckets = {}
for cx, cy, color, area in pieces:
    bucket = round(cy / cell)
    buckets.setdefault(bucket, []).append((cx, cy, color, area))
for k in sorted(buckets.keys()):
    print(f"  bucket {k}: {len(buckets[k])} 个")
    for p in sorted(buckets[k], key=lambda x: x[0]):
        print(f"    cx={p[0]:6.1f} cy={p[1]:6.1f} c={p[2]} area={p[3]:.0f}")
