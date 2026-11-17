"""1. 采样 row9 能检测到/不能检测到格子的 HSV，找覆盖不到的颜色；
2. 遍历所有 8 张截图看漏检情况。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import analyze_image

IMG_DIR = r'g:\chessCn\images'

# ========== 第一部分：103438 的 HSV 采样 ==========
print("====== Screenshot_20261117-103438 的 HSV 采样 ======")
path = os.path.join(IMG_DIR, 'Screenshot_20261117-103438.png')
a = analyze_image(path)
h, w = a.image.shape[:2]
hsv = cv2.cvtColor(a.image, cv2.COLOR_BGR2HSV)
row0, col0, cell = a.row0, a.col0, a.cell

# row9 所有格点
for c in range(9):
    cx = col0 + c * cell
    cy = row0 + 9 * cell
    y0s, y1s = max(0, int(cy)-25), min(h, int(cy)+25)
    x0s, x1s = max(0, int(cx)-25), min(w, int(cx)+25)
    patch = hsv[y0s:y1s, x0s:x1s]
    has = (a.grid_raw[9][c] != '.')
    print(f"row9 col{c}: detected={has}  "
          f"H=[{patch[...,0].min():>3},{patch[...,0].max():>3}]  "
          f"S=[{patch[...,1].min():>3},{patch[...,1].max():>3}]  "
          f"V=[{patch[...,2].min():>3},{patch[...,2].max():>3}]")

print()
# row0 所有格点
for c in range(9):
    cx = col0 + c * cell
    cy = row0 + 0 * cell
    y0s, y1s = max(0, int(cy)-25), min(h, int(cy)+25)
    x0s, x1s = max(0, int(cx)-25), min(w, int(cx)+25)
    patch = hsv[y0s:y1s, x0s:x1s]
    has = (a.grid_raw[0][c] != '.')
    print(f"row0 col{c}: detected={has}  "
          f"H=[{patch[...,0].min():>3},{patch[...,0].max():>3}]  "
          f"S=[{patch[...,1].min():>3},{patch[...,1].max():>3}]  "
          f"V=[{patch[...,2].min():>3},{patch[...,2].max():>3}]")

# ========== 第二部分：所有 8 张截图的漏检情况 ==========
print("\n\n====== 所有截图的漏检情况 ======")
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    try:
        a = analyze_image(os.path.join(IMG_DIR, fname))
    except Exception as e:
        print(f"{fname}: ERROR {e}")
        continue
    reds = sum(row.count('@') for row in a.grid_raw)
    blks = sum(row.count('#') for row in a.grid_raw)
    total = reds + blks
    # 总格子数有棋子的格 = sum(@, #)
    # 标准象棋开局 = 32 子
    print(f"{fname}: 红@={reds} 黑#={blks} 共{total}  view_side={a.view_side}")
    # 打印每行
    for r, row in enumerate(a.grid_raw):
        if '.' * 9 != ''.join(row):
            print(f"  row{r}: {''.join(row)}")
