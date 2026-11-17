"""用 S 中位数区分有无棋子，用位置区分红黑。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import detect_pieces, find_grid, infer_pieces, detect_red_bottom

IMG_DIR = r'g:\chessCn\images'

def classify_new(crop, row0, col0, cell, S_THRESH=92):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    rad = max(6, int(cell * 0.38))
    grid = []
    for r in range(10):
        cy = row0 + r * cell
        line = []
        for c in range(9):
            cx = col0 + c * cell
            y0s, y1s = max(0, int(cy)-rad), min(h, int(cy)+rad)
            x0s, x1s = max(0, int(cx)-rad), min(w, int(cx)+rad)
            if y1s - y0s < 4 or x1s - x0s < 4:
                line.append(".")
                continue
            patch = hsv[y0s:y1s, x0s:x1s]
            S_med = np.median(patch[..., 1])
            if S_med < S_THRESH:
                line.append(".")
            else:
                # 位置先验：图像 row 0~4 是红方（图像上方），row 5~9 是黑方（图像下方）
                line.append("@" if r <= 4 else "#")
        grid.append(line)
    return grid

print("=== 新方案（S median 92 + 位置先验）===")
print(f"{'文件':>35} {'R+B':>6} {'rbott':>5}  row0  row9  row3  row6")
print("=" * 90)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    
    pieces = detect_pieces(crop)
    if len(pieces) < 6:
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        continue
    try:
        row0, col0, cell, score, hit = find_grid(crop, pieces)
    except Exception as e:
        print(f"{fname}: ERROR find_grid {e}")
        continue
    
    grid_new = classify_new(crop, row0, col0, cell, S_THRESH=92)
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    red_bottom = detect_red_bottom(grid_new)
    
    row0_str = ''.join(grid_new[0])
    row3_str = ''.join(grid_new[3])
    row6_str = ''.join(grid_new[6])
    row9_str = ''.join(grid_new[9])
    
    # 标准开局参考：
    # 红方在上：row0=RNBAKABNR, row2=..C.C..C., row3=P.P.P.P.P.
    # 黑方在下：row6=p.p.p.p.p, row7=..c.c..c., row9=rnbakabnr
    std_top = "NB" not in fname  # 只是占位
    
    print(f"{fname:>35} {reds+blks:>3}{'R'+str(reds):>3}{'B'+str(blks):>4} {'red' if red_bottom else 'black':>5}  "
          f"{''.join(row0_str):>9} {''.join(row9_str):>9} {''.join(row3_str):>9} {''.join(row6_str):>9}")

# 看看 S median 的分布：空白格 vs 有棋子
print("\n\n=== S median 分布 ===")
path = os.path.join(IMG_DIR, 'Screenshot_20261117-103438.png')
img = cv2.imread(path)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]
pieces = detect_pieces(crop)
row0, col0, cell, score, hit = find_grid(crop, pieces)

from engine.recognize import classify as classify_old
grid_old, _ = classify_old(crop, row0, col0, cell)

hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
rad = int(cell * 0.38)
blank_S, piece_S = [], []
for r in range(10):
    for c in range(9):
        cy, cx = row0 + r*cell, col0 + c*cell
        y0s, y1s = max(0, int(cy)-rad), min(crop.shape[0], int(cy)+rad)
        x0s, x1s = max(0, int(cx)-rad), min(crop.shape[1], int(cx)+rad)
        patch = hsv[y0s:y1s, x0s:x1s]
        S_med = np.median(patch[..., 1])
        if grid_old[r][c] != '.':
            piece_S.append(S_med)
        else:
            blank_S.append(S_med)
print(f"旧 classify 判为有棋子 (n={len(piece_S)}): S_median=[{min(piece_S):.1f}, {max(piece_S):.1f}] 中位={np.median(piece_S):.1f}")
print(f"旧 classify 判为空白 (n={len(blank_S)}): S_median=[{min(blank_S):.1f}, {max(blank_S):.1f}] 中位={np.median(blank_S):.1f}")
print(f"建议阈值: {np.median(piece_S)+np.median(blank_S)/2:.1f}")
