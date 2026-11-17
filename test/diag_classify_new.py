"""新 classify 方案：用 S>=80 比例判断有无棋子，用 H 中位区分红黑。"""
import sys, os, cv2, numpy as np
sys.path.insert(0, r'g:\chessCn')
from engine.recognize import find_grid, infer_pieces, detect_red_bottom

IMG_DIR = r'g:\chessCn\images'

def classify_new(crop, row0, col0, cell, S_THRESH=80, S_RATIO=0.25):
    """新 classify：
    1. 采样半径 cell*0.42 的 patch
    2. S 中位数 > S_THRESH 判有棋子（棋盘底色 S≈68，棋子底色 S≈115）
    3. 用 H 中位数区分红黑
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    rad = max(6, int(cell * 0.42))
    
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
                continue
            # 有棋子，用 H 中位区分红黑
            H_med = np.median(patch[..., 0])
            # 但 patch 里有白色字（H 随机、S=0），需要排除
            nonwhite = patch[patch[..., 1] > 5]  # S>5 排除白色字
            if len(nonwhite) > 0:
                H_med = np.median(nonwhite[:, 0])
                S_med = np.median(nonwhite[:, 1])
            # 红方底色 H≈5~16，黑方底色 H≈10~22，重叠！
            # 换成看 V 中位数：红方底色 V≈244，黑方底色 V≈218
            V_med = np.median(nonwhite[:, 2]) if len(nonwhite) > 0 else np.median(patch[..., 2])
            
            # 方案 1: 用 H 中位
            if H_med <= 14 or H_med >= 165:
                line.append("@")
            else:
                line.append("#")
        grid.append(line)
    return grid

# 跑所有图
print(f"{'文件':>35} {'R+B':>6} {'view_side':>10}  grid (row 0, 9)")
print("=" * 80)
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.startswith('Screenshot') or not fname.endswith('.png'):
        continue
    path = os.path.join(IMG_DIR, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
    crop = img[y0:y1, 0:w]
    
    # 先用旧 detect_pieces 拿棋子位置
    from engine.recognize import detect_pieces
    pieces = detect_pieces(crop)
    if len(pieces) < 6:
        print(f"{fname}: 棋子太少 ({len(pieces)})")
        continue
    row0, col0, cell, score, hit = find_grid(crop, pieces)
    
    # 新 classify
    grid_new = classify_new(crop, row0, col0, cell)
    
    reds = sum(row.count('@') for row in grid_new)
    blks = sum(row.count('#') for row in grid_new)
    red_bottom = detect_red_bottom(grid_new)
    
    print(f"{fname:>35} {reds+blks:>3}{'R'+str(reds):>3}{'B'+str(blks):>4} {'red' if red_bottom else 'black':>10}  "
          f"row0={''.join(grid_new[0])}  row9={''.join(grid_new[9])}")
    
    # 简单统计：标准开局应该是 row0 和 row9 各 9 个，row2 和 row7 各 2 个炮，row3 和 row6 各 5 个兵
    # 如果漏太多，打印全部行
    if reds + blks < 28:
        print("  (漏子多)")
        for r, row in enumerate(grid_new):
            if '.'*9 != ''.join(row):
                print(f"    row{r}: {''.join(row)}")
