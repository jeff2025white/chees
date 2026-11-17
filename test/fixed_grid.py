"""更聪明的网格找法：先从横线算出 cell_h，再用 cell_h 约束找竖线（cell_w ≈ cell_h）。"""

import os
import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)

ROI = (0, 101, 720, 1264)


def local_minima(arr, min_gap=15):
    n = len(arr)
    peaks = [0]
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            peaks.append(i)
    peaks.append(n - 1)
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
        elif arr[p] < arr[filtered[-1]]:
            filtered[-1] = p
    return filtered


def pick_horizontal(gray_crop):
    """直接返回最好的 11 条横线 + cell_h 估计。"""
    h, w = gray_crop.shape
    row_proj = gray_crop.mean(axis=1)
    h_cands = local_minima(row_proj, min_gap=int(h * 0.04))
    # 穷举所有连续 11 条的窗口，选 std 最小且 median 最合理的
    best = None
    for i in range(len(h_cands) - 11 + 1):
        seq = h_cands[i:i + 11]
        gaps = np.diff(seq)
        med = np.median(gaps)
        std = np.std(gaps)
        # median 要在合理范围内（cell_h 通常 40-100）
        if not (40 <= med <= 100):
            continue
        # 比例要合理
        if abs((seq[-1] - seq[0]) - med * 10) > med * 1.5:
            continue
        score = std / med  # 相对偏差
        if best is None or score < best[0]:
            best = (score, seq, med)
    return best


def pick_vertical(gray_crop, cell_h_guess):
    """给定 cell_h 的先验，从竖线候选里挑出 10 条（cell_w ≈ cell_h_guess）。"""
    h, w = gray_crop.shape
    col_proj = gray_crop.mean(axis=0)
    v_cands = local_minima(col_proj, min_gap=max(int(cell_h_guess * 0.4), 15))

    # 贪心：从每个候选起点出发，每次选「距离 last + target_w 最近且 >= target_w*0.5」的下一个候选
    def greedy(start):
        target = cell_h_guess
        chosen = [start]
        last = start
        while len(chosen) < 10:
            next_candidates = [c for c in v_cands if c > last + target * 0.4]
            if not next_candidates:
                return None
            # 选距离 target 最近的
            next_candidates.sort(key=lambda c: abs(c - (last + target)))
            # 不能跳太远（超过 2 倍 target）
            if next_candidates[0] - last > target * 2.5:
                return None
            last = next_candidates[0]
            chosen.append(last)
        return chosen

    best = None
    for s in v_cands[:-9]:  # 起点候选
        seq = greedy(s)
        if seq is None:
            continue
        gaps = np.diff(seq)
        med = np.median(gaps)
        std = np.std(gaps)
        score = std / med + abs(med - cell_h_guess) / cell_h_guess
        if best is None or score < best[0]:
            best = (score, seq, med)
    return best


def main():
    img = cv2.imread(IMG)
    x0, y0, x1, y1 = ROI
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    res_h = pick_horizontal(gray)
    if res_h is None:
        print("横线失败")
        return
    _, horz, cell_h = res_h
    print(f"横线: {horz}  cell_h={cell_h:.0f}")

    res_v = pick_vertical(gray, cell_h)
    if res_v is None:
        print("竖线失败")
        return
    _, vert, cell_w = res_v
    print(f"竖线: {vert}  cell_w={cell_w:.0f}")

    # 画图
    vis = crop.copy()
    for y in horz:
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 2)
    for x in vert:
        cv2.line(vis, (x, 0), (h, x) if False else (x, h), (255, 0, 0), 2)
    # 画交点（90 个格子中心）
    for row in range(10):
        for col in range(9):
            cx = int(vert[col] + cell_w / 2)
            cy = int(horz[row] + cell_h / 2)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT, "12_fixed_grid.png"), vis)


if __name__ == "__main__":
    main()
