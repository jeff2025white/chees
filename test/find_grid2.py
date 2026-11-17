"""投影法找网格：先按列/行投影灰度，在投影的局部最小值处取网格线。
这样即使霍夫检测被中间按钮干扰也能找到真正的棋盘线。"""

import os

import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)


def local_minima(arr, min_gap=15):
    """找一维数组的局部最小值，距离 min_gap 以内只保留最低的。"""
    n = len(arr)
    peak = [0]
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            peak.append(i)
    peak.append(n - 1)
    # 去掉重复
    filtered = [peak[0]]
    for p in peak[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
        elif arr[p] < arr[filtered[-1]]:
            filtered[-1] = p
    return filtered


def find_grid(board, min_gap=50):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    bh, bw = gray.shape

    # 列投影（用于找水平线，也就是在灰度最低的行）
    row_proj = gray.mean(axis=1)  # 每一行的均值
    h_lines = local_minima(row_proj, min_gap=min_gap)
    print(f"候选横线: {h_lines}  间距: {np.diff(h_lines).round(1) if len(h_lines) > 1 else ''}")

    # 行投影（用于找垂直线）
    col_proj = gray.mean(axis=0)
    v_lines = local_minima(col_proj, min_gap=min_gap)
    print(f"候选竖线: {v_lines}  间距: {np.diff(v_lines).round(1) if len(v_lines) > 1 else ''}")

    # 画出来
    vis = board.copy()
    for y in h_lines:
        cv2.line(vis, (0, y), (bw, y), (0, 255, 0), 1)
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, bh), (255, 0, 0), 1)
    cv2.imwrite(os.path.join(OUT, "05_projection.png"), vis)
    # 保存投影曲线便于肉眼判断
    col_img = np.tile((row_proj / 255 * 200).astype(np.uint8)[None, :], (1, 1))
    cv2.imwrite(os.path.join(OUT, "06_row_proj.png"),
                (255 - row_proj).astype(np.uint8)[None, :].T.repeat(20, axis=1))
    cv2.imwrite(os.path.join(OUT, "07_col_proj.png"),
                (255 - col_proj).astype(np.uint8)[:, None].repeat(20, axis=1))
    return h_lines, v_lines


def main():
    img = cv2.imread(IMG)
    board = img[101:101 + 1163, :]
    bh, bw = board.shape[:2]
    print(f"棋盘 {bw}x{bh}")
    h, v = find_grid(board, min_gap=40)


if __name__ == "__main__":
    main()
