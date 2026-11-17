"""用霍夫直线 + 聚类找出 9×10 网格线，再用颜色分类棋子。"""

import os

import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)


def cluster_lines(values, gap=20):
    """把接近的ρ值聚类成中位数列表。"""
    if not values:
        return []
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def find_grid(board):
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200, L2gradient=True)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    h_rhos, v_rhos = [], []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            b = np.sin(theta)
            a = np.cos(theta)
            if abs(b) > 0.8:  # 水平线
                h_rhos.append(abs(rho))
            elif abs(a) > 0.8:  # 垂直线
                v_rhos.append(rho)

    h_lines = cluster_lines(h_rhos, gap=15)
    v_lines = cluster_lines([abs(v) for v in v_rhos], gap=15)
    print(f"聚类后水平线 {len(h_lines)}: {h_lines}")
    print(f"聚类后垂直线 {len(v_lines)}: {v_lines}")
    return sorted(h_lines), sorted(v_lines)


def pick_best_9(h_lines, need=10):
    """从候选横线里挑出等间距的 10 条（取平均间距、以中值对齐）。"""
    if len(h_lines) < need:
        return None
    gaps = np.diff(h_lines)
    g = np.median(gaps)
    # 找最密的位置当基准
    start = h_lines[0]
    best = None
    best_err = 1e9
    for i in range(len(h_lines) - need + 1):
        seq = h_lines[i:i + need]
        err = np.std(np.diff(seq))
        if err < best_err:
            best_err = err
            best = seq
    if best is None:
        return None
    return best


def main():
    img = cv2.imread(IMG)
    board = img[101:101 + 1163, :]
    bh, bw = board.shape[:2]

    h_lines, v_lines = find_grid(board)

    # 横线找 10 条
    if len(h_lines) >= 10:
        # 取最稳定的 10 条
        seq = None
        best_std = 1e9
        for i in range(len(h_lines) - 10 + 1):
            g = np.diff(h_lines[i:i + 10])
            if np.std(g) < best_std:
                best_std = np.std(g)
                seq = h_lines[i:i + 10]
        print(f"10 条横线: {seq}")
        print(f"  间距: {np.diff(seq).round(1)}  标准偏差: {best_std:.1f}")

    # 竖线找 9 条
    if len(v_lines) >= 9:
        seq2 = None
        best_std2 = 1e9
        for i in range(len(v_lines) - 9 + 1):
            g = np.diff(v_lines[i:i + 9])
            if np.std(g) < best_std2:
                best_std2 = np.std(g)
                seq2 = v_lines[i:i + 9]
        print(f"9 条竖线: {seq2}")
        print(f"  间距: {np.diff(seq2).round(1)}  标准偏差: {best_std2:.1f}")

    # 画出来看
    vis = board.copy()
    for y in h_lines:
        cv2.line(vis, (0, y), (bw, y), (0, 255, 0), 1)
    for x in v_lines:
        cv2.line(vis, (x, 0), (x, bh), (255, 0, 0), 1)
    cv2.imwrite(os.path.join(OUT, "04_grid_lines.png"), vis)


if __name__ == "__main__":
    main()
