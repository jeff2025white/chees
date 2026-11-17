"""完整识别流水线验证：9x10 网格 → 90 个棋子位置 → 颜色分类 → 走棋 → 画圈。
先用一张图完整跑通，再批量处理。"""

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import decide_position
from engine.constants import BLACK, RED

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)

# 上一轮投影法得到的稳定网格（参考点）
# 竖线: [0, 43, 131, 207, 284, 360, 437, 513, 590, 717] → 9 条内部: 131..590
# 横线: [0, 43, 135, 188, 263, 339, 411, 490, 569, 644, 720, 796, 864, 926, ...]
#       内部: 188..926 共 10 条


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


def detect_board_anchor(img):
    """返回 (x0, y0, cell_w, cell_h, grid_cols, grid_rows)。

    用投影法找到最稳定的 9×10 网格。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)

    h_lines = local_minima(row_proj, min_gap=30)
    v_lines = local_minima(col_proj, min_gap=30)

    # 选出等间距的那段
    def pick_uniform(lines, need):
        if len(lines) < need:
            return None
        best_seq = None
        best_std = 1e9
        for i in range(len(lines) - need + 1):
            seq = lines[i:i + need]
            gaps = np.diff(seq)
            if np.std(gaps) < best_std:
                best_std = np.std(gaps)
                best_seq = seq
        return best_seq

    vert = pick_uniform(v_lines, 10)  # 10 条竖线 → 9 格
    horz = pick_uniform(h_lines, 11)  # 11 条横线 → 10 格

    print(f"选中竖线: {vert}  std={np.std(np.diff(vert)):.1f}")
    print(f"选中横线: {horz}  std={np.std(np.diff(horz)):.1f}")

    cell_w = (vert[-1] - vert[0]) / 9
    cell_h = (horz[-1] - horz[0]) / 10
    return vert[0], horz[0], cell_w, cell_h, vert, horz


def classify_cell(img, cx, cy, cell_size):
    """在格子中心区域采样，返回 'r' | 'b' | '.'。"""
    r = int(cell_size * 0.30)
    patch = img[cy - r:cy + r, cx - r:cx + r]
    if patch.size == 0:
        return ".", None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 黑色：V 通道低
    v_low = (hsv[..., 2] < 70).mean()
    # 红色：H 在 0-10 或 160-180
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    red_ratio = red_mask.mean()
    return red_ratio, v_low


def recognize(img_path, side_bottom=None):
    """识别一张截图的棋盘。

    side_bottom: 'red' 红方在下 / 'black' 黑方在下 / None（自动检测：哪方棋子多且在中心区域）
    返回 (grid_string, cell_centers, board_x0, board_y0, cell_w, cell_h, view_side)
    """
    img = cv2.imread(img_path)
    x0, y0, cell_w, cell_h, vert, horz = detect_board_anchor(img)
    centers = []
    grid_chars = []
    for row in range(10):
        row_chars = []
        for col in range(9):
            cx = int(vert[col] + cell_w / 2)
            cy = int(horz[row] + cell_h / 2)
            red_ratio, black_ratio = classify_cell(img, cx, cy, cell_w)
            char = "."
            if red_ratio > 0.12 and red_ratio > black_ratio:
                char = "R"
            elif black_ratio > 0.15 and black_ratio > red_ratio:
                char = "r"
            row_chars.append(char)
            centers.append((cx, cy))
        grid_chars.append("".join(row_chars))

    grid_str = "\n".join(grid_chars)
    print("\n识别网格:")
    print(grid_str)

    # 统计双方棋子数
    red_count = sum(row.count("R") for row in grid_chars)
    black_count = sum(row.count("r") for row in grid_chars)
    print(f"红棋 {red_count} 颗，黑棋 {black_count} 颗")

    return grid_str, centers, x0, y0, cell_w, cell_h, img.shape[:2]


def main():
    grid_str, centers, bx, by, cw, ch, _ = recognize(IMG)
    print(f"\n棋盘区域 x={bx} y={by} cell={cw:.1f}x{ch:.1f}")


if __name__ == "__main__":
    main()
