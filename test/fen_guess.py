"""按位置猜兵种：图像 row 9 是红方底线（9 子），row 0 是黑方底线（9 子）。
从 strict 颜色识别里把颜色填进去，然后用位置逻辑补兵种。
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import decide_position

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)

HORZ = [188, 263, 339, 411, 490, 569, 644, 720, 796, 864, 926]
VERT = [43, 131, 207, 284, 360, 437, 513, 590, 666, 717]


def guess_fen_from_colors(color_grid):
    """根据 strict 识别的颜色 grid 猜 FEN。

    约定：
      图像 row 0 = 图像最上方 = 应该是黑方上方区域
      图像 row 9 = 图像最下方 = 应该是红方下方区域
      
      用位置+颜色猜兵种：
        红方底线（row 9）：9 个红子 → R N B A K A B N R
        黑方底线（row 0）：9 个黑子 → r n b a k a b n r
        次线：4 个子 → 两个炮（c3 和 c7），其余空
        第三层：5 个兵/卒
    """
    # 首先把 strict 里的深浅颜色都合并
    color_grid = [
        ["R" if c.upper() == "R" else ("B" if c.upper() == "B" else ".")
         for c in row] for row in color_grid
    ]

    # 按位置分兵种
    fen_lines = []
    for row in range(10):
        line = []
        for col in range(9):
            color = color_grid[row][col]
            if color == ".":
                line.append(".")
                continue
            # 红方（图像下方 row 6-9）还是黑方（图像上方 row 0-3）
            if row >= 6:  # 图像下方（红方区域）
                depth = row - 9  # row 9 → 底线（depth=0），row 6 → depth=-3（空）
            else:  # 图像上方（黑方区域）
                depth = row - 0  # row 0 → 底线（depth=0）
            if depth == 0:  # 底线
                name = ["R", "N", "B", "A", "K", "A", "B", "N", "R"][col]
                line.append(name if color == "R" else name.lower())
            elif depth == 1:  # 次线
                if col == 2 or col == 7:
                    line.append("C" if color == "R" else "c")
                else:
                    line.append(".")
            elif depth == 2:  # 兵/卒行
                if col in (0, 2, 4, 6, 8):
                    line.append("P" if color == "R" else "p")
                else:
                    line.append(".")
            elif depth < 0:  # 图像下方 row 5/6 不该有红方（strict 误判）
                # 看颜色
                if color == "R":
                    line.append("P")  # 暂时当兵
                else:
                    line.append(".")
            else:  # 黑方 depth=2 → 卒行
                if depth == 2:
                    if col in (0, 2, 4, 6, 8):
                        line.append("p")
                    else:
                        line.append(".")
                else:
                    line.append(".")
        fen_lines.append("".join(line))

    # FEN 行序：row 0 最上方（黑方底线），row 9 最下方（红方底线）——正好匹配我们的行序！
    return "/".join(fen_lines) + " w - - 0 1"


def main():
    img = cv2.imread(IMG)
    x0, y0, _, _ = ROI
    crop = img[y0:y0 + 1163, x0:x0 + 720]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 60)
    black_mask = hsv[..., 2] < 70

    color_grid = []
    for row in range(10):
        y_lo, y_hi = HORZ[row], HORZ[row + 1]
        row_chars = []
        for col in range(9):
            x_lo, x_hi = VERT[col], VERT[col + 1]
            rr = red_mask[y_lo:y_hi, x_lo:x_hi].mean()
            br = black_mask[y_lo:y_hi, x_lo:x_hi].mean()
            if rr > 0.03 and rr >= br * 1.1:
                row_chars.append("R")
            elif br > 0.05 and br >= rr:
                row_chars.append("B")
            elif rr > br * 0.7 and rr > 0.015:
                row_chars.append("R")
            elif br > rr * 0.7 and br > 0.02:
                row_chars.append("B")
            else:
                row_chars.append(".")
        color_grid.append(row_chars)

    fen = guess_fen_from_colors(color_grid)
    print(f"猜出的 FEN: {fen}")

    # 验证 FEN 能不能被引擎解析
    result = decide_position(fen=fen, time_limit=2.0, max_depth=5, top_n=1)
    best = result.get("best_move")
    print(f"引擎输出: best_move={best['chinese'] if best else None} ({best['iccs'] if best else None})")
    print(f"理由: {best['reason'] if best else result.get('message')}")
    print(f"局面状态: {result.get('status')}")


if __name__ == "__main__":
    main()
