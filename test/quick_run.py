"""把 strict 阈值的颜色识别结果喂给引擎，先跑通流程。
兵种暂时用一个简化假设：
  - row 0 (红方底线): R N B A K A B N R
  - row 1: . . . . . . . . .  （两个炮）
  - row 2: P P P P P （兵）
  - row 3+: 留空（或者逐格猜）
  
  等等，更聪明的方式：只填颜色，让引擎猜兵种？不行，引擎需要具体兵种。
  
  还是老老实实先跑：所有红色格子填 "R"（车），所有黑色填 "r"。
  虽然不合理，但至少 FEN 合法，引擎能给出一个着法。
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


def main():
    img = cv2.imread(IMG)
    x0, y0, _, _ = ROI
    crop = img[y0:y0 + 1163, x0:x0 + 720]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # strict 阈值
    red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 60)
    black = hsv[..., 2] < 70

    # 按行+列的区域分类
    grid = []  # 10 行 × 9 列
    for row in range(10):
        y_lo, y_hi = HORZ[row], HORZ[row + 1]
        row_chars = []
        for col in range(9):
            x_lo, x_hi = VERT[col], VERT[col + 1]
            rr = red[y_lo:y_hi, x_lo:x_hi].mean()
            br = black[y_lo:y_hi, x_lo:x_hi].mean()
            if rr > 0.03 and rr >= br * 1.1:
                row_chars.append("R")  # 临时：所有红方格子填 "R"（车）
            elif br > 0.05 and br >= rr:
                row_chars.append("r")  # 临时：所有黑方格子填 "r"（车）
            elif rr > br * 0.7 and rr > 0.015:
                row_chars.append("R")  # 浅红也当红方
            elif br > rr * 0.7 and br > 0.02:
                row_chars.append("r")
            else:
                row_chars.append(".")
        grid.append(row_chars)

    grid_str = "\n".join("".join(r) for r in grid)
    print("临时 FEN 网格：")
    print(grid_str)

    # 先让引擎跑一把，看输出
    result = decide_position(grid=grid_str, side="red", view_side="red",
                             time_limit=2.0, max_depth=5, top_n=1)
    best = result.get("best_move")
    print(f"\n引擎输出: best_move={best['chinese'] if best else None} reason={best['reason'] if best else None}")


if __name__ == "__main__":
    main()
