"""对每个格子区域（由相邻横线/竖线框定的矩形），找红/黑密度最高的点。"""

import os
import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work"

HORZ = [188, 263, 339, 411, 490, 569, 644, 720, 796, 864, 926]
VERT = [43, 131, 207, 284, 360, 437, 513, 590, 666, 717]


def main():
    img = cv2.imread(IMG)
    x0, y0, _, _ = ROI
    crop = img[y0:y0 + 1163, x0:x0 + 720]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 50)
    black = hsv[..., 2] < 70

    print("每个格子的红/黑密度 (max)：")
    print("       ", "  ".join([f"c{c:>5}" for c in range(9)]))
    for row in range(10):
        y_lo = HORZ[row]
        y_hi = HORZ[row + 1]
        line = f"row {row:>2}: "
        for col in range(9):
            x_lo = VERT[col]
            x_hi = VERT[col + 1]
            r_region = red[y_lo:y_hi, x_lo:x_hi]
            b_region = black[y_lo:y_hi, x_lo:x_hi]
            r_max = float(r_region.mean())
            b_max = float(b_region.mean())
            # 也找局部最大值的位置
            if r_max > 0.02:
                pos = np.unravel_index(r_region.argmax(), r_region.shape)
                line += f" R{r_max*100:4.0f}/{pos[0]:>2},{pos[1]:>2}"
            elif b_max > 0.02:
                pos = np.unravel_index(b_region.argmax(), b_region.shape)
                line += f" B{b_max*100:4.0f}/{pos[0]:>2},{pos[1]:>2}"
            else:
                line += f" .{r_max*100:4.0f}/{b_max*100:4.0f}"
        print(line)


if __name__ == "__main__":
    main()
