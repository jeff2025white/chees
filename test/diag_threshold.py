"""重新分类：用**更宽松的掩膜** + 格子内 argmax。"""

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

    # 多种阈值试试
    for name, r_lo, r_hi, s_lo, v_lo, b_v_hi in [
        ("strict", 10, 160, 50, 60, 70),
        ("mid", 15, 150, 30, 40, 90),
        ("loose", 20, 140, 15, 30, 110),
    ]:
        red = ((hsv[..., 0] < r_lo) | (hsv[..., 0] > r_hi)) & (hsv[..., 1] > s_lo) & (hsv[..., 2] > v_lo)
        black = hsv[..., 2] < b_v_hi

        print(f"\n=== {name} red(H<{r_lo} or H>{r_hi}, S>{s_lo}, V>{v_lo})  black(V<{b_v_hi}) ===")
        grid_chars = []
        for row in range(10):
            y_lo, y_hi = HORZ[row], HORZ[row + 1]
            row_chars = []
            for col in range(9):
                x_lo, x_hi = VERT[col], VERT[col + 1]
                rr = red[y_lo:y_hi, x_lo:x_hi].mean()
                br = black[y_lo:y_hi, x_lo:x_hi].mean()
                if rr > 0.03 and rr >= br * 1.1:
                    row_chars.append("R")
                elif br > 0.05 and br >= rr:
                    row_chars.append("B")
                elif rr > br * 0.7 and rr > 0.015:
                    row_chars.append("r")
                elif br > rr * 0.7 and br > 0.02:
                    row_chars.append("b")
                else:
                    row_chars.append(".")
            grid_chars.append("".join(row_chars))
        for line in grid_chars:
            print(line)
        R_count = sum(line.count("R") + line.count("r") for line in grid_chars)
        B_count = sum(line.count("B") + line.count("b") for line in grid_chars)
        print(f"红 {R_count} / 黑 {B_count}")


if __name__ == "__main__":
    main()
