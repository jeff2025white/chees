"""用正确的格子中心重新采样颜色。"""

import os
import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)
HORZ = [188, 263, 339, 411, 490, 569, 644, 720, 796, 864, 926]
VERT = [43, 131, 207, 284, 360, 437, 513, 590, 666, 717]


def main():
    img = cv2.imread(IMG)
    x0, y0, _, _ = ROI

    row_centers = [(HORZ[i] + HORZ[i + 1]) / 2 for i in range(10)]
    col_centers = [(VERT[i] + VERT[i + 1]) / 2 for i in range(9)]
    cell = 76

    print(f"row_centers = {row_centers}")
    print(f"col_centers = {col_centers}")

    results = []
    for row in range(10):
        for col in range(9):
            cx = int(x0 + col_centers[col])
            cy = int(y0 + row_centers[row])
            r = int(cell * 0.32)
            patch = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.size == 0:
                continue
            bgr = patch.reshape(-1, 3).mean(axis=0)
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            red_mask = ((hsv_patch[:, 0] < 10) | (hsv_patch[:, 0] > 160)) & (hsv_patch[:, 1] > 60) & (hsv_patch[:, 2] > 60)
            red_ratio = red_mask.mean()
            black_mask = hsv_patch[:, 2] < 75
            black_ratio = black_mask.mean()
            results.append((row, col, cx, cy, bgr, red_ratio, black_ratio))

    print("\n按格子扫描（只打印有嫌疑的）:")
    for row, col, cx, cy, bgr, rr, br in results:
        if rr > 0.03 or br > 0.05:
            print(f"  row={row} col={col} px=({cx},{cy}) BGR={tuple(bgr.astype(int))} R={rr:.2f} B={br:.2f}")

    print("\n全部 90 格的 red_ratio:")
    grid = []
    for row in range(10):
        line = ""
        for col in range(9):
            rr = [x[5] for x in results if x[0] == row and x[1] == col][0]
            br = [x[6] for x in results if x[0] == row and x[1] == col][0]
            line += f"{rr*100:4.0f}/{br*100:4.0f} "
        grid.append(line)
    for line in grid:
        print(line)


if __name__ == "__main__":
    main()
