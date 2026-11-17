"""把 ROI 区域渲染成 ASCII 艺术图，直接看棋盘结构。"""

import os
import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\roi_ascii.txt"


def main():
    img = cv2.imread(IMG)
    x0, y0, x1, y1 = ROI
    crop = img[y0:y1, x0:x1]
    # 缩放到 72×116（10px 一格）
    small = cv2.resize(crop, (72, 116))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 红色密度（按 10×10 块）
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 50)
    black = hsv[..., 2] < 70

    # 用字符映射灰度 + 红黑提示
    chars = " .:-=+*#%@"
    lines = []
    for row in range(gray.shape[0]):
        s = ""
        for col in range(gray.shape[1]):
            g = gray[row, col]
            r = red[row, col]
            b = black[row, col]
            if r:
                s += "R"
            elif b:
                s += "B"
            else:
                idx = min(int(g / 255 * (len(chars) - 1)), len(chars) - 1)
                s += chars[idx]
        lines.append(s)

    text = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已写入 {OUT}")
    print(text)


if __name__ == "__main__":
    main()
