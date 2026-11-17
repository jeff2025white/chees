"""整图分段诊断：每 100 行统计红+黑密度，看棋子集中在哪几段。"""

import os
import cv2
import numpy as np

IMG_DIR = r"g:\chessCn\images"


def main():
    files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    for fn in files[:3]:  # 看前 3 张就够
        path = os.path.join(IMG_DIR, fn)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
        black = hsv[..., 2] < 75
        # 每 50 行一段统计
        print(f"\n{fn} ({w}x{h})")
        for y0 in range(0, h, 50):
            y1 = min(y0 + 50, h)
            red_dens = red[y0:y1, :].mean()
            black_dens = black[y0:y1, :].mean()
            if red_dens > 0.005 or black_dens > 0.01:
                print(f"  y={y0:4d}: red={red_dens:.3f}  black={black_dens:.3f}")


if __name__ == "__main__":
    main()
