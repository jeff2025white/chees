"""诊断：对 ROI 的 90 个格子中心直接打印颜色统计。"""

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
    cell_w = np.median(np.diff(VERT))
    cell_h = np.median(np.diff(HORZ))
    print(f"cell_w={cell_w:.0f} cell_h={cell_h:.0f}")

    # 对所有 90 个格子中心采样
    results = []
    for row in range(10):
        for col in range(9):
            cx = int(x0 + VERT[col] + cell_w / 2)
            cy = int(y0 + HORZ[row] + cell_h / 2)
            r = int(min(cell_w, cell_h) * 0.32)
            patch = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            if patch.size == 0:
                continue
            bgr = patch.reshape(-1, 3).mean(axis=0)
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            mean_hsv = hsv_patch.mean(axis=0)
            # 红色掩膜
            red_mask = ((hsv_patch[:, 0] < 10) | (hsv_patch[:, 0] > 160)) & (hsv_patch[:, 1] > 50) & (hsv_patch[:, 2] > 60)
            red_ratio = red_mask.mean()
            # 黑色掩膜
            black_mask = hsv_patch[:, 2] < 75
            black_ratio = black_mask.mean()
            results.append((row, col, cx, cy, bgr, mean_hsv, red_ratio, black_ratio))

    # 打印有棋子嫌疑的格子（red_ratio>0.05 或 black_ratio>0.08）
    print("\n有颜色嫌疑的格子:")
    for row, col, cx, cy, bgr, hsv, rr, br in results:
        if rr > 0.05 or br > 0.08:
            print(f"  ({row},{col}) px=({cx},{cy}) BGR={tuple(bgr.astype(int))} HSV={tuple(hsv.astype(int))} R={rr:.2f} B={br:.2f}")

    # 也打印几个空格当对照
    print("\n对照空格（随机 10 个）:")
    sample_idx = sorted([(r, c) for r, c, *_ in results], key=lambda x: (x[0] * 7 + x[1]))
    for r, c in sample_idx[:10]:
        rr = [x for x in results if x[0] == r and x[1] == c][0]
        row, col, cx, cy, bgr, hsv, _, _ = rr
        print(f"  ({row},{col}) px=({cx},{cy}) BGR={tuple(bgr.astype(int))} HSV={tuple(hsv.astype(int))}")


if __name__ == "__main__":
    main()
