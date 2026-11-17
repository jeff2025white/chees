"""彻底看清 ROI 里是什么：把 ROI 裁出来、画网格交点、保存成 PNG。"""

import os
import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)

HORZ = [188, 263, 339, 411, 490, 569, 644, 720, 796, 864, 926]
VERT = [43, 131, 207, 284, 360, 437, 513, 590, 666, 717]


def main():
    img = cv2.imread(IMG)
    x0, y0, x1, y1 = ROI
    crop = img[y0:y1, x0:x1]

    # 原始 ROI
    cv2.imwrite(os.path.join(OUT, "20_roi_raw.png"), crop)

    # ROI 里所有红色像素的 mask（缩放到 1 条通道）
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    black = hsv[..., 2] < 75
    red_u8 = (red * 255).astype(np.uint8)
    black_u8 = (black * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "21_red_mask.png"), red_u8)
    cv2.imwrite(os.path.join(OUT, "22_black_mask.png"), black_u8)

    # 在 ROI 上画：横线（绿）、竖线（蓝）、格子中心（红圆）
    vis = crop.copy()
    for y in HORZ:
        cv2.line(vis, (0, y), (crop.shape[1], y), (0, 255, 0), 2)
    for x in VERT:
        cv2.line(vis, (x, 0), (x, crop.shape[0]), (255, 0, 0), 2)
    # 画格子中心（线中点）
    row_centers = [(HORZ[i] + HORZ[i + 1]) / 2 for i in range(10)]
    col_centers = [(VERT[i] + VERT[i + 1]) / 2 for i in range(9)]
    for rc in row_centers:
        for cc in col_centers:
            cv2.circle(vis, (int(cc), int(rc)), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT, "23_grid_on_roi.png"), vis)

    # 逐行投影图（灰度均值）
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)
    cv2.imwrite(os.path.join(OUT, "24_row_proj.png"),
                (255 - row_proj).astype(np.uint8)[:, None].repeat(30, axis=1))
    cv2.imwrite(os.path.join(OUT, "25_col_proj.png"),
                (255 - col_proj).astype(np.uint8)[None, :].repeat(30, axis=0))

    # 红色 mask 的投影（每行每列的红色密度）
    red_row = red.mean(axis=1)
    red_col = red.mean(axis=0)
    cv2.imwrite(os.path.join(OUT, "26_red_row.png"),
                (red_row * 255).astype(np.uint8)[:, None].repeat(30, axis=1))
    cv2.imwrite(os.path.join(OUT, "27_red_col.png"),
                (red_col * 255).astype(np.uint8)[None, :].repeat(30, axis=0))
    black_row = black.mean(axis=1)
    cv2.imwrite(os.path.join(OUT, "28_black_row.png"),
                (black_row * 255).astype(np.uint8)[:, None].repeat(30, axis=1))


if __name__ == "__main__":
    main()
