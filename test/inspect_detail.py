"""更深入的诊断：把截图的棋盘区域做 Canny 边缘、霍夫直线、角点检测，
把结果图和像素抽样一起保存，便于人工推断。
"""

import os

import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)


def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    print(f"原图 {w}x{h}")

    # 1) 棋盘区域（经验值：y 101 到 y 101+1163 整图宽）
    x0, y0, x1, y1 = 0, 101, w, 101 + 1163
    board = img[y0:y1, x0:x1]
    bh, bw = board.shape[:2]
    print(f"棋盘候选 {bw}x{bh}")
    cv2.imwrite(os.path.join(OUT, "01_board_candidate.png"), board)

    # 2) 灰度 + Canny
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, L2gradient=True)
    cv2.imwrite(os.path.join(OUT, "02_canny.png"), edges)

    # 3) 霍夫直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    vis = board.copy()
    hlines, vlines = [], []
    if lines is not None:
        for (rho, theta) in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            xc = a * rho
            yc = b * rho
            x1l = int(xc + 2000 * (-b))
            y1l = int(yc + 2000 * a)
            x2l = int(xc - 2000 * (-b))
            y2l = int(yc - 2000 * a)
            cv2.line(vis, (x1l, y1l), (x2l, y2l), (0, 255, 0), 2)
            if abs(b) > 0.8:
                hlines.append(rho)
            elif abs(a) > 0.8:
                vlines.append(rho)
    hlines = sorted(set([int(r) for r in hlines]))
    vlines = sorted(set([int(r) for r in vlines]))
    print(f"霍夫水平线 {len(hlines)}: {hlines}")
    print(f"霍夫垂直线 {len(vlines)}: {vlines}")
    cv2.imwrite(os.path.join(OUT, "03_hough.png"), vis)

    # 4) 打印中心 200x200 区域的像素样本（每隔20像素）
    center_y, center_x = bh // 2, bw // 2
    print("\n中心区域像素抽样 (BGR):")
    for dy in range(-100, 100, 20):
        row = []
        for dx in range(-100, 100, 20):
            py, px = center_y + dy, center_x + dx
            if 0 <= py < bh and 0 <= px < bw:
                bgr = board[py, px]
                row.append(f"{tuple(bgr)}")
        print("  ".join(row))

    # 5) 统计棋盘行/列投影（平均灰度），找条纹
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    print(f"\n行灰度投影（每 50 像素）: {row_mean[::50].round(1)}")
    print(f"列灰度投影（每 50 像素）: {col_mean[::50].round(1)}")


if __name__ == "__main__":
    main()
