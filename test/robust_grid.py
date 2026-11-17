"""用多步策略稳健定位棋盘：先找高频边缘区，再投影找 9×10 网格。"""

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT, exist_ok=True)


def local_minima(arr, min_gap=15):
    n = len(arr)
    peaks = [0]
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            peaks.append(i)
    peaks.append(n - 1)
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
        elif arr[p] < arr[filtered[-1]]:
            filtered[-1] = p
    return filtered


def pick_uniform(lines, need):
    if lines is None or len(lines) < need:
        return None
    best_seq = None
    best_err = 1e9
    for i in range(len(lines) - need + 1):
        seq = lines[i:i + need]
        gaps = np.diff(seq)
        if len(gaps) == 0:
            continue
        err = np.std(gaps)
        if err < best_err:
            best_err = err
            best_seq = seq
    return best_seq


def find_board_region_edges(img):
    """用 Sobel + 连通域找有很多线条的矩形区域。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel x + y
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    mag_u8 = np.clip(mag / 30, 0, 255).astype(np.uint8)
    _, edges = cv2.threshold(mag_u8, 30, 255, cv2.THRESH_BINARY)

    # 膨胀连通 + 取最大连通域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def find_board_region_color(img):
    """用颜色掩膜找棋盘区域（红蓝棋子+棋盘线集中的区域）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色棋子/棋盘线
    red = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) | cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    # 深棕色（棋子外框）+ 浅蓝色棋盘底
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))  # 深色部分
    mask = cv2.bitwise_or(red, dark)

    # 闭运算聚合
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def robust_find(img):
    """综合边缘密度和颜色掩膜找棋盘。"""
    r1 = find_board_region_edges(img)
    r2 = find_board_region_color(img)
    hh, ww = img.shape[:2]
    print(f"边缘方法: {r1}  颜色方法: {r2}")

    def score(r):
        if r is None:
            return 0, None
        x, y, w, h = r
        # 要占整图面积 30-70%，比例接近 0.5-0.8
        area = w * h / (ww * hh)
        ratio = w / h if h else 0
        if not (0.15 <= area <= 0.85):
            return 0, None
        if not (0.4 <= ratio <= 1.5):
            return 0, None
        return area, r

    s1 = score(r1)
    s2 = score(r2)
    print(f"边缘分数 {s1}  颜色分数 {s2}")

    # 取面积较大的有效候选
    chosen = None
    if s1[0] > 0 and s2[0] > 0:
        chosen = s1[1] if s1[0] > s2[0] else s2[1]
    elif s1[0] > 0:
        chosen = s1[1]
    elif s2[0] > 0:
        chosen = s2[1]
    else:
        return None

    # 给候选扩个边，避免切到棋盘边缘
    x, y, w, h = chosen
    pad = int(min(w, h) * 0.03)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(ww - x, w + 2 * pad)
    h = min(hh - y, h + 2 * pad)
    return x, y, w, h


def main():
    img = cv2.imread(IMG)
    h, w = img.shape[:2]
    print(f"整图 {w}x{h}")

    board = robust_find(img)
    if board is None:
        print("找不到有效棋盘区域")
        return
    x0, y0, bw, bh = board
    board_img = img[y0:y0 + bh, x0:x0 + bw]
    print(f"棋盘区域: ({x0},{y0}) {bw}x{bh}")

    # 投影找 10×9 网格
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)

    # 边缘密度投影（更能突出棋盘线）
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_row = np.abs(sy).mean(axis=1)  # 横边缘 → 垂直线位置
    edge_col = np.abs(sx).mean(axis=0)  # 竖边缘 → 水平线位置

    # 保存投影图
    cv2.imwrite(os.path.join(OUT, "08_row_proj.png"),
                np.uint8(255 - row_proj)[:, None].repeat(20, axis=1))
    cv2.imwrite(os.path.join(OUT, "09_col_proj.png"),
                np.uint8(255 - col_proj)[None, :].repeat(20, axis=0))

    h_cands = local_minima(row_proj, min_gap=int(bh * 0.04))
    v_cands = local_minima(col_proj, min_gap=int(bw * 0.04))
    print(f"行投影局部最小 {len(h_cands)}: {h_cands}")
    print(f"列投影局部最小 {len(v_cands)}: {v_cands}")

    h_need = 11
    v_need = 10
    horz = pick_uniform(h_cands, h_need)
    vert = pick_uniform(v_cands, v_need)
    print(f"horz {horz}  std={np.std(np.diff(horz) if horz else [])}")
    print(f"vert {vert}  std={np.std(np.diff(vert) if vert else [])}")

    # 画出来
    vis = board_img.copy()
    if horz:
        for y in horz:
            cv2.line(vis, (0, y), (bw, y), (0, 255, 0), 2)
    if vert:
        for x in vert:
            cv2.line(vis, (x, 0), (x, bh), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(OUT, "10_grid_guess.png"), vis)


if __name__ == "__main__":
    main()
