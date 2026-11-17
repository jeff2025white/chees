"""诊断图片尺寸与棋盘区域：把几张截图的关键区域缩放到 800px 以内后存为 PNG，
方便我们从文件尺寸/灰度分布推断棋盘布局（Read 工具无法直接加载时的备份手段）。
"""

import os
import sys

import cv2
import numpy as np

IMG_DIR = r"g:\chessCn\images"
OUT_DIR = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work\inspect"
os.makedirs(OUT_DIR, exist_ok=True)


def summarize(img: np.ndarray, name: str):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean = img.mean(axis=(0, 1))  # BGR 均值
    # 识别主要颜色区间（红/黑/木色背景）
    red_mask = (hsv[..., 0] > 160) | (hsv[..., 0] < 10)
    black_mask = hsv[..., 2] < 60
    total = h * w
    print(f"{name}: {w}x{h}, BGR=({mean[0]:.0f},{mean[1]:.0f},{mean[2]:.0f}), "
          f"red={red_mask.sum()/total:.3f}, black={black_mask.sum()/total:.3f}")


def save_preview(img: np.ndarray, path: str, max_side: int = 1200):
    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    out = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imwrite(path, out)


def try_find_board(img: np.ndarray, name: str):
    """用简单策略找棋盘：找大面积棕色/木色 + 红色棋盘线。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 木色背景：H 10-30, S 20-180, V 60-220
    wood = cv2.inRange(hsv, (10, 20, 60), (30, 200, 230))
    # 红色（棋子/棋盘线）
    red = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)) | cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    mask = cv2.bitwise_or(wood, red)
    # 形态学闭运算把细碎填充
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        print(f"  {name}: 没找到可疑棋盘区域")
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = np.argmax(areas) + 1
    x = stats[idx, cv2.CC_STAT_LEFT]
    y = stats[idx, cv2.CC_STAT_TOP]
    w = stats[idx, cv2.CC_STAT_WIDTH]
    h = stats[idx, cv2.CC_STAT_HEIGHT]
    ratio = w / h
    print(f"  {name}: 棋盘候选 x={x} y={y} w={w} h={h} 比例={ratio:.2f}")
    crop = img[y:y + h, x:x + w]
    return crop, (x, y, w, h)


def main():
    files = sorted(os.listdir(IMG_DIR))
    for fn in files:
        if not fn.lower().endswith(".png"):
            continue
        path = os.path.join(IMG_DIR, fn)
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取: {path}")
            continue
        summarize(img, fn)
        # 保存缩小预览（整图 + 中心 60%）
        save_preview(img, os.path.join(OUT_DIR, fn))
        h, w = img.shape[:2]
        cx1, cy1 = int(w * 0.1), int(h * 0.1)
        cx2, cy2 = int(w * 0.9), int(h * 0.9)
        save_preview(img[cy1:cy2, cx1:cx2], os.path.join(OUT_DIR, "c_" + fn))
        # 尝试自动定位棋盘
        res = try_find_board(img, fn)
        if res is not None:
            crop, (x, y, w, h) = res
            save_preview(crop, os.path.join(OUT_DIR, "board_" + fn))
            # 画矩形到原图
            vis = img.copy()
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 5)
            save_preview(vis, os.path.join(OUT_DIR, "boxed_" + fn))


if __name__ == "__main__":
    main()
