"""检测棋盘中部「墨云 / 吃 / 当头炮 / 将军」等特效遮挡。

供自动监听跳过脏帧；命令行扫描见 ``test/detect_fx_overlay.py``。

级别：
    strong  不透明黑色笔刷环
    soft    半透明墨云 + 金色大字
    faded   淡化阶段（中环变暗）
    clean   无明显特效
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def _center_roi(img):
    """棋盘河界一带：高约 40%–58%，宽约 25%–75%。"""
    h, w = img.shape[:2]
    return img[int(h * 0.40) : int(h * 0.58), int(w * 0.25) : int(w * 0.75)]


def detect_board_fx(img) -> dict:
    """检测中部特效遮挡。

    Parameters
    ----------
    img : ndarray
        BGR 整图（手机截图 / 自动监听客户区均可）。

    Returns
    -------
    dict
        dirty, level ('strong'|'soft'|'faded'|'clean'), score (0~1), metrics
    """
    if img is None or getattr(img, "size", 0) == 0:
        raise ValueError("空图像")

    roi = _center_roi(img)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    b, g, r = cv2.split(roi)

    # 近黑墨迹（不透明笔刷）
    ink = ((b < 40) & (g < 40) & (r < 40)).astype(np.uint8)
    ink_pct = float(ink.mean() * 100.0)
    n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    ink_cc = max((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_lbl)), default=0)
    ink_cc_pct = 100.0 * ink_cc / float(roi.shape[0] * roi.shape[1])

    v10 = float(np.percentile(v, 10))
    vstd = float(v.std())

    # 中环亮度：淡化墨云会把木纹中环拉暗并抬高方差
    rh, rw = roi.shape[:2]
    yy, xx = np.mgrid[:rh, :rw]
    rr = np.sqrt((yy - rh / 2.0) ** 2 + (xx - rw / 2.0) ** 2)
    rmax = min(rh, rw) / 2.0
    mid = (rr >= 0.20 * rmax) & (rr < 0.55 * rmax)
    ring_mean = float(v[mid].mean()) if mid.any() else float(v.mean())
    ring_std = float(v[mid].std()) if mid.any() else float(v.std())

    metrics = {
        "ink_pct": round(ink_pct, 3),
        "ink_cc_pct": round(ink_cc_pct, 3),
        "v10": round(v10, 2),
        "v_std": round(vstd, 2),
        "ring_v_mean": round(ring_mean, 2),
        "ring_v_std": round(ring_std, 2),
        "roi_wh": [int(rw), int(rh)],
    }

    if ink_cc_pct >= 3.0 and ink_pct >= 5.0:
        level = "strong"
        score = float(min(1.0, ink_cc_pct / 8.0))
    elif ink_pct >= 2.0 and ink_cc_pct >= 1.2 and v10 <= 150 and vstd >= 48:
        level = "soft"
        score = 0.55
    elif ring_mean <= 195 and ring_std >= 42 and v10 <= 155:
        level = "faded"
        score = 0.45
    else:
        level = "clean"
        score = 0.0

    return {
        "dirty": level != "clean",
        "level": level,
        "score": round(score, 3),
        "metrics": metrics,
    }


def detect_board_fx_path(path: str) -> dict:
    """读盘后检测；附加 path / name 字段。"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取: {path}")
    out = detect_board_fx(img)
    out["path"] = path
    out["name"] = os.path.basename(path)
    return out
