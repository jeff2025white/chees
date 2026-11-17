"""验证多张截图是否共享固定 ROI：对每张图用同一组候选裁剪做投影比对。"""

import os
import cv2
import numpy as np

IMG_DIR = r"g:\chessCn\images"


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
        return None, 1e9
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
    return best_seq, best_err


ROIS = [
    # 上一轮成功案例 (0, 101, 720, 1264) 加上几种变体
    (0, 100, 720, 1260),
    (0, 101, 720, 1264),
    (0, 80, 720, 1200),
    (100, 150, 600, 1150),
    (131, 188, 590, 926),  # 上一轮直接用投影在候选里找到的精确边界
]


def score(img_path, roi):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = roi
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape
    row_proj = crop.mean(axis=1)
    col_proj = crop.mean(axis=0)
    h_cands = local_minima(row_proj, min_gap=int(h * 0.04))
    v_cands = local_minima(col_proj, min_gap=int(w * 0.04))
    horz, eh = pick_uniform(h_cands, 11)
    vert, ev = pick_uniform(v_cands, 10)
    if horz is None or vert is None:
        return None
    cell_h = (horz[-1] - horz[0]) / 10
    cell_w = (vert[-1] - vert[0]) / 9
    return eh + ev + abs(cell_h - cell_w) * 0.2, horz, vert, cell_w, cell_h


def main():
    files = sorted(os.listdir(IMG_DIR))
    print(f"候选 ROI: {ROIS}\n")
    for fn in files:
        if not fn.lower().endswith(".png"):
            continue
        path = os.path.join(IMG_DIR, fn)
        best = None
        for roi in ROIS:
            r = score(path, roi)
            if r is None:
                continue
            s, h, v, cw, ch = r
            if best is None or s < best[0]:
                best = (s, roi, h, v, cw, ch)
        if best is None:
            print(f"{fn}: 所有 ROI 都失败")
        else:
            s, roi, h, v, cw, ch = best
            print(f"{fn}: 最佳 ROI={roi} score={s:.1f} cell={cw:.0f}x{ch:.0f}")


if __name__ == "__main__":
    main()
