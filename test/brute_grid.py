"""穷举多个候选裁剪区域，找到投影最稳定的那个（10×9 网格间距方差最小）。"""

import os

import cv2
import numpy as np

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


def score_crop(gray_crop):
    """给一个候选裁剪区域打分：返回 (std_h, std_v, h_count, v_count)。"""
    h, w = gray_crop.shape
    if h < 300 or w < 300:
        return None
    row_proj = gray_crop.mean(axis=1)
    col_proj = gray_crop.mean(axis=0)
    h_cands = local_minima(row_proj, min_gap=int(h * 0.04))
    v_cands = local_minima(col_proj, min_gap=int(w * 0.04))
    horz, eh = pick_uniform(h_cands, 11)
    vert, ev = pick_uniform(v_cands, 10)
    if horz is None or vert is None:
        return None
    # 综合分数：越低越好
    combined = eh + ev + abs((vert[-1] - vert[0]) / 9 - (horz[-1] - horz[0]) / 10) * 0.1
    return combined, (horz, vert)


def main():
    img = cv2.imread(IMG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # 穷举不同的裁剪区域
    best = None
    best_score = 1e9
    for y0 in range(0, 200, 20):
        for y1 in range(H - 200, H, 20):
            for x0 in range(0, 80, 20):
                for x1 in range(W - 80, W, 20):
                    if y1 - y0 < 500:
                        continue
                    if x1 - x0 < 400:
                        continue
                    crop = gray[y0:y1, x0:x1]
                    res = score_crop(crop)
                    if res is None:
                        continue
                    s, _ = res
                    if s < best_score:
                        best_score = s
                        best = (x0, y0, x1, y1, res)

    if best is None:
        print("没找到好区域")
        return
    x0, y0, x1, y1, (score, (horz, vert)) = best
    print(f"最佳区域: x={x0}:{x1} y={y0}:{y1}  score={score:.1f}")
    print(f"  横线: {[y0 + h for h in horz]}")
    print(f"  竖线: {[x0 + v for v in vert]}")

    # 画出来保存
    vis = img.copy()
    for h in horz:
        cv2.line(vis, (x0, y0 + h), (x1, y0 + h), (0, 255, 0), 2)
    for v in vert:
        cv2.line(vis, (x0 + v, y0), (x0 + v, y1), (255, 0, 0), 2)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 3)
    cv2.imwrite(os.path.join(OUT, "11_best_region.png"), vis)


if __name__ == "__main__":
    main()
