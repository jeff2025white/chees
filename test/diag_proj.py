"""诊断：在 ROI (0,101,720,1264) 上打印所有 candidate 与选中段。"""

import cv2
import numpy as np

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
ROI = (0, 101, 720, 1264)


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


def main():
    img = cv2.imread(IMG)
    x0, y0, x1, y1 = ROI
    crop = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)

    h_cands = local_minima(row_proj, min_gap=int(h * 0.04))
    v_cands = local_minima(col_proj, min_gap=int(w * 0.04))
    print(f"row_proj 候选 {len(h_cands)}: {h_cands}")
    print(f"col_proj 候选 {len(v_cands)}: {v_cands}")

    # 打印每个候选窗口的 std
    print("\n竖线候选窗口 std：")
    for i in range(len(v_cands) - 10 + 1):
        seq = v_cands[i:i + 10]
        gaps = np.diff(seq)
        print(f"  i={i} std={np.std(gaps):.1f} gaps={gaps}")

    print("\n横线候选窗口 std：")
    for i in range(len(h_cands) - 11 + 1):
        seq = h_cands[i:i + 11]
        gaps = np.diff(seq)
        print(f"  i={i} std={np.std(gaps):.1f} gaps={gaps}")


if __name__ == "__main__":
    main()
