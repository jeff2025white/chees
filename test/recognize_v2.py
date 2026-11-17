"""完整流水线：手动 ROI（已知正确） + 投影网格 + 颜色分类 + 引擎 + 画圈。
先跑通核心逻辑，再回头做自动 ROI。"""

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import decide_position


# ------------------------------------------------------------
# 网格找法
# ------------------------------------------------------------

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
    """在候选里挑出等间距的 need 条。"""
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


def pick_best_window(lines, need):
    if lines is None or len(lines) < need:
        return None, 1e9
    best_seq = None
    best_score = 1e9
    for i in range(len(lines) - need + 1):
        seq = lines[i:i + need]
        gaps = np.diff(seq)
        med = np.median(gaps)
        std = np.std(gaps)
        if med < 20 or med > 200:
            continue
        score = std / med
        if score < best_score:
            best_score = score
            best_seq = seq
    return best_seq, best_score


def pick_horizontal(gray_crop, need=11):
    """返回 (seq, cell_h_median) 或 None。"""
    h, w = gray_crop.shape
    row_proj = gray_crop.mean(axis=1)
    cands = local_minima(row_proj, min_gap=int(h * 0.04))
    best_seq, _ = pick_best_window(cands, need)
    if best_seq is None:
        return None
    return best_seq, float(np.median(np.diff(best_seq)))


def pick_vertical(gray_crop, target_cell_w, need=10):
    """给定 cell_w 先验，贪心挑出 need 条竖线。"""
    h, w = gray_crop.shape
    col_proj = gray_crop.mean(axis=0)
    cands = local_minima(col_proj, min_gap=max(int(target_cell_w * 0.35), 15))

    def greedy(start):
        chosen = [start]
        last = start
        while len(chosen) < need:
            cands_after = [c for c in cands if c > last + target_cell_w * 0.45]
            if not cands_after:
                return None
            cands_after.sort(key=lambda c: abs(c - (last + target_cell_w)))
            if cands_after[0] - last > target_cell_w * 2.5:
                return None
            last = cands_after[0]
            chosen.append(last)
        return chosen

    best = None
    for s in cands[:-need + 1]:
        seq = greedy(s)
        if seq is None:
            continue
        gaps = np.diff(seq)
        med = np.median(gaps)
        std = np.std(gaps)
        score = std / med + abs(med - target_cell_w) / target_cell_w
        if best is None or score < best[0]:
            best = (score, seq, float(med))
    return (best[1], best[2]) if best else None


def fit_lines_projection(gray_crop):
    h_crop = pick_horizontal(gray_crop, need=11)
    if h_crop is None:
        return None, None
    horz, cell_h = h_crop
    v_crop = pick_vertical(gray_crop, cell_h, need=10)
    if v_crop is None:
        return horz, None
    vert, cell_w = v_crop
    print(f"  网格: cell {cell_w:.0f}x{cell_h:.0f} px")
    return horz, vert


def dominant_period(proj):
    """自相关法找主周期（避免 FFT 的窗口泄漏问题）。"""
    n = len(proj)
    if n < 100:
        return 0, 0
    # 去均值
    p = proj - proj.mean()
    # 自相关
    ac = np.correlate(p, p, mode="full")
    ac = ac[n - 1:]
    # 从 lag=20 开始找次高峰（lag=0 是峰值自身）
    search = ac[20:n // 2]
    if len(search) == 0:
        return 0, 0
    peak_idx = np.argmax(search) + 20
    return peak_idx, ac[peak_idx] / ac[0]


# ------------------------------------------------------------
# 颜色分类
# ------------------------------------------------------------

def classify_cell(img, cx, cy, cell_size):
    """返回 ('R'红方 / 'r'黑方 / '.'空, 置信度)。"""
    r = max(int(cell_size * 0.32), 8)
    patch = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if patch.size == 0:
        return ".", 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 红色掩膜：H 在 0-10 或 160-180 且足够饱和
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 50) & (hsv[..., 2] > 60)
    red_ratio = float(red_mask.mean())

    # 黑掩膜：V 极低
    black_mask = hsv[..., 2] < 75
    black_ratio = float(black_mask.mean())

    # 红棋子通常中心还有白色"帅/将/马"等字样，可增加白色中心检测
    white_center = False
    if red_ratio > 0.05 or black_ratio > 0.05:
        inner_r = max(r // 2, 4)
        inner = patch[r - inner_r:r + inner_r, r - inner_r:r + inner_r]
        if inner.size > 0:
            hsv2 = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
            white_center = float((hsv2[..., 2] > 180).mean()) > 0.15

    if red_ratio > 0.08 and red_ratio >= black_ratio * 1.2:
        return "R", red_ratio
    if black_ratio > 0.13 and black_ratio >= red_ratio:
        return "r", black_ratio
    return ".", max(red_ratio, black_ratio)


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------

def auto_find_roi(img):
    """自动找 ROI：先看哪一行段的投影有明显的周期性自相关。"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 遍历不同的 ROI，找到最稳定的 10×9
    best = None
    for y0 in range(60, 200, 10):
        for y1 in range(h - 60, h - 100, -20):
            if y1 - y0 < 700:
                continue
            for x0 in range(0, 100, 20):
                for x1 in range(w - 20, w - 80, -20):
                    if x1 - x0 < 500:
                        continue
                    crop = gray[y0:y1, x0:x1]
                    row_proj = crop.mean(axis=1)
                    col_proj = crop.mean(axis=0)
                    ph, _ = dominant_period(row_proj)
                    pw, _ = dominant_period(col_proj)
                    if ph < 40 or pw < 40:
                        continue
                    h_cands = local_minima(row_proj, min_gap=max(int(ph * 0.5), 20))
                    v_cands = local_minima(col_proj, min_gap=max(int(pw * 0.5), 20))
                    horz = pick_uniform(h_cands, 11)
                    vert = pick_uniform(v_cands, 10)
                    if horz is None or vert is None:
                        continue
                    eh = np.std(np.diff(horz))
                    ev = np.std(np.diff(vert))
                    score = eh + ev
                    if best is None or score < best[0]:
                        best = (score, (x0, y0, x1, y1), horz, vert)
    if best is None:
        return None
    _, roi, horz, vert = best
    return roi, horz, vert


def process(img_path, out_dir, work_dir, roi=None):
    img = cv2.imread(img_path)
    print(f"\n{'=' * 60}\n处理: {os.path.basename(img_path)}  {img.shape[1]}x{img.shape[0]}")

    if roi is None:
        res = auto_find_roi(img)
        if res is None:
            print("❌ 自动找不到稳定的 10×9 网格")
            return None
        roi, horz, vert = res
    else:
        x0, y0, x1, y1 = roi
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        horz, vert = fit_lines_projection(gray)
        if horz is None or vert is None:
            print("❌ ROI 内投影仍找不到 10×9")
            return None
    (x0, y0, x1, y1) = roi
    bw, bh = x1 - x0, y1 - y0
    cell_w = (vert[-1] - vert[0]) / 9
    cell_h = (horz[-1] - horz[0]) / 10
    print(f"✅ ROI: x0={x0} y0={y0}  size {bw}x{bh}  cell {cell_w:.1f}x{cell_h:.1f}")

    # 棋子识别：horz/vert 是棋盘**线条**位置（暗条纹），
    # 格子中心是相邻两条线的中点：row_center[i] = (horz[i] + horz[i+1]) / 2
    row_centers = [(horz[i] + horz[i + 1]) / 2 for i in range(10)]
    col_centers = [(vert[i] + vert[i + 1]) / 2 for i in range(9)]
    cell_w = np.median(np.diff(vert))
    cell_h = np.median(np.diff(horz))

    grid_chars = []
    centers = []
    for row in range(10):
        row_chars = []
        for col in range(9):
            cx = int(x0 + col_centers[col])
            cy = int(y0 + row_centers[row])
            ch, conf = classify_cell(img, cx, cy, min(cell_w, cell_h))
            row_chars.append(ch)
            centers.append((cx, cy))
        grid_chars.append("".join(row_chars))

    grid_str = "\n".join(grid_chars)
    red_count = sum(row.count("R") for row in grid_chars)
    black_count = sum(row.count("r") for row in grid_chars)
    print(f"识别: 红棋 {red_count} / 黑棋 {black_count}")
    print(grid_str)

    if red_count < 3 or black_count < 3:
        print("❌ 识别太稀疏，放弃")
        # 仍存调试图
        vis = img.copy()
        for (cx, cy) in centers:
            cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(work_dir, "debug_" + os.path.basename(img_path)), vis)
        return None

    # 交给引擎
    # grid_str 约定：row 0 = 图最上方 = FEN row 0（黑方底线）；R 大写 = 红方
    try:
        result = decide_position(grid=grid_str, side="red", view_side="red",
                                 time_limit=3.0, max_depth=6, top_n=2)
    except Exception as exc:
        print(f"❌ 引擎失败: {exc}")
        return None

    # 画圈：界面格 (col,row) → 整图像素
    vis = img.copy()
    r = int(min(cell_w, cell_h) * 0.45)

    # 先画 90 个格子中心供肉眼核对
    for cx, cy in centers:
        cv2.circle(vis, (cx, cy), max(int(r * 0.15), 3), (160, 160, 160), -1)

    best = result.get("best_move")
    if best is not None:
        fc, fr = best["from"]["col"], best["from"]["row"]
        tc, tr = best["to"]["col"], best["to"]["row"]
        from_cx = int(x0 + vert[fc] + cell_w / 2)
        from_cy = int(y0 + horz[fr] + cell_h / 2)
        to_cx = int(x0 + vert[tc] + cell_w / 2)
        to_cy = int(y0 + horz[tr] + cell_h / 2)

        # 起点：红描边 + 白内圈
        cv2.circle(vis, (from_cx, from_cy), r + 6, (0, 0, 255), 6)
        cv2.circle(vis, (from_cx, from_cy), r + 1, (255, 255, 255), 2)
        # 目标：红实心圆 + 内圈
        cv2.circle(vis, (to_cx, to_cy), r + 6, (0, 0, 255), 6)
        cv2.circle(vis, (to_cx, to_cy), r - 2, (0, 0, 255), -1)
        cv2.line(vis, (from_cx, from_cy), (to_cx, to_cy), (0, 0, 255), 3)

    # 标注顶部
    best_line = "无着法"
    reason = result.get("message", "")
    if best is not None:
        best_line = f"{best['chinese']} ({best['iccs']})  {best['score_text']}"
        reason = best.get("reason", "")
    cv2.putText(vis, best_line, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, reason[:50], (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)

    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_dir, base + ".png"), vis)
    with open(os.path.join(work_dir, f"result_{base}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 输出: {out_dir}\\{base}.png")
    return result


def main():
    out_dir = r"g:\chessCn\images\annotated"
    work_dir = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    img_dir = r"g:\chessCn\images"

    # 先用手工 ROI 跑通核心流程，之后再回头优化自动检测
    HAND_ROIS = {
        "Screenshot_20261117-103346.png": (0, 101, 720, 1264),
        "Screenshot_20261117-103438.png": (0, 101, 720, 1264),
        "Screenshot_20261117-103459.png": (0, 101, 720, 1264),
        "Screenshot_20261117-103508.png": (0, 101, 720, 1264),
        "Screenshot_20261117-103516.png": (0, 101, 720, 1264),
    }

    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            for fn in sorted(os.listdir(sys.argv[1])):
                if fn.lower().endswith(".png"):
                    roi = HAND_ROIS.get(fn)
                    process(os.path.join(sys.argv[1], fn), out_dir, work_dir, roi)
        else:
            roi = HAND_ROIS.get(os.path.basename(sys.argv[1]))
            process(sys.argv[1], out_dir, work_dir, roi)
    else:
        first = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])[0]
        process(os.path.join(img_dir, first), out_dir, work_dir, HAND_ROIS.get(first))


if __name__ == "__main__":
    main()
