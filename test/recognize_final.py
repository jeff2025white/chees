"""识别 + 走棋 + 画圈 —— 完整流水线，一张图一张图地跑。"""

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

IMG = r"g:\chessCn\images\Screenshot_20261117-103346.png"
OUT_DIR = r"g:\chessCn\images\annotated"
WORK = r"g:\chessCn\2026-11-17\xiangqi-move-logic\work"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


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
    """从候选列表里挑出等间距的一段。"""
    if len(lines) < need:
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


def find_board_region(img):
    """在整图上找到棋盘所在的子矩形。

    策略：用红色+棕色的掩膜得到候选区域，取最大连通域。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 红色（棋子/棋盘线）
    red = (
        cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
        | cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    )
    # 浅棕/木色（棋盘背景）
    wood = cv2.inRange(hsv, (10, 20, 50), (30, 200, 240))
    # 蓝色 UI 背景也会有大面积亮蓝色区域（我们要排除它）
    blue = cv2.inRange(hsv, (90, 40, 120), (130, 255, 255))

    # 棋盘候选：非蓝 + (红 or 棕)
    non_blue = cv2.bitwise_not(blue)
    mask = cv2.bitwise_or(red, wood)
    mask = cv2.bitwise_and(mask, non_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 去掉太细的噪点
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

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


def detect_grid_in(board_region):
    """在裁出来的棋盘区域内用投影法找 10 横线 + 9 竖线。"""
    gray = cv2.cvtColor(board_region, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_proj = gray.mean(axis=1)
    col_proj = gray.mean(axis=0)

    h_cands = local_minima(row_proj, min_gap=int(min(h, w) * 0.05))
    v_cands = local_minima(col_proj, min_gap=int(min(h, w) * 0.05))

    # 找等间距段
    h_need = 11  # 10 格
    v_need = 10  # 9 格
    horz = pick_uniform(h_cands, h_need)
    vert = pick_uniform(v_cands, v_need)

    print(f"  横线候选 {len(h_cands)} → 选 {horz}  (间距 std={np.std(np.diff(horz) if horz else []):.1f})")
    print(f"  竖线候选 {len(v_cands)} → 选 {vert}  (间距 std={np.std(np.diff(vert) if vert else []):.1f})")
    return horz, vert


def classify_cell(img, cx, cy, cell_size):
    """返回 'R'（红方棋子）、'r'（黑方棋子）、'.'（空）。"""
    r = max(int(cell_size * 0.32), 8)
    patch = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    if patch.size == 0:
        return "."

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 红掩膜（H 0-10 或 160-180，且足够饱和）
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 60) & (hsv[..., 2] > 60)
    red_ratio = float(red_mask.mean())

    # 黑掩膜（V 极低）
    black_mask = hsv[..., 2] < 70
    black_ratio = float(black_mask.mean())

    if red_ratio > 0.10 and red_ratio >= black_ratio * 1.2:
        return "R", red_ratio
    if black_ratio > 0.14 and black_ratio >= red_ratio:
        return "r", black_ratio
    return ".", max(red_ratio, black_ratio)


def find_row_line(img, hsv, cx, cy, dx, dy, sign):
    """从格子中心沿 dir 方向找最近的横线/竖线（颜色突变点）。"""
    return None  # 暂不需要


def process_one(img_path):
    img = cv2.imread(img_path)
    print(f"\n{'='*60}")
    print(f"处理: {os.path.basename(img_path)}  {img.shape[1]}x{img.shape[0]}")

    region = find_board_region(img)
    if region is None:
        print("❌ 找不到棋盘区域，跳过")
        return None
    x0, y0, w0, h0 = region
    board = img[y0:y0 + h0, x0:x0 + w0]
    print(f"棋盘区域: ({x0},{y0}) {w0}x{h0}")

    horz, vert = detect_grid_in(board)
    if horz is None or vert is None:
        print("❌ 无法找到稳定的 9×10 网格，跳过")
        return None

    cell_w = (vert[-1] - vert[0]) / 9
    cell_h = (horz[-1] - horz[0]) / 10
    print(f"格子大小: {cell_w:.1f} x {cell_h:.1f} px")

    # 对每个格子：cx,cy 是整图坐标
    grid_chars = []
    centers = []  # [(cx, cy), ...]
    for row in range(10):
        row_chars = []
        for col in range(9):
            cx = x0 + int(vert[col] + cell_w / 2)
            cy = y0 + int(horz[row] + cell_h / 2)
            ch, conf = classify_cell(img, cx, cy, min(cell_w, cell_h))
            row_chars.append(ch)
            centers.append((cx, cy))
        grid_chars.append("".join(row_chars))

    grid_str = "\n".join(grid_chars)
    red_count = sum(row.count("R") for row in grid_chars)
    black_count = sum(row.count("r") for row in grid_chars)
    print(f"识别到红棋 {red_count} 颗，黑棋 {black_count} 颗")
    print("网格（R=红方大写，r=黑方小写）:")
    print(grid_str)

    if red_count < 3 and black_count < 3:
        print("❌ 棋子太少，可能识别失败，跳过")
        # 仍然存一份供调试
        vis = img.copy()
        for (cx, cy) in centers:
            cv2.circle(vis, (cx, cy), int(min(cell_w, cell_h) * 0.15), (255, 255, 0), -1)
        debug_path = os.path.join(OUT_DIR, "debug_" + os.path.basename(img_path))
        cv2.imwrite(debug_path, vis)
        return None

    # 调用引擎
    # 约定：识别出的大写 R = 红方棋子，小写 r = 黑方棋子。
    # 内部 FEN 约定：大写 = 红方，小写 = 黑方；行序 row 0 = 黑方底线 = 图像最上方
    # 图像网格 row 0 = 图最上方 → 正好对应 FEN row 0！
    # 图像网格 col 0 = 图最左方 → 红方视角 col 0 就是左方！
    # 但我们还需要决定行棋方：先假设红方先（需要 UI 状态识别或让用户指定）
    # 这里先默认红方走，如果想自动判行棋方，后续可加
    try:
        result = decide_position(grid=grid_str, side="red", view_side="red",
                                 time_limit=3.0, max_depth=6, top_n=2)
    except Exception as exc:
        print(f"❌ 引擎失败: {exc}")
        return None

    # 画圈
    vis = img.copy()
    best = result.get("best_move")
    radius = int(min(cell_w, cell_h) * 0.45)

    # 先画出所有 90 个格子中心（淡色），便于肉眼校验识别
    for (cx, cy) in centers:
        cv2.circle(vis, (cx, cy), int(min(cell_w, cell_h) * 0.18), (120, 120, 120), -1)

    # 画走子圈：起点（我方棋子描边）红色、目标格（实心红圈）略亮
    if best is not None:
        fc, fr = best["from"]["col"], best["from"]["row"]
        tc, tr = best["to"]["col"], best["to"]["row"]
        # 界面坐标 → 整图坐标：加上 x0,y0 + vert[fc]+horz[fr] 偏移
        # 这里 cell_w/h 是在 board 坐标系下的
        from_cx = x0 + int(vert[fc] + cell_w / 2)
        from_cy = y0 + int(horz[fr] + cell_h / 2)
        to_cx = x0 + int(vert[tc] + cell_w / 2)
        to_cy = y0 + int(horz[tr] + cell_h / 2)

        # 起点：红描边 + 白色内圈
        cv2.circle(vis, (from_cx, from_cy), radius + 6, (0, 0, 255), 6)
        cv2.circle(vis, (from_cx, from_cy), radius + 1, (255, 255, 255), 2)
        # 目标：红实心圆 + 内圈
        cv2.circle(vis, (to_cx, to_cy), radius + 6, (0, 0, 255), 6)
        cv2.circle(vis, (to_cx, to_cy), radius - 2, (0, 0, 255), -1)
        # 连线
        cv2.line(vis, (from_cx, from_cy), (to_cx, to_cy), (0, 0, 255), 3)

    # 在画面顶部标注着法信息
    best_line = "无着法"
    if best is not None:
        best_line = f"{best['chinese']} ({best['iccs']})  {best['score_text']}"
    cv2.putText(vis, best_line, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, best['reason'] if best else result.get('message', ''), (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 200), 2, cv2.LINE_AA)

    out_path = os.path.join(OUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, vis)

    # 同时存一份 JSON 结果
    base = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(WORK, f"result_{base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存到 {out_path}")
    return result


if __name__ == "__main__":
    single = IMG if len(sys.argv) < 2 else sys.argv[1]
    process_one(single)
