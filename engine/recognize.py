"""从手机截图识别中国象棋局面，并可把决策结果标注回图片。

对外接口：
    from engine.recognize import analyze_image, annotate, render_annotated

    a = analyze_image("images/xxx.png")      # 识别，自动判断我方颜色
    decision = decide_position(grid=a.grid_str, side=a.side, view_side=a.side)
    out = render_annotated(a, decision)      # 返回 BGR ndarray

约定：
    图像 row 0 = 图最上方，row 9 = 图最下方。
    我方颜色由 detect_our_side 在下半部分九宫（row 7-9, col 3-5）
    找主将确定，谁在下半边九宫里谁就是我方。
    九宫找不到时兜底扫描下半场（row 5-9）。
"""

import os

import cv2
import numpy as np

from .constants import FEN_TO_PIECE, PIECE_NAMES

# 基准分辨率（本批截图均为 720×1280）的比例参数
_REF_W = 720.0
_REF_H = 1280.0
_CELL_RATIO = 76.0 / _REF_W          # 格宽 / 图宽
_ROI_Y0 = 101.0 / _REF_H
_ROI_Y1 = 1264.0 / _REF_H
_TOP_UI = 0.075                       # 顶部 UI 屏蔽比例（头像/比分区）
_BOTTOM_UI = 0.92                     # 底部 UI 屏蔽比例（按钮区）

RED_BACKROW = ["R", "N", "B", "A", "K", "A", "B", "N", "R"]
BLACK_BACKROW = ["r", "n", "b", "a", "k", "a", "b", "n", "r"]

ROWS, COLS = 10, 9
RED = set("RNBAKCP")


# --------------------------------------------------------------------------
# 棋子检测
# --------------------------------------------------------------------------

def _piece_masks(hsv):
    """红 / 暗黑 / 亮棕黑 三路 HSV mask。

    亮棕（orange）覆盖部分屏幕上「黑子底色偏亮棕」的变体：V 很高，
    不进 black_mask（V<100），也不进 red_mask（H 在 10–20）。
    """
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (
        (hsv[..., 2] > 40) & (hsv[..., 2] < 100)
        & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
    )
    orange_mask = (
        (hsv[..., 0] >= 10) & (hsv[..., 0] <= 20)
        & (hsv[..., 1] >= 80) & (hsv[..., 2] >= 120) & (hsv[..., 2] <= 240)
    )
    orange_mask = orange_mask & ~red_mask
    return red_mask, black_mask, orange_mask


def _cell_color(rr, br, oran=0.0):
    """由窗口内红/暗黑 mask 均值判 '@' / '#' / '.'。

    oran 保留参数以兼容调用；亮棕补漏不在颜色阈值里做（与木纹不可分），
    而在 classify 之后用底线马位间隙修复。
    """
    if rr > 0.04 and rr > br * 1.2:
        return "@"
    if br > 0.05 and br > rr * 1.2:
        return "#"
    if rr > br * 0.8 and rr > 0.02:
        return "@"
    if br > rr * 0.8 and br > 0.02:
        return "#"
    return "."


def _horse_already_developed(grid, r, c, marker):
    """马位 (r,c) 是否已有同色子落在马的日字跳点上（说明马已跳出，勿补漏）。"""
    for dr, dc in ((-2, -1), (-2, 1), (2, -1), (2, 1),
                   (-1, -2), (-1, 2), (1, -2), (1, 2)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < ROWS and 0 <= cc < COLS and grid[rr][cc] == marker:
            return True
    return False


def _repair_backrow_horse_gaps(grid):
    """补回底线漏检的马位（col 1 / 7）。

    亮棕黑子常在 HSV 上与棋盘木纹不可分，连通域与格心 mask 都会漏。
    若某一行已有 ≥7 枚同色子且马位为空、左右邻为该色，则补上。
    只动 row 0-2 / 7-9，避免中局误补。

    若该马位的日字跳点上已有同色子，视为马已跳出，禁止补假马——
    否则会占掉 n/N 配额，把真马 OCR 成车（见 20261118 跳马局）。
    """
    for r in (0, 1, 2, 7, 8, 9):
        row = grid[r]
        occupied = [c for c in range(COLS) if row[c] != "."]
        if len(occupied) < 7:
            continue
        n_at = sum(1 for c in occupied if row[c] == "@")
        n_hash = sum(1 for c in occupied if row[c] == "#")
        if n_at >= 7 and n_at > n_hash:
            maj = "@"
        elif n_hash >= 7 and n_hash > n_at:
            maj = "#"
        else:
            continue
        for c in (1, 7):
            if row[c] != ".":
                continue
            if _horse_already_developed(grid, r, c, maj):
                continue
            left = row[c - 1] if c - 1 >= 0 else "."
            right = row[c + 1] if c + 1 < COLS else "."
            if left == maj or right == maj:
                row[c] = maj
    return grid


def detect_pieces(crop):
    """连通域 + 面积 + 圆形度过滤，返回 [(cx, cy, 'R'|'B', area)]。

    只用红 + 暗黑 mask 做连通域（亮棕与棋盘木纹接近，并入会污染网格）。
    亮棕黑子漏检由 classify() 格心二次判定补回。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask, black_mask, _orange = _piece_masks(hsv)
    piece_mask = (red_mask | black_mask).astype(np.uint8)

    num, _labels, stats, cents = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    scale = crop.shape[1] / _REF_W
    area_lo, area_hi = 80 * scale * scale, 800 * scale * scale
    min_side = max(8, int(15 * scale))

    pieces = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (area_lo <= area <= area_hi):
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w < min_side or h < min_side or min(w, h) / max(w, h) < 0.55:
            continue
        cx, cy = cents[i]
        r = int(15 * scale)
        y0, y1 = max(0, int(cy) - r), min(crop.shape[0], int(cy) + r)
        x0, x1 = max(0, int(cx) - r), min(crop.shape[1], int(cx) + r)
        rp = float(red_mask[y0:y1, x0:x1].mean())
        bp = float(black_mask[y0:y1, x0:x1].mean())
        pieces.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))
    return _nms_pieces(pieces, min_dist=28 * scale)


def _nms_pieces(pieces, min_dist: float):
    """合并碎裂连通域：同色且中心距 < min_dist 时保留面积更大者。"""
    if len(pieces) <= 1:
        return pieces
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i][3])
    kept = []
    for i in order:
        cx, cy, color, area = pieces[i]
        if any(
            color == kc and (cx - kx) ** 2 + (cy - ky) ** 2 < min_dist * min_dist
            for kx, ky, kc, _a in kept
        ):
            continue
        kept.append(pieces[i])
    return kept


# --------------------------------------------------------------------------
# 网格定位：在 (row0, col0, cell) 上做评分搜索
# --------------------------------------------------------------------------

def _score_grid(pieces, row0, col0, cell, tol):
    """返回落在格点上的棋子加权得分。"""
    score = 0.0
    hit = 0
    for cx, cy, _c, _a in pieces:
        c = (cx - col0) / cell
        r = (cy - row0) / cell
        ci, ri = round(c), round(r)
        if not (0 <= ci < COLS and 0 <= ri < ROWS):
            continue
        d = ((c - ci) ** 2 + (r - ri) ** 2) ** 0.5 * cell
        if d <= tol:
            score += 1.0 - 0.5 * d / tol
            hit += 1
    return score, hit


def _cluster(values, gap):
    """一维聚类（按间隙切分），返回各簇均值。"""
    vals = np.sort(np.asarray(values, dtype=float))
    if len(vals) == 0:
        return np.array([])
    breaks = np.where(np.diff(vals) > gap)[0]
    groups, start = [], 0
    for b in breaks:
        groups.append(vals[start:b + 1])
        start = b + 1
    groups.append(vals[start:])
    return np.array([g.mean() for g in groups if len(g) > 0])


def _score_axis(values, origin, cell, count, tol):
    """单轴评分：values 落在 origin + k*cell 格点上的加权得分。"""
    score, hit = 0.0, 0
    for v in values:
        k = round((v - origin) / cell)
        if not (0 <= k < count):
            continue
        d = abs(v - (origin + k * cell))
        if d <= tol:
            score += 1.0 - 0.5 * d / tol
            hit += 1
    return score, hit


def _unit_cells_from_gaps(centers, ref_cell):
    """把簇间距折算成单位格宽。

    残局相邻占用行常隔 2～3 格，直接取中位间距会把 cell 估大。
    用 round(d/ref) 还原格数后再取 d/n；ref 仅作分格参考，可略偏。
    """
    if len(centers) < 2 or ref_cell <= 1e-6:
        return []
    units = []
    for d in np.diff(np.asarray(centers, dtype=float)):
        d = float(d)
        if d < 0.70 * ref_cell or d > 3.60 * ref_cell:
            continue
        n = max(1, int(round(d / ref_cell)))
        unit = d / n
        if 0.75 * ref_cell <= unit <= 1.25 * ref_cell:
            units.append(unit)
    return units


def find_grid(crop, pieces=None):
    """定位 10×9 网格。

    先用棋子坐标聚类估计格宽，再在「哪一簇对应第几行/第几列」上做轴向评分，
    只有几百次评估（早期版本用全图四重暴力搜索，耗时几十秒）。

    残局棋子少时，相邻占用行间距常是 2～3 格；不能直接取中位间距，
    也不宜锁死「整图宽 × 76/720」——自动截图常含模拟器侧栏，该默认值会偏大，
    网格整体平移后 OCR 全错。应把多格间距折算成单位格宽后再搜原点。
    """
    if pieces is None:
        pieces = detect_pieces(crop)
    if len(pieces) < 6:
        raise RuntimeError(f"检测到的棋子过少（{len(pieces)}），无法定位棋盘")

    h, w = crop.shape[:2]
    scale = w / _REF_W
    # 宽度比例默认值：仅作聚类间隙/分格参考；letterbox 截图上会偏大
    width_cell = _CELL_RATIO * w

    # 排除顶部头像区 / 底部按钮区的杂点
    y_lo, y_hi = _TOP_UI * h, _BOTTOM_UI * h
    core = [p for p in pieces if y_lo <= p[1] <= y_hi] or pieces

    rows = _cluster([p[1] for p in core], 0.55 * width_cell)
    cols = _cluster([p[0] for p in core], 0.55 * width_cell)
    ys = [p[1] for p in core]
    xs = [p[0] for p in core]

    # 1 格间距 + 多格间距折算后的单位格宽
    near = []
    units = []
    for seq in (rows, cols):
        if len(seq) < 2:
            continue
        for d in np.diff(seq):
            d = float(d)
            if 0.85 * width_cell <= d <= 1.15 * width_cell:
                near.append(d)
        units.extend(_unit_cells_from_gaps(seq, width_cell))

    sparse = len(core) < 16
    if near:
        base_cell = float(np.median(near))
    elif units:
        base_cell = float(np.median(units))
    else:
        base_cell = width_cell

    # letterbox 时 width_cell 偏大；有足够间距样本时允许明显低于宽度默认值
    lo = 0.80 * width_cell if (sparse or len(units) >= 3 or len(near) >= 3) else 0.90 * width_cell
    hi = 1.15 * width_cell if sparse else 1.10 * width_cell
    if not (lo <= base_cell <= hi):
        if units:
            base_cell = float(np.median(units))
        elif not near:
            base_cell = width_cell
        if not (0.70 * width_cell <= base_cell <= 1.20 * width_cell):
            base_cell = width_cell

    # 稀疏残局略放宽格宽微调，原点仍靠占用点对齐
    deltas = (-1.0, -0.5, 0.0, 0.5, 1.0) if not sparse else (
        -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    cell_list = [base_cell + d * scale for d in deltas]
    # 稀疏时若宽度默认与估宽差一截，把宽度默认也纳入候选，避免估偏无退路
    if sparse and abs(width_cell - base_cell) > 1.5 * scale:
        cell_list.append(width_cell)
    tol = 0.38 * base_cell

    def search_axis(values, centers, count):
        best = (-1.0, 0, None)
        for cell in cell_list:
            span = (count - 1) * cell
            for c in centers:
                for k in range(count):
                    origin = c - k * cell
                    if origin < -2 or origin + span > (h if count == ROWS else w) + 2:
                        continue
                    s, ht = _score_axis(values, origin, cell, count, tol)
                    if ht > best[1] or (ht == best[1] and s > best[0]):
                        best = (s, ht, (origin, cell))
        return best

    y_score, y_hit, y_fit = search_axis(ys, rows, ROWS)
    x_score, x_hit, x_fit = search_axis(xs, cols, COLS)

    if y_fit is None or x_fit is None:
        raise RuntimeError("未找到可用网格")

    row0, cell_y = y_fit
    col0, cell_x = x_fit
    cell = (cell_y + cell_x) / 2.0

    # 二维复核（格宽取两轴平均后再评一次）
    s2, hit2 = _score_grid(core, row0, col0, cell, tol)
    if hit2 < min(y_hit, x_hit):
        cell = cell_y
        s2, hit2 = _score_grid(core, row0, col0, cell_y, tol)

    # 稀疏残局：用估得的 base_cell（不是偏大的 width_cell）重搜原点
    if sparse:
        cell_candidates = [base_cell]
        if abs(cell - base_cell) > 0.5 * scale:
            cell_candidates.append(cell)
        if abs(width_cell - base_cell) > 1.5 * scale:
            cell_candidates.append(width_cell)
        best = (hit2, s2, row0, col0, cell)
        for cand in cell_candidates:
            tol_s = 0.38 * cand
            for c in rows:
                for k in range(ROWS):
                    r0 = float(c - k * cand)
                    if r0 < -2 or r0 + 9 * cand > h + 2:
                        continue
                    for x in cols:
                        for j in range(COLS):
                            c0 = float(x - j * cand)
                            if c0 < -2 or c0 + 8 * cand > w + 2:
                                continue
                            sc, ht = _score_grid(core, r0, c0, cand, tol_s)
                            if ht > best[0] or (ht == best[0] and sc > best[1]):
                                best = (ht, sc, r0, c0, cand)
        hit2, s2, row0, col0, cell = best

    # 用已对齐的棋子反推原点，减小整体平移（透视/边距）
    row0, col0 = _refine_grid_origin(core, row0, col0, cell)

    return row0, col0, cell, s2, hit2


def _refine_grid_origin(pieces, row0, col0, cell):
    """按最近格点反推 row0/col0，使格心更贴近棋子中心。"""
    rs, cs = [], []
    for cx, cy, _color, _area in pieces:
        ci = int(round((cx - col0) / cell))
        ri = int(round((cy - row0) / cell))
        if 0 <= ci < COLS and 0 <= ri < ROWS:
            cs.append(cx - ci * cell)
            rs.append(cy - ri * cell)
    if len(rs) < 3:
        return row0, col0
    return float(np.mean(rs)), float(np.mean(cs))


# --------------------------------------------------------------------------
# 格子颜色分类
# --------------------------------------------------------------------------

def classify(crop, row0, col0, cell):
    """返回 (grid_raw, centers)：'@'=红 '#'=黑 '.'=空。

    格心用红/暗黑 mask；随后对底线马位做间隙修复，补亮棕黑子漏检。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    red_mask, black_mask, _orange = _piece_masks(hsv)

    rad = max(6, int(cell * 0.22))
    grid, centers = [], []
    for r in range(ROWS):
        cy = row0 + r * cell
        line = []
        for c in range(COLS):
            cx = col0 + c * cell
            centers.append((cx, cy))
            y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
            x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
            if y1 - y0 < 4 or x1 - x0 < 4:
                line.append(".")
                continue
            rr = float(red_mask[y0:y1, x0:x1].mean())
            br = float(black_mask[y0:y1, x0:x1].mean())
            line.append(_cell_color(rr, br))
        grid.append(line)
    _repair_backrow_horse_gaps(grid)
    return grid, centers


# --------------------------------------------------------------------------
# 兵种推断（开局位置种子 + 单格墨迹 OCR）
# --------------------------------------------------------------------------

def infer_pieces(grid_raw, side=None, crop=None, row0=None, col0=None, cell=None):
    """把 '@'/'#' 占位替换成真实兵种字母。

    side: 我方颜色，"red" 表示红方在下（row 9），"black" 表示黑方在下。
    若 side=None，用 detect_our_side(grid_raw) 自动识别。

    有 crop 时：近开局（底线≥8）仍位置种子；其余一律 ``piece_ocr``。
    无 crop 时才对真正底线 row0/9 做残缺九宫/象位种子。
    """
    from . import piece_ocr

    grid = [list(row) for row in grid_raw]
    if side is None:
        side = detect_our_side(grid_raw) or "red"
    red_bottom = (side == "red")
    red_rows = [sum(1 for ch in row if ch == "@") for row in grid]
    black_rows = [sum(1 for ch in row if ch == "#") for row in grid]

    def pick_backrow(counts, prefer_top):
        order = list(range(3)) if prefer_top else list(range(ROWS - 1, ROWS - 4, -1))
        for r in order:
            if counts[r] >= 7:
                return r
        order = list(range(ROWS // 2)) if prefer_top else list(range(ROWS - 1, ROWS // 2 - 1, -1))
        best_r, best_c = (0 if prefer_top else ROWS - 1), 0
        for r in order:
            if counts[r] > best_c:
                best_r, best_c = r, counts[r]
        return best_r if best_c >= 4 else None

    red_back = pick_backrow(red_rows, not red_bottom)
    black_back = pick_backrow(black_rows, red_bottom)

    def backrow_complete(back, marker):
        """近开局完整底线：两端车位都在，且同色 ≥8。

        缺角（如只有 ``.@@@@@@@@``）时不能套 RNB… 模板，否则会把错位的
        车标成马、占掉马配额，真马再被兜成车（见 103438）。
        缺马（``R.BAKABNR`` / ``RNBAKAB.R``）两端仍在，可以安全套模板。
        """
        if back is None:
            return False
        if grid[back][0] != marker or grid[back][8] != marker:
            return False
        return sum(1 for c in range(COLS) if grid[back][c] == marker) >= 8

    red_full = backrow_complete(red_back, "@")
    black_full = backrow_complete(black_back, "#")

    # 近开局：两端车位在且 ≥8 才套完整底线模板，并填经典炮/兵位
    if red_full:
        for c in range(COLS):
            if grid[red_back][c] == "@":
                grid[red_back][c] = RED_BACKROW[c]
    if black_full:
        for c in range(COLS):
            if grid[black_back][c] == "#":
                grid[black_back][c] = BLACK_BACKROW[c]

    def fill_cannon_pawn(backrow, marker, cannon, pawn):
        if backrow is None:
            return
        step = 1 if backrow <= 4 else -1
        cannon_row, pawn_row = backrow + 2 * step, backrow + 3 * step
        if 0 <= cannon_row < ROWS:
            for c in (1, 7):
                if grid[cannon_row][c] == marker:
                    grid[cannon_row][c] = cannon
        if 0 <= pawn_row < ROWS:
            for c in (0, 2, 4, 6, 8):
                if grid[pawn_row][c] == marker:
                    grid[pawn_row][c] = pawn

    if red_full:
        fill_cannon_pawn(red_back, "@", "C", "P")
    if black_full:
        fill_cannon_pawn(black_back, "#", "c", "p")

    # 单格 OCR：填充所有残留 @/#
    # 有 OCR 时不做「残缺底线」位置种子——pick_backrow 在中局常把 row2/7
    # 误当成底线，会把马标成象、把将士配额占掉（见 112752）。
    can_ocr = (
        crop is not None and row0 is not None and col0 is not None and cell is not None
    )
    if can_ocr:
        piece_ocr.ocr_fill_markers(
            grid, crop, row0, col0, cell, side,
            piece_masks_fn=_piece_masks,
            elephant_valid_fn=_valid_elephant_positions,
        )
    else:
        # 无图时才用残缺底线种子；且只允许真正底线 row0/9，禁止 row2/7
        def seed_partial_backrow(back, marker, backrow_letters):
            if back is None or back not in (0, ROWS - 1) or backrow_complete(back, marker):
                return
            for c, letter in ((4, backrow_letters[4]), (3, backrow_letters[3]),
                              (5, backrow_letters[5])):
                if grid[back][c] == marker:
                    grid[back][c] = letter
            for c in (2, 6):
                if grid[back][c] == marker:
                    grid[back][c] = backrow_letters[c]

        seed_partial_backrow(red_back, "@", RED_BACKROW)
        seed_partial_backrow(black_back, "#", BLACK_BACKROW)

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == "@":
                grid[r][c] = "R"
            elif grid[r][c] == "#":
                grid[r][c] = "r"
    return grid


# --------------------------------------------------------------------------
# 对外：识别
# --------------------------------------------------------------------------

def _palace_king_color(grid_raw, palace_rows):
    """在指定九宫范围（palace_rows 是行号集合，列固定 3-5）里找主将颜色。

    返回 '@' / '#' / None（没找到或两边都有时返回 None）。
    """
    found = set()
    for r in palace_rows:
        for c in (3, 4, 5):
            ch = grid_raw[r][c]
            if ch in ("@", "#"):
                found.add(ch)
    if len(found) == 1:
        return found.pop()
    return None


def detect_both_sides(grid_raw):
    """用双方主将颜色确定敌我颜色，带交叉验证。

    逻辑：
      - 我方主将在下半九宫（row 7-9, col 3-5）
      - 对方主将在上半九宫（row 0-2, col 3-5）
      - 我方颜色 = 我方主将颜色的大小写（@→red, #→black）
      - 对方颜色 = 对方主将颜色的对立面（我方 red → 对方 black，反之亦然）
      - 如果双方主将颜色看起来相同（都是 '@' 或都是 '#'），做交叉校验：
        用"我方半场 row5-9 中 @/# 的多数"来辅助判断

    返回 (side_str, opp_side_str)，都用 "red"/"black"。
    """
    lower_king = _palace_king_color(grid_raw, (7, 8, 9))
    upper_king = _palace_king_color(grid_raw, (0, 1, 2))

    # 基本映射
    if lower_king == "@":
        our_side = "red"
    elif lower_king == "#":
        our_side = "black"
    elif upper_king == "@":
        # 下半九宫漏检，用上半九宫的对立面
        our_side = "black"
    elif upper_king == "#":
        our_side = "red"
    else:
        # 两边都没找到任何主将——用半场颜色多数兜底
        our_rows = range(5, ROWS)
        our_at = sum(grid_raw[r].count("@") for r in our_rows)
        our_hash = sum(grid_raw[r].count("#") for r in our_rows)
        our_side = "red" if our_at >= our_hash else "black"

    opp_side = "black" if our_side == "red" else "red"

    # 交叉校验：如果下半九宫和上半九宫"主将颜色相同"，报不一致
    # （理论上不会发生，除非识别严重出错）
    if lower_king and upper_king and lower_king == upper_king:
        # 两个主将相了，信任下半九宫（我方视角）
        pass

    return our_side, opp_side


def detect_our_side(grid_raw):
    """确定我方颜色（下半九宫找主将）。

    逻辑：棋盘下半边是我方（拍照时用户坐下方）。九宫里一定有主将
    （残局也可靠），找到 '@' 就是红方在下，找到 '#' 就是黑方在下。

    返回 "red" / "black" / None（都没找到时返回 None）。
    """
    side, _ = detect_both_sides(grid_raw)
    return side


def detect_red_bottom(grid_raw):
    """兼容旧调用：返回红方是否在下（用新逻辑实现）。"""
    side = detect_our_side(grid_raw)
    return side == "red"


# --------------------------------------------------------------------------
# 棋局合法性验证
# --------------------------------------------------------------------------

_PALACE_COLS = {3, 4, 5}

# 红黑双方每种棋子的 (最少, 最多) 数量
_RED_COUNTS = {"R": (0, 2), "N": (0, 2), "B": (0, 2), "A": (0, 2),
               "K": (1, 1), "C": (0, 2), "P": (0, 5)}
_BLK_COUNTS = {"r": (0, 2), "n": (0, 2), "b": (0, 2), "a": (0, 2),
               "k": (1, 1), "c": (0, 2), "p": (0, 5)}


def _valid_elephant_positions(is_red, side):
    """该侧方的象/相合法坐标集合（开局 7 个）。"""
    red_bottom = (side == "red")
    if is_red and not red_bottom or (not is_red) and red_bottom:
        # 象在图像上半（row 0 底线）
        return {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)}
    # 象在图像下半（row 9 底线）
    return {(9, 2), (9, 6), (7, 0), (7, 4), (7, 8), (5, 2), (5, 6)}


def _palace_rows(is_red, side):
    """己方九宫的行范围 (row0, row1)。"""
    red_bottom = (side == "red")
    if is_red:
        return (7, 9) if red_bottom else (0, 2)
    return (0, 2) if red_bottom else (7, 9)


def validate_grid(grid, side=None):
    """校验兵种推断后的 grid 是否符合象棋规则。

    side: 我方颜色 "red"/"black"，None 时用 detect_our_side 自动识别。
    返回 (valid, messages, violations)：
      valid       bool  所有硬规则通过（计数、位置）。帅/将同列无子和总数超限
                        属于提示性信息，不算硬失败
      messages    [str]  所有问题描述（计数、位置、帅/将、总数）
      violations  {(r, c): [str]}  仅位置类问题，用于标注红 X
    """
    # side 应从 analyze_image 调用链传入；为兼容直接调用提供兜底
    if side is None:
        side = "red"
    red_bottom = (side == "red")
    messages = []
    violations = {}  # (row, col) -> [positional violation texts]
    hard_messages = []  # 计数 + 位置（硬失败）

    # --- 1. 计数 ---
    counts = {}
    for row in grid:
        for ch in row:
            if ch != ".":
                counts[ch] = counts.get(ch, 0) + 1

    for ch, (lo, hi) in {**_RED_COUNTS, **_BLK_COUNTS}.items():
        cnt = counts.get(ch, 0)
        if cnt < lo:
            hard_messages.append(f"缺少 {ch}：至少 {lo} 个，当前 {cnt}")
            messages.append(f"缺少 {ch}：至少 {lo} 个，当前 {cnt}")
        elif cnt > hi:
            hard_messages.append(f"{ch} 数量超限：最多 {hi} 个，当前 {cnt}")
            messages.append(f"{ch} 数量超限：最多 {hi} 个，当前 {cnt}")

    # --- 2. 位置 ---
    elep_red = _valid_elephant_positions(True, side)
    elep_blk = _valid_elephant_positions(False, side)

    for r in range(ROWS):
        for c in range(COLS):
            ch = grid[r][c]
            if ch == ".":
                continue
            is_red = ch.isupper()
            msgs = []

            if ch in ("K", "k"):
                r0, r1 = _palace_rows(is_red, side)
                if not (r0 <= r <= r1 and c in _PALACE_COLS):
                    msgs.append("帅/将不在九宫内")

            elif ch in ("A", "a"):
                r0, r1 = _palace_rows(is_red, side)
                if not (r0 <= r <= r1 and c in _PALACE_COLS):
                    msgs.append("仕/士不在九宫内")

            elif ch in ("B", "b"):
                valid_set = elep_red if is_red else elep_blk
                if (r, c) not in valid_set:
                    msgs.append("象/相位置非法（过河或落点不在田字）")

            if msgs:
                violations[(r, c)] = msgs
                for m in msgs:
                    hard_messages.append(f"{ch} 在 ({r},{c})：{m}")
                    messages.append(f"{ch} 在 ({r},{c})：{m}")

    # --- 3. 帅/将同列无子（合法局面，仅提示）---
    kings = [(r, c) for r in range(ROWS) for c in range(COLS)
             if grid[r][c] in ("K", "k")]
    if len(kings) == 2:
        r1, c1 = kings[0]
        r2, c2 = kings[1]
        if c1 == c2:
            lo, hi = min(r1, r2), max(r1, r2)
            btw = sum(1 for r in range(lo + 1, hi) if grid[r][c1] != ".")
            if btw == 0:
                messages.append(f"帅/将同列无子（({r1},{c1}) vs ({r2},{c2})）")

    # --- 4. 棋子总数提示（开局最多 32，但残局合法，只提示不算硬错）---
    total = sum(counts.values())
    if total > 32:
        messages.append(f"棋子总数 {total} > 32（棋盘最多 32 子，识别可能有误）")

    valid = len(hard_messages) == 0
    return valid, messages, violations


# --------------------------------------------------------------------------
# 对外：识别
# --------------------------------------------------------------------------

class Analysis:
    """一次识别结果。"""

    def __init__(self, path, image, crop_box, row0, col0, cell,
                 grid_raw, grid, score, hit, side,
                 valid=True, violations=None):
        self.path = path
        self.image = image              # 原图 BGR
        self.crop_box = crop_box        # (x0, y0, x1, y1)
        self.row0, self.col0, self.cell = row0, col0, cell
        self.grid_raw = grid_raw        # '@'/'#'/'.'
        self.grid = grid                # 已推断兵种
        self.grid_str = "\n".join("".join(row) for row in grid)
        self.score, self.hit = score, hit
        self.side = side                # 我方颜色："red" / "black"（由下半九宫主将确定）
        self.red_bottom = (side == "red")  # 兼容旧代码
        self.view_side = side           # 图下方是谁 = 我方
        self.valid = valid              # 识别结果是否通过合法性校验
        self.violations = violations or {}  # {(r,c): [str]} 非法格子

    @property
    def red_count(self):
        return sum(1 for row in self.grid for ch in row if ch in RED)

    @property
    def black_count(self):
        return sum(1 for row in self.grid for ch in row
                   if ch.islower() and ch != ".")

    def cell_center(self, row, col):
        """图像方向格子中心（整图像素坐标）。"""
        x0, y0, _x1, _y1 = self.crop_box
        return x0 + self.col0 + col * self.cell, y0 + self.row0 + row * self.cell


def analyze_image(path):
    """识别一张截图，返回 Analysis（含合法性校验）。"""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return analyze_array(img, path=path)


def analyze_array(img, path="<memory>"):
    """直接从内存中的 BGR ndarray 识别（不落盘）。

    用于每秒帧的自动监听场景，避免频繁 I/O。
    """
    h, w = img.shape[:2]
    y0, y1 = int(_ROI_Y0 * h), int(_ROI_Y1 * h)
    crop = img[y0:y1, 0:w]
    box = (0, y0, w, y1)

    pieces = detect_pieces(crop)
    row0, col0, cell, score, hit = find_grid(crop, pieces)
    grid_raw, _centers = classify(crop, row0, col0, cell)

    # 在下方九宫主将确定我方颜色
    side = detect_our_side(grid_raw) or "red"

    grid = infer_pieces(grid_raw, side=side, crop=crop, row0=row0, col0=col0, cell=cell)
    valid, _msgs, violations = validate_grid(grid, side=side)
    return Analysis(path, img, box, row0, col0, cell, grid_raw, grid,
                    score, hit, side, valid=valid, violations=violations)


# --------------------------------------------------------------------------
# 对外：标注
# --------------------------------------------------------------------------

_CN_FONT_CANDIDATES = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def _piece_cn(ch):
    """FEN 字母 -> 中文兵种名；未知则回退字母本身。"""
    piece = FEN_TO_PIECE.get(ch)
    if piece is None:
        return ch
    return PIECE_NAMES[(piece >> 3, piece & 7)]


def _load_cn_font(size):
    """按平台候选路径加载中文字体；失败返回 None。"""
    from PIL import ImageFont

    size = max(10, int(size))
    for path in _CN_FONT_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            return ImageFont.truetype(path, size=size, index=0)
        except OSError:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return None


def _draw_piece_labels(vis, analysis, half):
    """在每枚棋子描边方框左上角画中文兵种，颜色与棋子一致（红/黑）。

    字下垫高反差底色块，便于在木纹棋盘上核对识别结果。
    """
    from PIL import Image, ImageDraw

    font_px = max(12, int(analysis.cell * 0.32))
    font = _load_cn_font(font_px)
    pad = max(2, font_px // 8)
    labels = []
    for r in range(ROWS):
        for c in range(COLS):
            if analysis.grid_raw[r][c] == ".":
                continue
            ch = analysis.grid[r][c]
            if ch in (".", "@", "#"):
                continue
            cx, cy = analysis.cell_center(r, c)
            x = int(cx) - half + 2
            y = int(cy) - half + 1
            # RGB：红子红字+浅黄底，黑子黑字+浅青底
            if ch.isupper():
                color, bg = (200, 20, 20), (255, 245, 160)
            else:
                color, bg = (10, 10, 10), (180, 255, 240)
            labels.append((_piece_cn(ch), ch, x, y, color, bg))

    if not labels:
        return vis

    if font is None:
        # 无中文字体时回退 FEN 字母（OpenCV 拉丁字库）
        for _cn, ch, x, y, color, bg in labels:
            bgr = (color[2], color[1], color[0])
            bg_bgr = (bg[2], bg[1], bg[0])
            baseline_y = y + int(font_px * 0.85)
            (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_px / 28.0, 1)
            cv2.rectangle(vis, (x - pad, y - pad),
                          (x + tw + pad, y + th + pad), bg_bgr, -1)
            cv2.putText(vis, ch, (x, baseline_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_px / 28.0, bgr, 1, cv2.LINE_AA)
        return vis

    rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for text, _ch, x, y, color, bg in labels:
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
            fill=bg,
        )
        draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def annotate(analysis, decision, draw_grid=True):
    """把网格、棋子圈与最佳着法画到原图上，返回新图（BGR）。"""
    a = analysis
    vis = a.image.copy()
    x0, y0, x1, y1 = a.crop_box
    cell = a.cell
    half = int(cell * 0.46)

    # --- 所有检出棋子：黑色矩形描边 ---
    for r in range(ROWS):
        for c in range(COLS):
            if a.grid_raw[r][c] == ".":
                continue
            cx, cy = a.cell_center(r, c)
            cx, cy = int(cx), int(cy)
            cv2.rectangle(vis, (cx - half, cy - half), (cx + half, cy + half),
                          (0, 0, 0), 2)

    # --- 兵种文字：描边方框左上角，字色与棋子红/黑一致 ---
    vis = _draw_piece_labels(vis, a, half)

    # --- 合法性校验：非法棋子画红X ---
    for (r, c), _violations in a.violations.items():
        cx, cy = a.cell_center(r, c)
        cx, cy = int(cx), int(cy)
        cv2.line(vis, (cx - half, cy - half), (cx + half, cy + half),
                 (0, 0, 255), 3)
        cv2.line(vis, (cx + half, cy - half), (cx - half, cy + half),
                 (0, 0, 255), 3)

    if draw_grid:
        for r in range(ROWS + 1):
            y = int(y0 + a.row0 - cell / 2 + r * cell)
            cv2.line(vis, (x0, y), (x1, y), (90, 90, 90), 1)
        for c in range(COLS + 1):
            x = int(x0 + a.col0 - cell / 2 + c * cell)
            cv2.line(vis, (x, y0), (x, y1), (90, 90, 90), 1)

    best = decision.get("best_move") if decision else None
    if best:
        pv = decision.get("pv") or {}
        pv_steps = pv.get("steps") or []
        # 契约：推演第 1 手=我方、第 2 手=对方应手；两手都画在棋盘上
        rad = int(cell * 0.45)
        half = int(cell * 0.46)

        def _step_coords(step):
            fx, fy = a.cell_center(step["from"]["row"], step["from"]["col"])
            tx, ty = a.cell_center(step["to"]["row"], step["to"]["col"])
            return int(fx), int(fy), int(tx), int(ty)

        # 先画第 2 手（对方）：绿线 + 绿框起点 + 黄框终点
        if len(pv_steps) >= 2:
            rfx, rfy, rtx, rty = _step_coords(pv_steps[1])
            cv2.line(vis, (rfx, rfy), (rtx, rty), (0, 200, 0), 3)
            cv2.rectangle(vis, (rfx - half, rfy - half), (rfx + half, rfy + half),
                          (0, 200, 0), 4)
            cv2.rectangle(vis, (rtx - half, rty - half), (rtx + half, rty + half),
                          (0, 255, 255), 4)

        # 再画第 1 手（我方）：红线 + 红圈起点 + 蓝圈终点（盖在上层）
        if pv_steps:
            our = pv_steps[0]
        else:
            our = {
                "from": {"row": best["from"]["row"], "col": best["from"]["col"]},
                "to": {"row": best["to"]["row"], "col": best["to"]["col"]},
            }
        fx, fy, tx, ty = _step_coords(our)
        cv2.line(vis, (fx, fy), (tx, ty), (0, 0, 255), 3)
        cv2.circle(vis, (fx, fy), rad + 7, (0, 0, 255), 5)
        cv2.circle(vis, (fx, fy), rad + 1, (255, 255, 255), 2)
        cv2.circle(vis, (tx, ty), rad + 5, (255, 0, 0), 5)

        label = f"{best['chinese']} ({best['iccs']})  {best['score_text']}"
        if len(pv.get("chinese") or []) >= 2:
            label = f"{pv['chinese'][0]} / {pv['chinese'][1]}  {best['score_text']}"
        reason = best.get("reason", "")
        if reason:
            reason = reason[:70]

        # 字号随图片宽度自适应，文字衬半透明底以提升可读性
        fs = 0.9 * min(1.6, max(0.85, vis.shape[1] / 720.0))
        width, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
        roof = 12 + int(38 * fs)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (width + 24, roof), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
        cv2.putText(vis, label, (12, int(30 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 255), 2, cv2.LINE_AA)
        if reason:
            cv2.putText(vis, reason, (12, int(30 * fs) + baseline + int(14 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.62, (255, 200, 120), 1, cv2.LINE_AA)
    return vis
