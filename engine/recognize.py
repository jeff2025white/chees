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

# 基准分辨率（历史手机截图 720×1280；当前 images/* 多为 782×1332 模拟器）
_REF_W = 720.0
_REF_H = 1280.0
# 棋盘在整图中的相对布局（images/* 根目录 266 张、hit≥20 中位数标定）
# 随图宽高缩放；换皮/改布局后需重测
_BOARD_ROW0_ABS = 0.25443             # 首格中心 Y / 图高
_BOARD_COL0_ABS = 0.07344             # 首格中心 X / 图宽
_BOARD_CELL_W = 0.09782               # 格宽 / 图宽
_CELL_RATIO = _BOARD_CELL_W           # 兼容旧引用
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
    """由窗口内红/暗黑 mask 均值判 '@' / '#' / '.'（classify 诊断用）。

    oran 保留参数以兼容调用。主路径占用来自圆检吸附，不依赖本函数补漏。
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


def _side_occ_count(grid, marker):
    return sum(1 for row in grid for ch in row if ch == marker)


def _repair_backrow_horse_gaps(grid):
    """补回底线漏检的马位（col 1 / 7）。

    亮棕黑子常在 HSV 上与棋盘木纹不可分，连通域与格心 mask 都会漏。
    若某一行已有 ≥7 枚同色子且马位为空、左右邻为该色，则补上。
    只动 row 0-2 / 7-9，避免中局误补。

    若该行任一马位的日字跳点上已有同色子，视为马已出动，整行不再补——
    否则空马位会被假占位，占掉 n/N 配额后被改判成炮/车
    （见 20261118 跳马局、20261119_143731 马八空位）。
    同色占位已达 16（一方子力上限）时也不再补。
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
        # 任一侧马已跳出 → 本行禁止补漏（避免空马位变假子）
        if any(_horse_already_developed(grid, r, hc, maj) for hc in (1, 7)):
            continue
        for c in (1, 7):
            if row[c] != ".":
                continue
            # 同色最多 16 子；已满则不可再造占位
            if _side_occ_count(grid, maj) >= 16:
                break
            left = row[c - 1] if c - 1 >= 0 else "."
            right = row[c + 1] if c + 1 < COLS else "."
            if left == maj or right == maj:
                row[c] = maj
    return grid


def detect_pieces(crop):
    """连通域 + 面积 + 圆形度过滤，返回 [(cx, cy, 'R'|'B', area)]。

    主路径占用检测入口：先检棋子圆，再吸附到固定格点。
    只用红 + 暗黑 mask（亮棕与木纹接近，并入会污染）；漏检不造假子。
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
# 网格定位：整图相对固定布局 + 可选原点微调
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


def fixed_board_grid(full_w, full_h, crop_y0=0):
    """按标定比例得到 crop 坐标系下的 (row0, col0, cell)。"""
    cell = _BOARD_CELL_W * float(full_w)
    col0 = _BOARD_COL0_ABS * float(full_w)
    row0 = _BOARD_ROW0_ABS * float(full_h) - float(crop_y0)
    return float(row0), float(col0), float(cell)


def _infer_full_size(crop, crop_y0):
    """仅传 crop 时反推整图尺寸（假定仍按 _ROI_Y0/_ROI_Y1 裁切）。"""
    ch, cw = crop.shape[:2]
    full_w = cw
    if crop_y0 > 0 and _ROI_Y0 > 1e-6:
        full_h = int(round(crop_y0 / _ROI_Y0))
    else:
        span = max(_ROI_Y1 - _ROI_Y0, 1e-6)
        full_h = int(round(ch / span))
    return full_w, max(full_h, ch + crop_y0)


def find_grid(crop, pieces=None, full_w=None, full_h=None, crop_y0=0):
    """定位 10×9 网格。

    棋盘在截图中的相对位置固定：用 ``_BOARD_*`` 比例直接算格网，
    随 ``full_w/full_h`` 缩放。若传入 ``pieces``（连通域中心），再做原点微调。

    主识别路径不再依赖棋子聚类；``pieces`` 仅用于可选 refine / 诊断打分。
    """
    if full_w is None or full_h is None:
        fw, fh = _infer_full_size(crop, crop_y0)
        full_w = full_w or fw
        full_h = full_h or fh

    row0, col0, cell = fixed_board_grid(full_w, full_h, crop_y0=crop_y0)
    if cell <= 1e-6:
        raise RuntimeError("固定格网 cell 无效")

    score, hit = 0.0, 0
    if pieces:
        ch = crop.shape[0]
        y_lo, y_hi = _TOP_UI * ch, _BOTTOM_UI * ch
        core = [p for p in pieces if y_lo <= p[1] <= y_hi] or pieces
        row0, col0 = _refine_grid_origin(core, row0, col0, cell)
        score, hit = _score_grid(core, row0, col0, cell, 0.38 * cell)

    return row0, col0, cell, score, hit


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
# 占用：圆检吸附到固定格点（主路径）/ 格心采样（诊断）
# --------------------------------------------------------------------------

def occupy_from_pieces(pieces, row0, col0, cell, max_dist_ratio=0.42):
    """将棋子圆吸附到固定 10×9 格点，生成 grid_raw。

    无圆 = '.'；同格多圆保留面积更大者。不补空位、不做马位间隙修复。
    返回 (grid_raw, centers)，centers 为 90 个格心坐标。
    """
    grid = [["." for _ in range(COLS)] for _ in range(ROWS)]
    best_area = [[0.0 for _ in range(COLS)] for _ in range(ROWS)]
    centers = []
    for r in range(ROWS):
        for c in range(COLS):
            centers.append((col0 + c * cell, row0 + r * cell))

    max_dist = max_dist_ratio * float(cell)
    for cx, cy, color, area in pieces:
        cf = (cx - col0) / cell
        rf = (cy - row0) / cell
        ci, ri = int(round(cf)), int(round(rf))
        if not (0 <= ci < COLS and 0 <= ri < ROWS):
            continue
        dist = ((cf - ci) ** 2 + (rf - ri) ** 2) ** 0.5 * cell
        if dist > max_dist:
            continue
        marker = "@" if color == "R" else "#"
        if grid[ri][ci] != "." and area <= best_area[ri][ci]:
            continue
        grid[ri][ci] = marker
        best_area[ri][ci] = float(area)
    return grid, centers


def enrich_occupancy_cell_masks(grid, crop, row0, col0, cell):
    """圆检漏子时的格心二次确认：只认红/暗黑 mask，不用亮棕（与木纹不可分）。

    典型：黑士底色偏亮，连通域抓不到整圆，但格心墨迹仍进 black_mask。
    不做马位规则补漏；已有圆的格不改写。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask, black_mask, _orange = _piece_masks(hsv)
    h, w = crop.shape[:2]
    rad = max(6, int(cell * 0.22))
    out = [list(row) for row in grid]
    for r in range(ROWS):
        for c in range(COLS):
            if out[r][c] != ".":
                continue
            cy = row0 + r * cell
            cx = col0 + c * cell
            y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
            x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
            if y1 - y0 < 4 or x1 - x0 < 4:
                continue
            rr = float(red_mask[y0:y1, x0:x1].mean())
            br = float(black_mask[y0:y1, x0:x1].mean())
            if rr < 0.015 and br < 0.015:
                continue
            ch = _cell_color(rr, br)
            if ch != ".":
                out[r][c] = ch
    return out


def classify(crop, row0, col0, cell):
    """按固定格心采样占用（诊断对照用）：返回 (grid_raw, centers)。

    主识别路径已改为 detect_pieces → occupy_from_pieces；本函数不做马位补漏。
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
            if rr < 0.015 and br < 0.015:
                line.append(".")
                continue
            line.append(_cell_color(rr, br))
        grid.append(line)
    return grid, centers


# --------------------------------------------------------------------------
# 兵种推断（开局底线种子 + 单格 OCR；禁止兵行硬覆盖）
# --------------------------------------------------------------------------

def infer_pieces(grid_raw, side=None, crop=None, row0=None, col0=None, cell=None):
    """把 '@'/'#' 占位替换成真实兵种字母。

    side: 我方颜色，"red" 表示红方在下（row 9），"black" 表示黑方在下。
    若 side=None，用 detect_our_side(grid_raw) 自动识别。

    仅近开局（双方子力近满 + 完整底线）才对仍为 @/# 的底线格做位置种子；
    中局一律 OCR/CNN。禁止炮/兵行硬编码覆盖（避免真马被写成兵）。
    """
    from . import piece_ocr

    grid = [list(row) for row in grid_raw]
    if side is None:
        side = detect_our_side(grid_raw) or "red"
    red_bottom = (side == "red")
    red_n = sum(1 for row in grid for ch in row if ch == "@")
    black_n = sum(1 for row in grid for ch in row if ch == "#")
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
        """近开局完整底线：两端车位都在，且同色 ≥8。"""
        if back is None:
            return False
        if grid[back][0] != marker or grid[back][8] != marker:
            return False
        return sum(1 for c in range(COLS) if grid[back][c] == marker) >= 8

    red_full = backrow_complete(red_back, "@")
    black_full = backrow_complete(black_back, "#")
    # 近开局门控：双方都接近满盘且至少一方底线完整，才允许底线种子
    near_opening = (red_n >= 14 and black_n >= 14 and (red_full or black_full))

    def seed_backrow_occupied(back, marker, letters, require_full):
        """只改写仍为 marker 的格，不覆盖已有字母。"""
        if back is None:
            return False
        if require_full:
            if not backrow_complete(back, marker):
                return False
            for c in range(COLS):
                if grid[back][c] == marker:
                    grid[back][c] = letters[c]
            return True
        return False

    if near_opening:
        seed_backrow_occupied(red_back, "@", RED_BACKROW, require_full=True)
        seed_backrow_occupied(black_back, "#", BLACK_BACKROW, require_full=True)

    can_ocr = (
        crop is not None and row0 is not None and col0 is not None and cell is not None
    )
    if can_ocr:
        piece_ocr.ocr_fill_markers(
            grid, crop, row0, col0, cell, side,
            piece_masks_fn=_piece_masks,
            elephant_valid_fn=_valid_elephant_positions,
            validate_fn=validate_grid,
        )
    elif near_opening:
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

    # 仍残留的占位：弱默认兵/卒（有配额）；配额满则清空，禁止改判成其他兵种
    from .piece_ocr import TYPE_QUOTA, enforce_type_quota
    counts = {}
    for row in grid:
        for ch in row:
            if ch not in (".", "@", "#"):
                counts[ch] = counts.get(ch, 0) + 1
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == "@":
                if counts.get("P", 0) < TYPE_QUOTA["P"]:
                    grid[r][c] = "P"
                    counts["P"] = counts.get("P", 0) + 1
                else:
                    grid[r][c] = "."
            elif grid[r][c] == "#":
                if counts.get("p", 0) < TYPE_QUOTA["p"]:
                    grid[r][c] = "p"
                    counts["p"] = counts.get("p", 0) + 1
                else:
                    grid[r][c] = "."
    enforce_type_quota(grid)
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


def _valid_pawn_rows(is_red, side):
    """兵/卒允许出现的行号集合。

    兵卒只能向前，不能退回己方底线之后：
      - 该色在图下方：起点 row6，向前到 0 → 合法行 0..6
      - 该色在图上方：起点 row3，向前到 9 → 合法行 3..9
    """
    red_bottom = (side == "red")
    on_bottom = (is_red == red_bottom)
    return set(range(0, 7) if on_bottom else range(3, 10))


def check_piece_zones(grid, side="red"):
    """兵种位置带检查（识别后的简单规则测试）。

    规则：
      - 帅/将、仕/士：必须在己方九宫（3×3）
      - 象/相：必须在己方田字 7 点（不过河）
      - 兵/卒：不得出现在己方底线之后（见 ``_valid_pawn_rows``）

    返回 (ok, violations)，violations 为 {(r,c): [原因, ...]}。
    不含数量/飞将等其它规则；完整校验仍用 ``validate_grid``。
    """
    elep_red = _valid_elephant_positions(True, side)
    elep_blk = _valid_elephant_positions(False, side)
    pawn_red = _valid_pawn_rows(True, side)
    pawn_blk = _valid_pawn_rows(False, side)
    violations = {}

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
            elif ch in ("P", "p"):
                allowed = pawn_red if is_red else pawn_blk
                if r not in allowed:
                    msgs.append("兵/卒超出前进范围（疑似退回己方底线后）")
            if msgs:
                violations[(r, c)] = msgs
    return len(violations) == 0, violations


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

    # --- 2. 位置带（九宫 / 象田 / 兵卒行）---
    _zone_ok, zone_viol = check_piece_zones(grid, side)
    for (r, c), msgs in zone_viol.items():
        violations[(r, c)] = list(msgs)
        ch = grid[r][c]
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


# 自动监听：同占用图下兵种时序投票（最多 3 帧）+ 一着迁移
_TYPE_VOTE_BUF = []  # [(occ_key, grid_copy), ...]
_TYPE_VOTE_MAX = 3
_LAST_TRACK = None  # (occ_key, grid_copy)


def _occupancy_key(grid_raw):
    return tuple("".join(row) for row in grid_raw)


def _occ_color(ch):
    if ch == "@" or (ch.isupper() and ch != "."):
        return "R"
    if ch == "#" or (ch.islower() and ch != "."):
        return "B"
    return "."


def _vote_piece_types(grid, grid_raw):
    """占用不变时，对字母格做多数投票，抑制 OCR 闪错。"""
    global _TYPE_VOTE_BUF
    key = _occupancy_key(grid_raw)
    gcopy = [list(row) for row in grid]
    _TYPE_VOTE_BUF = [(k, g) for (k, g) in _TYPE_VOTE_BUF if k == key]
    _TYPE_VOTE_BUF.append((key, gcopy))
    if len(_TYPE_VOTE_BUF) > _TYPE_VOTE_MAX:
        _TYPE_VOTE_BUF = _TYPE_VOTE_BUF[-_TYPE_VOTE_MAX:]
    if len(_TYPE_VOTE_BUF) < 2:
        return grid
    out = [list(row) for row in grid]
    for r in range(ROWS):
        for c in range(COLS):
            if grid_raw[r][c] == ".":
                continue
            votes = {}
            for _k, g in _TYPE_VOTE_BUF:
                ch = g[r][c]
                if ch in (".", "@", "#"):
                    continue
                votes[ch] = votes.get(ch, 0) + 1
            if not votes:
                continue
            best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == out[r][c]))
            if best[1] >= 2:
                out[r][c] = best[0]
    return out


def _migrate_piece_identity(grid, grid_raw, prev_occ, prev_grid):
    """占用差分为「同色一格消失、一格出现」时迁移兵种，减少重识别抖动。"""
    if prev_occ is None or prev_grid is None:
        return grid
    disappeared = []
    appeared = []
    for r in range(ROWS):
        for c in range(COLS):
            oc = _occ_color(prev_occ[r][c])
            nc = _occ_color(grid_raw[r][c])
            if oc != nc:
                if oc != "." and nc == ".":
                    disappeared.append((r, c, oc))
                elif oc == "." and nc != ".":
                    appeared.append((r, c, nc))
    out = [list(row) for row in grid]
    if len(disappeared) == 1 and len(appeared) == 1:
        r0, c0, color0 = disappeared[0]
        r1, c1, color1 = appeared[0]
        if color0 == color1:
            piece = prev_grid[r0][c0]
            if piece not in (".", "@", "#") and _occ_color(piece) == color0:
                out[r1][c1] = piece
    return out


def _stabilize_types(grid, grid_raw):
    """投票 + 相对上一帧的一着迁移。"""
    global _LAST_TRACK
    key = _occupancy_key(grid_raw)
    out = [list(row) for row in grid]
    if _LAST_TRACK is not None:
        prev_key, prev_grid = _LAST_TRACK
        if prev_key == key:
            out = _vote_piece_types(out, grid_raw)
        else:
            out = _migrate_piece_identity(out, grid_raw, prev_key, prev_grid)
            # 迁移后仍投票缓冲重置为当前
            global _TYPE_VOTE_BUF
            _TYPE_VOTE_BUF = [(key, [list(row) for row in out])]
    else:
        out = _vote_piece_types(out, grid_raw)
    _LAST_TRACK = (key, [list(row) for row in out])
    return out


def analyze_array(img, path="<memory>"):
    """直接从内存中的 BGR ndarray 识别（不落盘）。

    用于每秒帧的自动监听场景，避免频繁 I/O。
    """
    h, w = img.shape[:2]
    y0, y1 = int(_ROI_Y0 * h), int(_ROI_Y1 * h)
    crop = img[y0:y1, 0:w]
    box = (0, y0, w, y1)

    row0, col0, cell, _score, _hit = find_grid(
        crop, pieces=None, full_w=w, full_h=h, crop_y0=y0)
    pieces = detect_pieces(crop)
    grid_raw, _centers = occupy_from_pieces(pieces, row0, col0, cell)
    # 圆检漏的亮底黑子（如士）：格心暗黑墨迹二次确认，不造空位假子
    grid_raw = enrich_occupancy_cell_masks(grid_raw, crop, row0, col0, cell)
    hit = sum(1 for row in grid_raw for ch in row if ch != ".")
    score = float(hit)

    side = detect_our_side(grid_raw) or "red"

    grid = infer_pieces(grid_raw, side=side, crop=crop, row0=row0, col0=col0, cell=cell)
    if path in ("<auto>", "<memory>") or (isinstance(path, str) and "auto" in path):
        grid = _stabilize_types(grid, grid_raw)
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
    """在每枚棋子格角画中文兵种，颜色与棋子一致（红/黑）。

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

    # --- 兵种文字：格角标注，字色与棋子红/黑一致（不再画外包黑框）---
    vis = _draw_piece_labels(vis, a, half)

    # --- 合法性校验：非法棋子画红X ---
    for (r, c), _violations in a.violations.items():
        cx, cy = a.cell_center(r, c)
        cx, cy = int(cx), int(cy)
        cv2.line(vis, (cx - half, cy - half), (cx + half, cy + half),
                 (0, 0, 255), 3)
        cv2.line(vis, (cx + half, cy - half), (cx - half, cy + half),
                 (0, 0, 255), 3)

    # --- 识别棋盘外轮廓（半格外扩，便于目检格网是否贴合）---
    board_l = int(round(x0 + a.col0 - cell / 2))
    board_r = int(round(x0 + a.col0 + (COLS - 1) * cell + cell / 2))
    board_t = int(round(y0 + a.row0 - cell / 2))
    board_b = int(round(y0 + a.row0 + (ROWS - 1) * cell + cell / 2))
    # 青色粗框 = 棋盘轮廓；四角再画短角标，避免和网格线混淆
    cv2.rectangle(vis, (board_l, board_t), (board_r, board_b), (255, 220, 0), 3)
    corner = max(10, int(cell * 0.35))
    thick = 3
    for cx, cy, dx, dy in (
        (board_l, board_t, 1, 1),
        (board_r, board_t, -1, 1),
        (board_l, board_b, 1, -1),
        (board_r, board_b, -1, -1),
    ):
        cv2.line(vis, (cx, cy), (cx + dx * corner, cy), (255, 220, 0), thick)
        cv2.line(vis, (cx, cy), (cx, cy + dy * corner), (255, 220, 0), thick)

    if draw_grid:
        for r in range(ROWS + 1):
            y = int(round(y0 + a.row0 - cell / 2 + r * cell))
            cv2.line(vis, (board_l, y), (board_r, y), (90, 90, 90), 1)
        for c in range(COLS + 1):
            x = int(round(x0 + a.col0 - cell / 2 + c * cell))
            cv2.line(vis, (x, board_t), (x, board_b), (90, 90, 90), 1)
        # 楚河汉界（row4 / row5 之间）加一条稍亮的中线，便于核对行对齐
        river_y = int(round(y0 + a.row0 + 4.5 * cell))
        cv2.line(vis, (board_l, river_y), (board_r, river_y), (0, 180, 255), 2)

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
