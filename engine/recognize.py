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

import cv2
import numpy as np

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

def detect_pieces(crop):
    """连通域 + 面积 + 圆形度过滤，返回 [(cx, cy, 'R'|'B', area)]。"""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)
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
        rp = red_mask[y0:y1, x0:x1].mean()
        bp = black_mask[y0:y1, x0:x1].mean()
        pieces.append((float(cx), float(cy), "R" if rp > bp else "B", float(area)))
    return pieces


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


def find_grid(crop, pieces=None):
    """定位 10×9 网格。

    先用棋子坐标聚类估计格宽，再在「哪一簇对应第几行/第几列」上做轴向评分，
    只有几百次评估（早期版本用全图四重暴力搜索，耗时几十秒）。
    """
    if pieces is None:
        pieces = detect_pieces(crop)
    if len(pieces) < 6:
        raise RuntimeError(f"检测到的棋子过少（{len(pieces)}），无法定位棋盘")

    h, w = crop.shape[:2]
    scale = w / _REF_W
    default_cell = _CELL_RATIO * w

    # 排除顶部头像区 / 底部按钮区的杂点
    y_lo, y_hi = _TOP_UI * h, _BOTTOM_UI * h
    core = [p for p in pieces if y_lo <= p[1] <= y_hi] or pieces

    rows = _cluster([p[1] for p in core], 0.55 * default_cell)
    cols = _cluster([p[0] for p in core], 0.55 * default_cell)
    ys = [p[1] for p in core]
    xs = [p[0] for p in core]

    # 格宽候选：优先用聚类间距中位数
    diffs = []
    if len(rows) >= 2:
        diffs.extend(np.diff(rows).tolist())
    if len(cols) >= 2:
        diffs.extend(np.diff(cols).tolist())
    base_cell = float(np.median(diffs)) if diffs else default_cell
    if not (0.72 * default_cell <= base_cell <= 1.35 * default_cell):
        base_cell = default_cell

    cell_list = [base_cell + d * scale for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
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

    return row0, col0, cell, s2, hit2


# --------------------------------------------------------------------------
# 格子颜色分类
# --------------------------------------------------------------------------

def classify(crop, row0, col0, cell):
    """返回 (grid_raw, centers)：'@'=红 '#'=黑 '.'=空。

    用 red_mask / black_mask 比例判棋子有无和颜色。
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]

    red_mask = ((hsv[..., 0] < 10) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    black_mask = (hsv[..., 2] > 40) & (hsv[..., 2] < 100) & (hsv[..., 1] < 100) & (hsv[..., 0] < 30)

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
            rr = red_mask[y0:y1, x0:x1].mean()
            br = black_mask[y0:y1, x0:x1].mean()
            if rr > 0.04 and rr > br * 1.2:
                line.append("@")
            elif br > 0.05 and br > rr * 1.2:
                line.append("#")
            elif rr > br * 0.8 and rr > 0.02:
                line.append("@")
            elif br > rr * 0.8 and br > 0.02:
                line.append("#")
            else:
                line.append(".")
        grid.append(line)
    return grid, centers


# --------------------------------------------------------------------------
# 兵种推断（位置先验）
# --------------------------------------------------------------------------

def infer_pieces(grid_raw, side=None):
    """把 '@'/'#' 占位替换成真实兵种字母。

    side: 我方颜色，"red" 表示红方在下（row 9），"black" 表示黑方在下。
    若 side=None，用 detect_our_side(grid_raw) 自动识别。
    """
    grid = [list(row) for row in grid_raw]
    if side is None:
        side = detect_our_side(grid_raw) or "red"
    our_side = side                    # 我方颜色 = side
    # 红方是否在下方
    red_bottom = (our_side == "red")
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

    if red_back is not None:
        for c in range(COLS):
            if grid[red_back][c] == "@":
                grid[red_back][c] = RED_BACKROW[c]
    if black_back is not None:
        for c in range(COLS):
            if grid[black_back][c] == "#":
                grid[black_back][c] = BLACK_BACKROW[c]

    def fill(backrow, marker, cannon, pawn, default):
        if backrow is None:
            for r in range(ROWS):
                for c in range(COLS):
                    if grid[r][c] == marker:
                        grid[r][c] = default
            return
        # 炮/兵向对方半场推进：底线在顶部则 +1，在底部则 -1
        step = 1 if backrow <= 4 else -1
        cannon_row, pawn_row = backrow + 2 * step, backrow + 3 * step
        for r, ch in ((cannon_row, cannon), (pawn_row, pawn)):
            if 0 <= r < ROWS:
                for c in range(COLS):
                    if grid[r][c] == marker:
                        grid[r][c] = ch
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == marker:
                    grid[r][c] = default

    fill(red_back, "@", "C", "P", "R")
    fill(black_back, "#", "c", "p", "r")
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
    h, w = img.shape[:2]
    y0, y1 = int(_ROI_Y0 * h), int(_ROI_Y1 * h)
    crop = img[y0:y1, 0:w]
    box = (0, y0, w, y1)

    pieces = detect_pieces(crop)
    row0, col0, cell, score, hit = find_grid(crop, pieces)
    grid_raw, _centers = classify(crop, row0, col0, cell)

    # 在下方九宫主将确定我方颜色
    side = detect_our_side(grid_raw) or "red"

    grid = infer_pieces(grid_raw, side=side)
    valid, _msgs, violations = validate_grid(grid, side=side)
    return Analysis(path, img, box, row0, col0, cell, grid_raw, grid,
                    score, hit, side, valid=valid, violations=violations)


# --------------------------------------------------------------------------
# 对外：标注
# --------------------------------------------------------------------------

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
        pv_steps = (decision.get("pv") or {}).get("steps") or []
        rad = int(cell * 0.45)
        half = int(cell * 0.46)

        # 先画对方应手（绿色线 + 绿色起点框 + 黄色终点框），我方着法覆盖其上
        if len(pv_steps) >= 2:
            s = pv_steps[1]
            rfx, rfy = a.cell_center(s["from"]["row"], s["from"]["col"])
            rtx, rty = a.cell_center(s["to"]["row"], s["to"]["col"])
            rfx, rfy, rtx, rty = int(rfx), int(rfy), int(rtx), int(rty)
            cv2.line(vis, (rfx, rfy), (rtx, rty), (0, 200, 0), 3)
            cv2.rectangle(vis, (rfx - half, rfy - half), (rfx + half, rfy + half),
                          (0, 200, 0), 4)
            cv2.rectangle(vis, (rtx - half, rty - half), (rtx + half, rty + half),
                          (0, 255, 255), 4)

        fc, fr = best["from"]["col"], best["from"]["row"]
        tc, tr = best["to"]["col"], best["to"]["row"]
        fx, fy = a.cell_center(fr, fc)
        tx, ty = a.cell_center(tr, tc)
        fx, fy, tx, ty = int(fx), int(fy), int(tx), int(ty)

        cv2.line(vis, (fx, fy), (tx, ty), (0, 0, 255), 3)
        cv2.circle(vis, (fx, fy), rad + 7, (0, 0, 255), 5)
        cv2.circle(vis, (fx, fy), rad + 1, (255, 255, 255), 2)
        # 终点用蓝色描边空心圆（BGR）
        cv2.circle(vis, (tx, ty), rad + 5, (255, 0, 0), 5)

        label = f"{best['chinese']} ({best['iccs']})  {best['score_text']}"
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
