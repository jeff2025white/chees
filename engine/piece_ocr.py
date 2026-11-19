"""单格兵种模板匹配：墨迹 NCC + 彩色块匹配 + 位置先验 + 校验重判。

模板目录（避开 macOS 大小写不敏感冲突）：
    engine/piece_templates/{red,black}/{R,N,...}_*.png          墨迹
    engine/piece_templates/color/{red,black}/{R,N,...}_*.png    彩色 40×40 BGR

对外：
    extract_cell_ink / extract_cell_color
    score_cell_types(ink, color_bgr, is_red) -> {ch: score}
    ocr_fill_markers(...) -> 填充数
    repair_by_validation(...) -> 校验失败时按候选分重判
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

_TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "piece_templates")
_COLOR_ROOT = os.path.join(_TEMPLATE_ROOT, "color")
_TEMPLATE_SIZE = 40
_TEMPLATE_CACHE = None
_COLOR_CACHE = None

_RED_CHARS = "RNBAKCP"
_BLK_CHARS = "rnbakcp"
_SHARED_PAIRS = (("R", "r"), ("N", "n"), ("C", "c"))  # 同字可跨色补漏
_MAX_TEMPLATES_PER_TYPE = 8

# 墨迹 / 彩色融合权重（无彩色模板时自动退回纯墨迹）
_INK_WEIGHT = 0.55
_COLOR_WEIGHT = 0.45

TYPE_QUOTA = {
    "K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5,
    "k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5,
}

_NAME_RE = re.compile(r"^([RNBAKCPrnbakcp])(?:_(\d+))?\.png$", re.IGNORECASE)


def clear_template_cache():
    global _TEMPLATE_CACHE, _COLOR_CACHE
    _TEMPLATE_CACHE = None
    _COLOR_CACHE = None


def _side_dir(is_red: bool, color: bool = False) -> str:
    root = _COLOR_ROOT if color else _TEMPLATE_ROOT
    return os.path.join(root, "red" if is_red else "black")


def _normalize_ink(img: np.ndarray) -> np.ndarray:
    if img.shape != (_TEMPLATE_SIZE, _TEMPLATE_SIZE):
        img = cv2.resize(img, (_TEMPLATE_SIZE, _TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
    return (img > 80).astype(np.uint8) * 255


def _normalize_color(bgr: np.ndarray) -> np.ndarray:
    if bgr.shape[0] != _TEMPLATE_SIZE or bgr.shape[1] != _TEMPLATE_SIZE:
        bgr = cv2.resize(bgr, (_TEMPLATE_SIZE, _TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
    return bgr


def load_templates():
    """返回 {ch: [ink, ...]}，多份模板取最高 NCC。"""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE
    templates = {ch: [] for ch in _RED_CHARS + _BLK_CHARS}
    for is_red, chars in ((True, _RED_CHARS), (False, _BLK_CHARS)):
        d = _side_dir(is_red, color=False)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = _NAME_RE.match(name)
            if not m:
                continue
            ch = m.group(1)
            ch = ch.upper() if is_red else ch.lower()
            if ch not in templates:
                continue
            path = os.path.join(d, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            templates[ch].append(_normalize_ink(img))
    if os.path.isdir(_TEMPLATE_ROOT):
        for name in os.listdir(_TEMPLATE_ROOT):
            if not name.endswith(".png"):
                continue
            path = os.path.join(_TEMPLATE_ROOT, name)
            if not os.path.isfile(path):
                continue
            stem = name[:-4]
            if stem in templates and not templates[stem]:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[stem].append(_normalize_ink(img))
    _TEMPLATE_CACHE = templates
    return templates


def load_color_templates():
    """返回 {ch: [bgr40, ...]}；目录不存在则空。"""
    global _COLOR_CACHE
    if _COLOR_CACHE is not None:
        return _COLOR_CACHE
    templates = {ch: [] for ch in _RED_CHARS + _BLK_CHARS}
    for is_red, chars in ((True, _RED_CHARS), (False, _BLK_CHARS)):
        d = _side_dir(is_red, color=True)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = _NAME_RE.match(name)
            if not m:
                continue
            ch = m.group(1)
            ch = ch.upper() if is_red else ch.lower()
            if ch not in templates:
                continue
            path = os.path.join(d, name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            templates[ch].append(_normalize_color(img))
    _COLOR_CACHE = templates
    return templates


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.sqrt((af * af).sum()) * np.sqrt((bf * bf).sum()) + 1e-6)
    return float((af * bf).sum() / denom)


def ncc_best(ink: np.ndarray, tmpl: np.ndarray, max_shift: int = 2) -> float:
    """允许小范围平移后的最高 NCC，减轻格心/裁切偏差。"""
    if max_shift <= 0:
        return ncc(ink, tmpl)
    pad = int(max_shift)
    padded = cv2.copyMakeBorder(ink, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    res = cv2.matchTemplate(
        padded.astype(np.float32), tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED
    )
    return float(res.max())


def color_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """三通道分别去均值相关后取平均。"""
    scores = []
    for i in range(3):
        scores.append(ncc(a[:, :, i], b[:, :, i]))
    return float(sum(scores) / 3.0)


def color_ncc_best(patch: np.ndarray, tmpl: np.ndarray, max_shift: int = 1) -> float:
    """各通道 TM_CCOEFF_NORMED 后平均；允许小平移。"""
    if max_shift <= 0:
        return color_ncc(patch, tmpl)
    pad = int(max_shift)
    padded = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    scores = []
    for i in range(3):
        res = cv2.matchTemplate(
            padded[:, :, i].astype(np.float32),
            tmpl[:, :, i].astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        scores.append(float(res.max()))
    return float(sum(scores) / 3.0)


def extract_cell_ink(crop, red_mask, black_mask, cx, cy, cell, is_red):
    """格心墨迹 → 40×40 二值图；失败返回 None。

    固定格网下用 ``0.32×cell`` 半径裁切（略紧，减少邻格渗入），
    圆盘遮罩后按墨迹外接框**正方形补边**再缩放，避免扁框把笔画拉歪。
    """
    h, w = crop.shape[:2]
    rad = max(8, int(round(cell * 0.32)))
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    patch = crop[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if is_red:
        ink = red_mask[y0:y1, x0:x1].astype(np.uint8) * 255
        thr = float(np.percentile(gray, 35))
        ink = np.maximum(ink, ((gray < thr) & (gray < 140)).astype(np.uint8) * 255)
    else:
        thr = float(np.percentile(gray, 40))
        ink = ((gray < thr) | black_mask[y0:y1, x0:x1]).astype(np.uint8) * 255
    yy, xx = np.ogrid[:ink.shape[0], :ink.shape[1]]
    cyi, cxi = ink.shape[0] // 2, ink.shape[1] // 2
    rad2 = int(min(ink.shape) * 0.42)
    ink = np.where((yy - cyi) ** 2 + (xx - cxi) ** 2 <= rad2 ** 2, ink, 0).astype(np.uint8)
    ys, xs = np.where(ink > 0)
    if len(ys) < 30:
        return None
    roi = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bh, bw = roi.shape[:2]
    side = max(bh, bw, 8)
    sq = np.zeros((side, side), dtype=np.uint8)
    oy = (side - bh) // 2
    ox = (side - bw) // 2
    sq[oy:oy + bh, ox:ox + bw] = roi
    return _normalize_ink(cv2.resize(sq, (_TEMPLATE_SIZE, _TEMPLATE_SIZE),
                                     interpolation=cv2.INTER_AREA))


def extract_cell_color(crop, cx, cy, cell):
    """格心彩色块 → 40×40 BGR；失败返回 None。"""
    h, w = crop.shape[:2]
    rad = max(8, int(round(cell * 0.36)))
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    patch = crop[y0:y1, x0:x1].copy()
    # 圆盘：压掉邻格一角
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cyi, cxi = patch.shape[0] // 2, patch.shape[1] // 2
    rad2 = int(min(patch.shape[:2]) * 0.48)
    mask = (yy - cyi) ** 2 + (xx - cxi) ** 2 <= rad2 ** 2
    patch[~mask] = 0
    return _normalize_color(patch)


def score_cell_types(ink, is_red, templates=None, color_bgr=None, color_templates=None) -> dict:
    """{本色兵种字母: 融合分}；有彩色模板时墨迹+彩色加权。"""
    if templates is None:
        templates = load_templates()
    if color_templates is None:
        color_templates = load_color_templates()
    own = _RED_CHARS if is_red else _BLK_CHARS
    ink_scores = {}
    for ch in own:
        tmpls = templates.get(ch, [])[-_MAX_TEMPLATES_PER_TYPE:]
        best = -1.0
        for tmpl in tmpls:
            if ink is not None:
                best = max(best, ncc_best(ink, tmpl))
        if tmpls and best >= 0:
            ink_scores[ch] = best
    for red_ch, blk_ch in _SHARED_PAIRS:
        mine, other = (red_ch, blk_ch) if is_red else (blk_ch, red_ch)
        for tmpl in templates.get(other, [])[-_MAX_TEMPLATES_PER_TYPE:]:
            if ink is None:
                continue
            ink_scores[mine] = max(ink_scores.get(mine, -1.0), ncc_best(ink, tmpl))

    color_scores = {}
    use_color = (
        color_bgr is not None
        and any(color_templates.get(ch) for ch in own)
    )
    if use_color:
        for ch in own:
            tmpls = color_templates.get(ch, [])[-_MAX_TEMPLATES_PER_TYPE:]
            best = -1.0
            for tmpl in tmpls:
                best = max(best, color_ncc_best(color_bgr, tmpl))
            if tmpls and best >= 0:
                color_scores[ch] = best
        for red_ch, blk_ch in _SHARED_PAIRS:
            mine, other = (red_ch, blk_ch) if is_red else (blk_ch, red_ch)
            for tmpl in color_templates.get(other, [])[-_MAX_TEMPLATES_PER_TYPE:]:
                color_scores[mine] = max(
                    color_scores.get(mine, -1.0), color_ncc_best(color_bgr, tmpl))

    if not use_color or not color_scores:
        return ink_scores

    keys = set(ink_scores) | set(color_scores)
    fused = {}
    for ch in keys:
        ink_s = ink_scores.get(ch, -0.2)
        col_s = color_scores.get(ch, -0.2)
        fused[ch] = _INK_WEIGHT * ink_s + _COLOR_WEIGHT * col_s
    return fused


def classify_cell_type(ink, is_red, templates=None, color_bgr=None, color_templates=None):
    """单格 OCR：返回 (best_ch, best_score, all_scores)。"""
    scores = score_cell_types(ink, is_red, templates, color_bgr, color_templates)
    if not scores:
        return None, -1.0, {}
    best_ch = max(scores, key=scores.get)
    return best_ch, scores[best_ch], scores


def save_template(ink, ch: str, index: int | None = None) -> str:
    """写入 red/ 或 black/ 下墨迹模板，返回路径。"""
    is_red = ch.isupper()
    d = _side_dir(is_red, color=False)
    os.makedirs(d, exist_ok=True)
    if index is None:
        existing = [f for f in os.listdir(d) if f.startswith(f"{ch}_") or f == f"{ch}.png"]
        index = len(existing)
    path = os.path.join(d, f"{ch}_{index}.png")
    cv2.imwrite(path, _normalize_ink(ink))
    clear_template_cache()
    return path


def save_color_template(bgr: np.ndarray, ch: str, index: int | None = None) -> str:
    """写入 color/red|black/ 下彩色模板。"""
    is_red = ch.isupper()
    d = _side_dir(is_red, color=True)
    os.makedirs(d, exist_ok=True)
    if index is None:
        existing = [f for f in os.listdir(d) if f.startswith(f"{ch}_") or f == f"{ch}.png"]
        index = len(existing)
    path = os.path.join(d, f"{ch}_{index}.png")
    cv2.imwrite(path, _normalize_color(bgr))
    clear_template_cache()
    return path


def position_type_bonus(r, c, ch, side, elephant_valid_fn):
    """模板分上叠位置先验（不单独决定兵种；拉开马/车/炮易混分差）。"""
    red_bottom = side == "red"
    is_red = ch.isupper()
    back = 9 if (is_red == red_bottom) else 0
    step = -1 if back == 9 else 1
    horse_home = (r == back and c in (1, 7))
    horse_jump = (r == back + 2 * step and c in (0, 2, 6, 8))
    rook_home = (r == back and c in (0, 8))
    cannon_home = (r == back + 2 * step and c in (1, 7))
    palace = (r in ((7, 8, 9) if (is_red == red_bottom) else (0, 1, 2))) and c in (3, 4, 5)

    bonus = 0.0
    if ch in ("K", "k"):
        r0, r1 = ((7, 9) if (is_red == red_bottom) else (0, 2))
        bonus += 0.40 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.55
        if r0 <= r <= r1 and c == 4:
            bonus += 0.30
    elif ch in ("A", "a"):
        r0, r1 = ((7, 9) if (is_red == red_bottom) else (0, 2))
        bonus += 0.28 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.40
        if r0 <= r <= r1 and c in (3, 5):
            bonus += 0.14
    elif ch in ("B", "b"):
        valid = elephant_valid_fn(is_red, side)
        bonus += 0.38 if (r, c) in valid else -0.50
    elif ch in ("C", "c"):
        if cannon_home:
            bonus += 0.32
        elif c in (1, 7):
            bonus += 0.06
        else:
            bonus -= 0.22
        if palace:
            bonus -= 0.35
        if horse_jump:
            bonus -= 0.28
    elif ch in ("P", "p"):
        home = back + 3 * step
        if r == home and c in (0, 2, 4, 6, 8):
            bonus += 0.42
        crossed = (r > 4) if (back == 0) else (r < 5)
        if crossed and c in (0, 2, 4, 6, 8):
            bonus += 0.15
        if palace:
            bonus -= 0.25
    elif ch in ("N", "n"):
        if horse_home:
            bonus += 0.28
        if horse_jump:
            bonus += 0.42
        if cannon_home:
            bonus -= 0.18
        if palace:
            bonus -= 0.30
    elif ch in ("R", "r"):
        if rook_home:
            bonus += 0.28
        if horse_jump:
            bonus -= 0.38
        if cannon_home:
            bonus -= 0.15
        if palace:
            bonus -= 0.20
    return bonus


def _position_hard_ok(r, c, ch, side, elephant_valid_fn) -> bool:
    """硬位置约束：将/士九宫、象田字。"""
    is_red = ch.isupper()
    red_bottom = side == "red"
    if ch in ("K", "k", "A", "a"):
        rows = (7, 8, 9) if (is_red == red_bottom) else (0, 1, 2)
        return r in rows and c in (3, 4, 5)
    if ch in ("B", "b"):
        return (r, c) in elephant_valid_fn(is_red, side)
    return True


def assign_by_scores(candidates, used_counts, side, elephant_valid_fn, min_raw=0.10):
    """candidates: [(r, c, scores_dict, is_red)] → {(r,c): ch}。

    同分接近时放大位置先验权重，专治马/车/炮墨迹接近。
    """
    assignments = {}
    opts = []
    for r, c, scores, _is_red in candidates:
        if not scores:
            continue
        ranked = sorted(scores.values(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else -1.0
        close = (top - second) < 0.08
        for ch, sc in scores.items():
            if not _position_hard_ok(r, c, ch, side, elephant_valid_fn):
                # 仍允许候选，但重罚（校验重判时再换）
                bonus = position_type_bonus(r, c, ch, side, elephant_valid_fn) - 0.8
            else:
                bonus = position_type_bonus(r, c, ch, side, elephant_valid_fn)
            if close:
                bonus *= 1.6
            opts.append((sc + bonus, sc, r, c, ch))
    opts.sort(reverse=True)
    claimed = set()
    counts = dict(used_counts)
    for _total, raw, r, c, ch in opts:
        if (r, c) in claimed:
            continue
        if counts.get(ch, 0) >= TYPE_QUOTA.get(ch, 0):
            continue
        if raw < min_raw:
            continue
        assignments[(r, c)] = ch
        claimed.add((r, c))
        counts[ch] = counts.get(ch, 0) + 1
    return assignments


def repair_by_validation(grid, cell_scores, side, elephant_valid_fn, validate_fn, rounds: int = 4):
    """校验失败时：超配额 / 非法位置格按候选次优重判。

    cell_scores: {(r,c): {ch: score}}
    返回是否最终 valid。
    """
    for _ in range(rounds):
        valid, _messages, violations = validate_fn(grid, side)
        if valid:
            return True
        counts = {}
        for row in grid:
            for ch in row:
                if ch != ".":
                    counts[ch] = counts.get(ch, 0) + 1
        changed = False

        # 1) 超配额：把多余的换成次优合法兵种
        for ch, (_lo, hi) in TYPE_QUOTA.items():
            excess = counts.get(ch, 0) - hi
            if excess <= 0:
                continue
            cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
                     if grid[r][c] == ch and (r, c) in cell_scores]
            # 优先改：当前位置对 ch 先验差、且有好替代的

            def _swap_key(rc):
                r, c = rc
                scores = cell_scores[rc]
                cur = scores.get(ch, 0.0) + position_type_bonus(
                    r, c, ch, side, elephant_valid_fn)
                alts = []
                for alt, sc in scores.items():
                    if alt == ch:
                        continue
                    if counts.get(alt, 0) >= TYPE_QUOTA.get(alt, 0):
                        continue
                    if not _position_hard_ok(r, c, alt, side, elephant_valid_fn):
                        continue
                    tot = sc + position_type_bonus(r, c, alt, side, elephant_valid_fn)
                    alts.append(tot)
                best_alt = max(alts) if alts else -9.0
                return (cur - best_alt, cur)  # 越小越该换

            cells.sort(key=_swap_key)
            for r, c in cells:
                if excess <= 0:
                    break
                scores = cell_scores[(r, c)]
                ranked = sorted(
                    ((sc + position_type_bonus(r, c, alt, side, elephant_valid_fn), sc, alt)
                     for alt, sc in scores.items() if alt != ch),
                    reverse=True,
                )
                for _tot, _raw, alt in ranked:
                    if counts.get(alt, 0) >= TYPE_QUOTA.get(alt, 0):
                        continue
                    if not _position_hard_ok(r, c, alt, side, elephant_valid_fn):
                        continue
                    grid[r][c] = alt
                    counts[ch] = counts.get(ch, 1) - 1
                    counts[alt] = counts.get(alt, 0) + 1
                    excess -= 1
                    changed = True
                    break

        # 2) 非法位置：该格换次优
        for (r, c) in list(violations.keys()):
            ch = grid[r][c]
            scores = cell_scores.get((r, c))
            if not scores:
                continue
            ranked = sorted(
                ((sc + position_type_bonus(r, c, alt, side, elephant_valid_fn), alt)
                 for alt, sc in scores.items()),
                reverse=True,
            )
            for _tot, alt in ranked:
                if alt == ch:
                    continue
                if counts.get(alt, 0) >= TYPE_QUOTA.get(alt, 0):
                    continue
                if not _position_hard_ok(r, c, alt, side, elephant_valid_fn):
                    continue
                grid[r][c] = alt
                counts[ch] = counts.get(ch, 1) - 1
                counts[alt] = counts.get(alt, 0) + 1
                changed = True
                break

        if not changed:
            break
    valid, _, _ = validate_fn(grid, side)
    return valid


def ocr_fill_markers(grid, crop, row0, col0, cell, side, piece_masks_fn, elephant_valid_fn,
                     validate_fn=None):
    """对 grid 中残留的 @/# 做单格 OCR 并写回字母。

    返回 (填充数量, cell_scores)。若传入 validate_fn，填充后做校验重判。
    """
    templates = load_templates()
    color_templates = load_color_templates()
    if not any(templates.values()) and not any(color_templates.values()):
        return 0, {}
    red_mask, black_mask, _ = piece_masks_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    used = {}
    for row in grid:
        for ch in row:
            if ch not in (".", "@", "#"):
                used[ch] = used.get(ch, 0) + 1
    candidates = []
    cell_scores = {}
    rows, cols = len(grid), len(grid[0]) if grid else 0
    rad = max(6, int(cell * 0.22))
    chh, cw = crop.shape[:2]
    for r in range(rows):
        for c in range(cols):
            marker = grid[r][c]
            if marker not in ("@", "#"):
                continue
            is_red = marker == "@"
            cx = col0 + c * cell
            cy = row0 + r * cell
            y0, y1 = max(0, int(cy) - rad), min(chh, int(cy) + rad)
            x0, x1 = max(0, int(cx) - rad), min(cw, int(cx) + rad)
            if y1 - y0 >= 4 and x1 - x0 >= 4:
                rr = float(red_mask[y0:y1, x0:x1].mean())
                br = float(black_mask[y0:y1, x0:x1].mean())
                if rr < 0.015 and br < 0.015:
                    continue
            ink = extract_cell_ink(crop, red_mask, black_mask, cx, cy, cell, is_red)
            color_bgr = extract_cell_color(crop, cx, cy, cell)
            if ink is None and color_bgr is None:
                continue
            scores = score_cell_types(
                ink, is_red, templates, color_bgr=color_bgr, color_templates=color_templates)
            if scores:
                candidates.append((r, c, scores, is_red))
                cell_scores[(r, c)] = scores
    filled = 0
    for (r, c), ch in assign_by_scores(candidates, used, side, elephant_valid_fn).items():
        grid[r][c] = ch
        filled += 1
        used[ch] = used.get(ch, 0) + 1

    if validate_fn is not None and cell_scores:
        # 兜底 R/r 也没有分数；先把仍是标记的填上再修
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "@":
                    grid[r][c] = "R"
                elif grid[r][c] == "#":
                    grid[r][c] = "r"
        repair_by_validation(grid, cell_scores, side, elephant_valid_fn, validate_fn)
    return filled, cell_scores
