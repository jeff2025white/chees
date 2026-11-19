"""单格兵种识别：优先 Torch CNN，回退墨迹/彩色模板 NCC。

模板目录（避开 macOS 大小写不敏感冲突）：
    engine/piece_templates/{red,black}/{R,N,...}_*.png          墨迹
    engine/piece_templates/color/{red,black}/{R,N,...}_*.png    彩色 40×40 BGR
模型权重：
    engine/models/piece_cls.pt

对外：
    extract_cell_ink / extract_cell_color
    score_cell_types(ink, color_bgr, is_red) -> {ch: score}
    ocr_fill_markers(...) -> 填充数（Torch 优先）
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


# 高置信 OCR：先验不得翻盘
_OCR_LOCK_MARGIN = 0.10
_OCR_LOCK_MIN = 0.55
# 软先验缩放（硬非法仍用大惩罚）；中局勿让马位/兵位 bonus 压过模板分
_PRIOR_SCALE = 0.35
# OCR 与已分配冲突：top 比 assigned 高出该值则尝试纠正（valid 也做）
_OCR_REPAIR_MARGIN = 0.12


def position_type_bonus(r, c, ch, side, elephant_valid_fn):
    """模板分上叠位置先验（软暗示；硬约束见 ``_position_hard_ok``）。

    正 bonus 已整体偏弱，避免中局把车判成马、炮判成卒。
    负惩罚（出九宫将/士、象不在田字）保持较强。
    """
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
        bonus += 0.18 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.55
        if r0 <= r <= r1 and c == 4:
            bonus += 0.08
    elif ch in ("A", "a"):
        r0, r1 = ((7, 9) if (is_red == red_bottom) else (0, 2))
        bonus += 0.12 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.40
        if r0 <= r <= r1 and c in (3, 5):
            bonus += 0.06
    elif ch in ("B", "b"):
        valid = elephant_valid_fn(is_red, side)
        bonus += 0.16 if (r, c) in valid else -0.50
    elif ch in ("C", "c"):
        if cannon_home:
            bonus += 0.14
        elif c in (1, 7):
            bonus += 0.03
        else:
            bonus -= 0.12
        if palace:
            bonus -= 0.35
        if horse_jump:
            bonus -= 0.18
    elif ch in ("P", "p"):
        home = back + 3 * step
        if r == home and c in (0, 2, 4, 6, 8):
            bonus += 0.16
        crossed = (r > 4) if (back == 0) else (r < 5)
        if crossed and c in (0, 2, 4, 6, 8):
            bonus += 0.06
        if palace:
            bonus -= 0.25
    elif ch in ("N", "n"):
        if horse_home:
            bonus += 0.10
        if horse_jump:
            bonus += 0.16
        if cannon_home:
            bonus -= 0.12
        if palace:
            bonus -= 0.30
    elif ch in ("R", "r"):
        if rook_home:
            bonus += 0.12
        if horse_jump:
            bonus -= 0.22
        if cannon_home:
            bonus -= 0.10
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


def _ocr_top(scores):
    """返回 (best_ch, best_sc, second_sc)。"""
    if not scores:
        return None, -1.0, -1.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_ch, best_sc = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    return best_ch, best_sc, second


def _scaled_bonus(r, c, ch, side, elephant_valid_fn, *, hard_fail_penalty: float = 0.8):
    bonus = position_type_bonus(r, c, ch, side, elephant_valid_fn)
    if not _position_hard_ok(r, c, ch, side, elephant_valid_fn):
        return bonus * _PRIOR_SCALE - hard_fail_penalty
    # 负惩罚不缩放（出宫将/士仍要挡住）；正 bonus 缩小
    if bonus >= 0:
        return bonus * _PRIOR_SCALE
    return bonus


def assign_by_scores(candidates, used_counts, side, elephant_valid_fn, min_raw=0.10,
                     opening_prior: bool = False):
    """candidates: [(r, c, scores_dict, is_red)] → {(r,c): ch}。

    两阶段：
      1) 高置信 OCR（margin/分数够）先锁格，先验不得翻盘；
      2) 其余格用缩小后的位置先验做全局贪心。
    opening_prior=False（中局）时正家位 bonus 置 0，只保留出宫/出田硬罚。
    """
    assignments = {}
    claimed = set()
    counts = dict(used_counts)

    # --- 阶段 1：锁高置信 ---
    locks = []
    for r, c, scores, _is_red in candidates:
        if not scores:
            continue
        best_ch, best_sc, second = _ocr_top(scores)
        if best_ch is None or best_sc < max(min_raw, _OCR_LOCK_MIN):
            continue
        if best_sc - second < _OCR_LOCK_MARGIN:
            continue
        if not _position_hard_ok(r, c, best_ch, side, elephant_valid_fn):
            continue
        locks.append((best_sc, r, c, best_ch))
    locks.sort(reverse=True)
    for raw, r, c, ch in locks:
        if (r, c) in claimed:
            continue
        if counts.get(ch, 0) >= TYPE_QUOTA.get(ch, 0):
            continue
        assignments[(r, c)] = ch
        claimed.add((r, c))
        counts[ch] = counts.get(ch, 0) + 1

    # --- 阶段 2：剩余格，弱先验 ---
    opts = []
    for r, c, scores, _is_red in candidates:
        if not scores or (r, c) in claimed:
            continue
        ranked = sorted(scores.values(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else -1.0
        close = (top - second) < 0.08
        for ch, sc in scores.items():
            bonus = _scaled_bonus(r, c, ch, side, elephant_valid_fn)
            if not opening_prior and bonus > 0:
                bonus = 0.0
            if close and opening_prior:
                bonus *= 1.15
            opts.append((sc + bonus, sc, r, c, ch))
    opts.sort(reverse=True)
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


def _recount(grid):
    counts = {}
    for row in grid:
        for ch in row:
            if ch != ".":
                counts[ch] = counts.get(ch, 0) + 1
    return counts


def repair_by_ocr_conflict(grid, cell_scores, side, elephant_valid_fn, rounds: int = 3):
    """valid 局面也修：高置信 OCR 与分配冲突时纠正，必要时与弱置信同兵种格对换。

    典型：马位先验把车锁成马占配额 → 真马只能改成兵；本函数把配额还给 OCR 更像马的格。
    """
    for _ in range(rounds):
        counts = _recount(grid)
        changed = False

        conflicts = []
        for (r, c), scores in cell_scores.items():
            ch = grid[r][c]
            if ch in (".", "@", "#") or not scores:
                continue
            best_ch, best_sc, _second = _ocr_top(scores)
            if best_ch is None or best_ch == ch:
                continue
            cur_sc = scores.get(ch, -1.0)
            if best_sc - cur_sc < _OCR_REPAIR_MARGIN:
                continue
            if not _position_hard_ok(r, c, best_ch, side, elephant_valid_fn):
                continue
            conflicts.append((best_sc - cur_sc, best_sc, r, c, ch, best_ch))
        conflicts.sort(reverse=True)

        for _gain, best_sc, r, c, cur, want in conflicts:
            if grid[r][c] != cur:
                continue
            if counts.get(want, 0) < TYPE_QUOTA.get(want, 0):
                grid[r][c] = want
                counts = _recount(grid)
                changed = True
                continue

            # 配额满：找 OCR 对 want 更弱的占位格，对换或改次优
            donors = []
            for rr in range(len(grid)):
                for cc in range(len(grid[0])):
                    if grid[rr][cc] != want or (rr, cc) == (r, c):
                        continue
                    other = cell_scores.get((rr, cc))
                    donors.append((other.get(want, -1.0) if other else -1.0, rr, cc))
            donors.sort()
            for donor_sc, rr, cc in donors:
                if donor_sc >= best_sc - 0.02:
                    continue
                donor_scores = cell_scores.get((rr, cc), {})
                if _position_hard_ok(rr, cc, cur, side, elephant_valid_fn):
                    grid[r][c] = want
                    grid[rr][cc] = cur
                    counts = _recount(grid)
                    changed = True
                    break
                moved = False
                for alt, _sc in sorted(donor_scores.items(), key=lambda kv: -kv[1]):
                    if alt == want:
                        continue
                    # 释放 want：本格变 want，donor 变 alt；cur 少 1、alt 多 1
                    if alt != cur and counts.get(alt, 0) >= TYPE_QUOTA.get(alt, 0):
                        continue
                    if not _position_hard_ok(rr, cc, alt, side, elephant_valid_fn):
                        continue
                    grid[r][c] = want
                    grid[rr][cc] = alt
                    counts = _recount(grid)
                    changed = True
                    moved = True
                    break
                if moved:
                    break

        if not changed:
            break
    return grid


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

        # 1) 超配额：优先改掉「OCR 最不像该兵种」的格
        for ch, hi in TYPE_QUOTA.items():
            excess = counts.get(ch, 0) - hi
            if excess <= 0:
                continue
            cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
                     if grid[r][c] == ch and (r, c) in cell_scores]

            def _swap_key(rc):
                r, c = rc
                scores = cell_scores[rc]
                cur = scores.get(ch, 0.0) + _scaled_bonus(r, c, ch, side, elephant_valid_fn)
                alts = []
                for alt, sc in scores.items():
                    if alt == ch:
                        continue
                    if counts.get(alt, 0) >= TYPE_QUOTA.get(alt, 0):
                        continue
                    if not _position_hard_ok(r, c, alt, side, elephant_valid_fn):
                        continue
                    tot = sc + _scaled_bonus(r, c, alt, side, elephant_valid_fn)
                    alts.append(tot)
                best_alt = max(alts) if alts else -9.0
                return (cur - best_alt, cur)  # 越小越该换

            cells.sort(key=_swap_key)
            for r, c in cells:
                if excess <= 0:
                    break
                scores = cell_scores[(r, c)]
                ranked = sorted(
                    ((sc + _scaled_bonus(r, c, alt, side, elephant_valid_fn), sc, alt)
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
                ((sc + _scaled_bonus(r, c, alt, side, elephant_valid_fn), alt)
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


def _refine_nr_with_templates(candidates, crop, row0, col0, cell, piece_masks_fn):
    """车/马易混：CNN 顶为马且模板更像车时，抬高车分压低马分。"""
    templates = load_templates()
    color_templates = load_color_templates()
    if not any(templates.values()) and not any(color_templates.values()):
        return candidates
    red_mask, black_mask, _ = piece_masks_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    out = []
    for r, c, scores, is_red in candidates:
        if not scores:
            out.append((r, c, scores, is_red))
            continue
        n_ch = "N" if is_red else "n"
        r_ch = "R" if is_red else "r"
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_ch, top_sc = ranked[0]
        second_sc = ranked[1][1] if len(ranked) > 1 else -1.0
        # 仅当 CNN 认马，或车马分接近时介入
        nr_close = (
            top_ch in (n_ch, r_ch)
            and scores.get(n_ch, -1) >= 0
            and scores.get(r_ch, -1) >= 0
            and abs(scores.get(n_ch, 0) - scores.get(r_ch, 0)) < 0.35
        )
        if top_ch != n_ch and not nr_close:
            out.append((r, c, scores, is_red))
            continue
        cx = col0 + c * cell
        cy = row0 + r * cell
        ink = extract_cell_ink(crop, red_mask, black_mask, cx, cy, cell, is_red)
        color_bgr = extract_cell_color(crop, cx, cy, cell)
        tmpl = score_cell_types(
            ink, is_red, templates, color_bgr=color_bgr, color_templates=color_templates)
        if not tmpl:
            out.append((r, c, scores, is_red))
            continue
        tr = tmpl.get(r_ch, -1.0)
        tn = tmpl.get(n_ch, -1.0)
        scores = dict(scores)
        if tr >= tn + 0.04:
            # 模板更像车：保证车分高于马
            scores[r_ch] = max(scores.get(r_ch, 0.0), scores.get(n_ch, 0.0) + 0.20)
            scores[n_ch] = min(scores.get(n_ch, 0.0), scores[r_ch] - 0.12)
        elif tn >= tr + 0.12 and top_ch == r_ch and top_sc - second_sc < 0.15:
            # 弱车、模板强马：略抬马（少见，防误伤）
            scores[n_ch] = max(scores.get(n_ch, 0.0), scores.get(r_ch, 0.0) + 0.05)
        out.append((r, c, scores, is_red))
    return out


def _score_markers_torch(grid, crop, row0, col0, cell, piece_masks_fn):
    """优先用 Torch CNN 给残留 @/# 打分；不可用则返回 None。"""
    from . import piece_model

    if piece_model.load_piece_model() is None:
        return None
    red_mask, black_mask, _ = piece_masks_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    rows, cols = len(grid), len(grid[0]) if grid else 0
    rad = max(6, int(cell * 0.22))
    chh, cw = crop.shape[:2]
    coords = []
    patches = []
    flags = []
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
            color_bgr = extract_cell_color(crop, cx, cy, cell)
            if color_bgr is None:
                continue
            coords.append((r, c, is_red))
            patches.append(color_bgr)
            flags.append(is_red)
    if not patches:
        return [], {}
    score_list = piece_model.predict_batch(patches, flags)
    candidates = []
    for (r, c, is_red), scores in zip(coords, score_list):
        if scores:
            candidates.append((r, c, scores, is_red))
    candidates = _refine_nr_with_templates(
        candidates, crop, row0, col0, cell, piece_masks_fn)
    cell_scores = {(r, c): scores for r, c, scores, _ in candidates}
    return candidates, cell_scores


def _score_markers_templates(grid, crop, row0, col0, cell, piece_masks_fn):
    """模板 NCC 回退路径。"""
    templates = load_templates()
    color_templates = load_color_templates()
    if not any(templates.values()) and not any(color_templates.values()):
        return [], {}
    red_mask, black_mask, _ = piece_masks_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
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
    return candidates, cell_scores


def _fill_unassigned_best(grid, cell_scores, used, side, elephant_valid_fn):
    """残留 @/#：按候选分填合法次优，禁止无脑兜成车。"""
    rows, cols = len(grid), len(grid[0]) if grid else 0
    for r in range(rows):
        for c in range(cols):
            marker = grid[r][c]
            if marker not in ("@", "#"):
                continue
            scores = cell_scores.get((r, c))
            if not scores:
                # 无分数时宁可不猜车：用兵/卒作弱默认（配额允许）
                weak = "P" if marker == "@" else "p"
                if used.get(weak, 0) < TYPE_QUOTA.get(weak, 0):
                    grid[r][c] = weak
                    used[weak] = used.get(weak, 0) + 1
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            for ch, _sc in ranked:
                if used.get(ch, 0) >= TYPE_QUOTA.get(ch, 0):
                    continue
                if not _position_hard_ok(r, c, ch, side, elephant_valid_fn):
                    continue
                grid[r][c] = ch
                used[ch] = used.get(ch, 0) + 1
                break


def ocr_fill_markers(grid, crop, row0, col0, cell, side, piece_masks_fn, elephant_valid_fn,
                     validate_fn=None):
    """对 grid 中残留的 @/# 做单格 OCR 并写回字母。

    优先 Torch CNN；不可用则模板 NCC。返回 (填充数量, cell_scores)。
    """
    used = {}
    for row in grid:
        for ch in row:
            if ch not in (".", "@", "#"):
                used[ch] = used.get(ch, 0) + 1

    torch_pack = _score_markers_torch(grid, crop, row0, col0, cell, piece_masks_fn)
    if torch_pack is not None:
        candidates, cell_scores = torch_pack
    else:
        candidates, cell_scores = _score_markers_templates(
            grid, crop, row0, col0, cell, piece_masks_fn)
    if not candidates and not cell_scores:
        return 0, {}

    # Torch 概率锁阈值略降；模板路径沿用原阈值
    min_raw = 0.08 if torch_pack is not None else 0.10
    occ = sum(1 for row in grid for ch in row if ch != ".")
    letters = sum(1 for row in grid for ch in row if ch not in (".", "@", "#"))
    # 已种子较多或占位近满 → 开局先验；否则中局只靠模型分
    opening_prior = (letters >= 20) or (occ >= 28 and letters >= 10)
    filled = 0
    for (r, c), ch in assign_by_scores(
            candidates, used, side, elephant_valid_fn,
            min_raw=min_raw, opening_prior=opening_prior).items():
        grid[r][c] = ch
        filled += 1
        used[ch] = used.get(ch, 0) + 1

    if cell_scores:
        _fill_unassigned_best(grid, cell_scores, used, side, elephant_valid_fn)
        repair_by_ocr_conflict(grid, cell_scores, side, elephant_valid_fn)
        if validate_fn is not None:
            repair_by_validation(grid, cell_scores, side, elephant_valid_fn, validate_fn)
    return filled, cell_scores
