"""单格兵种模板匹配（墨迹 NCC）。

模板目录（避开 macOS 大小写不敏感冲突）：
    engine/piece_templates/red/{R,N,B,A,K,C,P}_*.png
    engine/piece_templates/black/{r,n,b,a,k,c,p}_*.png

对外：
    extract_cell_ink(...)
    score_cell_types(ink, is_red) -> {ch: best_ncc}
    classify_cell_type(ink, is_red) -> (ch, score, all_scores)
    assign_board_types(...) -> 对残留 @/# 配额分配
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

_TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "piece_templates")
_TEMPLATE_SIZE = 40
_TEMPLATE_CACHE = None

_RED_CHARS = "RNBAKCP"
_BLK_CHARS = "rnbakcp"
_SHARED_PAIRS = (("R", "r"), ("N", "n"), ("C", "c"))  # 同字可跨色补漏

TYPE_QUOTA = {
    "K": 1, "A": 2, "B": 2, "N": 2, "R": 2, "C": 2, "P": 5,
    "k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5,
}

_NAME_RE = re.compile(r"^([RNBAKCPrnbakcp])(?:_(\d+))?\.png$", re.IGNORECASE)


def clear_template_cache():
    global _TEMPLATE_CACHE
    _TEMPLATE_CACHE = None


def _side_dir(is_red: bool) -> str:
    return os.path.join(_TEMPLATE_ROOT, "red" if is_red else "black")


def _normalize_ink(img: np.ndarray) -> np.ndarray:
    if img.shape != (_TEMPLATE_SIZE, _TEMPLATE_SIZE):
        img = cv2.resize(img, (_TEMPLATE_SIZE, _TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
    return (img > 80).astype(np.uint8) * 255


def load_templates():
    """返回 {ch: [ink, ...]}，多份模板取最高 NCC。"""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None:
        return _TEMPLATE_CACHE
    templates = {ch: [] for ch in _RED_CHARS + _BLK_CHARS}
    for is_red, chars in ((True, _RED_CHARS), (False, _BLK_CHARS)):
        d = _side_dir(is_red)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            m = _NAME_RE.match(name)
            if not m:
                continue
            ch = m.group(1)
            # 目录决定大小写
            ch = ch.upper() if is_red else ch.lower()
            if ch not in templates:
                continue
            path = os.path.join(d, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            templates[ch].append(_normalize_ink(img))
    # 兼容旧版扁平 piece_templates/{R}.png（仅红方大写，历史遗留）
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


def extract_cell_ink(crop, red_mask, black_mask, cx, cy, cell, is_red):
    """格心墨迹 → 40×40 二值图；失败返回 None。"""
    h, w = crop.shape[:2]
    rad = max(8, int(cell * 0.34))
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    patch = crop[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if is_red:
        ink = red_mask[y0:y1, x0:x1].astype(np.uint8) * 255
    else:
        thr = float(np.percentile(gray, 40))
        ink = ((gray < thr) | black_mask[y0:y1, x0:x1]).astype(np.uint8) * 255
    yy, xx = np.ogrid[:ink.shape[0], :ink.shape[1]]
    cyi, cxi = ink.shape[0] // 2, ink.shape[1] // 2
    rad2 = int(min(ink.shape) * 0.40)
    ink = np.where((yy - cyi) ** 2 + (xx - cxi) ** 2 <= rad2 ** 2, ink, 0).astype(np.uint8)
    ys, xs = np.where(ink > 0)
    if len(ys) < 30:
        return None
    roi = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return _normalize_ink(cv2.resize(roi, (_TEMPLATE_SIZE, _TEMPLATE_SIZE),
                                     interpolation=cv2.INTER_AREA))


def score_cell_types(ink, is_red, templates=None) -> dict:
    """{本色兵种字母: 最高 NCC}；車馬炮可回退对方同字模板。"""
    if templates is None:
        templates = load_templates()
    own = _RED_CHARS if is_red else _BLK_CHARS
    scores = {}
    for ch in own:
        best = -1.0
        for tmpl in templates.get(ch, []):
            best = max(best, ncc_best(ink, tmpl))
        if best >= 0:
            scores[ch] = best
    for red_ch, blk_ch in _SHARED_PAIRS:
        mine, other = (red_ch, blk_ch) if is_red else (blk_ch, red_ch)
        for tmpl in templates.get(other, []):
            scores[mine] = max(scores.get(mine, -1.0), ncc_best(ink, tmpl))
    return scores


def classify_cell_type(ink, is_red, templates=None):
    """单格 OCR：返回 (best_ch, best_score, all_scores)。ink 无效时 best_ch=None。"""
    if ink is None:
        return None, -1.0, {}
    scores = score_cell_types(ink, is_red, templates)
    if not scores:
        return None, -1.0, {}
    best_ch = max(scores, key=scores.get)
    return best_ch, scores[best_ch], scores


def save_template(ink, ch: str, index: int | None = None) -> str:
    """写入 red/ 或 black/ 下，返回路径。"""
    is_red = ch.isupper()
    d = _side_dir(is_red)
    os.makedirs(d, exist_ok=True)
    if index is None:
        existing = [f for f in os.listdir(d) if f.startswith(f"{ch}_") or f == f"{ch}.png"]
        index = len(existing)
    path = os.path.join(d, f"{ch}_{index}.png")
    cv2.imwrite(path, _normalize_ink(ink))
    clear_template_cache()
    return path


def position_type_bonus(r, c, ch, side, elephant_valid_fn):
    """模板分上叠少量位置先验（不单独决定兵种）。"""
    red_bottom = side == "red"
    is_red = ch.isupper()
    bonus = 0.0
    if ch in ("K", "k"):
        r0, r1 = ((7, 9) if (is_red == red_bottom) else (0, 2))
        bonus += 0.35 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.5
        if r0 <= r <= r1 and c == 4:
            bonus += 0.30  # 将/帅更常在九宫中线
    elif ch in ("A", "a"):
        r0, r1 = ((7, 9) if (is_red == red_bottom) else (0, 2))
        bonus += 0.25 if (r0 <= r <= r1 and c in (3, 4, 5)) else -0.35
        if r0 <= r <= r1 and c in (3, 5):
            bonus += 0.12  # 士位偏左右
    elif ch in ("B", "b"):
        valid = elephant_valid_fn(is_red, side)
        bonus += 0.35 if (r, c) in valid else -0.45
    elif ch in ("C", "c"):
        back = 9 if (is_red == red_bottom) else 0
        step = -1 if back == 9 else 1
        if r == back + 2 * step and c in (1, 7):
            bonus += 0.25
        else:
            # 非经典炮位抑制：残局马/车墨迹常被认成炮
            bonus -= 0.12
    elif ch in ("P", "p"):
        back = 9 if (is_red == red_bottom) else 0
        step = -1 if back == 9 else 1
        home = back + 3 * step
        if r == home and c in (0, 2, 4, 6, 8):
            bonus += 0.40
        crossed = (r > 4) if (back == 0) else (r < 5)
        if crossed and c in (0, 2, 4, 6, 8):
            bonus += 0.15
    elif ch in ("N", "n"):
        back = 9 if (is_red == red_bottom) else 0
        step = -1 if back == 9 else 1
        if r == back and c in (1, 7):
            bonus += 0.20
        # 开局跳马常见落点：底线马位的日字前进格
        if r == back + 2 * step and c in (0, 2, 6, 8):
            bonus += 0.30
    elif ch in ("R", "r"):
        back = 9 if (is_red == red_bottom) else 0
        if r == back and c in (0, 8):
            bonus += 0.20
        # 跳马落点上抑制「车」误抢（车马墨迹接近时）
        step = -1 if back == 9 else 1
        if r == back + 2 * step and c in (0, 2, 6, 8):
            bonus -= 0.25
    return bonus


def assign_by_scores(candidates, used_counts, side, elephant_valid_fn, min_raw=0.10):
    """candidates: [(r, c, scores_dict, is_red)] → {(r,c): ch}。"""
    assignments = {}
    opts = []
    for r, c, scores, _is_red in candidates:
        for ch, sc in scores.items():
            bonus = position_type_bonus(r, c, ch, side, elephant_valid_fn)
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


def ocr_fill_markers(grid, crop, row0, col0, cell, side, piece_masks_fn, elephant_valid_fn):
    """对 grid 中残留的 @/# 做单格 OCR 并写回字母。返回填充数量。"""
    templates = load_templates()
    if not any(templates.values()):
        return 0
    red_mask, black_mask, _ = piece_masks_fn(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    used = {}
    for row in grid:
        for ch in row:
            if ch not in (".", "@", "#"):
                used[ch] = used.get(ch, 0) + 1
    candidates = []
    rows, cols = len(grid), len(grid[0]) if grid else 0
    for r in range(rows):
        for c in range(cols):
            marker = grid[r][c]
            if marker not in ("@", "#"):
                continue
            is_red = marker == "@"
            cx = col0 + c * cell
            cy = row0 + r * cell
            ink = extract_cell_ink(crop, red_mask, black_mask, cx, cy, cell, is_red)
            if ink is None:
                continue
            scores = score_cell_types(ink, is_red, templates)
            if scores:
                candidates.append((r, c, scores, is_red))
    filled = 0
    for (r, c), ch in assign_by_scores(candidates, used, side, elephant_valid_fn).items():
        grid[r][c] = ch
        filled += 1
    return filled
