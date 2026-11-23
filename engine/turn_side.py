"""识别当前行棋方（我方 / 对方）。

两种办法（任一确定即采用；都不确定则兜底「我方」）：

1. 头像绿边：首次定位双方头像正方形并缓存；之后只取两边框环带的绿色像素，
   绿多的一方为行棋方
2. 局面差分：相对上一帧，若可判定「刚走完的一方」，则当前轮到另一方

返回的 to_move 只会是 ``\"our\"`` / ``\"opp\"``，不会是 None。

开关：``TURN_DETECT_ENABLED = False`` 时关闭判定，一律视为我方行棋、不因对方跳过计算。
界面「判断行棋方」勾选会读写该开关（偏好持久化）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 行棋方识别（头像绿边 / 局面差分 / 对方跳过计算）
TURN_DETECT_ENABLED = True

# 头像搜索区（相对整图比例）；对方左上，我方左下/右下二选一
_OPP_SEARCH = (0.0, 0.02, 0.30, 0.22)
_OUR_SEARCHES = (
    (0.0, 0.78, 0.30, 0.99),   # 左下
    (0.65, 0.78, 1.0, 0.995),  # 右下
)

# 头像边长相对图像宽度
_AVATAR_MIN_SIDE = 0.07
_AVATAR_MAX_SIDE = 0.16
# 取边框环带时相对边长的厚度；外扩几像素以包住绿光晕
_BORDER_RING_FRAC = 0.12
_BORDER_PAD_PX = 6
_GREEN_HSV_LO = (35, 60, 80)
_GREEN_HSV_HI = (100, 255, 255)
# 两边绿量分差过小则视为无法确定（相对 + 绝对）
_GREEN_MIN_DIFF = 40
_GREEN_MIN_RATIO = 1.25


Box = Tuple[float, float, float, float]  # 相对比例 (x0,y0,x1,y1)
AvatarBoxes = Dict[str, Box]


def _clamp_box(box: Box) -> Box:
    x0, y0, x1, y1 = box
    x0 = float(min(max(x0, 0.0), 1.0))
    y0 = float(min(max(y0, 0.0), 1.0))
    x1 = float(min(max(x1, 0.0), 1.0))
    y1 = float(min(max(y1, 0.0), 1.0))
    if x1 <= x0 + 1e-6 or y1 <= y0 + 1e-6:
        return box
    return (x0, y0, x1, y1)


def _box_to_px(img, box: Box) -> Tuple[int, int, int, int]:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _clamp_box(box)
    return (
        int(w * x0),
        int(h * y0),
        max(int(w * x0) + 1, int(w * x1)),
        max(int(h * y0) + 1, int(h * y1)),
    )


def _px_to_box(img, px: Tuple[int, int, int, int]) -> Box:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = px
    return _clamp_box((x0 / w, y0 / h, x1 / w, y1 / h))


def _find_square_candidates(
    img,
    search: Box,
    *,
    min_side_frac: float = _AVATAR_MIN_SIDE,
    max_side_frac: float = _AVATAR_MAX_SIDE,
) -> List[dict]:
    """在搜索区内找近似正方形轮廓（头像外框）。"""
    h, w = img.shape[:2]
    sx0, sy0, sx1, sy1 = _box_to_px(img, search)
    roi = img[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_s = int(w * min_side_frac)
    max_s = int(w * max_side_frac)
    out: List[dict] = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        side = max(bw, bh)
        if side < min_s or side > max_s:
            continue
        ar = bw / max(bh, 1)
        if ar < 0.75 or ar > 1.30:
            continue
        area = cv2.contourArea(c)
        if area < 0.15 * bw * bh:
            continue
        # 周界边缘密度
        t = max(2, int(min(bw, bh) * 0.12))
        pad = 2
        rx0, ry0 = max(0, x - pad), max(0, y - pad)
        rx1 = min(roi.shape[1], x + bw + pad)
        ry1 = min(roi.shape[0], y + bh + pad)
        sub = edges[ry0:ry1, rx0:rx1]
        ring = np.zeros_like(sub)
        ring[:t, :] = 1
        ring[-t:, :] = 1
        ring[:, :t] = 1
        ring[:, -t:] = 1
        edge_score = float(sub[ring > 0].mean()) if ring.any() else 0.0
        # 略偏爱更接近正方形的框
        square_bonus = 1.0 - min(abs(1.0 - ar), 0.3)
        score = edge_score * (0.7 + 0.3 * square_bonus)
        px = (sx0 + x, sy0 + y, sx0 + x + bw, sy0 + y + bh)
        out.append({
            "score": score,
            "px": px,
            "side": side,
            "ar": round(ar, 3),
            "box": _px_to_box(img, px),
        })
    out.sort(key=lambda d: -d["score"])
    return out


def _pick_best(
    cands: List[dict],
    *,
    ref_side: Optional[int] = None,
) -> Optional[dict]:
    if not cands:
        return None
    if ref_side is None:
        return cands[0]
    # 优先边长接近参考头像，再看边缘分
    scored = []
    for c in cands:
        ratio = c["side"] / max(ref_side, 1)
        if ratio < 0.70 or ratio > 1.45:
            continue
        size_fit = 1.0 - min(abs(1.0 - ratio), 0.45)
        scored.append((c["score"] * (0.55 + 0.45 * size_fit), c))
    if not scored:
        return cands[0]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def _locate_avatar_boxes(img) -> Tuple[Optional[AvatarBoxes], dict]:
    detail: dict = {"method": "locate"}
    if img is None or getattr(img, "size", 0) == 0:
        detail["reason"] = "empty"
        return None, detail

    opp_cands = _find_square_candidates(img, _OPP_SEARCH)
    opp = _pick_best(opp_cands)
    detail["opp_candidates"] = len(opp_cands)
    if opp is None:
        detail["reason"] = "no_opp"
        return None, detail

    our_cands: List[dict] = []
    for search in _OUR_SEARCHES:
        our_cands.extend(_find_square_candidates(img, search))
    our = _pick_best(our_cands, ref_side=opp["side"])
    detail["our_candidates"] = len(our_cands)
    if our is None:
        detail["reason"] = "no_our"
        detail["opp"] = {
            "box": opp["box"], "side": opp["side"], "score": round(opp["score"], 1),
        }
        return None, detail

    boxes: AvatarBoxes = {"opp": opp["box"], "our": our["box"]}
    detail.update({
        "opp": {"box": opp["box"], "side": opp["side"], "score": round(opp["score"], 1)},
        "our": {"box": our["box"], "side": our["side"], "score": round(our["score"], 1)},
    })
    return boxes, detail


def locate_avatar_boxes(img) -> Optional[AvatarBoxes]:
    """首次识别：在对方/我方搜索区定位头像正方形，返回相对比例框。"""
    boxes, _ = _locate_avatar_boxes(img)
    return boxes


def border_green_count(img, box: Box) -> dict:
    """统计头像框环带上的绿色像素数（外扩 pad 以包住绿光晕）。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _box_to_px(img, box)
    pad = _BORDER_PAD_PX
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    roi = img[y0:y1, x0:x1]
    empty = {"count": 0, "ring_px": 0, "pct": 0.0, "box_px": (x0, y0, x1, y1)}
    if roi.size == 0:
        return empty

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(_GREEN_HSV_LO, dtype=np.uint8),
        np.array(_GREEN_HSV_HI, dtype=np.uint8),
    )
    rh, rw = green.shape[:2]
    t = max(2, int(min(rh, rw) * _BORDER_RING_FRAC))
    ring = np.zeros((rh, rw), dtype=np.uint8)
    ring[:t, :] = 1
    ring[-t:, :] = 1
    ring[:, :t] = 1
    ring[:, -t:] = 1
    ring_px = int(ring.sum())
    count = int(((green > 0) & (ring > 0)).sum())
    pct = count / max(ring_px, 1)
    return {
        "count": count,
        "ring_px": ring_px,
        "pct": round(pct, 4),
        "box_px": (x0, y0, x1, y1),
    }


def detect_avatar_turn(
    img,
    avatar_boxes: Optional[AvatarBoxes] = None,
) -> Tuple[Optional[str], dict, Optional[AvatarBoxes]]:
    """从头像边框绿量判断行棋方。

    Parameters
    ----------
    avatar_boxes
        已缓存的双方头像框；缺省则先 ``locate`` 再比较。

    Returns
    -------
    (side, detail, boxes)
        side: ``\"our\"`` / ``\"opp\"`` / None
        boxes: 本次使用的头像框（可供上层缓存）
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None, {"reason": "empty"}, avatar_boxes

    locate_detail = None
    boxes = avatar_boxes
    if boxes is None or "opp" not in boxes or "our" not in boxes:
        boxes, locate_detail = _locate_avatar_boxes(img)
        if boxes is None:
            detail = {"method": "avatar", "locate": locate_detail, "reason": "locate_failed"}
            return None, detail, None

    our_g = border_green_count(img, boxes["our"])
    opp_g = border_green_count(img, boxes["opp"])
    detail = {
        "method": "avatar",
        "boxes": boxes,
        "our": our_g,
        "opp": opp_g,
        "cached": avatar_boxes is not None,
    }
    if locate_detail is not None:
        detail["locate"] = locate_detail

    our_n = our_g["count"]
    opp_n = opp_g["count"]
    detail["our_green"] = our_n
    detail["opp_green"] = opp_n

    if our_n == 0 and opp_n == 0:
        detail["reason"] = "no_green"
        return None, detail, boxes

    if our_n >= opp_n:
        leader, loser, side = our_n, opp_n, "our"
    else:
        leader, loser, side = opp_n, our_n, "opp"

    diff = leader - loser
    ratio = leader / max(loser, 1)
    detail["diff"] = diff
    detail["ratio"] = round(ratio, 3)
    if diff < _GREEN_MIN_DIFF and ratio < _GREEN_MIN_RATIO:
        detail["reason"] = "tie"
        return None, detail, boxes
    return side, detail, boxes


def _is_our_piece(ch: str, view_side: str) -> bool:
    if not ch or ch == ".":
        return False
    if view_side == "red":
        return ch.isupper()
    return ch.islower()


def _is_opp_piece(ch: str, view_side: str) -> bool:
    if not ch or ch == ".":
        return False
    return not _is_our_piece(ch, view_side)


def _side_lost_gained(
    prev: Sequence[Sequence[str]],
    curr: Sequence[Sequence[str]],
    is_side,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    lost, gained = [], []
    rows = min(len(prev), len(curr))
    for r in range(rows):
        cols = min(len(prev[r]), len(curr[r]))
        for c in range(cols):
            p, n = prev[r][c], curr[r][c]
            p_s, n_s = is_side(p), is_side(n)
            if p_s and not n_s:
                lost.append((r, c))
            if n_s and not p_s:
                gained.append((r, c))
    return lost, gained


def infer_turn_from_grid_diff(
    prev_grid: Optional[Sequence[Sequence[str]]],
    curr_grid: Sequence[Sequence[str]],
    view_side: str,
) -> Tuple[Optional[str], dict]:
    """由上一局面差分推断当前行棋方。

    若判定「我方刚走完」（我方一子迁移，对方至多少一子）→ 当前对方走；
    若判定「对方刚走完」→ 当前我方走；否则无法确定。
    """
    detail = {"method": "grid_diff", "view_side": view_side}
    if prev_grid is None or curr_grid is None:
        detail["reason"] = "no_prev"
        return None, detail

    our_lost, our_gained = _side_lost_gained(
        prev_grid, curr_grid, lambda ch: _is_our_piece(ch, view_side),
    )
    opp_lost, opp_gained = _side_lost_gained(
        prev_grid, curr_grid, lambda ch: _is_opp_piece(ch, view_side),
    )
    detail.update({
        "our_lost": len(our_lost),
        "our_gained": len(our_gained),
        "opp_lost": len(opp_lost),
        "opp_gained": len(opp_gained),
    })

    if not (our_lost or our_gained or opp_lost or opp_gained):
        detail["reason"] = "unchanged"
        return None, detail

    # 我方刚走（可吃子）：我方 1 失 1 得，对方不得子，对方至多失 1
    if (
        len(our_lost) == 1
        and len(our_gained) == 1
        and len(opp_gained) == 0
        and len(opp_lost) <= 1
    ):
        return "opp", detail

    # 对方刚走
    if (
        len(opp_lost) == 1
        and len(opp_gained) == 1
        and len(our_gained) == 0
        and len(our_lost) <= 1
    ):
        return "our", detail

    detail["reason"] = "ambiguous"
    return None, detail


def resolve_to_move(
    img,
    curr_grid: Sequence[Sequence[str]],
    view_side: str,
    prev_grid: Optional[Sequence[Sequence[str]]] = None,
    avatar_boxes: Optional[AvatarBoxes] = None,
) -> dict:
    """综合判定当前行棋方；无法确定时兜底为我方。

    Returns
    -------
    dict
        to_move: \"our\" | \"opp\"
        method: \"avatar\" | \"grid_diff\" | \"default\" | \"disabled\"
        detail: 各探测明细
        avatar_boxes: 本次头像框（上层应缓存，下次传入）
    """
    if not TURN_DETECT_ENABLED:
        return {
            "to_move": "our",
            "method": "disabled",
            "detail": {},
            "avatar_boxes": avatar_boxes,
        }

    side, avatar_detail, boxes = detect_avatar_turn(img, avatar_boxes)
    if side is not None:
        return {
            "to_move": side,
            "method": "avatar",
            "detail": {"avatar": avatar_detail},
            "avatar_boxes": boxes,
        }

    side, diff_detail = infer_turn_from_grid_diff(prev_grid, curr_grid, view_side)
    if side is not None:
        return {
            "to_move": side,
            "method": "grid_diff",
            "detail": {"avatar": avatar_detail, "grid_diff": diff_detail},
            "avatar_boxes": boxes,
        }

    return {
        "to_move": "our",
        "method": "default",
        "detail": {"avatar": avatar_detail, "grid_diff": diff_detail},
        "avatar_boxes": boxes,
    }


def to_move_label(to_move: str) -> str:
    return "我方" if to_move == "our" else "对方"


def method_label(method: str) -> str:
    """判定来源的中文说明（日志用）。"""
    if method == "avatar":
        return "头像绿边"
    if method == "grid_diff":
        return "局面差分"
    if method == "default":
        return "兜底（头像/差分均未确定，默认我方）"
    if method == "disabled":
        return "已关闭"
    return method


def format_turn_log(turn: dict, *, action: str = "") -> str:
    """拼一行行棋方日志。兜底时强制写明原因；关闭判定时返回空串。"""
    if turn.get("method") == "disabled" or not TURN_DETECT_ENABLED:
        return ""
    who = to_move_label(turn.get("to_move", "our"))
    how = method_label(turn.get("method", "default"))
    base = f"行棋方 · {who} · 判定={how}"
    avatar = (turn.get("detail") or {}).get("avatar") or {}
    if turn.get("method") == "avatar":
        our_n = avatar.get("our_green")
        opp_n = avatar.get("opp_green")
        if our_n is not None and opp_n is not None:
            base += f" · 绿量我{our_n}/对{opp_n}"
        if avatar.get("cached"):
            base += " · 框=缓存"
        else:
            base += " · 框=新定位"
    if action:
        return f"{base} · {action}"
    return base
