"""识别当前行棋方（我方 / 对方）。

两种办法（任一确定即采用；都不确定则兜底「我方」）：

1. 头像绿边：对方在左上、我方在左下；行棋方头像有绿色描边（随用时变短）
2. 局面差分：相对上一帧，若可判定「刚走完的一方」，则当前轮到另一方

返回的 to_move 只会是 ``\"our\"`` / ``\"opp\"``，不会是 None。

开关：``TURN_DETECT_ENABLED = False`` 时暂时关闭判定，一律视为我方行棋。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# 暂时关闭行棋方识别（头像绿边 / 局面差分 / 对方跳过计算）
TURN_DETECT_ENABLED = False

# 相对整图像素比例 (x0, y0, x1, y1)；多布局候选，取绿边分最高者
# 对方：偏上 / 偏下两种对局 UI；我方：左下 / 右下
_OPP_AVATAR_BOXES = (
    (0.020, 0.045, 0.165, 0.140),  # 对局页：头像靠上（绿边常在此）
    (0.012, 0.145, 0.145, 0.220),  # 另一布局：头像略低
)
_OUR_AVATAR_BOXES = (
    (0.012, 0.820, 0.140, 0.900),  # 左下
    (0.72, 0.82, 0.92, 0.94),      # 右下（部分皮肤）
)

# 绿边判定：环带绿占比需明显高于内部，避免头像内容里的绿色（如树木）误判
_RING_FRAC = 0.14
_GREEN_HSV_LO = (35, 60, 80)
_GREEN_HSV_HI = (100, 255, 255)
_AVATAR_MIN_SCORE = 0.05
_AVATAR_MIN_RING = 0.06
_AVATAR_DOMINANCE = 0.03  # 双方分差至少这么大才认


def _crop_frac(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]


def _ring_green_score(roi) -> dict:
    """外环绿色净分（环带 − 内部），抑制头像内容绿色干扰。"""
    if roi is None or getattr(roi, "size", 0) == 0:
        return {"score": 0.0, "ring": 0.0, "inner": 0.0, "box": None}
    h, w = roi.shape[:2]
    if h < 16 or w < 16:
        return {"score": 0.0, "ring": 0.0, "inner": 0.0}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array(_GREEN_HSV_LO, dtype=np.uint8),
        np.array(_GREEN_HSV_HI, dtype=np.uint8),
    )
    t = max(3, int(min(h, w) * _RING_FRAC))
    ring = np.zeros((h, w), dtype=np.uint8)
    ring[:t, :] = 1
    ring[-t:, :] = 1
    ring[:, :t] = 1
    ring[:, -t:] = 1
    inner = np.zeros((h, w), dtype=np.uint8)
    if h > 2 * t + 2 and w > 2 * t + 2:
        inner[t:-t, t:-t] = 1

    ring_pct = float(green[ring > 0].mean() / 255.0) if ring.any() else 0.0
    inner_pct = float(green[inner > 0].mean() / 255.0) if inner.any() else 0.0
    score = max(0.0, ring_pct - inner_pct)
    return {
        "score": round(score, 4),
        "ring": round(ring_pct, 4),
        "inner": round(inner_pct, 4),
    }


def _best_avatar_score(img, boxes) -> dict:
    """在多个候选框里取绿边净分最高者。"""
    best = {"score": 0.0, "ring": 0.0, "inner": 0.0, "box": None}
    for box in boxes:
        m = _ring_green_score(_crop_frac(img, box))
        m["box"] = box
        if m["score"] > best["score"] or (
            m["score"] == best["score"] and m["ring"] > best.get("ring", 0)
        ):
            best = m
    return best


def detect_avatar_turn(img) -> Tuple[Optional[str], dict]:
    """从头像绿边判断行棋方。

    Returns
    -------
    (side, detail)
        side: ``\"our\"`` / ``\"opp\"`` / None（无法确定）
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None, {"reason": "empty"}

    opp_m = _best_avatar_score(img, _OPP_AVATAR_BOXES)
    our_m = _best_avatar_score(img, _OUR_AVATAR_BOXES)
    detail = {"opp": opp_m, "our": our_m, "method": "avatar"}

    opp_ok = opp_m["score"] >= _AVATAR_MIN_SCORE and opp_m["ring"] >= _AVATAR_MIN_RING
    our_ok = our_m["score"] >= _AVATAR_MIN_SCORE and our_m["ring"] >= _AVATAR_MIN_RING

    cands = []
    if our_ok:
        cands.append(("our", our_m["score"]))
    if opp_ok:
        cands.append(("opp", opp_m["score"]))
    if not cands:
        return None, detail
    cands.sort(key=lambda x: -x[1])
    if len(cands) == 2 and (cands[0][1] - cands[1][1]) < _AVATAR_DOMINANCE:
        detail["reason"] = "tie"
        return None, detail
    return cands[0][0], detail


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

    # 无变化
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
) -> dict:
    """综合判定当前行棋方；无法确定时兜底为我方。

    Returns
    -------
    dict
        to_move: \"our\" | \"opp\"
        method: \"avatar\" | \"grid_diff\" | \"default\" | \"disabled\"
        detail: 各探测明细
    """
    if not TURN_DETECT_ENABLED:
        return {
            "to_move": "our",
            "method": "disabled",
            "detail": {},
        }

    side, avatar_detail = detect_avatar_turn(img)
    if side is not None:
        return {
            "to_move": side,
            "method": "avatar",
            "detail": {"avatar": avatar_detail},
        }

    side, diff_detail = infer_turn_from_grid_diff(prev_grid, curr_grid, view_side)
    if side is not None:
        return {
            "to_move": side,
            "method": "grid_diff",
            "detail": {"avatar": avatar_detail, "grid_diff": diff_detail},
        }

    return {
        "to_move": "our",
        "method": "default",
        "detail": {"avatar": avatar_detail, "grid_diff": diff_detail},
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
    if action:
        return f"{base} · {action}"
    return base
