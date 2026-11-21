"""局面解读：从棋盘抽取结构特征，生成有深度的总读文案（直播口播用）。

不依赖 LLM；特征来自当前 Board + decide 结果中的分数/开局库信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.board import Board
from engine.constants import (
    ADVISOR,
    BLACK,
    CANNON,
    CHARIOT,
    COLS,
    ELEPHANT,
    EMPTY,
    HORSE,
    KING,
    PAWN,
    PIECE_VALUE,
    RED,
    SQUARES,
)
from engine.movegen import in_check, is_attacked
from engine.notation import piece_name
from engine.verdict import phase_label, score_for_our_side


# 开局原位（内部坐标：红在下）
_RED_HORSE_HOME = (82, 88)
_RED_ROOK_HOME = (81, 89)
_BLACK_HORSE_HOME = (1, 7)
_BLACK_ROOK_HOME = (0, 8)


@dataclass
class PositionFeatures:
    """局面结构特征（我方视角）。"""

    phase: str = "中局"
    our_side: str = "red"
    our_cp: int = 0
    band: str = "均势"
    in_check: bool = False
    material_delta: int = 0          # 我方子力值 - 对方（粗算）
    our_missing_advisor: bool = False
    our_missing_elephant: bool = False
    opp_missing_advisor: bool = False
    opp_missing_elephant: bool = False
    our_undeveloped: list[str] = field(default_factory=list)   # ["马","车"]
    opp_undeveloped: list[str] = field(default_factory=list)
    our_crossed_pawns: int = 0
    opp_crossed_pawns: int = 0
    our_center_cannon: bool = False
    opp_center_cannon: bool = False
    hanging_ours: list[str] = field(default_factory=list)      # 无根被捉
    pressured_ours: list[str] = field(default_factory=list)    # 有根被盯
    from_book: bool = False
    book_hint: str = ""
    focus: str = ""                  # 本局核心矛盾一句


def extract_features(
    board: Board,
    our_side: str,
    *,
    score_cp: int | None = None,
    side_to_move: str | None = None,
    from_book: bool = False,
    verdict: dict | None = None,
) -> PositionFeatures:
    """扫描棋盘，抽出解说用特征。"""
    our = RED if our_side == "red" else BLACK
    opp = BLACK if our == RED else RED
    stm = side_to_move or ("red" if board.turn == RED else "black")
    stm_i = RED if stm == "red" else BLACK

    if score_cp is None:
        # 无搜索分时用 0，只做结构读
        our_cp = 0
    else:
        our_cp = score_for_our_side(score_cp, stm_i, our)

    verdict = verdict or {}
    phase = verdict.get("phase") or phase_label(board)
    band = verdict.get("band") or _band(our_cp)

    f = PositionFeatures(
        phase=phase,
        our_side=our_side,
        our_cp=our_cp,
        band=band,
        in_check=in_check(board, our),
        from_book=from_book,
    )

    mat_our = mat_opp = 0
    adv_our = ele_our = adv_opp = ele_opp = 0
    undev_our: list[str] = []
    undev_opp: list[str] = []

    for sq in range(SQUARES):
        p = board.cells[sq]
        if p == EMPTY:
            continue
        color = p >> 3
        ptype = p & 7
        r, c = divmod(sq, COLS)
        val = PIECE_VALUE[ptype] if ptype != KING else 0

        if color == our:
            mat_our += val
            if ptype == ADVISOR:
                adv_our += 1
            elif ptype == ELEPHANT:
                ele_our += 1
            elif ptype == PAWN and _pawn_crossed(our, r):
                f.our_crossed_pawns += 1
            elif ptype == CANNON and c == 4 and _cannon_own_half(our, r):
                f.our_center_cannon = True
            if ptype == HORSE and sq in (_RED_HORSE_HOME if our == RED else _BLACK_HORSE_HOME):
                undev_our.append("马")
            if ptype == CHARIOT and sq in (_RED_ROOK_HOME if our == RED else _BLACK_ROOK_HOME):
                undev_our.append("车")
        else:
            mat_opp += val
            if ptype == ADVISOR:
                adv_opp += 1
            elif ptype == ELEPHANT:
                ele_opp += 1
            elif ptype == PAWN and _pawn_crossed(opp, r):
                f.opp_crossed_pawns += 1
            elif ptype == CANNON and c == 4 and _cannon_own_half(opp, r):
                f.opp_center_cannon = True
            if ptype == HORSE and sq in (_RED_HORSE_HOME if opp == RED else _BLACK_HORSE_HOME):
                undev_opp.append("马")
            if ptype == CHARIOT and sq in (_RED_ROOK_HOME if opp == RED else _BLACK_ROOK_HOME):
                undev_opp.append("车")

    f.material_delta = mat_our - mat_opp
    f.our_missing_advisor = adv_our < 2
    f.our_missing_elephant = ele_our < 2
    f.opp_missing_advisor = adv_opp < 2
    f.opp_missing_elephant = ele_opp < 2
    f.our_undeveloped = _uniq(undev_our)
    f.opp_undeveloped = _uniq(undev_opp)

    hang, press = [], []
    for sq in range(SQUARES):
        p = board.cells[sq]
        if p == EMPTY or (p >> 3) != our or (p & 7) == KING:
            continue
        if not is_attacked(board, sq, opp):
            continue
        name = piece_name(p)
        if is_attacked(board, sq, our):
            press.append(name)
        else:
            hang.append(name)
    f.hanging_ours = _uniq(hang)
    f.pressured_ours = _uniq(press)

    if from_book:
        f.book_hint = _guess_book_line(board, our)

    f.focus = _pick_focus(f)
    return f


def render_overview(feat: PositionFeatures) -> str:
    """生成「局面解读」口播段落（多句、有层次）。"""
    sentences: list[str] = []

    # 1) 阶段 + 形势总括
    score_txt = f"{feat.our_cp / 100:+.2f}" if abs(feat.our_cp) < 90000 else (
        "绝杀" if feat.our_cp > 0 else "被绝杀"
    )
    head = f"{feat.phase}{feat.band}（我方 {score_txt}）"
    if feat.from_book and feat.book_hint:
        head += f"，布局偏「{feat.book_hint}」"
    elif feat.from_book:
        head += "，仍在开局库主线内"
    sentences.append(head)

    # 2) 结构 / 出子
    struct = []
    if feat.phase == "开局":
        if feat.our_undeveloped:
            struct.append("我方" + "、".join(feat.our_undeveloped) + "尚未出动")
        if feat.opp_undeveloped:
            struct.append("对方" + "、".join(feat.opp_undeveloped) + "仍在原位")
        if feat.our_center_cannon and not feat.opp_center_cannon:
            struct.append("我方已占中炮")
        elif feat.opp_center_cannon and not feat.our_center_cannon:
            struct.append("对方中炮占位，需重视中路")
        elif feat.our_center_cannon and feat.opp_center_cannon:
            struct.append("双方中炮对峙")
    else:
        if feat.our_crossed_pawns or feat.opp_crossed_pawns:
            struct.append(
                f"过河兵我{feat.our_crossed_pawns}/对{feat.opp_crossed_pawns}"
            )
        if feat.material_delta >= 200:
            struct.append("我方子力占优")
        elif feat.material_delta <= -200:
            struct.append("我方子力吃亏")
    if feat.our_missing_advisor or feat.our_missing_elephant:
        miss = []
        if feat.our_missing_advisor:
            miss.append("仕")
        if feat.our_missing_elephant:
            miss.append("相")
        struct.append("我方缺" + "、".join(miss) + "，帅门宜谨慎")
    if feat.opp_missing_advisor or feat.opp_missing_elephant:
        miss = []
        if feat.opp_missing_advisor:
            miss.append("仕")
        if feat.opp_missing_elephant:
            miss.append("相")
        struct.append("对方缺" + "、".join(miss) + "，可作为突破口")
    if struct:
        sentences.append("；".join(struct))

    # 3) 即时矛盾
    tension = []
    if feat.in_check:
        tension.append("我方正在被将，须优先解杀")
    if feat.hanging_ours:
        tension.append("无根受捉：" + "、".join(feat.hanging_ours[:3]))
    elif feat.pressured_ours:
        tension.append("受压子力：" + "、".join(feat.pressured_ours[:3]))
    if feat.focus and feat.focus not in "".join(tension):
        tension.append(feat.focus)
    if tension:
        sentences.append("当前矛盾：" + "；".join(tension))

    # 4) 阶段策略口吻
    if feat.phase == "开局":
        tip = "宜先出子占位，少贪打兵"
    elif feat.phase == "残局":
        tip = "宜细算交换与将帅活动，兵的价值上升"
    else:
        if feat.our_cp >= 40:
            tip = "宜保持压力，逼对方在弱点上表态"
        elif feat.our_cp <= -40:
            tip = "宜先稳固再纠缠，寻找兑子翻盘点"
        else:
            tip = "宜抢先手、争中路与肋道"
    sentences.append(tip)

    text = "。".join(s.rstrip("。") for s in sentences if s)
    return text + "。"


def render_situation(feat: PositionFeatures, verdict_summary: str = "") -> str:
    """加厚「形势」一行：分数带 + 核心矛盾。"""
    base = (verdict_summary or "").strip()
    if not base:
        base = f"{feat.band}（我方 {feat.our_cp / 100:+.2f}）· {feat.phase}"
    extra = []
    if feat.focus:
        extra.append(feat.focus)
    elif feat.hanging_ours:
        extra.append("先处理无根子")
    elif feat.in_check:
        extra.append("先解将")
    if not extra:
        return base
    return f"{base}；关键是{extra[0]}"


def features_from_decision(
    fen: str, our_side: str, decision: dict,
) -> PositionFeatures | None:
    """从 decide 结果取 FEN 并抽特征；失败返回 None。"""
    if not fen:
        return None
    try:
        board = Board.from_fen(fen)
    except Exception:  # noqa: BLE001
        return None
    inp = decision.get("input") or {}
    search = decision.get("search") or {}
    return extract_features(
        board,
        our_side,
        score_cp=search.get("score_cp"),
        side_to_move=inp.get("side_to_move"),
        from_book=bool(search.get("book")),
        verdict=decision.get("verdict") or {},
    )


# ------------------------------------------------------------------ 内部


def _band(our_cp: int) -> str:
    if our_cp >= 300:
        return "明显优势"
    if our_cp >= 120:
        return "优势"
    if our_cp >= 40:
        return "略优"
    if our_cp > -40:
        return "均势"
    if our_cp > -120:
        return "略劣"
    if our_cp > -300:
        return "劣势"
    return "明显劣势"


def _pawn_crossed(color: int, row: int) -> bool:
    return row <= 4 if color == RED else row >= 5


def _cannon_own_half(color: int, row: int) -> bool:
    return row >= 5 if color == RED else row <= 4


def _uniq(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _pick_focus(f: PositionFeatures) -> str:
    if f.in_check:
        return "解将压倒一切"
    if f.hanging_ours:
        return f"先救无根的{f.hanging_ours[0]}"
    if f.phase == "开局" and f.our_undeveloped:
        return "优先完成出子"
    if f.opp_center_cannon and not f.our_center_cannon:
        return "争夺中路主动"
    if f.our_cp <= -120:
        return "先稳住阵形再反击"
    if f.our_cp >= 120:
        return "扩大优势，不给对方兑子翻盘"
    return ""


def _guess_book_line(board: Board, our: int) -> str:
    """粗判常见开局名（中炮 / 屏风马痕迹）。"""
    cells = board.cells
    # 红中炮：己方半场 col4 有炮
    red_mid = any(
        cells[r * COLS + 4] == ((RED << 3) | CANNON) for r in range(5, 10)
    )
    black_mid = any(
        cells[r * COLS + 4] == ((BLACK << 3) | CANNON) for r in range(0, 5)
    )
    # 屏风马：双马离开原位到 2/6 一类（简化：双方马都离开原位且无中炮对中炮）
    if our == RED and red_mid and not black_mid:
        return "中炮类"
    if our == BLACK and black_mid and not red_mid:
        return "中炮类"
    if red_mid and black_mid:
        return "顺炮/中炮对攻"
    return "开局库主线"
