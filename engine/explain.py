"""着法解释：把一步棋的战术特征翻译成简短的人话（中文）。

输出用于最终产品里“为什么走这一步”的说明，也给出 tags 供程序判断。
"""

from .constants import (
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
    opponent,
)
from .movegen import generate_legal, in_check, is_attacked
from .notation import piece_name


def analyze_move(board, move) -> dict:
    """在走子前的局面 board 上分析 move 的战术含义。"""
    us = board.turn
    opp = opponent(us)
    frm, to = move
    piece = board.cells[frm]
    captured = board.cells[to]
    ptype = piece & 7

    was_check = in_check(board, us)
    captured_defended = bool(captured) and is_attacked(board, to, opp)

    after = board.clone()
    after.apply(move)

    gives_check = in_check(after, opp)
    opp_moves = generate_legal(after, opp)
    is_mate = len(opp_moves) == 0
    attacked_after = is_attacked(after, to, opp)
    defended_after = is_attacked(after, to, us)

    # 走完后我方对敌方的最大威胁（下一步能白吃/吃掉的最大子）
    threat = EMPTY
    threat_value = 0
    threat_sq = -1
    threat_defended = True
    if not is_mate:
        for sq in range(SQUARES):
            p = after.cells[sq]
            if p == EMPTY or (p >> 3) != opp:
                continue
            if (p & 7) == KING:
                continue  # 将/帅不可被吃，谈“吃将”没有意义
            if is_attacked(after, sq, us):
                value = PIECE_VALUE[p & 7]
                if value > threat_value:
                    threat_value = value
                    threat = p
                    threat_sq = sq
        if threat_sq >= 0:
            threat_defended = is_attacked(after, threat_sq, opp)

    crossed = False
    if ptype == PAWN:
        crossed = to // COLS <= 4 if us == RED else to // COLS >= 5

    return {
        "color": us,
        "piece": piece,
        "ptype": ptype,
        "captured": captured,
        "captured_defended": captured_defended,
        "was_check": was_check,
        "gives_check": gives_check,
        "is_mate": is_mate,
        "attacked_after": attacked_after,
        "defended_after": defended_after,
        "threat": threat,
        "threat_value": threat_value,
        "threat_defended": threat_defended,
        "crossed_river": crossed,
        "to_col": to % COLS,
        "forward_gain": (frm // COLS - to // COLS) if us == RED else (to // COLS - frm // COLS),
    }


def build_reason(facts: dict) -> tuple:
    """根据战术事实生成 (理由文本, tags 列表)。"""
    clauses = []
    tags = []

    if facts["is_mate"]:
        clauses.append("直接绝杀，对手已无合法应着")
        tags.append("mate")
    else:
        if facts["was_check"]:
            clauses.append("先解掉对方的将军")
            tags.append("escape_check")

        if facts["captured"]:
            name = piece_name(facts["captured"])
            if facts["captured_defended"]:
                clauses.append(f"吃掉对方一{name}")
                tags.append("capture")
            else:
                clauses.append(f"白吃对方一{name}")
                tags.append("capture")
                tags.append("free_capture")

        if facts["gives_check"]:
            clauses.append("形成将军，逼迫对方应付" if not facts["captured"] else "并同时将军")
            tags.append("check")

        # 无明确战术时不编布局套话（“出动马，抢占要道”等不准确）
        if not clauses:
            tags.append("quiet")

    # 落点安全性：仅在有其它战术条款时补充；单独一句“有保护”意义不大
    if clauses and facts["attacked_after"]:
        if facts["defended_after"]:
            clauses.append("落点有己方子力保护")
            tags.append("supported")
        else:
            clauses.append("注意落点暂无保护，可能被对方吃回")
            tags.append("risky")

    # 后续威胁（只提无根的子）
    if not facts["is_mate"] and facts["threat"] and not facts["threat_defended"]:
        name = piece_name(facts["threat"])
        clauses.append(f"下一步可白吃对方{name}")
        tags.append("threat")

    text = "；".join(clauses)
    if text:
        text += "。"
    return text, tags
