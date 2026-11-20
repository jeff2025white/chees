"""局势评价：面向「我方」的判词 / 建议 / 要点，供日志展示。"""

from .search import MATE


def phase_label(board) -> str:
    """开局 / 中局 / 残局（与 phase-strategy 一致）。"""
    if board.is_opening():
        return "开局"
    pc = getattr(board, "piece_count", 32)
    if pc <= 20:
        return "残局"
    return "中局"


def score_for_our_side(score_cp: int, side_to_move: int, our_side: int) -> int:
    """行棋方视角分数 → 我方（view_side）视角。"""
    if side_to_move == our_side:
        return score_cp
    return -score_cp


def _band_label(our_cp: int) -> str:
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


def _format_our_score(our_cp: int) -> str:
    if abs(our_cp) > MATE - 1000:
        steps = (MATE - abs(our_cp) + 1) // 2
        return f"绝杀({steps}步)" if our_cp > 0 else f"被绝杀({steps}步)"
    return f"{our_cp / 100:+.2f}"


def _suggestion(best: dict, phase: str, our_cp: int, from_book: bool) -> str:
    """压缩一句建议；禁止空泛「抢占要道」。"""
    move = (best or {}).get("chinese") or "该着"
    tags = set((best or {}).get("tags") or [])
    reason = ((best or {}).get("reason") or "").strip()

    if from_book or "book" in tags:
        # 去掉「开局库（我方）：」前缀，避免与建议句重复
        core = reason
        for prefix in ("开局库（我方）：", "开局库（对方）：", "开局库："):
            if core.startswith(prefix):
                core = core[len(prefix):].strip()
                break
        # 库着理由常以「炮X平Y。」开头
        if core.startswith(move):
            core = core[len(move):].lstrip("：:。;； ")
        tip = "走中炮/布局主线" if phase == "开局" else "按开局库续走"
        if "炮" in move and ("平五" in move or move.endswith("平5")):
            tip = "走中炮布局"
        elif "马" in move and ("进三" in move or "进7" in move or "进八" in move):
            tip = "跳出马位"
        elif "兵" in move or "卒" in move:
            tip = "挺兵展开"
        return f"{move}，{tip}（开局库）。"

    if "mate" in tags or (best or {}).get("is_mate"):
        return f"{move}，直接绝杀。"

    # 有实质战术理由（非空、非仅开局库套话）
    if reason and "开局库" not in reason:
        # 理由已含着法含义时直接用
        text = reason if reason.endswith("。") else reason + "。"
        if not text.startswith(move):
            text = f"{move}，{text}"
        return text

    # 安静着：按阶段 + 局势简短模板
    if phase == "开局":
        return f"先{move}，稳妥出子。"
    if phase == "残局":
        if our_cp >= 40:
            return f"{move}，扩大优势。"
        if our_cp <= -40:
            return f"{move}，尽量纠缠。"
        return f"{move}，细算残局。"
    if our_cp >= 40:
        return f"{move}，保持压力。"
    if our_cp <= -40:
        return f"{move}，寻求均势。"
    return f"{move}，维持均势。"


def _highlights(best: dict) -> str:
    """仅战术要点；无则空串。"""
    tags = set((best or {}).get("tags") or [])
    reason = ((best or {}).get("reason") or "").strip()
    parts = []
    if "free_capture" in tags:
        parts.append("有白吃机会")
    if "check" in tags:
        parts.append("形成将军")
    if "threat" in tags:
        # 尽量从 reason 抽「下一步可白吃…」
        for seg in reason.replace("。", "；").split("；"):
            if "白吃" in seg or "下一步" in seg:
                parts.append(seg.strip("；。 "))
                break
        else:
            parts.append("下一步有捉子威胁")
    if "risky" in tags:
        parts.append("落点暂无保护，注意被吃回")
    # 去重保序
    seen = set()
    uniq = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            uniq.append(p)
    if not uniq:
        return ""
    text = "；".join(uniq)
    return text if text.endswith("。") else text + "。"


def build_verdict(
    board,
    best: dict,
    score_cp: int,
    side_to_move: int,
    our_side: int,
    from_book: bool = False,
) -> dict:
    """生成结构化局势评价。

    返回：
        phase, our_score_cp, our_score_text, band,
        summary  (评价行),
        advice   (建议行),
        points   (要点行，可空),
        lines    (已排版的日志行列表，不含末尾多余空行)
    """
    phase = phase_label(board)
    our_cp = score_for_our_side(score_cp, side_to_move, our_side)
    our_text = _format_our_score(our_cp)

    if abs(our_cp) > MATE - 1000:
        steps = (MATE - abs(our_cp) + 1) // 2
        if our_cp > 0:
            band = "绝杀"
            summary = f"我方绝杀（{steps}步）· {phase}"
        else:
            band = "被绝杀"
            summary = f"我方被绝杀（{steps}步）· {phase}"
    else:
        band = _band_label(our_cp)
        if band == "均势":
            summary = f"均势（我方 {our_text}）· {phase}"
        else:
            summary = f"{band}（我方 {our_text}）· {phase}"

    advice = _suggestion(best, phase, our_cp, from_book)
    points = _highlights(best)

    lines = [f"评价 {summary}", f"建议 {advice}"]
    if points:
        lines.append(f"要点 {points}")

    return {
        "phase": phase,
        "our_score_cp": our_cp,
        "our_score_text": our_text,
        "band": band,
        "summary": summary,
        "advice": advice,
        "points": points,
        "lines": lines,
    }
