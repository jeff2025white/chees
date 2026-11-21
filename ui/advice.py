"""棋局解说模块（直播用）。

**主要工作**：输出解说面板文案，供直播口播。

固定栏目（顺序）：
    1. 形势（含局面解读：结构总读 + 分数带）
    2. 对方意图
    3. 我方方案（首步目的 + 结合推演的续着远见）
    4. 风险
    5. 对方应对
    6. 推演（最多 3 步）

主入口::

    content = format_advice(analysis, decision)

契约：我方 = view_side（棋盘下方）；推演 sides[i]==view_side 标「我方」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.board import Board
from engine.book import iccs_to_move
from engine.constants import (
    BLACK,
    EMPTY,
    KING,
    PIECE_VALUE,
    RED,
    SQUARES,
)
from engine.explain import analyze_move, build_reason
from engine.movegen import generate_legal, in_check, is_attacked
from engine.notation import chinese_notation, piece_name
from ui.position_read import features_from_decision, render_overview

TAG_HEAD = "head"
TAG_MOVE = "move"
TAG_OPP = "opp"
TAG_WARN = "warn"
TAG_PLAIN = ""

PV_DISPLAY_MAX = 3


@dataclass
class AdviceContent:
    """棋局解说面板一次完整输出。"""

    parts: list[tuple[str, str]] = field(default_factory=list)
    our_side: str = ""
    overview: str = ""        # 局面解读
    summary: str = ""         # 形势
    opp_intent: str = ""
    plan: str = ""
    risk: str = ""
    opp_reply: str = ""
    advice: str = ""          # 兼容：等同 plan
    points: str = ""          # 兼容：风险 + 应对摘要
    pv_count: int = 0
    ok: bool = True
    message: str = ""

    @property
    def text(self) -> str:
        return "".join(t for _, t in self.parts)


def format_advice(analysis: Any, decision: dict) -> AdviceContent:
    """输出棋局解说——本模块核心工作。"""
    best = decision.get("best_move")
    if not best:
        msg = decision.get("message", "没有可走的着法")
        line = msg if msg.endswith("\n") else msg + "\n"
        return AdviceContent(
            parts=[(TAG_WARN, line)],
            ok=False,
            message=msg,
        )

    inp = decision.get("input") or {}
    our_side = (
        inp.get("view_side")
        or inp.get("our_side")
        or getattr(analysis, "view_side", "red")
    )
    pv = decision.get("pv") or {}
    verdict = decision.get("verdict") or {}

    raw_summary = (verdict.get("summary") or "").strip()
    if not raw_summary:
        raw_summary = best.get("score_text") or "形势待判"

    fen = (inp.get("fen") or "").strip()
    feat = features_from_decision(fen, our_side, decision)
    if feat is not None:
        # 形势 = 总读段落（已含分数带与核心矛盾，不再单列「形势」行）
        summary = render_overview(feat)
    else:
        summary = f"{raw_summary}。结构信息不足，先按引擎着法应对。"
    overview = summary

    opp_intent = _read_opp_intent(fen, our_side)
    plan = _read_our_plan(best, pv, our_side, fen)
    risk = _read_risk(best)
    opp_reply = _read_opp_reply(pv, our_side)

    parts: list[tuple[str, str]] = []
    parts.extend(_line("形势 ", summary))
    parts.append((TAG_PLAIN, "\n"))
    parts.extend(_line("对方意图 ", opp_intent))
    parts.extend(_line("我方方案 ", plan))
    parts.extend(_line("风险 ", risk))
    parts.extend(_line("对方应对 ", opp_reply))
    parts.append((TAG_PLAIN, "\n"))
    parts.extend(_format_pv(pv, best, our_side, limit=PV_DISPLAY_MAX))

    names = (pv.get("chinese") or [])[:PV_DISPLAY_MAX]
    return AdviceContent(
        parts=parts,
        our_side=our_side,
        overview=overview,
        summary=summary,
        opp_intent=opp_intent,
        plan=plan,
        risk=risk,
        opp_reply=opp_reply,
        advice=plan,
        points=f"{risk} {opp_reply}".strip(),
        pv_count=len(names),
        ok=True,
    )


def format_status(kind: str, detail: str = "") -> list[tuple[str, str]]:
    """解说面板占位状态。"""
    if kind == "analyzing":
        return [(TAG_HEAD, "解说生成中…\n")]
    if kind == "waiting":
        return [
            (TAG_HEAD, "等待棋局…\n"),
            (TAG_PLAIN, "识别到棋盘并完成计算后，这里给出直播解说。"),
        ]
    if kind == "hint":
        text = detail or "点击「计算」，生成本手直播解说。"
        return [(TAG_PLAIN, text if text.endswith("\n") else text + "\n")]
    if kind == "error":
        return [(TAG_WARN, f"解说失败\n{detail}")]
    if kind == "warn":
        text = detail or ""
        return [(TAG_WARN, text if text.endswith("\n") else text + "\n")]
    return [(TAG_PLAIN, detail if detail.endswith("\n") else detail + "\n")]


# ------------------------------------------------------------------ 栏目文案


def _line(label: str, body: str) -> list[tuple[str, str]]:
    body = _ensure_period(body)
    return [(TAG_HEAD, label), (TAG_PLAIN, body + "\n")]


def _ensure_period(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "—"
    if text[-1] in "。！？.!":
        return text
    return text + "。"


def _side_int(name: str) -> int:
    return RED if name == "red" else BLACK


def _read_opp_intent(fen: str, our_side: str) -> str:
    """读当前局面：对方正在将军 / 捉子 / 控势。"""
    if not fen:
        return "局面信息不足，意图待观察"
    try:
        board = Board.from_fen(fen)
    except Exception:  # noqa: BLE001
        return "局面解析失败，意图待观察"

    our = _side_int(our_side)
    opp = BLACK if our == RED else RED

    if in_check(board, our):
        return "正在将军，逼我方解杀"

    hits: list[tuple[int, str]] = []
    for sq in range(SQUARES):
        p = board.cells[sq]
        if p == EMPTY or (p >> 3) != our:
            continue
        if (p & 7) == KING:
            continue
        if not is_attacked(board, sq, opp):
            continue
        name = piece_name(p)
        value = PIECE_VALUE[p & 7]
        if is_attacked(board, sq, our):
            hits.append((value, f"盯住我方{name}"))
        else:
            hits.append((value + 50, f"白捉我方{name}"))  # 无根优先

    if not hits:
        return "暂无明显捉杀，偏布局/控势"

    hits.sort(key=lambda x: x[0], reverse=True)
    # 去重文案，最多两句
    seen = set()
    bits = []
    for _, phrase in hits:
        if phrase in seen:
            continue
        seen.add(phrase)
        bits.append(phrase)
        if len(bits) >= 2:
            break
    return "，".join(bits)


def _clean_purpose(reason: str, move: str, tags: set, *, is_mate: bool = False) -> str:
    """从 reason/tags 抽出短目的句（不含着法名）。"""
    purpose = (reason or "").strip()
    for prefix in ("开局库（我方）：", "开局库（对方）：", "开局库："):
        if purpose.startswith(prefix):
            purpose = purpose[len(prefix):].strip()
            break
    if move and purpose.startswith(move):
        purpose = purpose[len(move):].lstrip("：:。;； ")
    purpose = purpose.rstrip("。；; ")
    if purpose:
        # 多条款时取前两句，避免太长
        segs = [s.strip() for s in purpose.replace("。", "；").split("；") if s.strip()]
        return "，".join(segs[:2])
    if is_mate or "mate" in tags:
        return "直接绝杀"
    if "free_capture" in tags:
        return "白吃对方子力"
    if "check" in tags:
        return "将军逼应"
    if "threat" in tags:
        return "制造下一步捉子"
    if "book" in tags:
        return "按开局库主线展开"
    if "escape_check" in tags:
        return "先解将军"
    return "改善结构、保持主动"


def _develop_gloss(chinese: str) -> str:
    """安静出子时的短目的，避免千篇一律「改善结构」。"""
    if "马" in chinese:
        return "跃马出子，呼应阵形"
    if "车" in chinese:
        return "出车占线，扩大活动空间"
    if "炮" in chinese:
        return "调整炮位，加强控制"
    if "兵" in chinese or "卒" in chinese:
        return "挺兵开路，争先手"
    if "仕" in chinese or "士" in chinese or "相" in chinese or "象" in chinese:
        return "补仕相，巩固防守"
    return "改善结构、保持主动"


def _purpose_on_board(board, move) -> str:
    """对推演中某步做战术解释，得到短目的。"""
    try:
        cn = chinese_notation(board, move)
        facts = analyze_move(board, move)
        reason, tags = build_reason(facts)
        purpose = _clean_purpose(
            reason, cn, set(tags), is_mate=bool(facts.get("is_mate")),
        )
        if purpose == "改善结构、保持主动":
            return _develop_gloss(cn)
        return purpose
    except Exception:  # noqa: BLE001
        return "衔接主变"


def _read_our_plan(best: dict, pv: dict, our_side: str, fen: str) -> str:
    """我方方案：首步目的 + 结合推演后续的我方续着（更有远见）。"""
    move0 = (best.get("chinese") or "该着").strip()
    tags0 = set(best.get("tags") or [])
    purpose0 = _clean_purpose(
        best.get("reason") or "",
        move0,
        tags0,
        is_mate=bool(best.get("is_mate")),
    )
    chunks = [f"先走{move0}，目的是{purpose0}"]

    names = pv.get("chinese") or []
    sides = pv.get("sides") or []
    iccs_list = pv.get("iccs") or []
    scores = pv.get("scores") or []

    opp_name = None
    follow_name = None
    follow_purpose = None

    # 沿主变复盘：解释「对方应着之后，我方第二步」的目的
    if fen and iccs_list:
        try:
            board = Board.from_fen(fen)
            n = min(PV_DISPLAY_MAX, len(iccs_list), len(names))
            for i in range(n):
                try:
                    mv = iccs_to_move(iccs_list[i])
                except Exception:  # noqa: BLE001
                    break
                if mv not in generate_legal(board, board.turn):
                    break
                side_i = sides[i] if i < len(sides) else (
                    our_side if i % 2 == 0 else ("black" if our_side == "red" else "red")
                )
                if side_i == our_side:
                    if i > 0:
                        follow_name = names[i]
                        follow_purpose = _purpose_on_board(board, mv)
                else:
                    if opp_name is None:
                        opp_name = names[i]
                board.apply(mv)
        except Exception:  # noqa: BLE001
            pass

    # 无 FEN 复盘时，仍用 PV 文本做远见衔接
    if follow_name is None and len(names) >= 3:
        # 约定我方先走：0 我、1 对、2 我
        if len(sides) < 3 or (sides[0] == our_side and sides[2] == our_side):
            opp_name = opp_name or names[1]
            follow_name = names[2]
            follow_purpose = "按主变继续展开"

    if opp_name and follow_name:
        bit = f"若对方{opp_name}，再{follow_name}"
        if follow_purpose:
            bit += f"，目的是{follow_purpose}"
        chunks.append(bit)
    elif opp_name:
        chunks.append(f"预判对方{opp_name}后继续按主变推进")
    elif follow_name:
        bit = f"下一步准备{follow_name}"
        if follow_purpose:
            bit += f"，目的是{follow_purpose}"
        chunks.append(bit)

    # 三步内形势起伏（我方视角静态分）
    if len(scores) >= 2:
        s0 = scores[0] / 100.0
        s_end = scores[min(len(scores), PV_DISPLAY_MAX) - 1] / 100.0
        delta = s_end - s0
        if delta >= 0.35:
            chunks.append("三步内形势有望抬升")
        elif delta <= -0.35:
            chunks.append("三步内分值可能回落，交换要细算")

    return "；".join(chunks)


def _read_risk(best: dict) -> str:
    """这手的主要风险。"""
    tags = set(best.get("tags") or [])
    reason = best.get("reason") or ""
    bits = []

    if "mate" in tags or best.get("is_mate"):
        return "风险很低，可直接取胜"
    if "risky" in tags or "暂无保护" in reason or "被对方吃回" in reason:
        bits.append("落点暂无保护，可能被吃回")
    if "capture" in tags and "free_capture" not in tags:
        bits.append("吃子处有根，需防兑子亏材")
    if "book" in tags and not bits:
        return "开局库着法，实战风险通常可控"
    if not bits:
        return "未见明显战术漏洞，仍防对方反扑"
    return "；".join(bits)


def _read_opp_reply(pv: dict, our_side: str) -> str:
    """对方最可能的应对 = 主变第 2 手（我方先走时）。"""
    names = pv.get("chinese") or []
    sides = pv.get("sides") or []
    if len(names) >= 2:
        reply = names[1]
        # 确认第 2 手是对方
        if len(sides) > 1 and sides[1] == our_side:
            # 异常：仍展示着法
            return f"{reply}（主变第 2 手）"
        return f"{reply}（主变）"
    if len(names) == 1:
        return "主变仅 1 步，对方应着待实战观察"
    return "主变未给出明确应着，需看实战"


def _format_pv(
    pv: dict, best: dict, our_side: str, *, limit: int = PV_DISPLAY_MAX,
) -> list[tuple[str, str]]:
    """推演序列，最多 limit 步。"""
    parts: list[tuple[str, str]] = [(TAG_HEAD, "推演\n")]
    names = (pv.get("chinese") or [])[:limit]
    sides = pv.get("sides") or []
    scores = pv.get("scores") or []
    if names:
        for i, name in enumerate(names):
            who = "我方" if (i < len(sides) and sides[i] == our_side) else "对方"
            score = scores[i] / 100.0 if i < len(scores) else 0.0
            tag = TAG_MOVE if who == "我方" else TAG_OPP
            parts.append((tag, f"{i + 1:>2}.{name} {who} {score:+.2f}\n"))
    else:
        who = "我方" if best.get("color") == our_side else "对方"
        tag = TAG_MOVE if who == "我方" else TAG_OPP
        parts.append((tag, f"{best['chinese']}  {who}  {best['iccs']}\n"))
    return parts


# ------------------------------------------------------------------ 面板写入


class AdviceView:
    """将棋局解说输出写到面板。"""

    def __init__(self, text_widget):
        self.widget = text_widget

    def apply(self, parts: list[tuple[str, str]]) -> None:
        w = self.widget
        w.configure(state="normal")
        w.delete("1.0", "end")
        for tag, text in parts:
            w.insert("end", text, tag or "")
        w.configure(state="disabled")

    def show(self, content: AdviceContent) -> None:
        self.apply(content.parts)

    def show_decision(self, analysis: Any, decision: dict) -> AdviceContent:
        content = format_advice(analysis, decision)
        self.show(content)
        return content

    def clear_analyzing(self) -> None:
        self.apply(format_status("analyzing"))

    def show_hint(self, text: str = "") -> None:
        self.apply(format_status("hint", text))

    def show_waiting(self) -> None:
        self.apply(format_status("waiting"))

    def show_error(self, error) -> None:
        self.apply(format_status("error", str(error)))

    def show_warn(self, msg: str) -> None:
        self.apply(format_status("warn", msg))

    def write(self, parts: list[tuple[str, str]]) -> None:
        self.apply(parts)
