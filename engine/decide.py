"""决策入口：局面 -> 结构化最佳着法 / 评分 / 备选 / 中文理由。

这是后续“截图识别模块”唯一需要调用的接口：
    decide_position(fen=..., side="red", view_side="red")
返回的 from/to 均为 0..8 列 × 0..9 行 的界面坐标（图下方为我方）。
"""

from .board import Board
from .constants import BLACK, COLS, RED
from .explain import analyze_move, build_reason
from .evaluate import evaluate_for
from .movegen import game_status, generate_legal, in_check
from .notation import (
    chinese_notation,
    move_to_iccs,
    move_to_ucci,
    piece_name,
    sq_to_iccs,
)
from .search import MATE, search

SIDE_NAMES = {RED: "red", BLACK: "black"}


def parse_side(value):
    """把 'red'/'black'/'w'/'b'/0/1 统一解析成 RED / BLACK。"""
    if value is None:
        return None
    if isinstance(value, int):
        if value in (RED, BLACK):
            return value
        raise ValueError(f"无法识别的行棋方: {value}")
    text = str(value).strip().lower()
    if text in ("red", "r", "w", "white", "红", "红方"):
        return RED
    if text in ("black", "b", "黑", "黑方"):
        return BLACK
    raise ValueError(f"无法识别的行棋方: {value}")


def _view_coords(sq: int, view_side: int):
    """内部坐标 -> 界面坐标（我方在下方，column 0 在最左）。"""
    r, c = divmod(sq, COLS)
    if view_side == RED:
        return c, r
    # 黑方在下：上下左右同时镜像
    return COLS - 1 - c, 9 - r


def mirror_grid(grid: str) -> str:
    """把“界面方向”的网格翻转成内部方向（行、列都反转，等同于旋转 180°）。

    截图识别模块按图像行序输出网格（第 0 行是图最上方）。当图下方是我方且
    我方执黑时，内部棋盘的方向与图像相反，需要先镜像再解析。
    """
    lines = [ln for ln in grid.strip().splitlines() if ln.strip()]
    mirrored = []
    for line in reversed(lines):
        chars = [c for c in line if c not in " \t|"]
        mirrored.append("".join(reversed(chars)))
    return "\n".join(mirrored)


def format_score(cp: int) -> str:
    """评分文本：普通局面为 +0.35 形式，杀棋为 绝杀(N步)。"""
    if abs(cp) > MATE - 1000:
        steps = (MATE - abs(cp) + 1) // 2
        return f"绝杀({steps}步)" if cp > 0 else f"被绝杀({steps}步)"
    return f"{cp / 100:+.2f}"


def _move_payload(board, move, score: int, view_side: int, exact: bool = True,
                  score_depth: int = None) -> dict:
    """把一步棋打包成对外的结构化描述（board 为走子前局面）。"""
    frm, to = move
    piece = board.cells[frm]
    facts = analyze_move(board, move)
    reason, tags = build_reason(facts)
    from_col, from_row = _view_coords(frm, view_side)
    to_col, to_row = _view_coords(to, view_side)
    mate_in = None
    if abs(score) > MATE - 1000 and score > 0:
        mate_in = (MATE - abs(score) + 1) // 2
    score_text = format_score(score)
    if not exact and abs(score) <= MATE - 1000:
        # 备选着法只做了 alpha-beta 边界搜索，这里是对手应着的评分上界
        score_text = "≤ " + score_text
    return {
        "from": {"col": from_col, "row": from_row},
        "to": {"col": to_col, "row": to_row},
        "from_sq": sq_to_iccs(frm),
        "to_sq": sq_to_iccs(to),
        "iccs": move_to_iccs(move),
        "ucci": move_to_ucci(move),
        "chinese": chinese_notation(board, move),
        "piece": piece_name(piece),
        "color": SIDE_NAMES[piece >> 3],
        "captured": piece_name(facts["captured"]) if facts["captured"] else None,
        "is_capture": bool(facts["captured"]),
        "gives_check": facts["gives_check"],
        "is_mate": facts["is_mate"],
        "mate_in": mate_in,
        "score_cp": score,
        "score_text": score_text,
        "score_exact": exact,
        "score_depth": score_depth,
        "reason": reason,
        "tags": tags,
    }


def _pv_payload(board, pv, view_side) -> dict:
    """主变序列 -> ICCS、中文记谱、走子方、每步后的静态评分与界面坐标。

    scores 与 chinese 等长，为“走完该步后”的静态评分（我方视角，厘兵）。
    steps 为每步的起止格子（行/列），已按 view_side 转为图像方向。
    """
    replay = board.clone()
    our_side = board.turn
    iccs, chinese, sides, scores, steps = [], [], [], [], []
    for move in pv:
        if move not in generate_legal(replay, replay.turn):
            break
        frm, to = move
        fc, fr = _view_coords(frm, view_side)
        tc, tr = _view_coords(to, view_side)
        steps.append({
            "from": {"row": fr, "col": fc},
            "to": {"row": tr, "col": tc},
        })
        iccs.append(move_to_iccs(move))
        chinese.append(chinese_notation(replay, move))
        sides.append(SIDE_NAMES[replay.turn])
        replay.apply(move)
        scores.append(evaluate_for(replay, our_side))
    return {"iccs": iccs, "chinese": chinese, "sides": sides,
            "scores": scores, "steps": steps}


def decide_position(
    fen: str = None,
    grid: str = None,
    side=None,
    view_side=None,
    time_limit: float = 4.0,
    max_depth: int = 8,
    top_n: int = 2,
) -> dict:
    """核心决策函数。

    参数：
        fen        标准中国象棋 FEN（与 grid 二选一）
        grid       自定义 10x9 网格文本（与 fen 二选一）
        side       轮到哪一方走，缺省用 FEN 中的行棋方
        view_side  哪一方在图下方（决定输出坐标方向），缺省等于行棋方
        time_limit 搜索时间预算（秒）
        max_depth  最大搜索深度
        top_n      备选着法数量

    返回：见 outputs/design.md 的“契约”一节。
    """
    side = parse_side(side)
    view_side = parse_side(view_side) if view_side is not None else None
    if view_side is None:
        view_side = side if side is not None else RED
    if fen:
        board = Board.from_fen(fen)
    elif grid:
        # 约定：grid 是“界面方向”（第 0 行 = 图最上方）。
        # 图下方是我方；若我方执黑，内部棋盘方向与图像相反，需镜像。
        board = Board.from_grid(mirror_grid(grid) if view_side == BLACK else grid,
                                side if side is not None else RED)
    else:
        raise ValueError("必须提供 fen 或 grid")
    if side is not None:
        board.turn = side
    our_side = board.turn

    status, legal = game_status(board, our_side)
    base = {
        "ok": True,
        "input": {
            "fen": board.to_fen(),
            "grid": board.to_grid(),
            "side_to_move": SIDE_NAMES[our_side],
            "our_side": SIDE_NAMES[our_side],
            "view_side": SIDE_NAMES[view_side],
            "in_check": in_check(board, our_side),
            "legal_move_count": len(legal),
        },
        "coordinate_contract": {
            "cols": COLS,
            "rows": 10,
            "col_range": "0..8，从左到右（图左为 0）",
            "row_range": "0..9，从上到下（图上方为 0）",
            "orientation": "row 9 为我方（图下方）底线；我方执黑时坐标已整体镜像",
            "view_side": SIDE_NAMES[view_side],
            "pixel_mapping": "x = margin_x + col * cell_w, y = margin_y + row * cell_h",
        },
        "status": status,
    }

    if not legal:
        if status == "checkmate":
            message = "我方已被将死，无合法着法。"
        else:
            message = "我方被困毙（无合法着法且未被将军），按规则判负。"
        base.update({
            "best_move": None,
            "alternatives": [],
            "pv": {"iccs": [], "chinese": [], "sides": [], "scores": [], "steps": []},
            "search": {"depth": 0, "nodes": 0, "time_ms": 0},
            "message": message,
        })
        return base

    outcome = search(board, time_limit=time_limit, max_depth=max_depth)

    best = _move_payload(board, outcome.best_move, outcome.score, view_side)
    ranked = sorted(outcome.ranked, key=lambda item: item[1], reverse=True)
    alternatives = []
    for move, score, exact in ranked:
        if move == outcome.best_move:
            continue
        alternatives.append(_move_payload(board, move, score, view_side, exact,
                                          outcome.alt_depth))
        if len(alternatives) >= top_n:
            break

    base.update({
        "best_move": best,
        "alternatives": alternatives,
        "pv": _pv_payload(board, outcome.pv, view_side),
        "search": {
            "depth": outcome.depth,
            "nodes": outcome.nodes,
            "time_ms": outcome.elapsed_ms,
            "score_cp": outcome.score,
            "mate_in": outcome.mate_in,
        },
        "message": best["reason"],
    })
    return base


def explain_position(fen: str) -> dict:
    """辅助：只返回局面状态（是否被将军、合法着法数、FEN 回显）。"""
    board = Board.from_fen(fen)
    status, legal = game_status(board)
    return {
        "fen": board.to_fen(),
        "side_to_move": SIDE_NAMES[board.turn],
        "status": status,
        "legal_move_count": len(legal),
        "in_check": in_check(board, board.turn),
        "grid": board.to_grid(),
    }
