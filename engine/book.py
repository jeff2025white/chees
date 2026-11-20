"""开局库：经典主线前 3–5 回合，避免浅层搜索在开局乱走。

着法用 ICCS（红方视角）。命中后 decide 第 1 手用库；
分析路径再对后续局面续搜，补满推演序列。
"""

from .board import Board
from .constants import BLACK, COLS, START_FEN
from .movegen import generate_legal
from .notation import move_to_iccs


def iccs_to_sq(text: str, start: int = 0):
    """解析 ICCS 坐标起点，返回 (sq, 下一字符下标)。"""
    if start >= len(text) or text[start] < "a" or text[start] > "i":
        raise ValueError(f"非法 ICCS: {text!r}")
    col = ord(text[start]) - ord("a")
    j = start + 1
    while j < len(text) and text[j].isdigit():
        j += 1
    if j == start + 1:
        raise ValueError(f"非法 ICCS: {text!r}")
    rank = int(text[start + 1:j])
    if not 1 <= rank <= 10:
        raise ValueError(f"非法 ICCS 行号: {text!r}")
    row = 10 - rank
    return row * 9 + col, j


def iccs_to_move(iccs: str):
    """ICCS 着法字符串 -> (from_sq, to_sq)，如 'h3e3' / 'i9i10'。"""
    text = iccs.strip().lower()
    frm, i = iccs_to_sq(text, 0)
    to, _ = iccs_to_sq(text, i)
    return (frm, to)


def _pos_key(fen: str) -> str:
    """局面键：棋子布局 + 行棋方。"""
    parts = fen.strip().split()
    if len(parts) < 2:
        return fen.strip()
    return f"{parts[0]} {parts[1]}"


def _color_swap_sq(sq: int) -> int:
    """上下翻转（同列），用于红先主线镜像成黑先布局。"""
    r, c = divmod(sq, COLS)
    return (9 - r) * COLS + c


def _color_swap_move(move):
    return (_color_swap_sq(move[0]), _color_swap_sq(move[1]))


def _mirror_iccs(iccs: str) -> str:
    return move_to_iccs(_color_swap_move(iccs_to_move(iccs)))


# 经典主线（红先）：每项 (名称, ICCS 序列, 主线权重)
# 只覆盖前 3–5 回合；对方偏离后脱库交还搜索。
_LINES = [
    (
        "中炮对屏风马",
        ["h3e3", "h10g8", "h1g3", "b10c8", "i1h1", "i10h10", "c4c5", "g7g6"],
        100,
    ),
    (
        "中炮对顺炮",
        ["h3e3", "h8e8", "h1g3", "h10g8", "i1h1", "i10i9", "c4c5"],
        95,
    ),
    (
        "仙人指路对卒底炮",
        ["c4c5", "b8c8", "h3e3", "h8e8", "h1g3", "h10g8"],
        85,
    ),
]

# 起始局面次选（不单独成线，权重低于中炮主着）
_START_ALTS_RED = [
    ("b3e3", 90),  # 炮八平五
]
_START_ALTS_BLACK = [
    ("b8e8", 90),  # 炮2平5（黑先镜像的左中炮）
]


def _add_entry(book: dict, key: str, iccs: str, weight: int) -> None:
    """向局面登记着法；同着法保留更高权重，不同着法并存。"""
    move = iccs_to_move(iccs)
    entries = book.setdefault(key, [])
    for i, (m, w, existing) in enumerate(entries):
        if m == move:
            if weight > w:
                entries[i] = (move, weight, iccs)
            return
    entries.append((move, weight, iccs))


def _ingest_line(book: dict, start_fen: str, iccs_list, weight: int, name: str) -> None:
    """从起始局面按序列走子，逐步登记下一手。"""
    board = Board.from_fen(start_fen)
    for ply, iccs in enumerate(iccs_list):
        key = _pos_key(board.to_fen())
        move = iccs_to_move(iccs)
        legal = generate_legal(board)
        if move not in legal:
            raise ValueError(
                f"开局库非法着法: 线={name!r} ply={ply} iccs={iccs!r} fen={board.to_fen()}"
            )
        _add_entry(book, key, iccs, weight)
        board.apply(move)


def _black_start_fen() -> str:
    board = Board.from_fen(START_FEN)
    board.turn = BLACK
    return board.to_fen()


def _build_book():
    book = {}
    # 红先主线
    for name, seq, weight in _LINES:
        _ingest_line(book, START_FEN, seq, weight, name)
    for iccs, weight in _START_ALTS_RED:
        _add_entry(book, _pos_key(START_FEN), iccs, weight)

    # 黑先（截图分析：我方在下先走）：主线上下镜像
    black_fen = _black_start_fen()
    for name, seq, weight in _LINES:
        mirrored = [_mirror_iccs(iccs) for iccs in seq]
        _ingest_line(book, black_fen, mirrored, weight, f"{name}(黑先)")
    for iccs, weight in _START_ALTS_BLACK:
        _add_entry(book, _pos_key(black_fen), iccs, weight)

    # 稳定排序：权重降序，便于 book_entries / 调试
    for key in book:
        book[key].sort(key=lambda item: (-item[1], item[2]))
    return book


BOOK = _build_book()


def lookup_book(board, legal_moves=None):
    """查开局库。命中返回 (move, weight, iccs)，否则 None。

    legal_moves 若给出，则只返回仍合法的着法。
    """
    key = _pos_key(board.to_fen())
    entries = BOOK.get(key)
    if not entries:
        return None
    legal = set(legal_moves) if legal_moves is not None else None
    best = None
    for move, weight, iccs in entries:
        if legal is not None and move not in legal:
            continue
        if best is None or weight > best[1]:
            best = (move, weight, iccs)
    return best


def is_book_position(board) -> bool:
    """当前局面是否在开局库中。"""
    return _pos_key(board.to_fen()) in BOOK


def book_entries(board):
    """返回当前局面的库着法列表 [(move, weight, iccs), ...]，无则空列表。"""
    return list(BOOK.get(_pos_key(board.to_fen()), []))


def book_lines():
    """返回主线定义副本，供测试与调试。"""
    return [(name, list(seq), weight) for name, seq, weight in _LINES]
