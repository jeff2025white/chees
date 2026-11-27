"""棋盘状态：FEN 解析、走子/撤销、Zobrist 哈希与重复局面检测。"""

import random

from .constants import (
    BLACK,
    COLS,
    EMPTY,
    FEN_TO_PIECE,
    PIECE_TO_FEN,
    RED,
    ROWS,
    SQUARES,
    color_of,
)
from .evaluate import compute_base_score, is_opening_cells, piece_base_term

# ---------------------------------------------------------------------------
# Zobrist 随机表：固定种子，保证可复现
# ---------------------------------------------------------------------------
_rng = random.Random(20261117)
ZOBRIST = [[_rng.getrandbits(64) for _ in range(SQUARES)] for _ in range(16)]
_ZOBRIST_SIDE = _rng.getrandbits(64)

PALACE_COLS = range(3, 6)


def in_board(row: int, col: int) -> bool:
    """判断行列是否落在棋盘内。"""
    return 0 <= row < ROWS and 0 <= col < COLS


def in_palace(color: int, row: int, col: int) -> bool:
    """判断 (row, col) 是否在指定方的九宫内。"""
    if col not in PALACE_COLS:
        return False
    if color == RED:
        return 7 <= row <= 9
    return 0 <= row <= 2


def has_crossed_river(color: int, row: int) -> bool:
    """判断该方棋子是否已过河（红方行号 <=4，黑方行号 >=5）。"""
    return row <= 4 if color == RED else row >= 5


class Board:
    """棋盘状态。

    cells 为长度 90 的一维数组；turn 表示轮到哪一方走（RED / BLACK）。
    undo_stack 保存 (move, captured, hash_before) 供撤销；
    hash_stack 保存每一步之后的 Zobrist 哈希，用于重复局面判断。
    gave_check / null_move 与走子一一对应：这一步是否将军、是否空着。
    piece_count / base_score（子力+PST）随走子增量维护，供评估加速。
    """

    __slots__ = ("cells", "turn", "undo_stack", "hash_stack", "kings",
                 "piece_count", "base_score", "gave_check", "null_move")

    def __init__(self, cells=None, turn: int = RED):
        self.cells = list(cells) if cells is not None else [EMPTY] * SQUARES
        self.turn = turn
        self.undo_stack = []
        self.hash_stack = [self._compute_hash()]
        self.gave_check = []
        self.null_move = []
        # 缓存双方将/帅位置，搜索中 O(1) 取用；-1 表示不在棋盘上
        self.kings = [-1, -1]
        self.piece_count = 0
        for sq, piece in enumerate(self.cells):
            if piece:
                self.piece_count += 1
                if (piece & 7) == 1:
                    self.kings[piece >> 3] = sq
        self.base_score = compute_base_score(self.cells)

    # -- 基础访问 ---------------------------------------------------------
    @property
    def hash(self) -> int:
        """当前局面的 Zobrist 哈希。"""
        return self.hash_stack[-1]

    def is_opening(self) -> bool:
        """开局阶段（将帅原位且子力仍多）。"""
        return is_opening_cells(self.cells, self.piece_count)

    def piece_at(self, col: int, row: int) -> int:
        """按列、行取棋子（越界返回 EMPTY）。"""
        if not in_board(row, col):
            return EMPTY
        return self.cells[row * COLS + col]

    def clone(self) -> "Board":
        """复制棋盘（不复制历史，用于推演）。"""
        b = Board(self.cells, self.turn)
        return b

    def _compute_hash(self) -> int:
        h = 0
        cells = self.cells
        for sq in range(SQUARES):
            p = cells[sq]
            if p:
                h ^= ZOBRIST[p][sq]
        if self.turn == BLACK:
            h ^= _ZOBRIST_SIDE
        return h

    # -- 走子 / 撤销 ------------------------------------------------------
    def apply(self, move) -> int:
        """走一步棋，返回被吃掉的棋子编码（无吃子返回 0）。"""
        frm, to = move
        cells = self.cells
        piece = cells[frm]
        captured = cells[to]
        h = self.hash_stack[-1] ^ ZOBRIST[piece][frm] ^ ZOBRIST[piece][to]
        if captured:
            h ^= ZOBRIST[captured][to]
            self.base_score -= piece_base_term(captured, to)
            self.piece_count -= 1
        h ^= _ZOBRIST_SIDE
        self.base_score -= piece_base_term(piece, frm)
        self.base_score += piece_base_term(piece, to)
        cells[frm] = EMPTY
        cells[to] = piece
        if (piece & 7) == 1:
            self.kings[piece >> 3] = to
        self.turn = BLACK if self.turn == RED else RED
        self.undo_stack.append((move, captured, h))
        self.hash_stack.append(h)
        self.gave_check.append(False)
        self.null_move.append(False)
        return captured

    def apply_null(self) -> None:
        """空着（只换行棋方），供空着剪枝使用。"""
        h = self.hash_stack[-1] ^ _ZOBRIST_SIDE
        self.turn = BLACK if self.turn == RED else RED
        self.undo_stack.append((None, 0, h))
        self.hash_stack.append(h)
        self.gave_check.append(False)
        self.null_move.append(True)

    def stamp_check(self, gave: bool) -> None:
        """把刚走的这一步标成是否将军。空着不算将军。"""
        if not self.null_move or self.null_move[-1]:
            return
        self.gave_check[-1] = bool(gave)

    def undo(self) -> None:
        """撤销最近一步（含空着）。"""
        move, captured, _ = self.undo_stack.pop()
        self.hash_stack.pop()
        self.gave_check.pop()
        self.null_move.pop()
        if move is None:
            self.turn = BLACK if self.turn == RED else RED
            return
        frm, to = move
        cells = self.cells
        piece = cells[to]
        self.base_score -= piece_base_term(piece, to)
        self.base_score += piece_base_term(piece, frm)
        if captured:
            self.base_score += piece_base_term(captured, to)
            self.piece_count += 1
        cells[frm] = piece
        cells[to] = captured
        if (piece & 7) == 1:
            self.kings[piece >> 3] = frm
        self.turn = BLACK if self.turn == RED else RED

    # -- 重复局面 ---------------------------------------------------------
    def is_repetition(self) -> bool:
        """当前哈希是否在历史中出现过。长将与和棋的区分在搜索里做。"""
        h = self.hash_stack[-1]
        n = len(self.hash_stack) - 1
        for i in range(n):
            if self.hash_stack[i] == h:
                return True
        return False

    # -- FEN --------------------------------------------------------------
    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        """解析 FEN。FEN 的 10 行依次对应 row 0（黑方底线）到 row 9（红方底线）。"""
        parts = fen.strip().split()
        if not parts:
            raise ValueError("FEN 为空")
        rows = parts[0].split("/")
        if len(rows) != ROWS:
            raise ValueError(f"FEN 行数应为 {ROWS}，实际 {len(rows)}")
        cells = [EMPTY] * SQUARES
        for r, row_text in enumerate(rows):
            col = 0
            for ch in row_text:
                if ch.isdigit():
                    col += int(ch)
                    if col > COLS:
                        raise ValueError(f"FEN 第 {r + 1} 行超出 9 列")
                else:
                    if ch not in FEN_TO_PIECE:
                        raise ValueError(f"FEN 含非法字符: {ch!r}")
                    if col >= COLS:
                        raise ValueError(f"FEN 第 {r + 1} 行超出 9 列")
                    cells[r * COLS + col] = FEN_TO_PIECE[ch]
                    col += 1
            if col != COLS:
                raise ValueError(f"FEN 第 {r + 1} 行不足 9 列")
        turn = RED
        if len(parts) > 1:
            side = parts[1].lower()
            if side in ("b", "black"):
                turn = BLACK
            elif side in ("w", "r", "red"):
                turn = RED
            else:
                raise ValueError(f"无法识别的行棋方: {parts[1]}")
        return cls(cells, turn)

    @classmethod
    def from_grid(cls, text: str, turn: int = RED) -> "Board":
        """从自定义网格文本解析棋盘。

        格式：10 行 × 9 个字符，可用 '.' 或空格表示空位；
        大写为红方（K/A/B/E/H/N/R/C/P），小写为黑方，字符含义同 FEN。
        """
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) != ROWS:
            raise ValueError(f"网格应有 {ROWS} 行，实际 {len(lines)}")
        cells = [EMPTY] * SQUARES
        for r, line in enumerate(lines):
            chars = [c for c in line if c not in " \t|"]
            if len(chars) != COLS:
                raise ValueError(f"网格第 {r + 1} 行应有 {COLS} 列，实际 {len(chars)}")
            for c, ch in enumerate(chars):
                if ch in ".-_0":
                    continue
                if ch not in FEN_TO_PIECE:
                    raise ValueError(f"网格含非法字符: {ch!r}")
                cells[r * COLS + c] = FEN_TO_PIECE[ch]
        return cls(cells, turn)

    def to_fen(self) -> str:
        """导出标准 FEN（行序 row 0 -> row 9，与解析一致）。"""
        rows = []
        for r in range(ROWS):
            line = ""
            empty = 0
            for c in range(COLS):
                p = self.cells[r * COLS + c]
                if p == EMPTY:
                    empty += 1
                else:
                    if empty:
                        line += str(empty)
                        empty = 0
                    line += PIECE_TO_FEN[p]
            if empty:
                line += str(empty)
            rows.append(line)
        side = "w" if self.turn == RED else "b"
        return "/".join(rows) + f" {side} - - 0 1"

    # -- 调试输出 ---------------------------------------------------------
    def to_grid(self) -> str:
        """导出 10×9 网格文本（'.' 表示空位），便于与截图识别结果对照。"""
        lines = []
        for r in range(ROWS):
            line = []
            for c in range(COLS):
                p = self.cells[r * COLS + c]
                line.append(PIECE_TO_FEN[p] if p else ".")
            lines.append("".join(line))
        return "\n".join(lines)

    def side_name(self) -> str:
        """当前行棋方的中文名。"""
        return "红方" if self.turn == RED else "黑方"

    def color_name(self, color: int) -> str:
        """指定颜色的中文名。"""
        return "红方" if color == RED else "黑方"


def piece_color(piece: int) -> int:
    """兼容外部调用的颜色读取。"""
    return color_of(piece)
