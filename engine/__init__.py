"""中国象棋决策内核（纯 Python，无外部引擎依赖）。

对外主接口：
    from engine import decide_position
    result = decide_position(fen="...", side="red")

坐标契约：输出中的 from/to 为 {col: 0..8, row: 0..9}，
col 向右递增、row 向下递增，row 9 为我方（图下方）底线。
"""

from .board import Board
from .decide import decide_position, explain_position
from .notation import chinese_notation, move_to_iccs, move_to_ucci
from .search import search

__all__ = [
    "Board",
    "decide_position",
    "explain_position",
    "search",
    "chinese_notation",
    "move_to_iccs",
    "move_to_ucci",
]

__version__ = "0.1.0"
