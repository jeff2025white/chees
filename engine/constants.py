"""中国象棋引擎常量：棋子编码、子力价值、中文名称与坐标约定。

内部坐标约定（全引擎统一）：
    下标 sq = row * 9 + col
    col: 0..8  从左到右（以红方视角，红方在下）
    row: 0..9  从上到下（row 0 = 黑方底线，row 9 = 红方底线）

棋子编码：
    piece = (color << 3) | type
    color: RED=0 / BLACK=1
    type : KING=1 .. PAWN=7
    因此红方棋子取值 1..7，黑方棋子取值 9..15，0 表示空。
"""

COLS = 9
ROWS = 10
SQUARES = COLS * ROWS

RED = 0
BLACK = 1

EMPTY = 0
KING = 1
ADVISOR = 2
ELEPHANT = 3
HORSE = 4
CHARIOT = 5
CANNON = 6
PAWN = 7


def make(color: int, ptype: int) -> int:
    """组合颜色与兵种，得到棋子编码。"""
    return (color << 3) | ptype


def color_of(piece: int) -> int:
    """取棋子颜色（空子返回 0，调用前应先判空）。"""
    return piece >> 3


def type_of(piece: int) -> int:
    """取棋子兵种（空子返回 0）。"""
    return piece & 7


def opponent(color: int) -> int:
    """返回对手颜色。"""
    return BLACK if color == RED else RED


# 子力价值（厘兵）。车、炮直线速度快，高于马。兵的名义值是 100，
# 开局到残局的实际吃兵收益在评估里再压低。将帅不参与交换评估。
PIECE_VALUE = {
    KING: 60000,
    ADVISOR: 200,
    ELEPHANT: 200,
    HORSE: 400,
    CHARIOT: 1200,
    CANNON: 650,
    PAWN: 100,
}

# 中文棋子名（红/黑两套写法）
PIECE_NAMES = {
    (RED, KING): "帅",
    (RED, ADVISOR): "仕",
    (RED, ELEPHANT): "相",
    (RED, HORSE): "马",
    (RED, CHARIOT): "车",
    (RED, CANNON): "炮",
    (RED, PAWN): "兵",
    (BLACK, KING): "将",
    (BLACK, ADVISOR): "士",
    (BLACK, ELEPHANT): "象",
    (BLACK, HORSE): "马",
    (BLACK, CHARIOT): "车",
    (BLACK, CANNON): "炮",
    (BLACK, PAWN): "卒",
}

# FEN 字符映射（大写红、小写黑）；兼容 b/e 两种“象”写法与 h 的“马”写法
FEN_TO_PIECE = {
    "K": make(RED, KING),
    "A": make(RED, ADVISOR),
    "B": make(RED, ELEPHANT),
    "E": make(RED, ELEPHANT),
    "H": make(RED, HORSE),
    "N": make(RED, HORSE),
    "R": make(RED, CHARIOT),
    "C": make(RED, CANNON),
    "P": make(RED, PAWN),
    "k": make(BLACK, KING),
    "a": make(BLACK, ADVISOR),
    "b": make(BLACK, ELEPHANT),
    "e": make(BLACK, ELEPHANT),
    "h": make(BLACK, HORSE),
    "n": make(BLACK, HORSE),
    "r": make(BLACK, CHARIOT),
    "c": make(BLACK, CANNON),
    "p": make(BLACK, PAWN),
}

# 输出 FEN 时统一使用的大写/小写字符
PIECE_TO_FEN = {
    make(RED, KING): "K",
    make(RED, ADVISOR): "A",
    make(RED, ELEPHANT): "B",
    make(RED, HORSE): "N",
    make(RED, CHARIOT): "R",
    make(RED, CANNON): "C",
    make(RED, PAWN): "P",
    make(BLACK, KING): "k",
    make(BLACK, ADVISOR): "a",
    make(BLACK, ELEPHANT): "b",
    make(BLACK, HORSE): "n",
    make(BLACK, CHARIOT): "r",
    make(BLACK, CANNON): "c",
    make(BLACK, PAWN): "p",
}

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"

# 中文数字（红方记谱用中文，黑方记谱用阿拉伯数字）
RED_NUMERALS = "零一二三四五六七八九"
BLACK_NUMERALS = "0123456789"
