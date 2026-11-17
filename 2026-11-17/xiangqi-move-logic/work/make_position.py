"""生成一个可复现的中局局面（中炮过河车 vs 屏风马），供示例使用。

仅用于产出示例素材，不属于引擎代码。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))

from engine.board import Board
from engine.constants import START_FEN
from engine.notation import chinese_notation

MOVES = [
    ((7, 7), (7, 4)),  # 红 炮二平五
    ((0, 7), (2, 6)),  # 黑 马8进7
    ((9, 7), (7, 6)),  # 红 马二进三
    ((0, 8), (0, 7)),  # 黑 车9平8
    ((9, 8), (9, 7)),  # 红 车一平二
    ((3, 6), (4, 6)),  # 黑 卒7进1
    ((9, 7), (3, 7)),  # 红 车二进六
]


def main():
    board = Board.from_fen(START_FEN)
    for frm, to in MOVES:
        move = (frm[0] * 9 + frm[1], to[0] * 9 + to[1])
        print("着法:", chinese_notation(board, move))
        board.apply(move)
    print("FEN:", board.to_fen())
    print("轮到:", board.side_name())


if __name__ == "__main__":
    main()
