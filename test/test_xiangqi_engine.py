"""中国象棋决策引擎的规则与接口测试。

运行：python -m unittest test.test_xiangqi_engine -v
（从仓库根目录执行）
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import decide_position
from engine.board import Board
from engine.constants import BLACK, RED, START_FEN
from engine.decide import parse_side
from engine.movegen import game_status, generate_legal_moves, perft
from engine.notation import chinese_notation

MATE_IN_ONE_FEN = "3ak4/8R/5a3/3N5/9/9/9/9/9/3K5 w - - 0 1"
STALEMATE_FEN = "4k4/3N5/2N6/3N5/9/9/9/9/9/3K5 b - - 0 1"


def sq(row, col):
    return row * 9 + col


class TestBoardAndRules(unittest.TestCase):
    def test_fen_roundtrip(self):
        board = Board.from_fen(START_FEN)
        self.assertEqual(board.to_fen(), START_FEN)

    def test_perft_start_position(self):
        board = Board.from_fen(START_FEN)
        self.assertEqual(perft(board, 1), 44)
        self.assertEqual(perft(board, 2), 1920)
        self.assertEqual(perft(board, 3), 79666)

    def test_horse_leg_blocked(self):
        board = Board.from_fen("4k4/9/9/9/4P4/4N4/9/9/9/4K4 w - - 0 1")
        targets = {m[1] for m in generate_legal_moves(board) if m[0] == sq(5, 4)}
        self.assertNotIn(sq(3, 3), targets)
        self.assertNotIn(sq(3, 5), targets)
        self.assertIn(sq(7, 3), targets)

    def test_elephant_eye_and_river(self):
        board = Board.from_fen("4k4/9/9/9/9/2B6/9/9/9/3K5 w - - 0 1")
        targets = {m[1] for m in generate_legal_moves(board) if m[0] == sq(5, 2)}
        # 过河的两个点都不允许
        self.assertNotIn(sq(3, 0), targets)
        self.assertNotIn(sq(3, 4), targets)
        self.assertIn(sq(7, 0), targets)

        board2 = Board.from_fen("4k4/9/9/9/9/9/9/9/3P5/2BK5 w - - 0 1")
        targets2 = {m[1] for m in generate_legal_moves(board2) if m[0] == sq(9, 2)}
        self.assertNotIn(sq(7, 4), targets2)  # 象眼 (8,3) 被塞
        self.assertIn(sq(7, 0), targets2)

    def test_cannon_screen(self):
        board = Board.from_fen("4k4/9/9/9/4r4/9/4p4/9/4C4/4K4 w - - 0 1")
        moves = {m[1] for m in generate_legal_moves(board) if m[0] == sq(8, 4)}
        self.assertIn(sq(4, 4), moves)      # 隔着卒吃车
        self.assertNotIn(sq(6, 4), moves)   # 不能直接吃炮架
        self.assertNotIn(sq(5, 4), moves)   # 不能越过炮架走子

    def test_flying_general_pins_rook(self):
        board = Board.from_fen("4k4/9/9/9/9/4R4/9/9/9/4K4 w - - 0 1")
        moves = [m for m in generate_legal_moves(board) if m[0] == sq(5, 4)]
        # 车被“将帅照面”钉住，只能沿中线走
        self.assertTrue(moves)
        for frm, to in moves:
            self.assertEqual(to % 9, 4)
        self.assertIn((sq(5, 4), sq(4, 4)), moves)

    def test_cannon_screen_created_by_own_move(self):
        # 黑将 e10 与红炮 e7 之间本无子；黑车若走到 e9，正好成为红炮的炮架，
        # 等于“走出”一个将军，因此这步棋不合法（perft(4) 曾因此多算 250 个节点）。
        board = Board.from_fen(
            "1nbakabnr/r8/1c5c1/p1p1C1p1p/9/9/P1P1P1P1P/7C1/9/RNBAKABNR b - - 0 1")
        targets = {m[1] for m in generate_legal_moves(board) if m[0] == sq(1, 0)}
        self.assertNotIn(sq(1, 4), targets)
        self.assertIn(sq(1, 3), targets)

    def test_stalemate_detected(self):
        board = Board.from_fen(STALEMATE_FEN)
        status, moves = game_status(board)
        self.assertEqual(status, "stalemate")
        self.assertEqual(moves, [])

    def test_checkmate_detected(self):
        board = Board.from_fen(MATE_IN_ONE_FEN)
        # 红先，黑方此时未被将死
        self.assertEqual(game_status(board)[0], "playing")
        board.apply((sq(1, 8), sq(0, 8)))
        self.assertEqual(game_status(board)[0], "checkmate")


class TestNotation(unittest.TestCase):
    def test_red_cannon_notation(self):
        board = Board.from_fen(START_FEN)
        self.assertEqual(chinese_notation(board, (sq(7, 7), sq(7, 4))), "炮二平五")
        self.assertEqual(chinese_notation(board, (sq(7, 1), sq(7, 4))), "炮八平五")

    def test_black_horse_notation(self):
        board = Board.from_fen(START_FEN)
        board.turn = BLACK
        # 黑方 8 路马进 7 路：列 7 -> 列 6，行 0 -> 行 2
        self.assertEqual(chinese_notation(board, (sq(0, 7), sq(2, 6))), "马8进7")

    def test_pawn_sideways_after_river(self):
        board = Board.from_fen("4k4/9/9/9/3P5/9/9/9/9/3K5 w - - 0 1")
        # 红兵已过河，横走用“平”；列 3 是红六路，列 4 是红五路
        self.assertEqual(chinese_notation(board, (sq(4, 3), sq(4, 4))), "兵六平五")


class TestDecisionAPI(unittest.TestCase):
    def test_mate_in_one(self):
        result = decide_position(fen=MATE_IN_ONE_FEN, time_limit=1.5)
        best = result["best_move"]
        self.assertIsNotNone(best)
        self.assertTrue(best["is_mate"])
        self.assertEqual(best["mate_in"], 1)
        self.assertEqual(best["iccs"], "i9i10")
        self.assertEqual(best["chinese"], "车一进一")
        self.assertEqual(best["from"], {"col": 8, "row": 1})
        self.assertEqual(best["to"], {"col": 8, "row": 0})
        self.assertIn("绝杀", best["reason"])

    def test_view_side_mirror(self):
        result = decide_position(fen=MATE_IN_ONE_FEN, time_limit=1.5, view_side="black")
        best = result["best_move"]
        # 黑方在下时坐标整体镜像：col -> 8-col, row -> 9-row
        self.assertEqual(best["from"], {"col": 0, "row": 8})
        self.assertEqual(best["to"], {"col": 0, "row": 9})
        # ICCS 仍以红方为基准
        self.assertEqual(best["iccs"], "i9i10")

    def test_output_is_json_serializable(self):
        result = decide_position(fen=START_FEN, time_limit=1.0, max_depth=2)
        text = json.dumps(result, ensure_ascii=False)
        self.assertIn("coordinate_contract", text)
        self.assertEqual(result["input"]["legal_move_count"], 44)
        self.assertIn("alternatives", result)
        self.assertGreaterEqual(len(result["alternatives"]), 1)

    def test_checkmate_position_has_no_move(self):
        result = decide_position(fen=MATE_IN_ONE_FEN, time_limit=1.0)
        board = Board.from_fen(MATE_IN_ONE_FEN)
        board.apply((sq(1, 8), sq(0, 8)))
        result2 = decide_position(fen=board.to_fen(), time_limit=1.0)
        self.assertEqual(result2["status"], "checkmate")
        self.assertIsNone(result2["best_move"])

    def test_parse_side(self):
        self.assertEqual(parse_side("red"), RED)
        self.assertEqual(parse_side("黑"), BLACK)
        self.assertEqual(parse_side("b"), BLACK)
        with self.assertRaises(ValueError):
            parse_side("green")

    def test_grid_input_and_view_mirror(self):
        board = Board.from_fen(MATE_IN_ONE_FEN)
        base = decide_position(fen=MATE_IN_ONE_FEN, time_limit=1.0)
        # 红方在下：界面网格就是内部网格
        same = decide_position(grid=board.to_grid(), side="red", view_side="red",
                               time_limit=1.0)
        self.assertEqual(same["input"]["fen"], base["input"]["fen"])
        self.assertEqual(same["best_move"]["iccs"], base["best_move"]["iccs"])
        # 黑方在下：界面网格 = 内部网格旋转 180°，引擎应还原成同一局面
        flipped = "\n".join("".join(reversed(line))
                            for line in reversed(board.to_grid().splitlines()))
        mirrored = decide_position(grid=flipped, side="red", view_side="black",
                                   time_limit=1.0)
        self.assertEqual(mirrored["input"]["fen"], base["input"]["fen"])
        self.assertEqual(mirrored["best_move"]["from"], {"col": 0, "row": 8})
        self.assertEqual(mirrored["best_move"]["iccs"], base["best_move"]["iccs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
