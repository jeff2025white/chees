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
from engine.book import book_entries, book_lines, iccs_to_move, lookup_book
from engine.constants import BLACK, CHARIOT, HORSE, PIECE_VALUE, RED, START_FEN
from engine.decide import endgame_search_budget, parse_side
from engine.evaluate import evaluate, see_value
from engine.movegen import game_status, generate_legal, generate_legal_moves, perft
from engine.notation import chinese_notation, move_to_iccs
from engine.search import MATE, search
from engine.verdict import build_verdict, score_for_our_side

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
    def test_opening_book_start(self):
        result = decide_position(fen=START_FEN, time_limit=0.1, max_depth=1)
        best = result["best_move"]
        self.assertTrue(result["search"]["book"])
        self.assertEqual(best["iccs"], "h3e3")
        self.assertEqual(best["chinese"], "炮二平五")
        self.assertIn("book", best["tags"])

    def test_grid_bottom_side_moves_first(self):
        board = Board.from_fen(START_FEN)
        # 黑方在下：第一步固定为我方（黑）先走 + 开局库（镜像中炮）
        flipped = "\n".join("".join(reversed(line))
                            for line in reversed(board.to_grid().splitlines()))
        result = decide_position(grid=flipped, side=None, view_side="black",
                                 time_limit=0.1, max_depth=1)
        self.assertEqual(result["input"]["side_to_move"], "black")
        self.assertEqual(result["input"]["our_side"], "black")
        self.assertEqual(result["input"]["view_side"], "black")
        self.assertTrue(result["search"]["book"])
        self.assertEqual(result["best_move"]["iccs"], "h8e8")
        self.assertEqual(result["best_move"]["color"], "black")
        # PV 第一步必须是我方
        self.assertEqual(result["pv"]["sides"][0], "black")

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

    def test_endgame_budget_adds_on_manual_depth(self):
        """残局加深必须叠在手动深度上，而不是被 12 封顶吃掉。"""
        d, t = endgame_search_budget(12, max_depth=8, time_limit=10.0)
        self.assertEqual(d, 11)  # 8+3
        self.assertEqual(t, 80.0)  # 10 * 2^3
        d2, t2 = endgame_search_budget(12, max_depth=12, time_limit=10.0)
        self.assertEqual(d2, 15)  # 12+3，封顶 15
        self.assertEqual(t2, 80.0)
        d3, _ = endgame_search_budget(18, max_depth=8, time_limit=10.0)
        self.assertEqual(d3, 9)  # 8+1
        d4, t4 = endgame_search_budget(24, max_depth=8, time_limit=10.0)
        self.assertEqual(d4, 8)
        self.assertEqual(t4, 10.0)

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


class TestOpeningBook(unittest.TestCase):
    def test_principal_lines_stay_in_book(self):
        """沿 3 条主线逐步走子：每步着法都在库中；权重最高主线逐步等于 lookup。"""
        lines = book_lines()
        top_weight = max(w for _n, _s, w in lines)
        for name, seq, weight in lines:
            board = Board.from_fen(START_FEN)
            for ply, iccs in enumerate(seq):
                entries = book_entries(board)
                self.assertTrue(entries, f"{name} ply={ply} 应在开局库")
                entry_iccs = {e[2] for e in entries}
                self.assertIn(iccs, entry_iccs, f"{name} ply={ply}")
                if weight == top_weight:
                    hit = lookup_book(board)
                    self.assertEqual(hit[2], iccs, f"{name} ply={ply} 主线应优先")
                    result = decide_position(fen=board.to_fen(), time_limit=0.05,
                                             max_depth=1, fill_pv=False)
                    self.assertTrue(result["search"]["book"], f"{name} ply={ply}")
                    self.assertEqual(result["best_move"]["iccs"], iccs)
                board.apply(iccs_to_move(iccs))

    def test_deviation_leaves_book(self):
        """故意偏一手后应脱谱。"""
        board = Board.from_fen(START_FEN)
        board.apply(iccs_to_move("h3e3"))  # 炮二平五
        board.apply(iccs_to_move("a10a9"))  # 偏着：车1进1（非主变）
        self.assertIsNone(lookup_book(board))
        result = decide_position(fen=board.to_fen(), time_limit=0.2,
                                 max_depth=2, fill_pv=False)
        self.assertFalse(result["search"]["book"])

    def test_start_alternatives_include_left_cannon(self):
        board = Board.from_fen(START_FEN)
        result = decide_position(fen=board.to_fen(), time_limit=0.05,
                                 max_depth=1, top_n=3, fill_pv=False)
        self.assertTrue(result["search"]["book"])
        self.assertEqual(result["best_move"]["iccs"], "h3e3")
        alt_iccs = {a["iccs"] for a in result["alternatives"]}
        self.assertIn("b3e3", alt_iccs)


class TestVerdict(unittest.TestCase):
    def test_score_flips_to_our_side(self):
        # 红行棋 +50 → 黑为我方时应为 -50
        self.assertEqual(score_for_our_side(50, RED, RED), 50)
        self.assertEqual(score_for_our_side(50, RED, BLACK), -50)

    def test_band_and_summary_our_perspective(self):
        board = Board.from_fen(START_FEN)
        best = {"chinese": "炮二平五", "reason": "", "tags": ["quiet"]}
        v = build_verdict(board, best, score_cp=80, side_to_move=RED,
                          our_side=RED, from_book=False)
        self.assertEqual(v["band"], "略优")
        self.assertIn("我方 +0.80", v["summary"])
        self.assertIn("开局", v["summary"])
        self.assertTrue(v["advice"].startswith("先炮二平五") or "炮二平五" in v["advice"])

    def test_black_bottom_our_advantage_wording(self):
        board = Board.from_fen(START_FEN)
        board.turn = BLACK
        best = {"chinese": "炮8平5", "reason": "", "tags": ["book"]}
        # 行棋方黑，分数 +60（黑优）→ 我方黑仍为略优
        v = build_verdict(board, best, score_cp=60, side_to_move=BLACK,
                          our_side=BLACK, from_book=True)
        self.assertIn("我方 +0.60", v["summary"])
        self.assertIn("开局库", v["advice"])

    def test_mate_summary_not_vague_advantage(self):
        board = Board.from_fen(MATE_IN_ONE_FEN)
        best = {"chinese": "车一进一", "reason": "直接绝杀", "tags": ["mate"],
                "is_mate": True}
        v = build_verdict(board, best, score_cp=MATE - 1, side_to_move=RED,
                          our_side=RED, from_book=False)
        self.assertEqual(v["band"], "绝杀")
        self.assertIn("绝杀", v["summary"])
        self.assertNotIn("优势", v["summary"])

    def test_decide_includes_verdict(self):
        result = decide_position(fen=START_FEN, time_limit=0.1, max_depth=1)
        self.assertIn("verdict", result)
        self.assertIn("lines", result["verdict"])
        self.assertTrue(result["verdict"]["lines"][0].startswith("评价 "))
        self.assertTrue(result["search"]["book"])
        self.assertIn("开局库", result["verdict"]["advice"])


class TestOpeningAntiPawnGrab(unittest.TestCase):
    def test_early_cannon_raid_eval_not_rewarded(self):
        """开局炮打中兵后的静态分不应明显高于出车发展。"""
        from engine.book import iccs_to_move
        base = Board.from_fen(START_FEN)
        for iccs in ("h3e3", "h10g8", "h1g3", "b10c8"):
            base.apply(iccs_to_move(iccs))
        self.assertTrue(base.is_opening())

        develop = base.clone()
        develop.apply(iccs_to_move("i1h1"))  # 车一平二
        raid = base.clone()
        raid.apply(iccs_to_move("e3e7"))  # 炮五进四打中卒

        # 贪打兵不应凭子力压过出车
        self.assertLessEqual(evaluate(raid) - evaluate(develop), 20)

    def test_rook_lateral_develop_beats_sitting(self):
        """车一平二虽仍在底线，也应比窝在原位更好。"""
        from engine.book import iccs_to_move
        base = Board.from_fen(START_FEN)
        for iccs in ("h3e3", "h10g8", "h1g3", "b10c8"):
            base.apply(iccs_to_move(iccs))
        sit = evaluate(base)
        moved = base.clone()
        moved.apply(iccs_to_move("i1h1"))
        self.assertGreater(evaluate(moved), sit + 10)

    def test_search_prefers_develop_over_cannon_pawn_grab(self):
        """开局搜索（绕过书）不应首选炮打中卒。"""
        from engine.book import iccs_to_move
        board = Board.from_fen(START_FEN)
        for iccs in ("h3e3", "h10g8", "h1g3", "b10c8"):
            board.apply(iccs_to_move(iccs))
        outcome = search(board, time_limit=1.5, max_depth=5, top_n=0, fill_pv=False)
        self.assertIsNotNone(outcome.best_move)
        self.assertNotEqual(move_to_iccs(outcome.best_move), "e3e7")


class TestSeeAndDeep(unittest.TestCase):
    def test_see_rook_takes_defended_horse_is_negative(self):
        """车吃被马护的马：SEE 应为 马价值−车价值。"""
        # e8 马由 g7 马保护；车在 e2 可吃
        fen = "4k4/9/4n4/6n2/9/9/9/9/4R4/4K4 w - - 0 1"
        board = Board.from_fen(fen)
        move = None
        for mv in generate_legal(board):
            if ((board.cells[mv[0]] & 7) == CHARIOT
                    and board.cells[mv[1]]
                    and (board.cells[mv[1]] & 7) == HORSE):
                move = mv
                break
        self.assertIsNotNone(move)
        self.assertEqual(
            see_value(board, move),
            PIECE_VALUE[HORSE] - PIECE_VALUE[CHARIOT],
        )

    def test_see_rook_takes_undefended_horse_is_gain(self):
        """车吃无根马：SEE 应为整只马价值。"""
        fen = "4k4/9/4n4/9/9/9/9/9/4R4/4K4 w - - 0 1"
        board = Board.from_fen(fen)
        move = None
        for mv in generate_legal(board):
            if ((board.cells[mv[0]] & 7) == CHARIOT
                    and board.cells[mv[1]]
                    and (board.cells[mv[1]] & 7) == HORSE):
                move = mv
                break
        self.assertIsNotNone(move)
        self.assertEqual(see_value(board, move), PIECE_VALUE[HORSE])

    def test_endgame_deep_rook_not_heavily_penalized(self):
        """残局双车深入不应比底线双车低约 2×180。"""
        safe = Board.from_fen("4k4/9/5n3/9/9/9/9/9/9/R3K3R w - - 0 1")
        deep = Board.from_fen("4k4/R7R/5n3/9/9/9/9/9/9/4K4 w - - 0 1")
        self.assertFalse(safe.is_opening())
        self.assertFalse(deep.is_opening())
        # 深入仅 PST 差，不应再各扣 180
        self.assertGreater(evaluate(deep), evaluate(safe) - 200)

    def test_avoid_rook_for_defended_horse_when_ahead(self):
        """大优时不应首选车换被保护的马。"""
        fen = "3aka3/9/4n4/6n2/9/9/9/9/4R4/3AKA3 w - - 0 1"
        board = Board.from_fen(fen)
        result = decide_position(fen=board.to_fen(), time_limit=2.0,
                                 max_depth=6, top_n=0, fill_pv=False)
        best = result["best_move"]
        self.assertIsNotNone(best)
        self.assertNotEqual(best.get("captured"), "马")


class TestPvAntiOscillation(unittest.TestCase):
    def test_single_retreat_allowed_cycle_blocked(self):
        """单次原路退回允许；走回已出现局面才禁止。"""
        from engine.search import Searcher
        board = Board.from_fen("3k5/9/9/9/9/9/1R7/9/9/4K4 w - - 0 1")
        rook_from = sq(6, 1)
        rook_to = sq(5, 1)
        first = (rook_from, rook_to)
        self.assertIn(first, generate_legal(board))
        board.apply(first)
        king_move = (sq(0, 3), sq(1, 3))
        self.assertIn(king_move, generate_legal(board))
        board.apply(king_move)
        back = (rook_to, rook_from)
        # 车退回：尚未重复局面，应允许
        self.assertFalse(Searcher._causes_repetition(board, back))
        board.apply(back)
        king_back = (sq(1, 3), sq(0, 3))
        # 将也退回：回到出发局面，应禁止继续循环
        self.assertTrue(Searcher._causes_repetition(board, king_back))

    def test_extract_pv_allows_retreat_stops_cycle(self):
        """TT 植入进退循环时：允许车退一步，但不得把闭环重复局面写进主变。"""
        from engine.search import Searcher
        board = Board.from_fen("3k5/9/9/9/9/9/1R7/9/9/4K4 w - - 0 1")
        searcher = Searcher(board, time_limit=0.5, max_depth=4)
        a = (sq(6, 1), sq(5, 1))
        b = (sq(0, 3), sq(1, 3))
        a2 = (sq(5, 1), sq(6, 1))
        b2 = (sq(1, 3), sq(0, 3))
        scratch = board.clone()
        searcher._tt_store(scratch.hash, 4, 0, 0, a)
        scratch.apply(a)
        searcher._tt_store(scratch.hash, 4, 0, 0, b)
        scratch.apply(b)
        searcher._tt_store(scratch.hash, 4, 0, 0, a2)
        scratch.apply(a2)
        searcher._tt_store(scratch.hash, 4, 0, 0, b2)

        pv = searcher.extract_pv(a, max_len=6, fill=True)
        self.assertGreaterEqual(len(pv), 3)
        self.assertEqual(pv[:3], [a, b, a2])
        self.assertNotIn(b2, pv)
        replay = board.clone()
        for mv in pv:
            replay.apply(mv)
            self.assertFalse(replay.is_repetition(), f"PV 含重复局面: {pv}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
