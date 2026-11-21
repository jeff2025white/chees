"""行棋方识别：局面差分 + 兜底。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.turn_side as turn_side
from engine.turn_side import (
    format_turn_log,
    infer_turn_from_grid_diff,
    method_label,
    resolve_to_move,
    to_move_label,
)


def _empty():
    return [["."] * 9 for _ in range(10)]


def _copy(g):
    return [list(row) for row in g]


class TestGridDiffTurn(unittest.TestCase):
    def test_our_move_then_opp_to_play(self):
        # 黑在下（我方=black）：黑马从 (9,1) 走到 (7,2)
        prev = _empty()
        prev[9][4] = "k"
        prev[0][4] = "K"
        prev[9][1] = "n"
        curr = _copy(prev)
        curr[9][1] = "."
        curr[7][2] = "n"
        side, detail = infer_turn_from_grid_diff(prev, curr, "black")
        self.assertEqual(side, "opp")
        self.assertEqual(detail["our_lost"], 1)
        self.assertEqual(detail["our_gained"], 1)

    def test_opp_move_then_our_to_play(self):
        prev = _empty()
        prev[9][4] = "k"
        prev[0][4] = "K"
        prev[0][1] = "N"
        curr = _copy(prev)
        curr[0][1] = "."
        curr[2][2] = "N"
        side, _ = infer_turn_from_grid_diff(prev, curr, "black")
        self.assertEqual(side, "our")

    def test_our_capture(self):
        prev = _empty()
        prev[9][4] = "k"
        prev[0][4] = "K"
        prev[5][4] = "n"
        prev[4][4] = "P"  # 对方兵
        curr = _copy(prev)
        curr[5][4] = "."
        curr[4][4] = "n"  # 吃掉
        side, detail = infer_turn_from_grid_diff(prev, curr, "black")
        self.assertEqual(side, "opp")
        self.assertEqual(detail["opp_lost"], 1)

    def test_ambiguous_defaults_our(self):
        prev = _empty()
        prev[9][4] = "k"
        prev[0][4] = "K"
        prev[9][1] = "n"
        prev[0][1] = "N"
        curr = _copy(prev)
        curr[9][1] = "."
        curr[7][2] = "n"
        curr[0][1] = "."
        curr[2][2] = "N"
        side, detail = infer_turn_from_grid_diff(prev, curr, "black")
        self.assertIsNone(side)
        self.assertEqual(detail["reason"], "ambiguous")

        old = turn_side.TURN_DETECT_ENABLED
        turn_side.TURN_DETECT_ENABLED = True
        try:
            out = resolve_to_move(
                None, curr, "black", prev_grid=prev,
            )
            self.assertEqual(out["to_move"], "our")
            self.assertEqual(out["method"], "default")
        finally:
            turn_side.TURN_DETECT_ENABLED = old

    def test_no_prev_defaults_our(self):
        curr = _empty()
        curr[9][4] = "k"
        curr[0][4] = "K"
        old = turn_side.TURN_DETECT_ENABLED
        turn_side.TURN_DETECT_ENABLED = True
        try:
            out = resolve_to_move(None, curr, "black", prev_grid=None)
            self.assertEqual(out["to_move"], "our")
            self.assertEqual(out["method"], "default")
        finally:
            turn_side.TURN_DETECT_ENABLED = old

    def test_disabled_skips_judgment(self):
        curr = _empty()
        curr[9][4] = "k"
        curr[0][4] = "K"
        old = turn_side.TURN_DETECT_ENABLED
        turn_side.TURN_DETECT_ENABLED = False
        try:
            out = resolve_to_move(None, curr, "black", prev_grid=None)
            self.assertEqual(out["to_move"], "our")
            self.assertEqual(out["method"], "disabled")
            self.assertEqual(format_turn_log(out, action="开始搜索"), "")
        finally:
            turn_side.TURN_DETECT_ENABLED = old

    def test_label(self):
        self.assertEqual(to_move_label("our"), "我方")
        self.assertEqual(to_move_label("opp"), "对方")
        self.assertIn("兜底", method_label("default"))
        self.assertIn("头像", method_label("avatar"))
        old = turn_side.TURN_DETECT_ENABLED
        turn_side.TURN_DETECT_ENABLED = True
        try:
            log = format_turn_log(
                {"to_move": "our", "method": "default"}, action="开始搜索",
            )
            self.assertIn("兜底", log)
            self.assertIn("默认我方", log)
        finally:
            turn_side.TURN_DETECT_ENABLED = old


if __name__ == "__main__":
    unittest.main()
