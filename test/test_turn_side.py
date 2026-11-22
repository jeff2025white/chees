"""行棋方识别：头像绿边（定位+绿量）+ 局面差分 + 兜底。"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.turn_side as turn_side
from engine.turn_side import (
    border_green_count,
    detect_avatar_turn,
    format_turn_log,
    infer_turn_from_grid_diff,
    locate_avatar_boxes,
    method_label,
    resolve_to_move,
    to_move_label,
)

SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images", "auto_20261118_171632.png",
)


def _empty():
    return [["."] * 9 for _ in range(10)]


def _copy(g):
    return [list(row) for row in g]


class TestGridDiffTurn(unittest.TestCase):
    def test_our_move_then_opp_to_play(self):
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
        prev[4][4] = "P"
        curr = _copy(prev)
        curr[5][4] = "."
        curr[4][4] = "n"
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


class TestAvatarGreenCompare(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(SAMPLE), "sample screenshot missing")
    def test_locate_and_compare_sample(self):
        img = cv2.imread(SAMPLE)
        self.assertIsNotNone(img)
        boxes = locate_avatar_boxes(img)
        self.assertIsNotNone(boxes)
        self.assertIn("opp", boxes)
        self.assertIn("our", boxes)
        # 我方头像应在右下
        self.assertGreater(boxes["our"][0], 0.5)
        self.assertGreater(boxes["our"][1], 0.85)

        our_g = border_green_count(img, boxes["our"])
        opp_g = border_green_count(img, boxes["opp"])
        self.assertGreater(our_g["count"], opp_g["count"])
        self.assertGreater(our_g["count"], 100)

        side, detail, used = detect_avatar_turn(img)
        self.assertEqual(side, "our")
        self.assertFalse(detail.get("cached"))
        self.assertEqual(used["our"], boxes["our"])

        # 缓存框再次判定，结果一致且标记 cached
        side2, detail2, _ = detect_avatar_turn(img, avatar_boxes=boxes)
        self.assertEqual(side2, "our")
        self.assertTrue(detail2.get("cached"))

    @unittest.skipUnless(os.path.isfile(SAMPLE), "sample screenshot missing")
    def test_resolve_returns_avatar_boxes(self):
        img = cv2.imread(SAMPLE)
        grid = _empty()
        grid[9][4] = "k"
        grid[0][4] = "K"
        old = turn_side.TURN_DETECT_ENABLED
        turn_side.TURN_DETECT_ENABLED = True
        try:
            out = resolve_to_move(img, grid, "red", prev_grid=None)
            self.assertEqual(out["to_move"], "our")
            self.assertEqual(out["method"], "avatar")
            self.assertIsNotNone(out.get("avatar_boxes"))
            log = format_turn_log(out, action="开始搜索")
            self.assertIn("绿量我", log)
            self.assertIn("框=新定位", log)

            out2 = resolve_to_move(
                img, grid, "red", prev_grid=None,
                avatar_boxes=out["avatar_boxes"],
            )
            self.assertEqual(out2["method"], "avatar")
            self.assertIn("框=缓存", format_turn_log(out2))
        finally:
            turn_side.TURN_DETECT_ENABLED = old

    def test_synthetic_green_border_compare(self):
        # 构造：左上对方框无绿，右下我方框有绿边
        img = np.zeros((400, 300, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        # opp square top-left
        cv2.rectangle(img, (20, 20), (80, 80), (180, 180, 180), 2)
        # our square bottom-right with green border
        cv2.rectangle(img, (200, 300), (270, 370), (0, 255, 0), 3)
        boxes = {
            "opp": (20 / 300, 20 / 400, 80 / 300, 80 / 400),
            "our": (200 / 300, 300 / 400, 270 / 300, 370 / 400),
        }
        side, detail, _ = detect_avatar_turn(img, avatar_boxes=boxes)
        self.assertEqual(side, "our")
        self.assertGreater(detail["our_green"], detail["opp_green"])


if __name__ == "__main__":
    unittest.main()
