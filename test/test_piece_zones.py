"""兵种位置带简单测试：九宫 / 象田 / 兵卒前进范围。

识别流水线在 ``validate_grid`` 里已调用 ``check_piece_zones``；
本文件用合成局面 + images/auto 样张做回归。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.recognize import (  # noqa: E402
    analyze_image,
    check_piece_zones,
    validate_grid,
)
from engine.piece_ocr import clear_template_cache  # noqa: E402


def _empty():
    return [["."] * 9 for _ in range(10)]


def test_palace_king_advisor_ok_and_bad():
    g = _empty()
    # 红在下：九宫 row7-9, col3-5
    g[9][4] = "K"
    g[8][3] = "A"
    g[0][4] = "k"
    g[1][5] = "a"
    ok, viol = check_piece_zones(g, side="red")
    assert ok, viol

    g[5][4] = "K"  # 帅出九宫
    ok, viol = check_piece_zones(g, side="red")
    assert not ok
    assert (5, 4) in viol


def test_elephant_field_ok_and_bad():
    g = _empty()
    g[9][4] = "K"
    g[0][4] = "k"
    g[7][0] = "B"   # 红相合法
    g[2][4] = "b"   # 黑象合法
    ok, viol = check_piece_zones(g, side="red")
    assert ok, viol

    g[5][4] = "B"  # 过河/非田字
    ok, viol = check_piece_zones(g, side="red")
    assert not ok
    assert (5, 4) in viol


def test_pawn_forward_range():
    g = _empty()
    g[9][4] = "K"
    g[0][4] = "k"
    # 红在下：兵只能在 row0..6
    g[6][0] = "P"
    g[3][4] = "P"
    g[0][8] = "P"
    # 黑卒只能在 row3..9
    g[3][2] = "p"
    g[9][2] = "p"
    ok, viol = check_piece_zones(g, side="red")
    assert ok, viol

    g[8][0] = "P"  # 红兵退到己方底线后
    ok, viol = check_piece_zones(g, side="red")
    assert not ok
    assert (8, 0) in viol

    g2 = _empty()
    g2[9][4] = "K"
    g2[0][4] = "k"
    g2[1][0] = "p"  # 黑卒退回己方底线后（红在下时黑在上，row0-2 非法）
    ok, viol = check_piece_zones(g2, side="red")
    assert not ok
    assert (1, 0) in viol


def test_black_bottom_orientation():
    """黑在下时九宫/象/兵行翻转。"""
    g = _empty()
    g[9][4] = "k"
    g[8][3] = "a"
    g[0][4] = "K"
    g[7][0] = "b"
    g[2][0] = "B"
    g[6][4] = "p"  # 黑在下，卒行 0..6
    g[3][4] = "P"  # 红在上，兵行 3..9
    ok, viol = check_piece_zones(g, side="black")
    assert ok, viol

    g[1][4] = "P"  # 红在上起点 row3，退到 row1 → 非法
    ok, viol = check_piece_zones(g, side="black")
    assert not ok
    assert (1, 4) in viol


def test_validate_grid_includes_zones():
    g = _empty()
    g[9][4] = "K"
    g[0][4] = "k"
    g[7][7] = "A"  # 仕出九宫
    valid, msgs, viol = validate_grid(g, side="red")
    assert not valid
    assert (7, 7) in viol
    assert any("九宫" in m for m in msgs)


def test_auto_frames_pass_zones_after_ocr():
    """images/auto 两张样张：识别后位置带应通过。"""
    clear_template_cache()
    paths = sorted((ROOT / "images" / "auto").glob("auto_*.png"))
    assert paths, "缺少 images/auto 样张"
    for path in paths:
        a = analyze_image(str(path))
        ok, viol = check_piece_zones(a.grid, side=a.view_side)
        assert ok, (path.name, viol, a.grid_str)


if __name__ == "__main__":
    test_palace_king_advisor_ok_and_bad()
    print("ok palace")
    test_elephant_field_ok_and_bad()
    print("ok elephant")
    test_pawn_forward_range()
    print("ok pawn")
    test_black_bottom_orientation()
    print("ok black-bottom")
    test_validate_grid_includes_zones()
    print("ok validate_grid")
    test_auto_frames_pass_zones_after_ocr()
    print("ok auto frames")
    print("ALL PASS")
