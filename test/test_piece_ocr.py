"""单格兵种 OCR / 模板匹配回归。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.piece_ocr import clear_template_cache, classify_cell_type, extract_cell_ink, load_templates
from engine.recognize import _piece_masks, analyze_image, validate_grid
import cv2

GT_103459 = {
    (0, 3): "A", (0, 4): "K", (0, 5): "A", (0, 6): "B", (0, 7): "R",
    (2, 0): "B", (2, 2): "N", (2, 4): "C", (2, 6): "N",
    (3, 0): "P", (3, 4): "P", (3, 6): "c", (3, 8): "P",
    (4, 7): "C",
    (6, 0): "p", (6, 1): "R", (6, 4): "p", (6, 6): "p", (6, 8): "p",
    (7, 0): "r", (7, 2): "n", (7, 4): "c", (7, 6): "n",
    (9, 2): "b", (9, 3): "a", (9, 4): "k", (9, 5): "a", (9, 6): "b", (9, 7): "r",
}


def test_templates_loaded_both_colors():
    clear_template_cache()
    tm = load_templates()
    for ch in "RNBAKCP":
        assert tm[ch], f"missing red templates for {ch}"
    for ch in "rnbakcp":
        assert tm[ch], f"missing black templates for {ch}"


def test_103459_grid_matches_gt():
    clear_template_cache()
    a = analyze_image(str(ROOT / "images" / "Screenshot_20261117-103459.png"))
    assert a.valid, validate_grid(a.grid, side=a.side)[1]
    wrong = [(r, c, gt, a.grid[r][c]) for (r, c), gt in GT_103459.items() if a.grid[r][c] != gt]
    assert not wrong, wrong


def test_opening_still_valid():
    clear_template_cache()
    a = analyze_image(str(ROOT / "images" / "Screenshot_20261117-103346.png"))
    assert a.red_count == 16 and a.black_count == 16
    assert a.valid
    assert a.grid[0] == list("RNBAKABNR")
    assert a.grid[9] == list("rnbakabnr")


def test_103438_left_rook_horse():
    """缺角底线勿套 RNB 模板：左侧红车(r0c1)、红马(r2c2)。"""
    clear_template_cache()
    a = analyze_image(str(ROOT / "images" / "Screenshot_20261117-103438.png"))
    assert a.grid[0][1] == "R", a.grid[0]
    assert a.grid[2][2] == "N", a.grid[2]
    assert a.grid[0][7] == "N", a.grid[0]
    assert a.grid[0][8] == "R", a.grid[0]
    assert a.grid[9] == list("rnbakabnr"), a.grid[9]


def test_112752_black_horse_elephant():
    """中局勿把 row2 当底线：r2c6=马、r2c4=象、r0c4=将。"""
    clear_template_cache()
    a = analyze_image(str(ROOT / "images" / "Screenshot_20261117-112752.png"))
    assert a.valid
    assert a.grid[2][6] == "n", a.grid[2][6]
    assert a.grid[2][4] == "b", a.grid[2][4]
    assert a.grid[0][4] == "k", a.grid[0][4]
    assert a.grid[0][6] == "b", a.grid[0][6]


def test_classify_cell_type_rook():
    clear_template_cache()
    a = analyze_image(str(ROOT / "images" / "Screenshot_20261117-103459.png"))
    x0, y0, x1, y1 = a.crop_box
    crop = a.image[y0:y1, x0:x1]
    rm, bm, _ = _piece_masks(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    # r0c7 = 红车
    ink = extract_cell_ink(crop, rm, bm, a.col0 + 7 * a.cell, a.row0 + 0 * a.cell, a.cell, True)
    ch, sc, _ = classify_cell_type(ink, True)
    assert ch == "R" and sc > 0.5, (ch, sc)


if __name__ == "__main__":
    test_templates_loaded_both_colors()
    print("ok templates")
    test_classify_cell_type_rook()
    print("ok classify_cell_type")
    test_103459_grid_matches_gt()
    print("ok 103459")
    test_opening_still_valid()
    print("ok opening")
    test_103438_left_rook_horse()
    print("ok 103438")
    test_112752_black_horse_elephant()
    print("ok 112752")
    print("ALL PASS")
