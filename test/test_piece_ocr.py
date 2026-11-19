"""单格兵种 OCR / 模板匹配 / Torch CNN 回归。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.piece_ocr import clear_template_cache, classify_cell_type, extract_cell_ink, load_templates
from engine.piece_model import clear_model_cache, load_piece_model, torch_available
from engine.recognize import _piece_masks, analyze_image, validate_grid
import cv2

# 标定分辨率下的自动截图（782×1332 系）
OPENING = ROOT / "images" / "auto_20261118_172439.png"
EARLY = ROOT / "images" / "auto_20261118_171632.png"


def test_templates_loaded_both_colors():
    clear_template_cache()
    tm = load_templates()
    for ch in "RNBAKCP":
        assert tm[ch], f"missing red templates for {ch}"
    for ch in "rnbakcp":
        assert tm[ch], f"missing black templates for {ch}"


def test_opening_auto_full():
    clear_template_cache()
    clear_model_cache()
    assert OPENING.is_file(), OPENING
    a = analyze_image(str(OPENING))
    assert a.red_count == 16 and a.black_count == 16, (a.red_count, a.black_count)
    assert a.valid, validate_grid(a.grid, side=a.side)[1]
    assert a.grid[0] == list("rnbakabnr"), a.grid[0]
    assert a.grid[9] == list("RNBAKABNR"), a.grid[9]


def test_early_mid_has_kings():
    clear_template_cache()
    clear_model_cache()
    if not EARLY.is_file():
        return
    a = analyze_image(str(EARLY))
    assert a.valid or ("K" in a.grid_str and "k" in a.grid_str)
    assert sum(1 for row in a.grid for ch in row if ch == "K") == 1
    assert sum(1 for row in a.grid for ch in row if ch == "k") == 1


def test_classify_cell_type_rook():
    clear_template_cache()
    if not OPENING.is_file():
        return
    a = analyze_image(str(OPENING))
    x0, y0, x1, y1 = a.crop_box
    crop = a.image[y0:y1, x0:x1]
    rm, bm, _ = _piece_masks(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    # 红底车 r9c0
    ink = extract_cell_ink(crop, rm, bm, a.col0 + 0 * a.cell, a.row0 + 9 * a.cell, a.cell, True)
    ch, sc, _ = classify_cell_type(ink, True)
    assert ch == "R" and sc > 0.35, (ch, sc)


def test_torch_model_loads():
    clear_model_cache()
    if not torch_available():
        print("skip torch (not installed)")
        return
    loaded = load_piece_model()
    assert loaded is not None, "expected engine/models/piece_cls.pt"


def test_gt_boards_cell_accuracy():
    """对照 gt_boards 计算单格准确率（有模型时要求 ≥0.90）。"""
    clear_template_cache()
    clear_model_cache()
    gt_dir = ROOT / "test" / "gt_boards"
    total = correct = 0
    for jf in sorted(gt_dir.glob("*.json")):
        meta = json.loads(jf.read_text(encoding="utf-8"))
        img = ROOT / meta["image"]
        if not img.is_file():
            alt = ROOT / "images" / Path(meta["image"]).name
            if alt.is_file():
                img = alt
            else:
                continue
        # 跳过错误分辨率的旧 inspect 图
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        if not (h >= 1200 and 700 <= w <= 850):
            continue
        a = analyze_image(str(img))
        for r, line in enumerate(meta["grid"].strip().splitlines()):
            for c, gt in enumerate(line.strip()):
                if gt == ".":
                    continue
                total += 1
                if a.grid[r][c] == gt:
                    correct += 1
                elif not meta.get("partial", False):
                    assert a.grid[r][c] == gt, (jf.name, r, c, gt, a.grid[r][c])
    if total == 0:
        print("skip gt_boards (no usable images)")
        return
    acc = correct / total
    print(f"gt_boards acc={acc:.3f} ({correct}/{total})")
    if torch_available() and load_piece_model() is not None:
        assert acc >= 0.90, acc
    else:
        assert acc >= 0.80, acc


def test_20261118_jumped_horses_not_rooks():
    clear_template_cache()
    clear_model_cache()
    cases = [
        ("capture_20261118_130754.png", [(2, 2, "N"), (2, 6, "N"), (7, 2, "n"), (7, 6, "n")]),
        ("capture_20261118_130659.png", [(2, 2, "N"), (7, 2, "n")]),
        ("Screenshot_20261118-123743.png", [(2, 2, "N"), (2, 6, "N"), (7, 2, "n"), (7, 6, "n")]),
    ]
    for name, cells in cases:
        path = ROOT / "images" / name
        if not path.exists():
            continue
        a = analyze_image(str(path))
        for r, c, expect in cells:
            assert a.grid[r][c] == expect, (name, r, c, expect, a.grid[r][c], a.grid)


if __name__ == "__main__":
    test_templates_loaded_both_colors()
    print("ok templates")
    test_torch_model_loads()
    print("ok torch model")
    test_classify_cell_type_rook()
    print("ok classify_cell_type")
    test_opening_auto_full()
    print("ok opening")
    test_early_mid_has_kings()
    print("ok early kings")
    test_gt_boards_cell_accuracy()
    print("ok gt_boards")
    test_20261118_jumped_horses_not_rooks()
    print("ok optional horse cases")
    print("ALL PASS")
