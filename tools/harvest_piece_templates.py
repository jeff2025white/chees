#!/usr/bin/env python3
"""从截图采集单格墨迹模板到 engine/piece_templates/{red,black}/。

用法：
  python tools/harvest_piece_templates.py
  python tools/harvest_piece_templates.py images/Screenshot_xxx.png --labels path/to/labels.txt

默认：开局图用位置先验标全盘；103459 用内置中局 GT。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.piece_ocr import clear_template_cache, extract_cell_ink, save_template
from engine.recognize import (
    RED_BACKROW, BLACK_BACKROW, ROWS, COLS,
    _piece_masks, analyze_image,
)

# 中局人工核对过的局面（图坐标 row0=顶）
GT_103459 = {
    (0, 3): "A", (0, 4): "K", (0, 5): "A", (0, 6): "B", (0, 7): "R",
    (2, 0): "B", (2, 2): "N", (2, 4): "C", (2, 6): "N",
    (3, 0): "P", (3, 4): "P", (3, 6): "c", (3, 8): "P",
    (4, 7): "C",
    (6, 0): "p", (6, 1): "R", (6, 4): "p", (6, 6): "p", (6, 8): "p",
    (7, 0): "r", (7, 2): "n", (7, 4): "c", (7, 6): "n",
    (9, 2): "b", (9, 3): "a", (9, 4): "k", (9, 5): "a", (9, 6): "b", (9, 7): "r",
}


def _opening_labels(grid_raw, side):
    """近开局：按标准布局标兵种（仅当双方底线都较完整时可靠）。"""
    red_bottom = side == "red"
    labels = {}
    # 找底线
    def back_of(marker, prefer_top):
        order = range(3) if prefer_top else range(ROWS - 1, ROWS - 4, -1)
        for r in order:
            if sum(1 for c in range(COLS) if grid_raw[r][c] == marker) >= 8:
                return r
        return None

    red_back = back_of("@", not red_bottom)
    blk_back = back_of("#", red_bottom)
    if red_back is not None:
        for c in range(COLS):
            if grid_raw[red_back][c] == "@":
                labels[(red_back, c)] = RED_BACKROW[c]
        step = 1 if red_back <= 4 else -1
        cr, pr = red_back + 2 * step, red_back + 3 * step
        if 0 <= cr < ROWS:
            for c in (1, 7):
                if grid_raw[cr][c] == "@":
                    labels[(cr, c)] = "C"
        if 0 <= pr < ROWS:
            for c in (0, 2, 4, 6, 8):
                if grid_raw[pr][c] == "@":
                    labels[(pr, c)] = "P"
    if blk_back is not None:
        for c in range(COLS):
            if grid_raw[blk_back][c] == "#":
                labels[(blk_back, c)] = BLACK_BACKROW[c]
        step = 1 if blk_back <= 4 else -1
        cr, pr = blk_back + 2 * step, blk_back + 3 * step
        if 0 <= cr < ROWS:
            for c in (1, 7):
                if grid_raw[cr][c] == "#":
                    labels[(cr, c)] = "c"
        if 0 <= pr < ROWS:
            for c in (0, 2, 4, 6, 8):
                if grid_raw[pr][c] == "#":
                    labels[(pr, c)] = "p"
    return labels


def harvest(path: Path, labels: dict, counters: dict) -> int:
    a = analyze_image(str(path))
    x0, y0, x1, y1 = a.crop_box
    crop = a.image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    rm, bm, _ = _piece_masks(hsv)
    n = 0
    for (r, c), ch in labels.items():
        if a.grid_raw[r][c] == ".":
            print(f"  skip empty r{r}c{c} want {ch}")
            continue
        is_red = ch.isupper()
        # 颜色一致性检查
        marker = a.grid_raw[r][c]
        if (marker == "@") != is_red:
            print(f"  skip color mismatch r{r}c{c} raw={marker} label={ch}")
            continue
        ink = extract_cell_ink(crop, rm, bm, a.col0 + c * a.cell, a.row0 + r * a.cell,
                               a.cell, is_red)
        if ink is None:
            print(f"  skip no-ink r{r}c{c} {ch}")
            continue
        idx = counters.get(ch, 0)
        path_out = save_template(ink, ch, index=idx)
        counters[ch] = idx + 1
        print(f"  + {ch}_{idx} <- {path.name} r{r}c{c} -> {path_out}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="截图路径；默认采内置集")
    ap.add_argument("--clear", action="store_true", help="先清空 red/ black/ 再采")
    args = ap.parse_args()

    tmpl_root = ROOT / "engine" / "piece_templates"
    if args.clear:
        for sub in ("red", "black"):
            d = tmpl_root / sub
            if d.exists():
                for f in d.glob("*.png"):
                    f.unlink()
        # 清掉旧扁平大写残留，避免混淆
        for f in tmpl_root.glob("*.png"):
            f.unlink()
        clear_template_cache()
        print("cleared templates")

    jobs = []
    if args.images:
        for p in args.images:
            path = Path(p)
            a = analyze_image(str(path))
            side = a.side
            labels = _opening_labels(a.grid_raw, side)
            if len(labels) < 10:
                print(f"{path.name}: opening labels too few ({len(labels)}), skip")
                continue
            jobs.append((path, labels))
    else:
        opening = ROOT / "images" / "Screenshot_20261117-103346.png"
        mid = ROOT / "images" / "Screenshot_20261117-103459.png"
        if opening.exists():
            a = analyze_image(str(opening))
            jobs.append((opening, _opening_labels(a.grid_raw, a.side)))
        if mid.exists():
            jobs.append((mid, dict(GT_103459)))

    # 未 --clear 时从已有最大序号继续，避免覆盖旧模板
    counters = {}
    for sub, chars in (("red", "RNBAKCP"), ("black", "rnbakcp")):
        d = tmpl_root / sub
        if not d.exists():
            continue
        for ch in chars:
            idxs = []
            for f in d.glob(f"{ch}_*.png"):
                try:
                    idxs.append(int(f.stem.split("_", 1)[1]))
                except (IndexError, ValueError):
                    continue
            if idxs:
                counters[ch] = max(idxs) + 1
    total = 0
    for path, labels in jobs:
        print(f"\n== {path.name} ({len(labels)} labels) ==")
        total += harvest(path, labels, counters)
    print(f"\nharvested {total} templates; counts={dict(sorted(counters.items()))}")


if __name__ == "__main__":
    main()
