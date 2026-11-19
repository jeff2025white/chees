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

from engine.piece_ocr import (
    clear_template_cache, extract_cell_ink, extract_cell_color,
    save_template, save_color_template,
)
from engine.recognize import (
    RED_BACKROW, BLACK_BACKROW, ROWS, COLS,
    _piece_masks, analyze_image, classify, find_grid, detect_our_side,
    _ROI_Y0, _ROI_Y1,
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


def harvest(path: Path, labels: dict, counters: dict, color_counters: dict,
            do_ink: bool = True, do_color: bool = True) -> int:
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
        marker = a.grid_raw[r][c]
        if (marker == "@") != is_red:
            print(f"  skip color mismatch r{r}c{c} raw={marker} label={ch}")
            continue
        cx = a.col0 + c * a.cell
        cy = a.row0 + r * a.cell
        if do_ink:
            ink = extract_cell_ink(crop, rm, bm, cx, cy, a.cell, is_red)
            if ink is None:
                print(f"  skip no-ink r{r}c{c} {ch}")
            else:
                idx = counters.get(ch, 0)
                path_out = save_template(ink, ch, index=idx)
                counters[ch] = idx + 1
                print(f"  + ink {ch}_{idx} <- {path.name} r{r}c{c}")
                n += 1
        if do_color:
            bgr = extract_cell_color(crop, cx, cy, a.cell)
            if bgr is None:
                print(f"  skip no-color r{r}c{c} {ch}")
            else:
                idx = color_counters.get(ch, 0)
                path_out = save_color_template(bgr, ch, index=idx)
                color_counters[ch] = idx + 1
                print(f"  + color {ch}_{idx} <- {path.name} r{r}c{c} -> {path_out}")
                n += 1
    return n


def _labels_from_grid_raw(path: Path):
    """不跑完整兵种 OCR，只用占用图 + 开局位置先验标号（采模板更干净）。"""
    img = cv2.imread(str(path))
    if img is None:
        return None, {}
    h, w = img.shape[:2]
    y0 = int(_ROI_Y0 * h)
    crop = img[y0:int(_ROI_Y1 * h), 0:w]
    row0, col0, cell, _, _ = find_grid(crop, full_w=w, full_h=h, crop_y0=y0)
    grid_raw, _ = classify(crop, row0, col0, cell)
    side = detect_our_side(grid_raw) or "red"
    return side, _opening_labels(grid_raw, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="截图路径；默认采内置集")
    ap.add_argument("--clear", action="store_true", help="先清空 ink/color 模板再采")
    ap.add_argument("--ink-only", action="store_true")
    ap.add_argument("--color-only", action="store_true")
    ap.add_argument("--max-per-type", type=int, default=6, help="每种最多保留新采数量上限提示")
    args = ap.parse_args()

    do_ink = not args.color_only
    do_color = not args.ink_only

    tmpl_root = ROOT / "engine" / "piece_templates"
    color_root = tmpl_root / "color"
    if args.clear:
        for sub in ("red", "black"):
            for root in (tmpl_root, color_root):
                d = root / sub
                if d.exists():
                    for f in d.glob("*.png"):
                        f.unlink()
        for f in tmpl_root.glob("*.png"):
            f.unlink()
        clear_template_cache()
        print("cleared templates")

    jobs = []
    if args.images:
        for p in args.images:
            path = Path(p)
            _side, labels = _labels_from_grid_raw(path)
            if len(labels) < 10:
                print(f"{path.name}: opening labels too few ({len(labels)}), skip")
                continue
            jobs.append((path, labels))
    else:
        # 默认：images/ 根目录里占用较满的开局样张（最多 12 张）
        cands = sorted((ROOT / "images").glob("*.png"))
        for path in cands:
            _side, labels = _labels_from_grid_raw(path)
            if len(labels) >= 20:
                jobs.append((path, labels))
            if len(jobs) >= 12:
                break
        mid = ROOT / "images" / "Screenshot_20261117-103459.png"
        if mid.exists():
            jobs.append((mid, dict(GT_103459)))

    def _next_counters(root: Path):
        counters = {}
        for sub, chars in (("red", "RNBAKCP"), ("black", "rnbakcp")):
            d = root / sub
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
        return counters

    counters = _next_counters(tmpl_root) if do_ink else {}
    color_counters = _next_counters(color_root) if do_color else {}
    total = 0
    for path, labels in jobs:
        print(f"\n== {path.name} ({len(labels)} labels) ==")
        total += harvest(path, labels, counters, color_counters, do_ink, do_color)
    print(f"\nharvested {total}; ink={dict(sorted(counters.items()))} "
          f"color={dict(sorted(color_counters.items()))}")


if __name__ == "__main__":
    main()
