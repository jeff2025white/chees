#!/usr/bin/env python3
"""导出兵种训练样本到 datasets/pieces/{R,N,...}/。

来源：
  1) engine/piece_templates/color/{red,black}/*.png（已有彩色模板）
  2) test/gt_boards/*.json 标注图切格
  3) 开局图位置先验（占用完整时）

用法：
  python tools/export_piece_dataset.py
  python tools/export_piece_dataset.py --clear
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.piece_model import CLASSES, INPUT_SIZE  # noqa: E402
from engine.piece_ocr import extract_cell_color  # noqa: E402
from engine.recognize import (  # noqa: E402
    COLS, ROWS, _ROI_Y0, _ROI_Y1,
    classify, detect_our_side, find_grid,
)

OUT_ROOT = ROOT / "datasets" / "pieces"
COLOR_TMPL = ROOT / "engine" / "piece_templates" / "color"
GT_DIR = ROOT / "test" / "gt_boards"
INSPECT = ROOT / "2026-11-17" / "xiangqi-move-logic" / "work" / "inspect"

RED_BACK = list("RNBAKABNR")
BLK_BACK = list("rnbakabnr")


def _type_key(ch: str) -> str:
    return ch.upper()


def _save(out_dir: Path, ch: str, img, counters: dict) -> Path:
    key = _type_key(ch)
    d = out_dir / key
    d.mkdir(parents=True, exist_ok=True)
    side = "R" if ch.isupper() else "B"
    idx = counters.get(f"{key}_{side}", 0)
    counters[f"{key}_{side}"] = idx + 1
    path = d / f"{side}_{idx:04d}.png"
    if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img)
    return path


def export_color_templates(out_dir: Path, counters: dict) -> int:
    n = 0
    for side, upper in (("red", True), ("black", False)):
        d = COLOR_TMPL / side
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.png")):
            stem = f.stem.split("_")[0]
            if not stem or stem.upper() not in CLASSES:
                continue
            ch = stem.upper() if upper else stem.lower()
            img = cv2.imread(str(f))
            if img is None:
                continue
            _save(out_dir, ch, img, counters)
            n += 1
    return n


def _opening_labels(grid_raw, side: str) -> dict:
    red_bottom = side == "red"
    labels = {}

    def back_of(marker, prefer_top):
        order = range(3) if prefer_top else range(ROWS - 1, ROWS - 4, -1)
        for r in order:
            if sum(1 for c in range(COLS) if grid_raw[r][c] == marker) >= 8:
                return r
        return None

    red_back = back_of("@", not red_bottom)
    blk_back = back_of("#", red_bottom)
    if red_back is not None and sum(1 for c in range(COLS) if grid_raw[red_back][c] == "@") >= 8:
        if grid_raw[red_back][0] == "@" and grid_raw[red_back][8] == "@":
            for c in range(COLS):
                if grid_raw[red_back][c] == "@":
                    labels[(red_back, c)] = RED_BACK[c]
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
    if blk_back is not None and sum(1 for c in range(COLS) if grid_raw[blk_back][c] == "#") >= 8:
        if grid_raw[blk_back][0] == "#" and grid_raw[blk_back][8] == "#":
            for c in range(COLS):
                if grid_raw[blk_back][c] == "#":
                    labels[(blk_back, c)] = BLK_BACK[c]
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


def export_from_image(path: Path, labels: dict, out_dir: Path, counters: dict) -> int:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  skip unreadable {path}")
        return 0
    h, w = img.shape[:2]
    y0 = int(_ROI_Y0 * h)
    crop = img[y0:int(_ROI_Y1 * h), 0:w]
    row0, col0, cell, _, _ = find_grid(crop, full_w=w, full_h=h, crop_y0=y0)
    grid_raw, _ = classify(crop, row0, col0, cell)
    n = 0
    for (r, c), ch in labels.items():
        if not (0 <= r < ROWS and 0 <= c < COLS):
            continue
        marker = grid_raw[r][c]
        if marker == ".":
            continue
        if (marker == "@") != ch.isupper():
            continue
        cx = col0 + c * cell
        cy = row0 + r * cell
        patch = extract_cell_color(crop, cx, cy, cell)
        if patch is None:
            continue
        _save(out_dir, ch, patch, counters)
        n += 1
    return n


def export_gt_boards(out_dir: Path, counters: dict) -> int:
    n = 0
    if not GT_DIR.is_dir():
        return 0
    for jf in sorted(GT_DIR.glob("*.json")):
        meta = json.loads(jf.read_text(encoding="utf-8"))
        img_path = ROOT / meta["image"]
        if not img_path.is_file():
            print(f"  missing image for {jf.name}: {img_path}")
            continue
        grid_lines = meta["grid"].strip().splitlines()
        labels = {}
        for r, line in enumerate(grid_lines):
            line = line.strip()
            for c, ch in enumerate(line):
                if ch != "." and ch.upper() in CLASSES:
                    labels[(r, c)] = ch
        got = export_from_image(img_path, labels, out_dir, counters)
        print(f"  gt {jf.name}: {got} cells")
        n += got
    return n


def export_openings(out_dir: Path, counters: dict, limit: int = 30) -> int:
    n = 0
    cands = []
    cands.extend(sorted(INSPECT.glob("Screenshot*.png")))
    cands.extend(sorted((ROOT / "images").glob("auto_*.png"))[:80])
    seen = 0
    for path in cands:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        y0 = int(_ROI_Y0 * h)
        crop = img[y0:int(_ROI_Y1 * h), 0:w]
        row0, col0, cell, _, _ = find_grid(crop, full_w=w, full_h=h, crop_y0=y0)
        grid_raw, _ = classify(crop, row0, col0, cell)
        side = detect_our_side(grid_raw) or "red"
        labels = _opening_labels(grid_raw, side)
        if len(labels) < 20:
            continue
        got = export_from_image(path, labels, out_dir, counters)
        if got:
            print(f"  opening {path.name}: {got}")
            n += got
            seen += 1
        if seen >= limit:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    args = ap.parse_args()

    if args.clear and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    counters = {}
    total = 0
    print("== color templates ==")
    total += export_color_templates(args.out, counters)
    print(f"  templates: {total}")
    print("== gt_boards ==")
    total += export_gt_boards(args.out, counters)
    print("== openings ==")
    total += export_openings(args.out, counters)
    print(f"\nexported {total} patches → {args.out}")
    for ch in CLASSES:
        d = args.out / ch
        print(f"  {ch}: {len(list(d.glob('*.png'))) if d.is_dir() else 0}")


if __name__ == "__main__":
    main()
