"""批量处理：识别 images 下所有截图，计算着法并输出标注图。

    python test/batch_run.py

输出：images/annotated/<原名>.png，以及 work/result_<原名>.json
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2

from engine import decide_position
from engine.recognize import analyze_image, annotate

IMG_DIR = os.path.join(_ROOT, "images")
OUT_DIR = os.path.join(IMG_DIR, "annotated")
WORK_DIR = os.path.join(_ROOT, "2026-11-17", "xiangqi-move-logic", "work")
RED = set("RNBAKCP")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png"))
    for fn in files:
        path = os.path.join(IMG_DIR, fn)
        print(f"\n{'=' * 60}\n{fn}")

        analysis = analyze_image(path)
        print(f"  网格 row0={analysis.row0:.0f} col0={analysis.col0:.0f} "
              f"cell={analysis.cell:.1f} 命中 {analysis.hit}")
        print(f"  红 {analysis.red_count} / 黑 {analysis.black_count}  视角: {analysis.view_side}")
        for line in analysis.grid_str.splitlines():
            print("    " + line)

        decision = decide_position(
            grid=analysis.grid_str,
            side=analysis.view_side,          # 我方 = 棋盘下半边一方
            view_side=analysis.view_side,
            time_limit=3.0, max_depth=8, top_n=3,
        )
        best = decision.get("best_move")
        if best:
            fc, fr = best["from"]["col"], best["from"]["row"]
            tc, tr = best["to"]["col"], best["to"]["row"]
            # 起点必须是本方棋子
            own = (lambda ch: ch in RED) if analysis.view_side == "red" \
                else (lambda ch: ch.islower() and ch != ".")
            ok = own(analysis.grid[fr][fc]) and not own(analysis.grid[tr][tc])
            print(f"  {'OK' if ok else 'BAD'} {best['chinese']} {best['iccs']} "
                  f"{best['score_text']}")
        else:
            print(f"  无着法: {decision.get('message')}")

        base = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(OUT_DIR, base + ".png"), annotate(analysis, decision))
        with open(os.path.join(WORK_DIR, f"result_{base}.json"), "w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
