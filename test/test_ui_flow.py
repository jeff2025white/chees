"""界面逻辑冒烟测试：不启动窗口，复现「计算」按钮的完整流程。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cv2
from engine import decide_position
from engine.recognize import analyze_image, annotate

img_dir = os.path.join(_ROOT, "images")
out_dir = os.path.join(_ROOT, "images", "annotated")
os.makedirs(out_dir, exist_ok=True)

RED = set("RNBAKCP")
files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
for fn in files:
    path = os.path.join(img_dir, fn)
    analysis = analyze_image(path)
    decision = decide_position(grid=analysis.grid_str, side=analysis.view_side,
                               view_side=analysis.view_side,
                               time_limit=2.0, max_depth=6, top_n=3)
    best = decision.get("best_move")
    status = "NO-MOVE"
    if best:
        fc, fr = best["from"]["col"], best["from"]["row"]
        tc, tr = best["to"]["col"], best["to"]["row"]
        src = analysis.grid[fr][fc]
        dst = analysis.grid[tr][tc]
        own = (lambda ch: ch in RED) if analysis.view_side == "red" \
            else (lambda ch: ch.islower() and ch != ".")
        ok = own(src) and not own(dst)
        status = f"{'OK ' if ok else 'BAD'} {best['chinese']} {best['iccs']} from'{src}' to'{dst}'"
    vis = annotate(analysis, decision)
    cv2.imwrite(os.path.join(out_dir, "ui_" + fn), vis)
    print(f"{fn}: view={analysis.view_side} 红{analysis.red_count}/黑{analysis.black_count}  {status}")

print("\n标注图已输出到 images/annotated/ui_*.png")
