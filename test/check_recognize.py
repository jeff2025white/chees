"""验证 engine.recognize 在全部截图上的网格定位与识别结果。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.recognize import analyze_image

img_dir = os.path.join(_ROOT, "images")
for fn in sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png")):
    try:
        a = analyze_image(os.path.join(img_dir, fn))
    except Exception as e:
        print(f"{fn}: ERROR {e}")
        continue
    print(f"\n{fn}  row0={a.row0:.0f} col0={a.col0:.0f} cell={a.cell:.1f} "
          f"score={a.score:.1f} hit={a.hit}  红{a.red_count}/黑{a.black_count}")
    for line in a.grid_str.splitlines():
        print("  " + line)
