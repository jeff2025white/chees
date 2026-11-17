"""GUI 冒烟测试：真实创建窗口、渲染、截图界面，然后退出。"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "ui"))

import tkinter as tk
from PIL import ImageGrab

from app import ChessApp, _enable_dpi_awareness

_enable_dpi_awareness()
root = tk.Tk()
app = ChessApp(root)
root.lift()
root.attributes("-topmost", True)
root.focus_force()
root.update()
time.sleep(0.8)
root.update()

# 切到第 2 张，模拟点击「计算」
app.next_image()
root.update()
time.sleep(0.3)
app.depth_var.set(5)          # 用拉杆选深度
app.scale_depth.set(5)
root.update()
app.calculate()

deadline = time.time() + 40
t0 = time.time()
while app._busy and time.time() < deadline:
    root.update()
    time.sleep(0.1)
    if int((time.time() - t0) * 1000) % 2000 < 100:
        print(f"  [{time.time()-t0:.1f}s] busy={app._busy} qsize={app._queue.qsize()}")
root.update()
time.sleep(0.4)
root.update()

x, y = root.winfo_rootx(), root.winfo_rooty()
w, h = root.winfo_width(), root.winfo_height()
print(f"窗口尺寸: {w}x{h}   屏幕: {root.winfo_screenwidth()}x{root.winfo_screenheight()}")
shot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
out = os.path.join(_ROOT, "images", "annotated", "ui_window.png")
shot.save(out)
print(f"窗口截图已保存: {out}")
print(f"busy={app._busy}  file={os.path.basename(app.current_path or '')}")
print(f"analysis={'ok' if app._analysis else 'None'}  "
      f"best={(app._decision or {}).get('best_move', {}).get('chinese', 'None')}")
if app._decision:
    si = app._decision.get("search") or {}
    print(f"拉杆深度=5 → 实际搜索深度={si.get('depth')} 节点={si.get('nodes')} "
          f"耗时={si.get('time_ms', 0) / 1000:.1f}s")
    pv = app._decision.get("pv") or {}
    for i, (ch, sd, sc) in enumerate(zip(pv.get("chinese", []),
                                         pv.get("sides", []),
                                         pv.get("scores", [])), 1):
        print(f"  {i}. {ch}  {sd}  {sc / 100:+.2f}")
    for i, st in enumerate((pv.get("steps") or [])[:3], 1):
        f, t = st["from"], st["to"]
        print(f"  step{i} from({f['row']},{f['col']})='{app._analysis.grid[f['row']][f['col']]}'"
              f" to({t['row']},{t['col']})='{app._analysis.grid[t['row']][t['col']]}'")
root.destroy()
