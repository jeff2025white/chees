"""中国象棋截图分析 · 图形界面（Tkinter）。

运行：
    python ui/app.py

布局（左右）：
    ┌──────────────────────────────┬──────────────────┐
    │                              │  中国象棋助手     │
    │         图片显示窗口          │  [ 打开图片 ]     │
    │                              │  [  截图  ]       │
    │                              │  [  计算  ]       │
    │                              │  ──────────       │
    ├──────────────────────────────┤  识别 / 着法信息  │
    │  [◀]  文件名 (3/8)  [▶]      │                  │
    └──────────────────────────────┴──────────────────┘
"""

import os
import queue
import sys
import threading

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import decide_position
from engine.recognize import analyze_image, annotate

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
BG = "#1f2430"
PANEL = "#2b3242"
FG = "#e6e6e6"
ACCENT = "#e23b3b"
UI_FONT = ("Microsoft YaHei", 10)
UI_FONT_B = ("Microsoft YaHei", 10, "bold")
TITLE_FONT = ("Microsoft YaHei", 13, "bold")


# 启动窗口尺寸（物理像素）
WINDOW_W, WINDOW_H = 1024, 768


def _enable_dpi_awareness():
    """让 Tkinter 按物理像素计算窗口尺寸。

    Windows 默认会对未声明 DPI 感知的程序做位图拉伸，系统缩放 125%/150% 时
    geometry("1024x768") 得到的实际像素并非 1024×768。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:                                    # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                                # noqa: BLE001
            pass


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中国象棋截图分析")
        self.root.configure(bg=BG)

        # 用屏幕尺寸约束窗口，避免在小屏或高缩放环境下超出可视区域
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(WINDOW_W, sw - 20)
        h = min(WINDOW_H, sh - 80)          # 预留标题栏与任务栏
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 20)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(800, 600)

        self.files = []            # 当前目录下的图片列表
        self.index = -1
        self.dir_path = ""
        self.current_path = None
        self.base_image = None     # PIL Image，当前显示源图（原图或标注图）
        self._photo = None         # 防止被 GC
        self._busy = False
        self._analysis = None
        self._decision = None
        self._queue = queue.Queue()   # 工作线程 → 主线程 的结果通道

        self._build_ui()
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.canvas.bind("<Configure>", lambda e: self._render())

        # 默认载入 images 目录
        default_dir = os.path.join(_ROOT, "images")
        if os.path.isdir(default_dir):
            self.load_directory(default_dir)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        # ---------- 左：图片区 ----------
        left = tk.Frame(outer, bg=PANEL)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#11141b", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 4))

        nav = tk.Frame(left, bg=PANEL)
        nav.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        self.btn_prev = tk.Button(nav, text="◀", width=4, font=UI_FONT_B,
                                  bg="#3b4459", fg=FG, activebackground="#4b5570",
                                  relief="flat", command=self.prev_image)
        self.btn_prev.pack(side="left")

        self.btn_next = tk.Button(nav, text="▶", width=4, font=UI_FONT_B,
                                  bg="#3b4459", fg=FG, activebackground="#4b5570",
                                  relief="flat", command=self.next_image)
        self.btn_next.pack(side="right")

        self.lbl_file = tk.Label(nav, text="未加载图片", bg=PANEL, fg=FG,
                                 font=UI_FONT, anchor="center")
        self.lbl_file.pack(side="left", fill="x", expand=True, padx=8)

        # ---------- 右：操作区 ----------
        right = tk.Frame(outer, bg=PANEL, width=340)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        tk.Label(right, text="中国象棋助手", bg=PANEL, fg=FG,
                 font=TITLE_FONT).pack(pady=(16, 14), padx=16, anchor="w")

        self.btn_open = self._mk_button(right, "打开图片", self.open_image)
        self.btn_shot = self._mk_button(right, "截图（暂未实现）", self.not_implemented)
        self.btn_shot.configure(state="disabled")
        self.btn_calc = self._mk_button(right, "计算", self.calculate)

        # 迭代深度拉杆
        depth_box = tk.Frame(right, bg=PANEL)
        depth_box.pack(fill="x", padx=16, pady=(6, 0))
        self.lbl_depth = tk.Label(depth_box, text="迭代深度：5 层", bg=PANEL, fg=FG,
                                  font=UI_FONT, anchor="w")
        self.lbl_depth.pack(fill="x")

        self.depth_var = tk.IntVar(value=5)
        self.depth_var.trace_add("write", self._on_depth_change)
        self.scale_depth = tk.Scale(
            depth_box, from_=2, to=12, orient="horizontal", variable=self.depth_var,
            showvalue=False, resolution=1, bg=PANEL, fg=FG,
            troughcolor="#3b4459", activebackground=ACCENT, highlightthickness=0,
            bd=0, sliderrelief="flat",
        )
        self.scale_depth.pack(fill="x")

        tk.Frame(right, bg="#3b4459", height=1).pack(fill="x", padx=16, pady=12)

        info_wrap = tk.Frame(right, bg=PANEL)
        info_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 12))

        sb = tk.Scrollbar(info_wrap, orient="vertical")
        sb.pack(side="right", fill="y")

        self.info = tk.Text(info_wrap, bg=PANEL, fg=FG, font=UI_FONT, wrap="word",
                            relief="flat", borderwidth=0, height=20,
                            insertbackground=FG, yscrollcommand=sb.set)
        self.info.pack(side="left", fill="both", expand=True)
        sb.configure(command=self.info.yview)
        self.info.tag_configure("head", foreground="#8fd3ff", font=UI_FONT_B)
        self.info.tag_configure("move", foreground=ACCENT, font=UI_FONT_B)
        self.info.tag_configure("opp", foreground="#b8c0d0")
        self.info.tag_configure("warn", foreground="#ffb86b")
        self.info.configure(state="disabled")

        self._write_info([("head", "使用说明\n"),
                          ("", "1. 打开图片或直接使用 images 目录\n"
                               "2. ◀ ▶ 切换上一张 / 下一张\n"
                               "3. 点击计算，走位线标注在图上\n"
                               "（方向键左右也可切换）")])

    def _mk_button(self, parent, text, cmd):
        btn = tk.Button(parent, text=text, font=UI_FONT_B, bg="#3b4459", fg=FG,
                        activebackground=ACCENT, activeforeground="#ffffff",
                        relief="flat", height=2, cursor="hand2", command=cmd)
        btn.pack(fill="x", padx=16, pady=4)
        return btn

    # -------------------------------------------------------------- 数据加载
    def load_directory(self, path, select=None):
        self.dir_path = path
        self.files = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(path, f))
        )
        if not self.files:
            self._write_info([("warn", "该目录下没有图片文件")])
            return
        self.index = self.files.index(select) if select in self.files else 0
        self._load_current()

    def _load_current(self):
        name = self.files[self.index]
        self.current_path = os.path.join(self.dir_path, name)
        self._analysis = None
        self._decision = None
        # 丢弃上一张图遗留的计算结果
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # 清空旧标注，显示原图
        bgr = cv2.imread(self.current_path)
        if bgr is None:
            messagebox.showerror("错误", f"无法读取图片：\n{self.current_path}")
            return
        self.base_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.lbl_file.configure(text=f"{name}   ({self.index + 1}/{len(self.files)})")
        self._render()
        self._write_info([("head", "已加载\n"), ("", f"{name}\n\n"),
                          ("", "点击「计算」识别局面并给出走位建议。")])

    def open_image(self):
        path = filedialog.askopenfilename(
            title="选择象棋截图",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        folder = os.path.dirname(path)
        self.load_directory(folder, select=os.path.basename(path))

    def prev_image(self):
        if not self.files:
            return
        self.index = (self.index - 1) % len(self.files)
        self._load_current()

    def next_image(self):
        if not self.files:
            return
        self.index = (self.index + 1) % len(self.files)
        self._load_current()

    def not_implemented(self):
        messagebox.showinfo("提示", "截图功能暂未实现。")

    def _on_depth_change(self, *_args):
        self.lbl_depth.configure(text=f"迭代深度：{self.depth_var.get()} 层")

    # ---------------------------------------------------------------- 渲染
    def _render(self):
        if self.base_image is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        img = self.base_image
        ratio = min(cw / img.width, ch / img.height)
        size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
        self._photo = ImageTk.PhotoImage(img.resize(size, Image.LANCZOS))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    # ---------------------------------------------------------------- 计算
    def calculate(self):
        if self._busy or not self.current_path:
            return
        self._busy = True
        self.btn_calc.configure(state="disabled", text="计算中…")
        self._write_info([("head", "分析中\n"), ("", "正在识别棋盘并搜索最佳着法…")])
        path = self.current_path
        depth = self.depth_var.get()
        threading.Thread(target=self._calc_worker, args=(path, depth), daemon=True).start()
        self.root.after(80, self._poll_result)

    def _calc_worker(self, path, max_depth):
        """后台线程：只做计算，结果放进队列，绝不触碰 Tk 对象。

        time_limit 随 depth 动态扩展：每层耗时约上一层 3 倍，
        3×depth 秒能保证深度够用时跑完，最低 4s，最高 60s。
        """
        try:
            analysis = analyze_image(path)
            # time_limit 随 depth 指数扩展：每层耗时约上一层 3 倍，
            # 每层 3 秒基准 × 2.5 留余量，最高 90 秒封顶避免太夸张。
            time_limit = min(90.0, max(4.0, 2.5 * (3 ** (max_depth - 4))))
            decision = decide_position(
                grid=analysis.grid_str,
                # 我方是棋盘下半边一方，即为它计算着法
                side=analysis.view_side,
                view_side=analysis.view_side,
                time_limit=time_limit,
                max_depth=max_depth,
                top_n=0,
            )
            vis = annotate(analysis, decision)
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            self._queue.put((analysis, decision, pil, None, time_limit))
        except Exception as exc:                      # noqa: BLE001
            self._queue.put((None, None, None, exc, None))

    def _poll_result(self):
        """主线程轮询：把后台结果取出来更新界面。"""
        try:
            analysis, decision, pil, error, _tl = self._queue.get_nowait()
        except queue.Empty:
            if self._busy:
                self.root.after(80, self._poll_result)
            return
        self._calc_done(analysis, decision, pil, error)

    def _calc_done(self, analysis, decision, pil, error):
        self._busy = False
        self.btn_calc.configure(state="normal", text="计算")

        if error is not None:
            self._write_info([("warn", "计算失败\n"), ("", str(error))])
            messagebox.showerror("计算失败", str(error))
            return

        self._analysis = analysis
        self._decision = decision
        self.base_image = pil
        self._render()

        side_txt = "红方在下" if analysis.red_bottom else "黑方在下"
        search_info = decision.get("search") or {}
        pv = decision.get("pv") or {}
        our_side = (decision.get("input") or {}).get("our_side", "red")

        lines = [
            ("head", "识别结果\n"),
            ("", f"红{analysis.red_count}/黑{analysis.black_count} · {side_txt}"
                 f" · 命中{analysis.hit}\n\n"),
            ("head", "推演序列\n"),
        ]

        best = decision.get("best_move")
        if best:
            names = pv.get("chinese") or []
            sides = pv.get("sides") or []
            scores = pv.get("scores") or []
            if names:
                for i, name in enumerate(names):
                    who = "我方" if (i < len(sides) and sides[i] == our_side) else "对方"
                    score = scores[i] / 100.0 if i < len(scores) else 0.0
                    tag = "move" if who == "我方" else "opp"
                    lines.append((tag, f"{i + 1:>2}.{name} {who} {score:+.2f}\n"))
            else:
                lines.append(("move", f"{best['chinese']}  {best['iccs']}\n"))

            lines.append(("", f"\n最终评价 {best['score_text']}"
                              f"{'  · 将军' if best.get('gives_check') else ''}"
                              f"{'  · 绝杀' if best.get('is_mate') else ''}\n"))
            lines.append(("", f"深度 {search_info.get('depth', 0)} 层"
                              f" · {search_info.get('nodes', 0):,} 节点"
                              f" · {search_info.get('time_ms', 0) / 1000:.1f}s\n"))
            if best.get("reason"):
                lines.append(("", f"{best['reason']}\n"))
        else:
            lines.append(("warn", decision.get("message", "没有可走的着法")))
        self._write_info(lines)

    # --------------------------------------------------------------- 工具
    def _write_info(self, parts):
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        for tag, text in parts:
            self.info.insert("end", text, tag or "")
        self.info.configure(state="disabled")


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    app = ChessApp(root)

    def report_size():
        root.update_idletasks()
        print(f"窗口尺寸: {root.winfo_width()}x{root.winfo_height()}  "
              f"屏幕: {root.winfo_screenwidth()}x{root.winfo_screenheight()}")

    root.after(300, report_size)
    root.mainloop()


if __name__ == "__main__":
    main()
