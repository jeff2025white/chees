"""中国象棋截图分析 · 图形界面（Tkinter）。

运行：
    python ui/app.py

布局（左右；右侧三等分等高）：
    ┌──────────────────────────────┬──────────────────┐
    │                              │  棋局解说（直播） │
    │         图片显示窗口          ├──────────────────┤
    │                              │  操作按钮 / 拉杆  │
    │                              ├──────────────────┤
    ├──────────────────────────────┤  日志（处理步骤）│
    │  [◀]  文件名 (3/8)  [▶]      │                  │
    └──────────────────────────────┴──────────────────┘

自动监听（Windows）：找「手游助手」窗口 → 每秒截客户区 → 特效过滤
→ analyze_array → 主将判定棋盘 → 行棋方（头像绿边/局面差分，兜底我方）
→ 对方行棋则跳过计算；我方行棋且 grid 变化 → decide + annotate。
默认不落盘；「自动截图」拉杆 >0 时写入 images/auto/ 并环形保留。
本窗标题可显示为「QQ游戏」，与截图目标关键字分开。
"""

import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import decide_position
from engine.fx_overlay import detect_board_fx
from engine.recognize import analyze_array, analyze_image, annotate
from engine.turn_side import (
    TURN_DETECT_ENABLED,
    format_turn_log,
    method_label,
    resolve_to_move,
    to_move_label,
)
from ui.advice import AdviceView
from ui.prefs import get_depth, get_show_piece_labels, set_depth, set_show_piece_labels

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
# 本窗标题栏（可与截图目标不同，避免监听截到自己）
APP_WINDOW_TITLE = "QQ游戏"
# 自动监听 / 截图：按窗口标题关键字匹配（雷电「手游助手 x …」）
TARGET_WINDOW_NAME = "手游助手"
BG = "#1f2430"
PANEL = "#2b3242"
FG = "#e6e6e6"
ACCENT = "#e23b3b"
BTN_BG = "#3b4459"
BTN_HOVER = "#4b5570"
BTN_DISABLED_BG = "#252b38"
BTN_DISABLED_FG = "#7a8499"
UI_FONT = ("Microsoft YaHei", 10)
UI_FONT_B = ("Microsoft YaHei", 10, "bold")
TITLE_FONT = ("Microsoft YaHei", 13, "bold")

# 自动监听落盘：0=不保存（默认）；>0 时写入 images/auto/ 并只保留最近 N 张
AUTO_SAVE_KEEP_DEFAULT = 0
AUTO_SAVE_KEEP_MAX = 50


# 启动窗口尺寸（物理像素）
WINDOW_W, WINDOW_H = 1064, 768


class FlatButton(tk.Label):
    """自绘按钮。macOS Aqua 会忽略 tk.Button 的 bg/fg，导致浅底浅字不可读。"""

    def __init__(self, parent, text, command=None, **kwargs):
        self._command = command
        self._enabled = True
        self._bg = kwargs.pop("bg", BTN_BG)
        self._fg = kwargs.pop("fg", FG)
        self._hover = kwargs.pop("activebackground", BTN_HOVER)
        kwargs.setdefault("cursor", "hand2")
        super().__init__(parent, text=text, bg=self._bg, fg=self._fg, **kwargs)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _event=None):
        if self._enabled:
            super().configure(bg=self._hover)

    def _on_leave(self, _event=None):
        super().configure(bg=self._bg if self._enabled else BTN_DISABLED_BG)

    def _on_click(self, _event=None):
        if self._enabled and self._command:
            self._command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs = {**cnf, **kwargs}
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._enabled = str(state) not in ("disabled", str(tk.DISABLED))
            if self._enabled:
                kwargs.setdefault("bg", self._bg)
                kwargs.setdefault("fg", self._fg)
                kwargs.setdefault("cursor", "hand2")
            else:
                kwargs.setdefault("bg", BTN_DISABLED_BG)
                kwargs.setdefault("fg", BTN_DISABLED_FG)
                kwargs.setdefault("cursor", "arrow")
        if kwargs:
            return super().configure(**kwargs)
        return None

    config = configure


def _enable_dpi_awareness():
    """让 Tkinter 按物理像素计算窗口尺寸。

    Windows 默认会对未声明 DPI 感知的程序做位图拉伸，系统缩放 125%/150% 时
    geometry("1064x768") 得到的实际像素并非标称宽高。
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


def _set_console_title(title: str) -> None:
    """把背后黑控制台标题改掉（默认常带脚本名/截图分析字样）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(str(title))
    except Exception:  # noqa: BLE001
        pass


def _find_game_window(keyword=TARGET_WINDOW_NAME, exclude_hwnds=()):
    """枚举可见顶层窗口，返回标题包含 keyword 的句柄列表（仅 Windows）。

    exclude_hwnds：排除本助手等窗口，防止标题关键字撞车时截到自己。
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []
    skip = {int(h) for h in exclude_hwnds if h}
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        if int(hwnd) in skip:
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if keyword in buf.value:
            found.append(hwnd)
        return True

    user32.EnumWindows(enum_proc(_cb), 0)
    return found


def _bring_to_front(hwnd):
    """恢复最小化窗口并置前，便于从屏幕截取客户区。"""
    import ctypes

    user32 = ctypes.windll.user32
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)       # SW_RESTORE
    else:
        user32.ShowWindow(hwnd, 5)       # SW_SHOW
    user32.SetForegroundWindow(hwnd)


def _capture_window_client(hwnd):
    """截取窗口客户区（不含标题栏与边框），返回 PIL Image。

    模拟器多为 DirectX/OpenGL 渲染，PrintWindow / 窗口 DC BitBlt 常得到黑图，
    因此直接按客户区矩形从屏幕截取；调用前需确保窗口已置前且未被遮挡。
    进程已声明 DPI 感知，客户区坐标与 ImageGrab 同为物理像素。
    """
    import ctypes
    from ctypes import wintypes
    from PIL import ImageGrab

    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError("GetClientRect 失败")
    if rect.right <= 0 or rect.bottom <= 0:
        raise OSError("窗口客户区尺寸为 0，可能已最小化")
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise OSError("ClientToScreen 失败")
    box = (origin.x, origin.y, origin.x + rect.right, origin.y + rect.bottom)
    return ImageGrab.grab(bbox=box, all_screens=True)


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_WINDOW_TITLE)
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

        # ---- 自动监听模式 ----
        self._auto_running = False
        self._auto_thread = None
        self._auto_stop = threading.Event()
        self._computing = threading.Event()     # 计算中 → 暂停截图
        self._last_grid_str = None               # 上一帧棋盘串（帧差）
        self._last_grid = None                   # 上一帧 grid（行棋方差分）
        self._auto_skip_reason = None            # 等待态日志去重
        self._target_hwnd = None                 # 找到的目标游戏窗口句柄

        # 操作栏工作状态动画（等待 / 计算中）
        self._work_mode = None                   # None | "wait" | "calc"
        self._work_label = ""
        self._work_phase = 0
        self._work_anim_after = None

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
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1, uniform="halves")
        outer.columnconfigure(1, weight=1, uniform="halves")

        # ---------- 左：图片区 ----------
        left = tk.Frame(outer, bg=PANEL)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.canvas = tk.Canvas(left, bg="#11141b", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 4))

        nav = tk.Frame(left, bg=PANEL)
        nav.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        self.btn_prev = FlatButton(nav, text="◀", width=4, font=UI_FONT_B,
                                   command=self.prev_image, padx=8, pady=4)
        self.btn_prev.pack(side="left")

        self.btn_next = FlatButton(nav, text="▶", width=4, font=UI_FONT_B,
                                   command=self.next_image, padx=8, pady=4)
        self.btn_next.pack(side="right")

        self.lbl_file = tk.Label(nav, text="未加载图片", bg=PANEL, fg=FG,
                                 font=UI_FONT, anchor="center")
        self.lbl_file.pack(side="left", fill="x", expand=True, padx=8)

        # ---------- 右：棋局解说 / 操作 / 日志 三等分 ----------
        right = tk.Frame(outer, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right.columnconfigure(0, weight=1)
        for r in range(3):
            right.rowconfigure(r, weight=1, uniform="right_thirds")

        top = tk.Frame(right, bg=PANEL)
        top.grid(row=0, column=0, sticky="nsew", padx=16, pady=(12, 4))
        tk.Label(top, text="中国象棋助手", bg=PANEL, fg=FG,
                 font=TITLE_FONT, anchor="w").pack(fill="x", pady=(0, 6))
        self.verdict = self._mk_text_panel(top, "棋局解说")
        self.advice = AdviceView(self.verdict)

        # 外包线：与上下解说/日志区分
        mid_border = tk.Frame(right, bg="#5a6578")
        mid_border.grid(row=1, column=0, sticky="nsew", padx=16, pady=6)
        mid = tk.Frame(mid_border, bg=PANEL)
        mid.pack(fill="both", expand=True, padx=1, pady=1)
        ops_head = tk.Frame(mid, bg=PANEL)
        ops_head.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(ops_head, text="操作", bg=PANEL, fg="#8fd3ff",
                 font=UI_FONT_B, anchor="w").pack(side="left")
        self.lbl_work = tk.Label(
            ops_head, text="", bg=PANEL, fg="#ffb86b",
            font=UI_FONT, anchor="e",
        )
        self.lbl_work.pack(side="right", fill="x", expand=True, padx=(8, 0))
        mid_body = tk.Frame(mid, bg=PANEL)
        mid_body.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        # 第一行：打开图片 | 计算
        self.btn_open, self.btn_calc = self._mk_button_row(
            mid_body,
            [("打开图片", self.open_image), ("计算", self.calculate)],
        )
        # 第二行：自动监听/停止 | 重新截图
        self.btn_shot, self.btn_resnap = self._mk_button_row(
            mid_body,
            [("自动监听", self.toggle_auto_listen), ("重新截图", self.resnapshot)],
        )

        depth_box = tk.Frame(mid_body, bg=PANEL)
        depth_box.pack(fill="x", pady=(4, 0))
        _depth0 = get_depth()
        self.lbl_depth = tk.Label(depth_box, text=f"迭代深度：{_depth0} 层", bg=PANEL, fg=FG,
                                  font=UI_FONT, anchor="w")
        self.lbl_depth.pack(fill="x")

        self.depth_var = tk.IntVar(value=_depth0)
        self.depth_var.trace_add("write", self._on_depth_change)
        self._ensure_scale_style()
        self.scale_depth = ttk.Scale(
            depth_box, from_=2, to=12, orient="horizontal",
            variable=self.depth_var, style="Accent.Horizontal.TScale",
            command=lambda v: self.depth_var.set(int(round(float(v)))),
        )
        self.scale_depth.pack(fill="x")

        keep_box = tk.Frame(mid_body, bg=PANEL)
        keep_box.pack(fill="x", pady=(6, 0))
        self.lbl_auto_keep = tk.Label(
            keep_box, text=self._auto_keep_label(AUTO_SAVE_KEEP_DEFAULT),
            bg=PANEL, fg=FG, font=UI_FONT, anchor="w",
        )
        self.lbl_auto_keep.pack(fill="x")
        self.auto_keep_var = tk.IntVar(value=AUTO_SAVE_KEEP_DEFAULT)
        self.auto_keep_var.trace_add("write", self._on_auto_keep_change)
        self.scale_auto_keep = ttk.Scale(
            keep_box, from_=0, to=AUTO_SAVE_KEEP_MAX, orient="horizontal",
            variable=self.auto_keep_var, style="Accent.Horizontal.TScale",
            command=lambda v: self.auto_keep_var.set(int(round(float(v)))),
        )
        self.scale_auto_keep.pack(fill="x")

        # 棋子兵种文字开关（记忆偏好；切换后立刻重绘当前标注）
        self._show_piece_labels = get_show_piece_labels(True)
        self.show_labels_var = tk.BooleanVar(value=self._show_piece_labels)
        self.chk_labels = tk.Checkbutton(
            mid_body,
            text="棋子上显示兵种",
            variable=self.show_labels_var,
            command=self._on_show_labels_change,
            bg=PANEL, fg=FG, selectcolor=BTN_BG,
            activebackground=PANEL, activeforeground=FG,
            font=UI_FONT, anchor="w", highlightthickness=0, bd=0,
        )
        self.chk_labels.pack(fill="x", pady=(8, 0))

        self.btn_play = FlatButton(
            mid_body, text="人机对弈 →", font=UI_FONT_B, command=self.open_play,
            activebackground=ACCENT, padx=8, pady=8,
        )
        self.btn_play.pack(fill="x", pady=(8, 0))

        bottom = tk.Frame(right, bg=PANEL)
        bottom.grid(row=2, column=0, sticky="nsew", padx=16, pady=(4, 12))
        self.log = self._mk_text_panel(bottom, "日志")

        self.advice.write([
            ("head", "直播解说面板\n"),
            ("", f"1. 打开图片 / 监听；动画干扰时可点「重新截图」立刻重算\n"
                 "2. 计算后这里给出口播建议、形势与推演\n"
                 "3. 图上标注前两手走位\n"
                 "（方向键 ◀ ▶ 可切换图片）"),
        ])
        self._write_log([])

    def _mk_button_row(self, parent, specs):
        """同一行并排按钮；specs 为 [(text, cmd), ...]。"""
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=3)
        btns = []
        n = len(specs)
        for i, (text, cmd) in enumerate(specs):
            row.columnconfigure(i, weight=1, uniform="ops_btn")
            btn = FlatButton(
                row, text=text, font=UI_FONT_B, command=cmd,
                activebackground=ACCENT, padx=8, pady=8,
            )
            pad_l = 0 if i == 0 else 3
            pad_r = 0 if i == n - 1 else 3
            btn.grid(row=0, column=i, sticky="ew", padx=(pad_l, pad_r))
            btns.append(btn)
        return btns

    def _mk_button(self, parent, text, cmd):
        btn = FlatButton(parent, text=text, font=UI_FONT_B, command=cmd,
                         activebackground=ACCENT, padx=8, pady=8)
        btn.pack(fill="x", pady=3)
        return btn

    def _mk_text_panel(self, parent, title):
        """在父容器内铺满的只读文本区（父容器负责三等分高度）。"""
        tk.Label(parent, text=title, bg=PANEL, fg="#8fd3ff",
                 font=UI_FONT_B, anchor="w").pack(fill="x", pady=(0, 4))
        wrap = tk.Frame(parent, bg=PANEL)
        wrap.pack(fill="both", expand=True)
        sb = tk.Scrollbar(wrap, orient="vertical")
        sb.pack(side="right", fill="y")
        text = tk.Text(
            wrap, bg=PANEL, fg=FG, font=UI_FONT, wrap="word",
            relief="flat", borderwidth=0, height=1,
            insertbackground=FG, yscrollcommand=sb.set,
        )
        text.pack(side="left", fill="both", expand=True)
        sb.configure(command=text.yview)
        text.tag_configure("head", foreground="#8fd3ff", font=UI_FONT_B)
        text.tag_configure("move", foreground=ACCENT, font=UI_FONT_B)
        text.tag_configure("opp", foreground="#b8c0d0")
        text.tag_configure("warn", foreground="#ffb86b")
        text.configure(state="disabled")
        return text

    # -------------------------------------------------------------- 数据加载
    def load_directory(self, path, select=None):
        self.dir_path = path
        self.files = sorted(
            f for f in os.listdir(path)
            if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(path, f))
        )
        if not self.files:
            self.advice.show_warn("该目录下没有图片文件")
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
        self._log_step(f"已加载 · {name}")
        self.advice.show_hint("点击「计算」，生成本手直播解说建议。")

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

    def _own_hwnds(self):
        """本助手顶层窗句柄，查找目标窗时排除，避免截到自己。"""
        try:
            return (int(self.root.winfo_id()),)
        except tk.TclError:
            return ()

    def _resolve_target_hwnd(self):
        """优先用监听中的目标窗；否则按标题关键字查找。"""
        import ctypes

        if self._target_hwnd and ctypes.windll.user32.IsWindow(self._target_hwnd):
            return self._target_hwnd
        hwnds = _find_game_window(exclude_hwnds=self._own_hwnds())
        return hwnds[0] if hwnds else None

    def resnapshot(self):
        """重新截取目标窗并立刻识别+计算。

        用于自动监听抓到走子动画帧、识别失误时：棋子落定后手动纠正。
        """
        if sys.platform != "win32":
            messagebox.showinfo("提示", "窗口截图目前仅支持 Windows。")
            return
        if self._busy:
            self._log_step("计算中，请稍后再重新截图", warn=True)
            return
        hwnd = self._resolve_target_hwnd()
        if not hwnd:
            messagebox.showwarning(
                "未找到窗口",
                f"没有找到标题包含「{TARGET_WINDOW_NAME}」的窗口。\n"
                f"请先启动手游助手并打开象棋游戏。",
            )
            return
        try:
            _bring_to_front(hwnd)
        except OSError as exc:
            messagebox.showerror("截图失败", str(exc))
            return
        self._log_step(f"重新截图 · 置前后抓取并计算（{TARGET_WINDOW_NAME}）")
        # 等窗口完成前台渲染再截，避免拿到旧帧或被本窗遮挡的画面
        self.root.after(500, lambda: self._resnap_and_calc(hwnd))

    def _resnap_and_calc(self, hwnd):
        """主线程：抓一帧 → 预览 → 后台识别计算。"""
        import ctypes
        import numpy as np

        if self._busy:
            return
        if not ctypes.windll.user32.IsWindow(hwnd):
            self._log_step("重新截图失败：目标窗口已关闭", warn=True)
            messagebox.showwarning("窗口丢失", f"{TARGET_WINDOW_NAME}窗口已关闭。")
            return
        try:
            pil_img = _capture_window_client(hwnd)
        except Exception as exc:                          # noqa: BLE001
            self._log_step(f"截图失败：{exc}", warn=True)
            messagebox.showerror("截图失败", str(exc))
            return

        folder = self.dir_path or os.path.join(_ROOT, "images")
        os.makedirs(folder, exist_ok=True)
        name = time.strftime("capture_%Y%m%d_%H%M%S") + ".png"
        save_path = os.path.join(folder, name)
        try:
            pil_img.save(save_path)
        except OSError as exc:
            self._log_step(f"截图保存失败：{exc}", warn=True)
            messagebox.showerror("截图失败", str(exc))
            return

        # 更新当前图片列表，但不走「请点计算」提示
        self.dir_path = folder
        self.files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f))
        )
        self.index = self.files.index(name) if name in self.files else 0
        self.current_path = save_path
        self._analysis = None
        self._decision = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self.base_image = pil_img
        self.lbl_file.configure(text=f"{name}   ({self.index + 1}/{len(self.files)})")
        self._render()

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        depth = self.depth_var.get()
        self._busy = True
        self.btn_calc.configure(state="disabled", text="计算中…")
        self.btn_resnap.configure(state="disabled")
        # 监听中：暂停自动截帧，避免与本次手动重算抢结果
        if self._auto_running:
            self._computing.set()
        self._begin_round(
            f"重新截图 · {name}（{pil_img.width}×{pil_img.height}）"
            f" · 深度{depth} · 识别行棋方"
        )
        self.root.after(80, self._poll_result)
        threading.Thread(
            target=self._resnap_calc_worker,
            args=(frame, save_path, depth),
            daemon=True,
        ).start()

    def _resnap_calc_worker(self, frame, path, max_depth):
        """后台：对重新截取的帧做识别+搜索（对方行棋则跳过计算）。"""
        try:
            analysis = analyze_array(frame, path=path)
            turn = resolve_to_move(
                frame, analysis.grid, analysis.view_side,
                prev_grid=self._last_grid,
            )
            self._last_grid_str = analysis.grid_str
            self._last_grid = [list(row) for row in analysis.grid]
            if TURN_DETECT_ENABLED and turn["to_move"] == "opp":
                # 对方走棋：只更新局面基准，不搜索
                self._queue.put((
                    analysis, None,
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    None, None,
                ))
                msg = format_turn_log(turn, action="对方行棋，跳过计算")
                if msg:
                    self.root.after(0, lambda m=msg: self._log_step(m, warn=True))
                return
            msg = format_turn_log(turn, action="开始计算")
            if msg:
                self.root.after(0, lambda m=msg: self._log_step(m))
            time_limit = min(90.0, max(4.0, 2.5 * (3 ** (max_depth - 4))))
            decision = decide_position(
                grid=analysis.grid_str,
                side=analysis.view_side,
                view_side=analysis.view_side,
                time_limit=time_limit,
                max_depth=max_depth,
                top_n=0,
            )
            vis = annotate(
                analysis, decision, draw_labels=self._show_piece_labels,
            )
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            self._queue.put((analysis, decision, pil, None, time_limit))
        except Exception as exc:                      # noqa: BLE001
            self._queue.put((None, None, None, exc, None))
        finally:
            self._computing.clear()

    # --------------------------------------------------------- 自动监听
    def toggle_auto_listen(self):
        """切换自动监听模式：每秒截图 → 主将判断棋盘 → 帧差 → 自动计算。"""
        if self._auto_running:
            self.stop_auto_listen()
        else:
            self.start_auto_listen()

    def start_auto_listen(self):
        if sys.platform != "win32":
            messagebox.showinfo("提示", "窗口截图目前仅支持 Windows。")
            return
        hwnds = _find_game_window(exclude_hwnds=self._own_hwnds())
        if not hwnds:
            messagebox.showwarning(
                "未找到窗口",
                f"没有找到标题包含「{TARGET_WINDOW_NAME}」的窗口。\n"
                f"请先启动手游助手并打开象棋游戏。",
            )
            return
        hwnd = hwnds[0]
        try:
            _bring_to_front(hwnd)
        except OSError as exc:
            messagebox.showerror("置前失败", str(exc))
            return
        self._target_hwnd = hwnd
        self._auto_running = True
        self._auto_stop.clear()
        self._last_grid_str = None
        self._last_grid = None
        self._auto_skip_reason = None  # 等待态去重：仅状态变化时打一行
        self._computing.clear()
        # UI 切换
        self.btn_shot.configure(text="停止监听", activebackground="#ff6b6b")
        self.btn_open.configure(state="disabled")
        self.btn_calc.configure(state="disabled")
        keep = int(self.auto_keep_var.get())
        keep_txt = "不落盘" if keep <= 0 else f"保留最近{keep}张"
        win_title = self._hwnd_title(hwnd) or "?"
        self._write_log([])
        self._log_step(f"启动自动监听 · 截图{keep_txt}")
        self._log_step(
            f"目标窗 · hwnd={int(hwnd)} · 「{win_title}」"
            f" · 关键字「{TARGET_WINDOW_NAME}」 · 深度{self.depth_var.get()}"
        )
        self.advice.show_waiting()
        self._set_work("wait", "监听中 · 等待局面")
        # 等前台渲染就绪再开循环
        self.root.after(500, self._spawn_auto_thread)

    def _spawn_auto_thread(self):
        if not self._auto_running:
            return
        self._auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
        self._auto_thread.start()

    def stop_auto_listen(self):
        """停止自动监听。"""
        self._auto_running = False
        self._auto_stop.set()
        self._computing.set()          # 如果卡在计算等待，解除阻塞让循环退出
        self.btn_shot.configure(text="自动监听", activebackground=ACCENT)
        self.btn_open.configure(state="normal")
        self.btn_calc.configure(state="normal")
        self._set_work(None)
        self._log_step("停止自动监听")

    def _auto_loop(self):
        """后台线程：每秒截图 → 特效过滤 → 识别 → 主将判断棋盘 → 帧差 → 触发计算。"""
        import ctypes
        import numpy as np

        # 首次启动后等窗口稳定，再截第 1 帧
        time.sleep(0.3)

        while not self._auto_stop.is_set():
            # 计算中 → 暂停截图，每 200ms 轮询一次
            while self._computing.is_set():
                if self._auto_stop.is_set():
                    break
                time.sleep(0.2)
            if self._auto_stop.is_set():
                break

            # 窗口有效性检查
            if not ctypes.windll.user32.IsWindow(self._target_hwnd):
                self.root.after(0, self._auto_window_lost)
                return

            # 截图（内存中 PIL → BGR ndarray，不落盘）
            try:
                pil_img = _capture_window_client(self._target_hwnd)
            except Exception as exc:                      # noqa: BLE001
                self.root.after(0, lambda e=exc: self._log_step(
                    f"截图失败：{type(e).__name__}: {e}", warn=True))
                time.sleep(1.0)
                continue
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            fh, fw = frame.shape[:2]

            # 中部墨云/吃/将军等特效 → 跳过识别与计算，等下一帧
            try:
                fx = detect_board_fx(frame)
            except Exception as exc:                      # noqa: BLE001
                self.root.after(0, lambda e=exc: self._log_step(
                    f"特效检测异常：{type(e).__name__}: {e}", warn=True))
                time.sleep(1.0)
                continue
            if fx.get("dirty"):
                level = fx.get("level", "?")
                reason = f"fx:{level}"
                if reason != self._auto_skip_reason:
                    self._auto_skip_reason = reason
                    detail = self._fx_log_detail(fx, fw, fh)
                    self.root.after(0, lambda d=detail, lv=level: (
                        self._log_step(f"跳过 · 特效中 · {d}", warn=True),
                        self._set_work("wait", f"监听中 · 特效（{lv}）"),
                    ))
                time.sleep(1.0)
                continue

            # 识别
            try:
                analysis = analyze_array(frame, path="<auto>")
            except Exception as exc:                      # noqa: BLE001
                self.root.after(0, lambda e=exc, w=fw, h=fh: self._log_step(
                    f"识别异常 · 帧{w}x{h} · {type(e).__name__}: {e}", warn=True))
                time.sleep(1.0)
                continue

            # 主将判断：同时有红帅 K 和黑将 k → 有棋盘
            grid_str = analysis.grid_str
            has_board = ("K" in grid_str) and ("k" in grid_str)
            r_cnt = sum(1 for row in analysis.grid for ch in row if ch in "RNBAKCP")
            b_cnt = sum(1 for row in analysis.grid for ch in row
                        if ch.islower() and ch != ".")

            if not has_board:
                # 无棋盘（首页/结算页等）→ 不触发计算，继续等下一帧
                detail = (
                    f"帧{fw}x{fh} · 命中{analysis.hit}"
                    f" · 红{r_cnt}/黑{b_cnt}"
                    f" · 有帅{'K' if 'K' in grid_str else '×'}"
                    f"/有将{'k' if 'k' in grid_str else '×'}"
                    f" · view={getattr(analysis, 'view_side', '?')}"
                )
                if self._auto_skip_reason != "no_board":
                    self._auto_skip_reason = "no_board"
                    self.root.after(0, lambda d=detail: (
                        self._log_step(f"跳过 · 无棋盘 · {d}", warn=True),
                        self._set_work("wait", "监听中 · 无棋盘"),
                    ))
                self._last_grid_str = None
                self._last_grid = None
                time.sleep(1.0)
                continue

            # 行棋方：头像绿边 → 局面差分 → 兜底我方
            turn = resolve_to_move(
                frame, analysis.grid, analysis.view_side,
                prev_grid=self._last_grid,
            )
            turn_txt = to_move_label(turn["to_move"])

            # 跟上次棋盘比
            changed = (self._last_grid_str is not None) and (grid_str != self._last_grid_str)
            first_time = (self._last_grid_str is None)

            # 清除等待态去重（对方行棋且局面未变时保留「对方行棋」提示）
            if self._auto_skip_reason is not None:
                keep_opp = (
                    str(self._auto_skip_reason).startswith("opp_turn")
                    and turn["to_move"] == "opp"
                    and not changed
                    and not first_time
                )
                if not keep_opp:
                    self._auto_skip_reason = None
                    if not self._computing.is_set() and turn["to_move"] == "our":
                        self.root.after(
                            0, lambda: self._set_work("wait", "监听中 · 等待局面"),
                        )

            # 局面未变：静默等待；首次/变化 → 视行棋方决定是否计算
            if first_time or changed:
                self.root.after(0, lambda p=pil_img: self._auto_show_raw(p))
                side_txt = "红方在下" if analysis.red_bottom else "黑方在下"
                # 先更新帧差基准（对方行棋跳过计算时也要跟上，便于下次差分）
                self._last_grid_str = grid_str
                self._last_grid = [list(row) for row in analysis.grid]

                if TURN_DETECT_ENABLED and turn["to_move"] == "opp":
                    reason = f"opp_turn:{turn['method']}"
                    if reason != self._auto_skip_reason:
                        self._auto_skip_reason = reason
                        how = method_label(turn["method"])
                        step = (
                            f"{'首次识别' if first_time else '局面变化'}"
                            f" · 帧{fw}x{fh} · 红{r_cnt}/黑{b_cnt}"
                            f" · {side_txt} · 行棋={turn_txt} · 判定={how}"
                            f" · 跳过计算"
                        )
                        self.root.after(0, lambda s=step: (
                            self._log_step(s, warn=True),
                            self._set_work("wait", "监听中 · 对方行棋"),
                        ))
                    # 等满 1 秒再截下一帧
                    self._auto_stop.wait(1.0)
                    continue

                if TURN_DETECT_ENABLED:
                    how = method_label(turn["method"])
                    turn_part = f" · 行棋={turn_txt} · 判定={how}"
                else:
                    turn_part = ""
                step = (
                    f"{'首次识别' if first_time else '局面变化'}"
                    f" · 帧{fw}x{fh} · 红{r_cnt}/黑{b_cnt}"
                    f" · {side_txt} · 命中{analysis.hit}"
                    f" · view={analysis.view_side}"
                    f"{turn_part}"
                    f" · 深度{self.depth_var.get()}"
                    f" · 开始计算"
                )
                self.root.after(0, lambda s=step: self._begin_round(s))
                self._computing.set()     # 计算期间暂停截图
                self._busy = True         # 让轮询保持运行
                self.root.after(80, self._poll_result)
                # 可选落盘（默认不保存；计算用内存中的 analysis）
                save_path = self._save_auto_frame_if_needed(pil_img)
                if save_path:
                    analysis.path = save_path
                    self.root.after(0, lambda p=save_path: self._log_step(
                        f"已落盘 · {os.path.basename(p)}"))

                depth = self.depth_var.get()
                threading.Thread(
                    target=self._auto_calc_worker,
                    args=(analysis, depth),
                    daemon=True,
                ).start()

            # 等满 1 秒再截下一帧
            self._auto_stop.wait(1.0)

    @staticmethod
    def _auto_keep_label(n: int) -> str:
        if n <= 0:
            return "自动截图：不保存"
        return f"自动截图：保留最近 {n} 张"

    def _on_auto_keep_change(self, *_args):
        self.lbl_auto_keep.configure(
            text=self._auto_keep_label(int(self.auto_keep_var.get()))
        )

    def _auto_save_dir(self) -> str:
        return os.path.join(_ROOT, "images", "auto")

    def _prune_auto_saves(self, keep: int) -> None:
        """只保留 images/auto/auto_*.png 中最新的 keep 张。"""
        if keep <= 0:
            return
        folder = self._auto_save_dir()
        if not os.path.isdir(folder):
            return
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.startswith("auto_") and name.lower().endswith(".png")
        ]
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for path in files[keep:]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _save_auto_frame_if_needed(self, pil_img):
        """按拉杆决定是否落盘；返回路径或 None。"""
        try:
            keep = int(self.auto_keep_var.get())
        except (TypeError, ValueError, tk.TclError):
            keep = 0
        if keep <= 0:
            return None
        folder = self._auto_save_dir()
        os.makedirs(folder, exist_ok=True)
        save_name = time.strftime("auto_%Y%m%d_%H%M%S") + ".png"
        save_path = os.path.join(folder, save_name)
        try:
            pil_img.save(save_path)
        except OSError as exc:
            self.root.after(0, lambda e=exc: self._log_step(f"截图保存失败：{e}", warn=True))
            return None
        self._prune_auto_saves(keep)
        return save_path

    def _auto_calc_worker(self, analysis, max_depth):
        """自动模式下的计算 worker：完成后恢复截图循环。"""
        try:
            # 用已经识别好的 analysis 直接算，不用再识别
            time_limit = min(90.0, max(4.0, 2.5 * (3 ** (max_depth - 4))))
            decision = decide_position(
                grid=analysis.grid_str,
                side=analysis.view_side,
                view_side=analysis.view_side,
                time_limit=time_limit,
                max_depth=max_depth,
                top_n=0,
            )
            vis = annotate(
                analysis, decision, draw_labels=self._show_piece_labels,
            )
            pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            self._queue.put((analysis, decision, pil, None, time_limit))
        except Exception as exc:                      # noqa: BLE001
            self._queue.put((None, None, None, exc, None))
        finally:
            self._computing.clear()    # 解除截图暂停

    def _auto_show_raw(self, pil_img):
        """主线程：显示原始帧（不等计算完成）。"""
        self.base_image = pil_img
        self._render()

    def _begin_round(self, first_step=None):
        """每轮计算开始：解说面板与日志一并清空，再写入本轮首条步骤。"""
        self._set_work("calc", "计算中")
        self.advice.clear_analyzing()
        self._write_log([])
        if first_step:
            self._log_step(first_step)

    def _auto_window_lost(self):
        self.stop_auto_listen()
        self._log_step("目标窗口已关闭，监听停止", warn=True)
        messagebox.showwarning(
            "窗口丢失",
            f"{TARGET_WINDOW_NAME}窗口已关闭，自动监听停止。",
        )

    def open_play(self):
        """另开人机对弈窗口（与本窗并存）。"""
        script = os.path.join(_HERE, "play.py")
        try:
            subprocess.Popen(
                [sys.executable, script],
                cwd=_ROOT,
                start_new_session=True,
            )
        except OSError as exc:
            messagebox.showerror("无法启动", str(exc))

    @staticmethod
    def _ensure_scale_style():
        """ttk.Scale 在 clam 下可设滑块色；Windows 原生 tk.Scale 滑块几乎不可着色。"""
        style = ttk.Style()
        if style.theme_use() != "clam":
            style.theme_use("clam")
        style.configure(
            "Accent.Horizontal.TScale",
            background=ACCENT,
            troughcolor="#1a2030",
            bordercolor=PANEL,
            lightcolor=ACCENT,
            darkcolor="#b82e2e",
            sliderthickness=18,
            gripcount=0,
        )

    def _on_depth_change(self, *_args):
        d = self.depth_var.get()
        self.lbl_depth.configure(text=f"迭代深度：{d} 层")
        set_depth(d)

    def _on_show_labels_change(self):
        self._show_piece_labels = bool(self.show_labels_var.get())
        set_show_piece_labels(self._show_piece_labels)
        self._refresh_annotation()
        self._log_step(
            "兵种文字：" + ("开" if self._show_piece_labels else "关")
        )

    def _refresh_annotation(self):
        """开关切换后，用当前分析结果重画标注图。"""
        if self._analysis is None:
            return
        try:
            vis = annotate(
                self._analysis,
                self._decision,
                draw_labels=self._show_piece_labels,
            )
        except Exception as exc:  # noqa: BLE001
            self._log_step(f"重绘标注失败：{exc}", warn=True)
            return
        self.base_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self._render()

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
        self.btn_resnap.configure(state="disabled")
        name = os.path.basename(self.current_path)
        depth = self.depth_var.get()
        path = self.current_path
        self._begin_round(f"开始计算 · {name} · 深度{depth} · path={path}")
        threading.Thread(target=self._calc_worker, args=(path, depth), daemon=True).start()
        self.root.after(80, self._poll_result)

    def _calc_worker(self, path, max_depth):
        """后台线程：识别 → 行棋方判定 → 我方才搜索；结果进队列，不碰 Tk。"""
        try:
            analysis = analyze_image(path)
            turn = resolve_to_move(
                analysis.image, analysis.grid, analysis.view_side,
                prev_grid=self._last_grid,
            )
            self._last_grid_str = analysis.grid_str
            self._last_grid = [list(row) for row in analysis.grid]

            if TURN_DETECT_ENABLED and turn["to_move"] == "opp":
                pil = Image.fromarray(
                    cv2.cvtColor(analysis.image, cv2.COLOR_BGR2RGB),
                )
                self._queue.put((analysis, None, pil, None, None))
                msg = format_turn_log(turn, action="对方行棋，跳过计算")
                if msg:
                    self.root.after(0, lambda m=msg: self._log_step(m, warn=True))
                return

            msg = format_turn_log(turn, action="开始搜索")
            if msg:
                self.root.after(0, lambda m=msg: self._log_step(m))
            # time_limit 随 depth 指数扩展：每层耗时约上一层 3 倍，
            # 每层 3 秒基准 × 2.5 留余量，最高 90 秒封顶避免太夸张。
            time_limit = min(90.0, max(4.0, 2.5 * (3 ** (max_depth - 4))))
            decision = decide_position(
                grid=analysis.grid_str,
                # 第一步固定为我方：图下方先走（与红黑无关）
                side=analysis.view_side,
                view_side=analysis.view_side,
                time_limit=time_limit,
                max_depth=max_depth,
                top_n=0,
            )
            vis = annotate(
                analysis, decision, draw_labels=self._show_piece_labels,
            )
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
        auto_mode = self._auto_running
        self._busy = False
        self.btn_resnap.configure(state="normal")
        if not auto_mode:
            self.btn_calc.configure(state="normal", text="计算")
            self._set_work(None)
        else:
            self.btn_calc.configure(state="disabled", text="计算")
            self._set_work("wait", "监听中 · 等待局面")

        img_name = os.path.basename(self.current_path or "")
        if error is not None:
            self._log_step(
                f"计算失败 · {img_name or '<auto>'} · {type(error).__name__}: {error}",
                warn=True,
            )
            self.advice.show_error(error)
            if not auto_mode:
                messagebox.showerror("计算失败", str(error))
            return

        self._analysis = analysis
        self._decision = decision
        self.base_image = pil
        self._render()

        # 对方行棋跳过搜索时 decision 为空，只展示原图
        if decision is None:
            self.advice.show_hint("对方行棋中，暂不分析。轮到我方后再计算。")
            if auto_mode:
                self._set_work("wait", "监听中 · 对方行棋")
            return

        # 棋局解说由 ui.advice 主输出；过程信息只进日志
        self.advice.show_decision(analysis, decision)
        self._log_calc_process(analysis, decision, img_name)

    def _log_calc_process(self, analysis, decision, img_name=""):
        """过程日志：识别概况 / 最佳着 / 搜索统计（不进解说面板）。"""
        side_txt = "红方在下" if analysis.red_bottom else "黑方在下"
        inp = decision.get("input") or {}
        our_side = inp.get("view_side") or inp.get("our_side") or analysis.view_side
        turn = inp.get("side_to_move", "red")
        turn_txt = "红方行棋" if turn == "red" else "黑方行棋"
        our_txt = "红" if our_side == "red" else "黑"
        if img_name:
            self._log_step(f"当前图片 · {img_name}")
        self._log_step(
            f"识别结果 · 红{analysis.red_count}/黑{analysis.black_count}"
            f" · {side_txt}（我方={our_txt}） · {turn_txt}"
            f" · 命中{analysis.hit} · view={analysis.view_side}"
            f" · path={getattr(analysis, 'path', '') or '?'}"
        )
        best = decision.get("best_move")
        search_info = decision.get("search") or {}
        if not best:
            self._log_step(
                f"计算无着法：{decision.get('message', '没有可走的着法')}",
                warn=True,
            )
            return
        self._log_step(
            f"最佳着 · {best.get('chinese', '?')} ({best.get('iccs', '?')})"
            f" · {best.get('score_text', '?')}"
        )
        src = "开局库+" if search_info.get("book") else ""
        depth_n = search_info.get("depth", 0)
        nodes = search_info.get("nodes", 0)
        secs = search_info.get("time_ms", 0) / 1000.0
        pv_len = len((decision.get("pv") or {}).get("chinese") or [])
        self._log_step(
            f"计算完成 · {src}深度{depth_n}"
            f" · 推演{pv_len}步 · {nodes:,}节点 · {secs:.1f}s"
            f" · book={'是' if search_info.get('book') else '否'}"
        )

    # --------------------------------------------------------------- 工具
    def _write_text(self, widget, parts):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for tag, text in parts:
            widget.insert("end", text, tag or "")
        widget.configure(state="disabled")

    def _write_log(self, parts):
        self._write_text(self.log, parts)

    # ----------------------------------------------------------- 工作动画
    # ASCII 旋转符（避免部分中文字体缺 braille 字形）
    _WORK_FRAMES = ("|", "/", "-", "\\", "|", "/", "-", "\\")

    def _set_work(self, mode, label=""):
        """操作栏状态：None 空闲 / wait 等待 / calc 计算中（带动画）。"""
        self._work_mode = mode
        self._work_label = label or ""
        if mode is None:
            self._stop_work_anim()
            self.lbl_work.configure(text="", fg="#ffb86b")
            return
        fg = ACCENT if mode == "calc" else "#ffb86b"
        self.lbl_work.configure(fg=fg)
        self._tick_work_anim()

    def _stop_work_anim(self):
        if self._work_anim_after is not None:
            try:
                self.root.after_cancel(self._work_anim_after)
            except tk.TclError:
                pass
            self._work_anim_after = None

    def _tick_work_anim(self):
        if not self._work_mode:
            self._stop_work_anim()
            return
        frame = self._WORK_FRAMES[self._work_phase % len(self._WORK_FRAMES)]
        self._work_phase += 1
        self.lbl_work.configure(text=f"{frame}  {self._work_label}")
        self._stop_work_anim()
        self._work_anim_after = self.root.after(100, self._tick_work_anim)

    @staticmethod
    def _hwnd_title(hwnd) -> str:
        if sys.platform != "win32" or not hwnd:
            return ""
        import ctypes
        user32 = ctypes.windll.user32
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    @staticmethod
    def _fx_log_detail(fx, fw, fh) -> str:
        """特效跳过时的诊断串：级别 + 墨迹指标 + 帧尺寸。"""
        m = fx.get("metrics") or {}
        return (
            f"{fx.get('level', '?')} score={fx.get('score', 0)}"
            f" · 帧{fw}x{fh}"
            f" · ink={m.get('ink_pct', '?')}%"
            f" · ink_cc={m.get('ink_cc_pct', '?')}%"
            f" · v10={m.get('v10', '?')}"
            f" · ring={m.get('ring_v_mean', '?')}/{m.get('ring_v_std', '?')}"
            f" · roi={m.get('roi_wh', '?')}"
        )

    def _log_step(self, msg, *, warn=False):
        """处理日志：一行一条，带时间戳；含诊断字段便于定位。"""
        tag = "warn" if warn else ""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        self.log.configure(state="normal")
        self.log.insert("end", line, tag)
        total = int(self.log.index("end-1c").split(".")[0])
        if total > 500:
            self.log.delete("1.0", f"{total - 500}.0")
        self.log.see("end")
        self.log.configure(state="disabled")


def main():
    from tk_boot import reexec_with_modern_tk_if_needed

    reexec_with_modern_tk_if_needed(__file__)

    # 人机对弈进行中时不要抢焦点弹出截图分析窗；
    # 对弈页「截图分析」按钮会带 CHEES_ALLOW_WITH_PLAY=1 显式放行。
    play_lock = os.path.join(_ROOT, ".play_session.lock")
    allow = os.environ.get("CHEES_ALLOW_WITH_PLAY") == "1"
    if os.path.isfile(play_lock) and not allow:
        print("人机对弈进行中，跳过启动截图分析界面。")
        return

    _enable_dpi_awareness()
    _set_console_title(APP_WINDOW_TITLE)
    root = tk.Tk()
    root.title(APP_WINDOW_TITLE)  # 尽早设标题，避免闪过旧名
    app = ChessApp(root)

    def on_close():
        app.stop_auto_listen()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    def report_size():
        root.update_idletasks()
        print(f"窗口尺寸: {root.winfo_width()}x{root.winfo_height()}  "
              f"屏幕: {root.winfo_screenwidth()}x{root.winfo_screenheight()}")

    root.after(300, report_size)
    root.mainloop()


if __name__ == "__main__":
    main()
