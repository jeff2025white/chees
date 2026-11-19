"""中国象棋截图分析 · 图形界面（Tkinter）。

运行：
    python ui/app.py

布局（左右）：
    ┌──────────────────────────────┬──────────────────┐
    │                              │  中国象棋助手     │
    │         图片显示窗口          │  [ 打开图片 ]     │
    │                              │  [ 自动监听 ]     │
    │                              │  [  计算  ]       │
    │                              │  迭代深度拉杆     │
    ├──────────────────────────────┤  识别 / 着法信息  │
    │  [◀]  文件名 (3/8)  [▶]      │                  │
    └──────────────────────────────┴──────────────────┘

自动监听（Windows）：找「手游助手」窗口 → 每秒截客户区 → 特效过滤
→ analyze_array → 主将判定棋盘 → grid_str 帧差 → decide + annotate。
默认不落盘；「自动截图」拉杆 >0 时写入 images/auto/ 并环形保留。
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

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
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
WINDOW_W, WINDOW_H = 1024, 768


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


def _find_game_window(keyword="手游助手"):
    """枚举可见顶层窗口，返回标题包含 keyword 的句柄列表（仅 Windows）。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
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

        # ---- 自动监听模式 ----
        self._auto_running = False
        self._auto_thread = None
        self._auto_stop = threading.Event()
        self._computing = threading.Event()     # 计算中 → 暂停截图
        self._last_grid_str = None               # 上一帧棋盘串（帧差）
        self._target_hwnd = None                 # 找到的手游助手窗口句柄

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

        self.btn_prev = FlatButton(nav, text="◀", width=4, font=UI_FONT_B,
                                   command=self.prev_image, padx=8, pady=4)
        self.btn_prev.pack(side="left")

        self.btn_next = FlatButton(nav, text="▶", width=4, font=UI_FONT_B,
                                   command=self.next_image, padx=8, pady=4)
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

        # 底栏链接先 pack，保证贴在右侧面板底部
        footer = tk.Frame(right, bg=PANEL)
        footer.pack(side="bottom", fill="x", padx=16, pady=(8, 16))
        self.btn_play = FlatButton(
            footer, text="人机对弈 →", font=UI_FONT_B, command=self.open_play,
            activebackground=ACCENT, padx=8, pady=12,
        )
        self.btn_play.pack(fill="x")

        self.btn_open = self._mk_button(right, "打开图片", self.open_image)
        self.btn_shot = self._mk_button(right, "自动监听", self.toggle_auto_listen)
        self.btn_calc = self._mk_button(right, "计算", self.calculate)

        # 迭代深度拉杆
        depth_box = tk.Frame(right, bg=PANEL)
        depth_box.pack(fill="x", padx=16, pady=(6, 0))
        self.lbl_depth = tk.Label(depth_box, text="迭代深度：7 层", bg=PANEL, fg=FG,
                                  font=UI_FONT, anchor="w")
        self.lbl_depth.pack(fill="x")

        self.depth_var = tk.IntVar(value=7)
        self.depth_var.trace_add("write", self._on_depth_change)
        self._ensure_scale_style()
        self.scale_depth = ttk.Scale(
            depth_box, from_=2, to=12, orient="horizontal",
            variable=self.depth_var, style="Accent.Horizontal.TScale",
            command=lambda v: self.depth_var.set(int(round(float(v)))),
        )
        self.scale_depth.pack(fill="x")

        # 自动截图保留张数：0=不落盘
        keep_box = tk.Frame(right, bg=PANEL)
        keep_box.pack(fill="x", padx=16, pady=(8, 0))
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
                          ("", "1. 打开图片 / 截图（手游助手窗口）\n"
                               "2. ◀ ▶ 切换上一张 / 下一张\n"
                               "3. 点击计算，走位线标注在图上\n"
                               "（方向键左右也可切换）")])

    def _mk_button(self, parent, text, cmd):
        btn = FlatButton(parent, text=text, font=UI_FONT_B, command=cmd,
                         activebackground=ACCENT, padx=8, pady=12)
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

    def capture_screen(self):
        """查找标题含「手游助手」的窗口并截图，保存到当前图片目录后载入显示。"""
        if sys.platform != "win32":
            messagebox.showinfo("提示", "窗口截图目前仅支持 Windows。")
            return
        hwnds = _find_game_window("手游助手")
        if not hwnds:
            messagebox.showwarning(
                "未找到窗口",
                "没有找到标题包含「手游助手」的窗口。\n请先启动手游助手并打开象棋游戏。",
            )
            return
        hwnd = hwnds[0]
        try:
            _bring_to_front(hwnd)
        except OSError as exc:
            messagebox.showerror("截图失败", str(exc))
            return
        self._write_info([("head", "截图中\n"),
                          ("", "已置前手游助手窗口，正在截取客户区…")])
        # 等窗口完成前台渲染再截，避免拿到旧帧或被本窗遮挡的画面
        self.root.after(500, lambda: self._grab_window(hwnd))

    def _grab_window(self, hwnd):
        try:
            img = _capture_window_client(hwnd)
        except Exception as exc:                          # noqa: BLE001
            messagebox.showerror("截图失败", str(exc))
            return
        folder = self.dir_path or os.path.join(_ROOT, "images")
        os.makedirs(folder, exist_ok=True)
        name = time.strftime("capture_%Y%m%d_%H%M%S") + ".png"
        img.save(os.path.join(folder, name))
        self.load_directory(folder, select=name)
        self._write_info([("head", "截图完成\n"),
                          ("", f"{name}\n尺寸 {img.width}×{img.height}\n\n"),
                          ("", "点击「计算」识别局面并给出走位建议。")])

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
        hwnds = _find_game_window("手游助手")
        if not hwnds:
            messagebox.showwarning(
                "未找到窗口",
                "没有找到标题包含「手游助手」的窗口。\n请先启动手游助手并打开象棋游戏。",
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
        self._computing.clear()
        # UI 切换
        self.btn_shot.configure(text="停止监听", activebackground="#ff6b6b")
        self.btn_open.configure(state="disabled")
        self.btn_calc.configure(state="disabled")
        keep = int(self.auto_keep_var.get())
        keep_txt = "不落盘" if keep <= 0 else f"保留最近 {keep} 张 → images/auto/"
        self._write_info([
            ("head", "自动监听中\n"),
            ("", "每秒截图；特效帧跳过；主将判断棋盘；帧差触发计算。\n"),
            ("", f"自动截图：{keep_txt}\n"),
            ("", "点击「停止监听」退出。\n\n"),
            ("warn", "首次截图需要窗口完全渲染，约 1 秒后开始。"),
        ])
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
        self._write_info([("head", "已停止自动监听\n")])

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
                self.root.after(0, lambda: self._auto_status(f"截图失败：{exc}"))
                time.sleep(1.0)
                continue
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 中部墨云/吃/将军等特效 → 跳过识别与计算，等下一帧
            try:
                fx = detect_board_fx(frame)
            except Exception as exc:                      # noqa: BLE001
                self.root.after(0, lambda: self._auto_status(f"特效检测异常：{exc}"))
                time.sleep(1.0)
                continue
            if fx.get("dirty"):
                level = fx.get("level", "?")
                self.root.after(0, lambda lv=level: self._auto_status(
                    f"[{time.strftime('%H:%M:%S')}] 特效遮挡（{lv}），跳过"))
                time.sleep(1.0)
                continue

            # 识别
            try:
                analysis = analyze_array(frame, path="<auto>")
            except Exception as exc:                      # noqa: BLE001
                self.root.after(0, lambda: self._auto_status(f"识别异常：{exc}"))
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
                self.root.after(0, lambda: self._auto_status(
                    f"[{time.strftime('%H:%M:%S')}] 无棋盘（红{r_cnt}/黑{b_cnt}），跳过"))
                self._last_grid_str = None
                time.sleep(1.0)
                continue

            # 跟上次棋盘比
            changed = (self._last_grid_str is not None) and (grid_str != self._last_grid_str)
            first_time = (self._last_grid_str is None)

            # 首次 or 变化 → 先显示原始帧（不等计算）
            if first_time or changed:
                self.root.after(0, lambda p=pil_img: self._auto_show_raw(p))

            if first_time:
                self.root.after(0, lambda: self._auto_status(
                    f"[{time.strftime('%H:%M:%S')}] 有棋盘 红{r_cnt}/黑{b_cnt}，首次触发计算"))
            elif changed:
                self.root.after(0, lambda: self._auto_status(
                    f"[{time.strftime('%H:%M:%S')}] 有棋盘 红{r_cnt}/黑{b_cnt}，局面变化 → 触发计算"))
            else:
                self.root.after(0, lambda: self._auto_status(
                    f"[{time.strftime('%H:%M:%S')}] 有棋盘 红{r_cnt}/黑{b_cnt}，局面未变"))

            # 首次 or 变化 → 清空旧日志 + 触发计算；算完自动覆盖为标注图
            if first_time or changed:
                self._last_grid_str = grid_str
                self._computing.set()     # 计算期间暂停截图
                self._busy = True         # 让轮询保持运行
                self.root.after(0, self._auto_clear)   # 新计算先清旧日志
                self.root.after(80, self._poll_result)
                # 可选落盘（默认不保存；计算用内存中的 analysis）
                save_path = self._save_auto_frame_if_needed(pil_img)
                if save_path:
                    analysis.path = save_path

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
            self.root.after(0, lambda: self._auto_status(f"截图保存失败：{exc}"))
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
            vis = annotate(analysis, decision)
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

    def _auto_clear(self):
        """自动模式：清空日志（新计算开始前调用）。"""
        self.info.configure(state="normal")
        self.info.delete("1.0", "end")
        self.info.configure(state="disabled")

    def _auto_append(self, parts):
        """自动模式：追加若干行到日志末尾，不覆盖历史；自动滚动到底。"""
        self.info.configure(state="normal")
        for tag, text in parts:
            self.info.insert("end", text, tag or "")
        # 限制日志最多 500 行，超过就删旧行防止无限膨胀
        total = int(self.info.index("end-1c").split(".")[0])
        if total > 500:
            self.info.delete("1.0", f"{total - 500}.0")
        self.info.see("end")
        self.info.configure(state="disabled")

    def _auto_status(self, msg):
        """自动模式：追加一行状态（简化版日志）。"""
        self._auto_append([("", msg + "\n")])

    def _auto_window_lost(self):
        self.stop_auto_listen()
        self._write_info([
            ("warn", "目标窗口已关闭\n"),
            ("", "手游助手窗口消失，自动监听停止。\n"
                 "窗口句柄：" + str(self._target_hwnd)),
        ])
        messagebox.showwarning("窗口丢失", "手游助手窗口已关闭，自动监听停止。")

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
        name = os.path.basename(self.current_path)
        self._write_info([
            ("head", "分析中\n"),
            ("", f"图片：{name}\n"),
            ("", f"迭代深度：{self.depth_var.get()} 层\n"),
            ("", "正在识别棋盘并搜索最佳着法…\n"),
        ])
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
                # 第一步固定为我方：图下方先走（与红黑无关）
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
        auto_mode = self._auto_running
        self._busy = False
        if not auto_mode:
            self.btn_calc.configure(state="normal", text="计算")

        img_name = os.path.basename(self.current_path or "")
        if error is not None:
            if auto_mode:
                self._auto_append([
                    ("warn", f"计算失败：{error}\n"),
                ])
            else:
                self._write_info([
                    ("warn", "计算失败\n"),
                    ("", f"图片：{img_name}\n"),
                    ("", str(error)),
                ])
                messagebox.showerror("计算失败", str(error))
            return

        self._analysis = analysis
        self._decision = decision
        self.base_image = pil
        self._render()

        # 我方 = 棋盘下半部（view_side），不能按红/黑或行棋方判断
        side_txt = "红方在下" if analysis.red_bottom else "黑方在下"
        search_info = decision.get("search") or {}
        pv = decision.get("pv") or {}
        inp = decision.get("input") or {}
        our_side = inp.get("view_side") or inp.get("our_side") or analysis.view_side
        turn = inp.get("side_to_move", "red")
        turn_txt = "红方行棋" if turn == "red" else "黑方行棋"

        best = decision.get("best_move")

        if auto_mode:
            # 自动模式 → 追加到日志末尾，不清空
            ts = time.strftime("%H:%M:%S")
            block = [
                ("head", f"── [{ts}] 局面分析 ──\n"),
                ("", f"红{analysis.red_count}/黑{analysis.black_count}"
                     f" · {side_txt}（我方） · {turn_txt}\n"),
                ("head", "推演序列\n"),
            ]
            if best:
                names = pv.get("chinese") or []
                sides = pv.get("sides") or []
                scores = pv.get("scores") or []
                if names:
                    for i, name in enumerate(names):
                        who = "我方" if (i < len(sides) and sides[i] == our_side) else "对方"
                        score = scores[i] / 100.0 if i < len(scores) else 0.0
                        tag = "move" if who == "我方" else "opp"
                        block.append((tag, f"{i + 1:>2}.{name} {who} {score:+.2f}\n"))
                else:
                    who = "我方" if best.get("color") == our_side else "对方"
                    tag = "move" if who == "我方" else "opp"
                    block.append((tag, f"{best['chinese']}  {who}  {best['iccs']}\n"))
                block.append(("", f"\n评价 {best['score_text']}"
                                  f"{' · 将' if best.get('gives_check') else ''}"
                                  f"{' · 杀' if best.get('is_mate') else ''}\n"))
                src = "开局库+" if search_info.get("book") else ""
                block.append(("", f"{src}深度{search_info.get('depth', 0)}"
                                  f" · {search_info.get('nodes', 0):,}节点"
                                  f" · {search_info.get('time_ms', 0) / 1000:.1f}s\n"))
            else:
                block.append(("warn", decision.get("message", "没有可走的着法")))
            self._auto_append(block)
        else:
            # 手动模式 → 全量重写（保持原样）
            lines = [
                ("head", "当前图片\n"),
                ("", f"{img_name}\n\n"),
                ("head", "识别结果\n"),
                ("", f"红{analysis.red_count}/黑{analysis.black_count} · {side_txt}（我方）"
                     f" · {turn_txt} · 命中{analysis.hit}\n\n"),
                ("head", "推演序列\n"),
            ]
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
                    who = "我方" if best.get("color") == our_side else "对方"
                    tag = "move" if who == "我方" else "opp"
                    lines.append((tag, f"{best['chinese']}  {who}  {best['iccs']}\n"))

                lines.append(("", f"\n最终评价 {best['score_text']}"
                                  f"{'  · 将军' if best.get('gives_check') else ''}"
                                  f"{'  · 绝杀' if best.get('is_mate') else ''}\n"))
                src = "开局库+" if search_info.get("book") else ""
                lines.append(("", f"{src}深度 {search_info.get('depth', 0)} 层"
                                  f" · 推演 {len(pv.get('chinese') or [])} 步"
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
    root = tk.Tk()
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
