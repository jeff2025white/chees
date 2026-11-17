"""中国象棋人机对弈 · 图形界面（Tkinter）。

运行：
    python ui/play.py

布局（左右）：
    ┌──────────────────────────────┬──────────────────┐
    │                              │  中国象棋对弈     │
    │         可点击棋盘            │  难度（搜索深度） │
    │                              │  [====○====]     │
    │                              │                  │
    ├──────────────────────────────┤                  │
    │  状态：红方行棋 / 电脑思考中  │                  │
    └──────────────────────────────┴──────────────────┘

人（红，下方）先走；电脑执黑。右侧难度拉杆对应引擎迭代深度。
"""

import atexit
import os
import queue
import subprocess
import sys
import threading

import tkinter as tk
from tkinter import messagebox

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 对弈进行中时阻止截图分析窗（其它对话常会自动拉起 ui/app.py）
PLAY_LOCK = os.path.join(_ROOT, ".play_session.lock")

from engine import decide_position
from engine.board import Board
from engine.constants import (
    BLACK,
    COLS,
    PIECE_NAMES,
    RED,
    ROWS,
    START_FEN,
    color_of,
    type_of,
)
from engine.movegen import generate_legal, game_status
from engine.notation import chinese_notation

# 与 ui/app.py 保持同一视觉基调
BG = "#1f2430"
PANEL = "#2b3242"
FG = "#e6e6e6"
ACCENT = "#e23b3b"
UI_FONT = ("Microsoft YaHei", 10)
UI_FONT_B = ("Microsoft YaHei", 10, "bold")
TITLE_FONT = ("Microsoft YaHei", 13, "bold")

WINDOW_W, WINDOW_H = 1024, 768

# 棋盘绘制
BOARD_BG = "#c9a46a"
BOARD_LINE = "#3a2a18"
RIVER_FG = "#5a4030"
PIECE_FILL = "#f3e6c8"
RED_PIECE = "#c62828"
BLACK_PIECE = "#1a1a1a"
SEL_FILL = "#ffeb3b"
HINT_FILL = "#66bb6a"
LAST_BLUE = "#42a5f5"     # 上一步高亮蓝；与棋盘底色混合模拟半透明


def _hex_rgb(color: str):
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _blend(fg: str, bg: str, alpha: float) -> str:
    """把 fg 以 alpha 叠在 bg 上，得到 Tk 可用的不透明色。"""
    fr, fg_, fb = _hex_rgb(fg)
    br, bg_, bb = _hex_rgb(bg)
    r = int(fr * alpha + br * (1 - alpha))
    g = int(fg_ * alpha + bg_ * (1 - alpha))
    b = int(fb * alpha + bb * (1 - alpha))
    return f"#{r:02x}{g:02x}{b:02x}"


def _enable_dpi_awareness():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:                                    # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                                # noqa: BLE001
            pass


def _acquire_play_lock():
    with open(PLAY_LOCK, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _release_play_lock():
    try:
        if os.path.isfile(PLAY_LOCK):
            os.remove(PLAY_LOCK)
    except OSError:
        pass


class FlatButton(tk.Label):
    """与 ui/app.py 同款自绘按钮（避免 import app 拉入识别依赖）。"""

    def __init__(self, parent, text, command=None, **kwargs):
        self._command = command
        self._enabled = True
        self._bg = kwargs.pop("bg", "#3b4459")
        self._fg = kwargs.pop("fg", FG)
        self._hover = kwargs.pop("activebackground", "#4b5570")
        kwargs.setdefault("cursor", "hand2")
        super().__init__(parent, text=text, bg=self._bg, fg=self._fg, **kwargs)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, _event=None):
        if self._enabled:
            super().configure(bg=self._hover)

    def _on_leave(self, _event=None):
        super().configure(bg=self._bg)

    def _on_click(self, _event=None):
        if self._enabled and self._command:
            self._command()


class PlayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中国象棋人机对弈")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        atexit.register(_release_play_lock)
        _acquire_play_lock()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w = min(WINDOW_W, sw - 20)
        h = min(WINDOW_H, sh - 80)
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - 20)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(800, 600)

        # 人在下方执红；电脑执黑。view_side=红 = 我方在下
        self.human_side = RED
        self.engine_side = BLACK
        self.view_side = RED

        self.board = Board.from_fen(START_FEN)
        self.selected = None          # sq or None
        self.legal_targets = set()    # to_sq
        self.last_move = None         # (frm, to)
        self._busy = False
        self._game_over = False
        self._queue = queue.Queue()
        self._layout = None           # 画布格子几何，供点击换算

        self._build_ui()
        self.canvas.bind("<Configure>", lambda _e: self._draw_board())
        self.canvas.bind("<Button-1>", self._on_click)
        self._set_status("红方行棋 · 点击棋子走棋")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(outer, bg=PANEL)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#11141b", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 4))

        status = tk.Frame(left, bg=PANEL)
        status.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
        self.lbl_status = tk.Label(status, text="", bg=PANEL, fg=FG,
                                   font=UI_FONT, anchor="center")
        self.lbl_status.pack(fill="x")

        right = tk.Frame(outer, bg=PANEL, width=340)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # 底栏链接先 pack，贴在右侧面板底部
        footer = tk.Frame(right, bg=PANEL)
        footer.pack(side="bottom", fill="x", padx=16, pady=(8, 16))
        FlatButton(
            footer, text="截图分析 →", font=UI_FONT_B, command=self.open_analysis,
            activebackground=ACCENT, padx=8, pady=12,
        ).pack(fill="x")

        tk.Label(right, text="中国象棋对弈", bg=PANEL, fg=FG,
                 font=TITLE_FONT).pack(pady=(16, 14), padx=16, anchor="w")

        depth_box = tk.Frame(right, bg=PANEL)
        depth_box.pack(fill="x", padx=16, pady=(6, 0))
        self.lbl_depth = tk.Label(depth_box, text="难度：6（深度 6 层）",
                                  bg=PANEL, fg=FG, font=UI_FONT, anchor="w")
        self.lbl_depth.pack(fill="x")

        self.depth_var = tk.IntVar(value=6)
        self.depth_var.trace_add("write", self._on_depth_change)
        self.scale_depth = tk.Scale(
            depth_box, from_=2, to=12, orient="horizontal", variable=self.depth_var,
            showvalue=False, resolution=1, bg=PANEL, fg=FG,
            troughcolor="#3b4459", activebackground=ACCENT, highlightthickness=0,
            bd=0, sliderrelief="flat",
        )
        self.scale_depth.pack(fill="x")

        tk.Label(depth_box, text="浅 ←　　　　　　→ 深", bg=PANEL, fg="#8a93a8",
                 font=("Microsoft YaHei", 9), anchor="center").pack(fill="x", pady=(2, 0))

    def _on_depth_change(self, *_args):
        d = self.depth_var.get()
        self.lbl_depth.configure(text=f"难度：{d}（深度 {d} 层）")

    def _set_status(self, text):
        self.lbl_status.configure(text=text)

    def open_analysis(self):
        """另开截图分析窗口（显式放行对弈锁）。"""
        script = os.path.join(_HERE, "app.py")
        env = os.environ.copy()
        env["CHEES_ALLOW_WITH_PLAY"] = "1"
        try:
            subprocess.Popen(
                [sys.executable, script],
                cwd=_ROOT,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            messagebox.showerror("无法启动", str(exc))

    # -------------------------------------------------------------- 绘制
    def _board_geometry(self):
        """根据画布尺寸算棋盘外框与格点间距。"""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 40 or ch < 40:
            return None
        pad = 28
        usable_w = cw - 2 * pad
        usable_h = ch - 2 * pad
        # 9 列 10 行 → 8 格宽、9 格高；外圈再给棋子留边
        cell = min(usable_w / 8.0, usable_h / 9.0)
        board_w = cell * 8
        board_h = cell * 9
        x0 = (cw - board_w) / 2
        y0 = (ch - board_h) / 2
        return {"x0": x0, "y0": y0, "cell": cell, "cw": cw, "ch": ch}

    def _sq_center(self, sq, geo):
        r, c = divmod(sq, COLS)
        x = geo["x0"] + c * geo["cell"]
        y = geo["y0"] + r * geo["cell"]
        return x, y

    def _draw_board(self):
        geo = self._board_geometry()
        self._layout = geo
        self.canvas.delete("all")
        if geo is None:
            return

        x0, y0, cell = geo["x0"], geo["y0"], geo["cell"]
        # 底板略大于线网，给边缘棋子留白
        margin = cell * 0.55
        self.canvas.create_rectangle(
            x0 - margin, y0 - margin,
            x0 + 8 * cell + margin, y0 + 9 * cell + margin,
            fill=BOARD_BG, outline=BOARD_LINE, width=2,
        )

        # 横线 10 条、竖线：上下各 9 列贯通，河界断开
        for r in range(ROWS):
            y = y0 + r * cell
            self.canvas.create_line(x0, y, x0 + 8 * cell, y, fill=BOARD_LINE, width=1)
        for c in range(COLS):
            x = x0 + c * cell
            self.canvas.create_line(x, y0, x, y0 + 4 * cell, fill=BOARD_LINE, width=1)
            self.canvas.create_line(x, y0 + 5 * cell, x, y0 + 9 * cell,
                                    fill=BOARD_LINE, width=1)

        # 九宫斜线
        for (r0, c0, r1, c1) in (
            (0, 3, 2, 5), (0, 5, 2, 3),
            (7, 3, 9, 5), (7, 5, 9, 3),
        ):
            self.canvas.create_line(
                x0 + c0 * cell, y0 + r0 * cell,
                x0 + c1 * cell, y0 + r1 * cell,
                fill=BOARD_LINE, width=1,
            )

        # 楚河汉界
        river_y = y0 + 4.5 * cell
        self.canvas.create_text(
            x0 + 2 * cell, river_y, text="楚 河", fill=RIVER_FG,
            font=("Microsoft YaHei", max(12, int(cell * 0.38)), "bold"),
        )
        self.canvas.create_text(
            x0 + 6 * cell, river_y, text="汉 界", fill=RIVER_FG,
            font=("Microsoft YaHei", max(12, int(cell * 0.38)), "bold"),
        )

        # 上一步起止：半透明蓝方块（与棋盘底色混合）
        if self.last_move:
            fill = _blend(LAST_BLUE, BOARD_BG, 0.38)
            for sq in self.last_move:
                cx, cy = self._sq_center(sq, geo)
                rad = cell * 0.42
                self.canvas.create_rectangle(
                    cx - rad, cy - rad, cx + rad, cy + rad,
                    fill=fill, outline="",
                )

        # 选中格：半透明黄方块
        if self.selected is not None:
            cx, cy = self._sq_center(self.selected, geo)
            rad = cell * 0.42
            self.canvas.create_rectangle(
                cx - rad, cy - rad, cx + rad, cy + rad,
                fill=_blend(SEL_FILL, BOARD_BG, 0.42), outline="",
            )

        # 合法落点提示
        for to in self.legal_targets:
            cx, cy = self._sq_center(to, geo)
            if self.board.cells[to]:
                rad = cell * 0.42
                self.canvas.create_oval(
                    cx - rad, cy - rad, cx + rad, cy + rad,
                    outline=HINT_FILL, width=3,
                )
            else:
                self.canvas.create_oval(
                    cx - cell * 0.12, cy - cell * 0.12,
                    cx + cell * 0.12, cy + cell * 0.12,
                    fill=HINT_FILL, outline="",
                )

        # 棋子
        piece_r = cell * 0.38
        font_size = max(14, int(cell * 0.42))
        for sq, piece in enumerate(self.board.cells):
            if not piece:
                continue
            cx, cy = self._sq_center(sq, geo)
            side = color_of(piece)
            outline = RED_PIECE if side == RED else BLACK_PIECE
            name = PIECE_NAMES.get((side, type_of(piece)), "?")
            self.canvas.create_oval(
                cx - piece_r, cy - piece_r, cx + piece_r, cy + piece_r,
                fill=PIECE_FILL, outline=outline, width=3,
            )
            self.canvas.create_text(
                cx, cy, text=name, fill=outline,
                font=("Microsoft YaHei", font_size, "bold"),
            )

    # -------------------------------------------------------------- 点击
    def _hit_square(self, event):
        geo = self._layout
        if geo is None:
            return None
        cell = geo["cell"]
        # 落在格点半径内才算选中
        best, best_d2 = None, (cell * 0.48) ** 2
        for sq in range(ROWS * COLS):
            cx, cy = self._sq_center(sq, geo)
            d2 = (event.x - cx) ** 2 + (event.y - cy) ** 2
            if d2 <= best_d2:
                best_d2 = d2
                best = sq
        return best

    def _on_click(self, event):
        if self._busy or self._game_over:
            return
        if self.board.turn != self.human_side:
            return
        sq = self._hit_square(event)
        if sq is None:
            return

        piece = self.board.cells[sq]
        if self.selected is None:
            if piece and color_of(piece) == self.human_side:
                self._select(sq)
            return

        if sq == self.selected:
            self._clear_selection()
            self._draw_board()
            return

        if piece and color_of(piece) == self.human_side:
            self._select(sq)
            return

        move = (self.selected, sq)
        legal = generate_legal(self.board, self.human_side)
        if move not in legal:
            self._clear_selection()
            self._draw_board()
            return

        self._apply_human_move(move)

    def _select(self, sq):
        self.selected = sq
        self.legal_targets = {to for frm, to in generate_legal(self.board, self.human_side)
                              if frm == sq}
        self._draw_board()

    def _clear_selection(self):
        self.selected = None
        self.legal_targets = set()

    def _apply_human_move(self, move):
        note = chinese_notation(self.board, move)
        self.board.apply(move)
        self.last_move = move
        self._clear_selection()
        self._draw_board()

        status, _ = game_status(self.board)
        if status in ("checkmate", "stalemate"):
            self._game_over = True
            self._set_status(f"{note} · 红方胜（{'将死' if status == 'checkmate' else '困毙'}）")
            return
        if status == "check":
            self._set_status(f"{note} · 将军！电脑思考中…")
        else:
            self._set_status(f"{note} · 电脑思考中…")
        self._start_engine()

    # -------------------------------------------------------------- 引擎
    def _start_engine(self):
        self._busy = True
        depth = self.depth_var.get()
        fen = self.board.to_fen()
        threading.Thread(target=self._engine_worker, args=(fen, depth),
                         daemon=True).start()
        self.root.after(80, self._poll_engine)

    def _engine_worker(self, fen, max_depth):
        try:
            # 对弈要快响应：比截图分析更短的时间预算（深度 6 ≈ 4–8s）
            time_limit = min(30.0, max(2.5, 1.2 * (2.5 ** (max_depth - 4))))
            decision = decide_position(
                fen=fen,
                side="black",
                view_side="red",   # 人在下方，输出坐标与内部一致
                time_limit=time_limit,
                max_depth=max_depth,
                top_n=0,
                fill_pv=False,     # 对弈只要最佳一手，不做 PV 续搜
            )
            self._queue.put((decision, None))
        except Exception as exc:                         # noqa: BLE001
            self._queue.put((None, exc))

    def _poll_engine(self):
        try:
            decision, error = self._queue.get_nowait()
        except queue.Empty:
            if self._busy:
                self.root.after(80, self._poll_engine)
            return
        self._engine_done(decision, error)

    def _engine_done(self, decision, error):
        self._busy = False
        if error is not None:
            self._set_status(f"引擎出错：{error}")
            messagebox.showerror("引擎出错", str(error))
            return

        best = (decision or {}).get("best_move")
        if not best:
            # 无着法：红方胜
            self._game_over = True
            msg = (decision or {}).get("message", "电脑无子可走")
            self._set_status(f"{msg} · 红方胜")
            self._draw_board()
            return

        # view_side=红时 from/to 行列即内部坐标
        frm = best["from"]["row"] * COLS + best["from"]["col"]
        to = best["to"]["row"] * COLS + best["to"]["col"]
        move = (frm, to)
        legal = generate_legal(self.board, self.engine_side)
        if move not in legal:
            self._set_status(f"引擎着法非法：{best.get('chinese', move)}")
            messagebox.showerror("引擎着法非法", str(best))
            return

        note = best.get("chinese") or chinese_notation(self.board, move)
        self.board.apply(move)
        self.last_move = move
        self._draw_board()

        status, _ = game_status(self.board)
        if status in ("checkmate", "stalemate"):
            self._game_over = True
            self._set_status(
                f"{note} · 黑方胜（{'将死' if status == 'checkmate' else '困毙'}）"
            )
            return
        if status == "check":
            self._set_status(f"电脑：{note} · 将军！轮到红方")
        else:
            self._set_status(f"电脑：{note} · 轮到红方")

    def _on_close(self):
        _release_play_lock()
        self.root.destroy()


def main():
    _enable_dpi_awareness()
    root = tk.Tk()
    PlayApp(root)
    try:
        root.mainloop()
    finally:
        _release_play_lock()


if __name__ == "__main__":
    main()
