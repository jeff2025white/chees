"""macOS Tk 启动辅助：避开系统 Python 自带的 Tcl/Tk 8.5 白屏。"""

from __future__ import annotations

import os
import sys


def reexec_with_modern_tk_if_needed(script_file: str) -> None:
    """系统 /usr/bin/python3 的 Tk 8.5 在新 macOS 上常整窗白屏。

    检测到 Tk < 8.6 且存在 Homebrew Python 时，自动切换重跑。
    """
    if sys.platform != "darwin":
        return
    try:
        import tkinter as tk
        ver = float(tk.TkVersion)
    except Exception:  # noqa: BLE001
        ver = 0.0
    if ver >= 8.6:
        return
    candidates = (
        "/opt/homebrew/bin/python3.14",
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    )
    here = os.path.realpath(sys.executable)
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        if os.path.realpath(cand) == here:
            continue
        os.execv(cand, [cand, os.path.realpath(script_file), *sys.argv[1:]])
