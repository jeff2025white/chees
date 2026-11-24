"""界面偏好：本地 JSON 记忆（深度等），下次启动恢复。"""

from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREFS_PATH = os.path.join(_HERE, "user_prefs.json")

_DEFAULT_DEPTH = 9
_DEPTH_MIN = 2
_DEPTH_MAX = 12


def _clamp_depth(value) -> int:
    try:
        d = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_DEPTH
    return max(_DEPTH_MIN, min(_DEPTH_MAX, d))


def load_prefs() -> dict:
    """读取偏好；文件缺失或损坏时返回空 dict。"""
    try:
        with open(_PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def save_prefs(**updates) -> None:
    """合并写入偏好（原子替换）。"""
    data = load_prefs()
    data.update(updates)
    try:
        tmp = _PREFS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, _PREFS_PATH)
    except OSError:
        pass


def get_depth(default: int = _DEFAULT_DEPTH) -> int:
    """上次记忆的迭代深度；无则用 default。"""
    data = load_prefs()
    if "depth" in data:
        return _clamp_depth(data["depth"])
    return _clamp_depth(default)


def set_depth(depth: int) -> None:
    """保存迭代深度（分析页 / 对弈页共用）。"""
    save_prefs(depth=_clamp_depth(depth))


def get_show_piece_labels(default: bool = True) -> bool:
    """棋子上是否显示识别兵种文字。"""
    data = load_prefs()
    if "show_piece_labels" in data:
        return bool(data["show_piece_labels"])
    return bool(default)


def set_show_piece_labels(on: bool) -> None:
    """保存兵种文字开关。"""
    save_prefs(show_piece_labels=bool(on))


def get_advice_enabled(default: bool = False) -> bool:
    """是否生成/刷新棋局解说面板；默认关闭。"""
    data = load_prefs()
    if "advice_enabled" in data:
        return bool(data["advice_enabled"])
    return bool(default)


def set_advice_enabled(on: bool) -> None:
    """保存棋局解说开关。"""
    save_prefs(advice_enabled=bool(on))


def get_turn_detect_enabled(default: bool = True) -> bool:
    """是否根据头像/差分判断行棋方；关则不因对方行棋跳过计算。默认开。"""
    data = load_prefs()
    if "turn_detect_enabled" in data:
        return bool(data["turn_detect_enabled"])
    return bool(default)


def set_turn_detect_enabled(on: bool) -> None:
    """保存行棋方判定开关。"""
    save_prefs(turn_detect_enabled=bool(on))


def get_fx_detect_enabled(default: bool = False) -> bool:
    """是否在自动监听里检测墨迹动画并跳过该帧。默认关闭。"""
    data = load_prefs()
    if "fx_detect_enabled" in data:
        return bool(data["fx_detect_enabled"])
    return bool(default)


def set_fx_detect_enabled(on: bool) -> None:
    """保存墨迹动画检测开关。"""
    save_prefs(fx_detect_enabled=bool(on))
