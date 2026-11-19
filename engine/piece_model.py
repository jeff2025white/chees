"""轻量兵种 CNN：定义、加载、批量推理（PyTorch）。

类别固定为 RNBAKCP（0..6）；红/黑由 is_red 通道区分，输出再映射大小写。
无 torch 或无权重时 load_piece_model() 返回 None，调用方回退模板 OCR。
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

CLASSES = "RNBAKCP"
CLASS_TO_IDX = {ch: i for i, ch in enumerate(CLASSES)}
INPUT_SIZE = 48
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "piece_cls.pt")

_MODEL_CACHE = None  # (model, device, meta) | False 表示已尝试失败


def preprocess_bgr(bgr: np.ndarray, is_red: bool) -> np.ndarray:
    """BGR 圆盘切图 → float32 CHW (4, 48, 48)，值域约 [0,1]。"""
    if bgr is None:
        raise ValueError("empty patch")
    img = bgr
    if img.shape[0] != INPUT_SIZE or img.shape[1] != INPUT_SIZE:
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    bgr_f = img.astype(np.float32) / 255.0
    flag = np.full((INPUT_SIZE, INPUT_SIZE), 1.0 if is_red else 0.0, dtype=np.float32)
    return np.stack([bgr_f[:, :, 0], bgr_f[:, :, 1], bgr_f[:, :, 2], flag], axis=0)


def build_model(n_classes: int = len(CLASSES)):
    from torch import nn

    class PieceCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return PieceCNN()


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def clear_model_cache():
    global _MODEL_CACHE
    _MODEL_CACHE = None


def load_piece_model(path: Optional[str] = None):
    """返回 (model, device, meta)；不可用则 None。结果会缓存。"""
    global _MODEL_CACHE
    if _MODEL_CACHE is False:
        return None
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not torch_available():
        _MODEL_CACHE = False
        return None
    import torch

    weight_path = path or MODEL_PATH
    if not os.path.isfile(weight_path):
        _MODEL_CACHE = False
        return None
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            ckpt = torch.load(weight_path, map_location="cpu")
        except Exception:
            _MODEL_CACHE = False
            return None
    except Exception:
        _MODEL_CACHE = False
        return None

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        state = ckpt
        meta = {"classes": CLASSES, "input_size": INPUT_SIZE}

    classes = meta.get("classes", CLASSES)
    model = build_model(n_classes=len(classes))
    model.load_state_dict(state)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _MODEL_CACHE = (model, device, meta)
    return _MODEL_CACHE


def predict_batch(patches_bgr, is_red_flags) -> list:
    """对多格批量推理，返回 list[dict[ch, float]]（已按颜色大小写）。"""
    loaded = load_piece_model()
    if loaded is None or not patches_bgr:
        return []
    import torch

    model, device, meta = loaded
    classes = meta.get("classes", CLASSES)
    tensors = []
    for patch, is_red in zip(patches_bgr, is_red_flags):
        if patch is None:
            tensors.append(None)
            continue
        tensors.append(preprocess_bgr(patch, is_red))

    valid_idx = [i for i, t in enumerate(tensors) if t is not None]
    if not valid_idx:
        return [{} for _ in patches_bgr]

    batch = np.stack([tensors[i] for i in valid_idx], axis=0)
    x = torch.from_numpy(batch).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    out = [{} for _ in patches_bgr]
    for row, i in enumerate(valid_idx):
        is_red = is_red_flags[i]
        scores = {}
        for j, ch in enumerate(classes):
            letter = ch.upper() if is_red else ch.lower()
            scores[letter] = float(probs[row, j])
        out[i] = scores
    return out


def predict_one(bgr: np.ndarray, is_red: bool) -> dict:
    results = predict_batch([bgr], [is_red])
    return results[0] if results else {}
