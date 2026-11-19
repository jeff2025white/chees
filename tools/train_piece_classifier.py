#!/usr/bin/env python3
"""训练兵种 TinyCNN，写出 engine/models/piece_cls.pt。

用法：
  pip install torch
  python tools/export_piece_dataset.py --clear
  python tools/train_piece_classifier.py --epochs 25
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.piece_model import (  # noqa: E402
    CLASSES, CLASS_TO_IDX, INPUT_SIZE, MODEL_PATH, build_model, clear_model_cache,
    preprocess_bgr,
)


def _augment(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    if random.random() < 0.7:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), scale)
        M[0, 2] += random.uniform(-2, 2)
        M[1, 2] += random.uniform(-2, 2)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if random.random() < 0.35:
        q = random.randint(40, 90)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if random.random() < 0.25:
        noise = np.random.normal(0, 6, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _is_red_name(name: str) -> bool:
    """文件名 R_* / R_hard_* 为红；B_* 为黑。"""
    return name.startswith("R_")


def load_samples(data_dir: Path, hard_dir: Path | None = None, hard_repeat: int = 6):
    items = []  # (path, class_idx, is_red)
    for ch in CLASSES:
        d = data_dir / ch
        if not d.is_dir():
            continue
        for f in d.glob("*.png"):
            items.append((f, CLASS_TO_IDX[ch], _is_red_name(f.name)))
    if hard_dir and hard_dir.is_dir():
        for ch in CLASSES:
            d = hard_dir / ch
            if not d.is_dir():
                continue
            for f in d.glob("*.png"):
                row = (f, CLASS_TO_IDX[ch], _is_red_name(f.name))
                for _ in range(max(1, hard_repeat)):
                    items.append(row)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "datasets" / "pieces")
    ap.add_argument("--out", type=Path, default=Path(MODEL_PATH))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-dir", type=Path, default=ROOT / "datasets" / "pieces_hard")
    ap.add_argument("--hard-repeat", type=int, default=8)
    args = ap.parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        raise SystemExit("需要 PyTorch：pip install torch") from e

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = load_samples(args.data, hard_dir=args.hard_dir, hard_repeat=args.hard_repeat)
    if len(items) < 50:
        raise SystemExit(f"样本过少 ({len(items)})，先跑 python tools/export_piece_dataset.py")

    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    # 难例不要全进 val：按路径拆
    val_items, train_items = [], []
    for it in items:
        if "hard_" in it[0].name and random.random() < 0.85:
            train_items.append(it)
        elif len(val_items) < n_val and "hard_" not in it[0].name:
            val_items.append(it)
        else:
            train_items.append(it)
    if not val_items:
        val_items = items[:n_val]
        train_items = items[n_val:]

    class PieceDS(Dataset):
        def __init__(self, rows, augment=False):
            self.rows = rows
            self.augment = augment

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            path, y, is_red = self.rows[idx]
            bgr = cv2.imread(str(path))
            if bgr is None:
                bgr = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            if self.augment:
                bgr = _augment(bgr)
            x = preprocess_bgr(bgr, is_red)
            return torch.from_numpy(x), y

    train_loader = DataLoader(PieceDS(train_items, augment=True),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(PieceDS(val_items, augment=False),
                            batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 车/马易混：抬高 R 权重；炮次之
    class_w = torch.ones(len(CLASSES), dtype=torch.float32)
    class_w[CLASS_TO_IDX["R"]] = 2.5
    class_w[CLASS_TO_IDX["N"]] = 1.4
    class_w[CLASS_TO_IDX["C"]] = 1.3
    crit = nn.CrossEntropyLoss(weight=class_w.to(device))

    best_acc = -1.0
    best_state = None
    print(f"train={len(train_items)} val={len(val_items)} device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        model.eval()
        correct, total = 0, 0
        conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += len(yb)
                for t, p in zip(yb.cpu().tolist(), pred.cpu().tolist()):
                    conf[t, p] += 1
        acc = correct / max(1, total)
        print(f"epoch {epoch:02d}  loss={total_loss/max(1,n):.4f}  val_acc={acc:.3f}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "classes": CLASSES,
        "input_size": INPUT_SIZE,
        "val_acc": best_acc,
    }
    torch.save(ckpt, args.out)
    clear_model_cache()
    print(f"saved {args.out}  best_val_acc={best_acc:.3f}")
    print("confusion (rows=true):")
    print("     " + " ".join(f"{c:>4}" for c in CLASSES))
    for i, ch in enumerate(CLASSES):
        print(f"{ch:>4} " + " ".join(f"{conf[i, j]:4d}" for j in range(len(CLASSES))))


if __name__ == "__main__":
    main()
