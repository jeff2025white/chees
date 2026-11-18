"""检测棋盘中部「墨云 / 吃 / 当头炮 / 将军」等特效遮挡的脏截图。

用法：
    python test/detect_fx_overlay.py images/
    python test/detect_fx_overlay.py images/auto_*.png
    python test/detect_fx_overlay.py images/ --json
    python test/detect_fx_overlay.py --selftest

判定分级：
    strong  不透明黑色笔刷环（吃 / 当头炮等），置信度高
    soft    半透明墨云 + 金色大字（将军等）
    faded   淡化阶段：中环变暗、对比升高，但近黑像素很少
    clean   无明显中部特效
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.fx_overlay import detect_board_fx, detect_board_fx_path  # noqa: F401

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

# 用户提供的已知脏样例（用于 --selftest）；已迁到 images/fx_dirty/ 亦可
KNOWN_DIRTY = (
    "auto_20261118_171910.png",
    "auto_20261118_171928.png",
    "auto_20261118_173908.png",
    "auto_20261118_174057.png",
    "auto_20261118_174124.png",
    "auto_20261118_174236.png",
)

_LEVEL_RANK = {"clean": 0, "faded": 1, "soft": 2, "strong": 3}


def _iter_images(args_paths):
    files = []
    for raw in args_paths:
        if any(ch in raw for ch in "*?[]"):
            import glob as _glob

            files.extend(sorted(_glob.glob(raw)))
            continue
        if os.path.isdir(raw):
            for name in sorted(os.listdir(raw)):
                if name.lower().endswith(IMAGE_EXTS) and not name.startswith("."):
                    # 跳过标注输出子目录里的文件（若被展开）
                    files.append(os.path.join(raw, name))
        elif os.path.isfile(raw):
            files.append(raw)
    # 去重且只要图片
    seen = set()
    out = []
    for p in files:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        if not ap.lower().endswith(IMAGE_EXTS):
            continue
        # 默认不扫 annotated/
        if f"{os.sep}annotated{os.sep}" in ap:
            continue
        seen.add(ap)
        out.append(ap)
    return out


def _resolve_sample(name: str) -> str | None:
    """脏样例可能在 images/ 或 images/fx_dirty/。"""
    for folder in (
        os.path.join(_ROOT, "images", "fx_dirty"),
        os.path.join(_ROOT, "images"),
    ):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def _selftest() -> int:
    img_dir = os.path.join(_ROOT, "images")
    missing = []
    failed = []
    for name in KNOWN_DIRTY:
        path = _resolve_sample(name)
        if path is None:
            missing.append(name)
            continue
        r = detect_board_fx_path(path)
        ok = r["dirty"]
        mark = "OK" if ok else "MISS"
        print(f"  [{mark}] {name}  level={r['level']} score={r['score']}  "
              f"ink={r['metrics']['ink_pct']:.1f}% ringV={r['metrics']['ring_v_mean']:.0f}")
        if not ok:
            failed.append(name)

    # 抽几张干净开局图，不应误报
    clean_probe = [
        "auto_20261118_171632.png",
        "auto_20261118_171653.png",
        "auto_20261118_181101.png",
    ]
    fps = []
    for name in clean_probe:
        path = os.path.join(img_dir, name)
        if not os.path.isfile(path):
            continue
        r = detect_board_fx_path(path)
        mark = "FP" if r["dirty"] else "OK"
        print(f"  [{mark}] clean? {name}  level={r['level']}")
        if r["dirty"]:
            fps.append(name)

    print()
    if missing:
        print(f"缺少样例文件: {missing}")
    if failed:
        print(f"漏检: {failed}")
    if fps:
        print(f"误报: {fps}")
    if missing or failed or fps:
        print("SELFTEST FAIL")
        return 1
    print(f"SELFTEST PASS  known_dirty={len(KNOWN_DIRTY)} clean_probe={len(clean_probe)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检测象棋截图中部特效遮挡（墨云）")
    ap.add_argument("paths", nargs="*", help="图片文件或目录（默认 images/）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 数组")
    ap.add_argument(
        "--min-level",
        choices=("faded", "soft", "strong"),
        default="faded",
        help="最低命中级别（默认 faded，全部脏图）",
    )
    ap.add_argument("--only-dirty", action="store_true", help="只打印命中的脏图")
    ap.add_argument("--selftest", action="store_true", help="用已知脏/净样例自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    paths = args.paths or [os.path.join(_ROOT, "images")]
    files = _iter_images(paths)
    if not files:
        print("未找到图片", file=sys.stderr)
        return 2

    min_rank = _LEVEL_RANK[args.min_level]
    results = []
    for path in files:
        try:
            r = detect_board_fx_path(path)
        except Exception as exc:  # noqa: BLE001
            r = {
                "path": path,
                "name": os.path.basename(path),
                "dirty": False,
                "level": "error",
                "score": 0.0,
                "error": str(exc),
                "metrics": {},
            }
        results.append(r)

    hits = [
        r for r in results
        if r.get("level") in _LEVEL_RANK and _LEVEL_RANK[r["level"]] >= min_rank
    ]

    if args.json:
        payload = hits if args.only_dirty else results
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        show = hits if args.only_dirty else results
        for r in show:
            if r.get("level") == "error":
                print(f"ERR  {r['name']}: {r.get('error')}")
                continue
            tag = r["level"].upper() if r["dirty"] else "clean"
            m = r.get("metrics") or {}
            print(
                f"{tag:6} {r['name']:40} score={r['score']:.2f}  "
                f"ink={m.get('ink_pct', 0):5.2f}%  "
                f"icc={m.get('ink_cc_pct', 0):5.2f}%  "
                f"ringV={m.get('ring_v_mean', 0):5.1f}/{m.get('ring_v_std', 0):4.1f}"
            )
        print(
            f"\nscanned={len(results)}  dirty>={args.min_level}: {len(hits)}  "
            f"(strong={sum(1 for r in hits if r['level']=='strong')}  "
            f"soft={sum(1 for r in hits if r['level']=='soft')}  "
            f"faded={sum(1 for r in hits if r['level']=='faded')})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
