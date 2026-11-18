"""对比纯 Python 搜索核与 Cython 原生核的耗时 / NPS。

用法：
    python test/bench_native.py
    python test/bench_native.py --depth 5 --fen start
"""

import argparse
import importlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.board import Board
from engine.constants import START_FEN
from engine.search import native_available, search_python


def _clear_py_tt():
    S = importlib.import_module("engine.search")
    for lst, val in zip(
        (S._TT_KEY, S._TT_DEPTH, S._TT_SCORE, S._TT_FLAG, S._TT_MOVE),
        (0, -1, 0, 0, None),
    ):
        lst[:] = [val] * S.TT_SIZE


def main():
    parser = argparse.ArgumentParser(description="Python vs Cython 搜索核对照")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--fen", default="start",
                        help="FEN 或 start")
    args = parser.parse_args()

    fen = START_FEN if args.fen == "start" else args.fen
    board = Board.from_fen(fen)

    print(f"FEN: {board.to_fen()}")
    print(f"深度 {args.depth}  native={native_available()}")

    _clear_py_tt()
    t0 = time.perf_counter()
    py = search_python(board.clone(), time_limit=1e9, max_depth=args.depth,
                       top_n=0, fill_pv=False)
    py_t = time.perf_counter() - t0
    py_nps = int(py.nodes / py_t) if py_t > 0 else 0
    print(f"  Python  {py_t:7.3f}s  nodes={py.nodes:>10,}  "
          f"nps={py_nps:>10,}  move={py.best_move} score={py.score:+d}")

    if not native_available():
        print("  （未编译 engine._xiangqi，跳过原生对照）")
        print("  构建：pip install 'Cython>=3.0' && python setup.py build_ext --inplace")
        return 1

    from engine._xiangqi import search_cells

    t1 = time.perf_counter()
    nat = search_cells(list(board.cells), board.turn, 1e9, args.depth, 0, False)
    nat_t = time.perf_counter() - t1
    nat_nps = int(nat["nodes"] / nat_t) if nat_t > 0 else 0
    print(f"  Native  {nat_t:7.3f}s  nodes={nat['nodes']:>10,}  "
          f"nps={nat_nps:>10,}  move={nat['best_move']} score={nat['score']:+d}")
    speedup = py_t / nat_t if nat_t > 0 else 0.0
    same = (py.best_move == nat["best_move"] and py.score == nat["score"])
    print(f"  加速 {speedup:.1f}x  着法/评分一致={same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
