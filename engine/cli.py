"""命令行入口：输入局面，输出最佳着法、评分、备选与理由。

用法示例：
    python -m engine.cli --fen "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w" --time 3
    python -m engine.cli --grid-file board.txt --side black --view-side black
    python -m engine.cli --demo
"""

import argparse
import json
import sys

from .board import Board
from .constants import BLACK, PIECE_TO_FEN, RED, ROWS, COLS
from .decide import decide_position


def print_board(board: Board, view_side: int) -> None:
    """按“我方在下方”的视角打印 ASCII 棋盘。"""
    print("    " + " ".join(str(c) for c in range(COLS)))
    for r in range(ROWS):
        if view_side == RED:
            row = r
        else:
            row = ROWS - 1 - r
        cells = []
        for c in range(COLS):
            col = c if view_side == RED else COLS - 1 - c
            p = board.cells[row * COLS + col]
            cells.append(PIECE_TO_FEN[p] if p else ".")
        print(f"{row:>2}  " + " ".join(cells))
    print()


def format_move_line(move: dict) -> str:
    """单步棋的紧凑描述。"""
    cap = f"，吃{move['captured']}" if move["captured"] else ""
    check = "，将军" if move["gives_check"] else ""
    depth = f"，{move['score_depth']}层复核" if move.get("score_depth") else ""
    return (f"{move['chinese']}  ({move['iccs']} / {move['ucci']})  "
            f"评分 {move['score_text']}{cap}{check}{depth}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.cli",
        description="中国象棋决策引擎：给出最佳着法与结构化坐标",
    )
    parser.add_argument("--fen", help="标准中国象棋 FEN")
    parser.add_argument("--grid-file", help="10 行 x 9 列的网格文本文件（'.' 表示空位）")
    parser.add_argument("--side", help="轮到哪一方走：red / black（缺省取 FEN 中的行棋方）")
    parser.add_argument("--view-side", help="哪一方在图下方：red / black（缺省同 --side）")
    parser.add_argument("--time", type=float, default=4.0, help="搜索时间预算（秒），默认 4.0")
    parser.add_argument("--depth", type=int, default=9, help="最大搜索深度，默认 9")
    parser.add_argument("--json", dest="json_path", help="把完整结果写入 JSON 文件")
    parser.add_argument("--demo", action="store_true", help="用初始局面跑一次")
    parser.add_argument("--quiet", action="store_true", help="不打印棋盘")
    args = parser.parse_args(argv)

    grid = None
    fen = args.fen
    if args.grid_file:
        with open(args.grid_file, "r", encoding="utf-8") as f:
            grid = f.read()
    if args.demo and not fen and not grid:
        from .constants import START_FEN
        fen = START_FEN
    if not fen and not grid:
        parser.print_help()
        return 2

    try:
        result = decide_position(
            fen=fen,
            grid=grid,
            side=args.side,
            view_side=args.view_side,
            time_limit=args.time,
            max_depth=args.depth,
        )
    except ValueError as exc:
        print(f"输入错误: {exc}", file=sys.stderr)
        return 2

    # 还原棋盘用于展示
    board = Board.from_fen(result["input"]["fen"])
    view_side = RED if result["input"]["view_side"] == "red" else BLACK
    if not args.quiet:
        print_board(board, view_side)

    print(f"轮到: {result['input']['side_to_move']}")
    print(f"局面状态: {result['status']}（合法着法 {result['input']['legal_move_count']} 个）")

    best = result["best_move"]
    if best is None:
        print(result["message"])
    else:
        s = result["search"]
        print(f"搜索: 深度 {s['depth']}，节点 {s['nodes']}，用时 {s['time_ms']} ms")
        print(f"最佳着法: {format_move_line(best)}")
        print(f"  坐标: 从 (col={best['from']['col']}, row={best['from']['row']}) "
              f"到 (col={best['to']['col']}, row={best['to']['row']})")
        print(f"  理由: {best['reason']}")
        if result["pv"]["iccs"]:
            print(f"  预计后续: {' '.join(result['pv']['chinese'])}")
        for i, alt in enumerate(result["alternatives"], 1):
            print(f"备选{i}: {format_move_line(alt)}")
            print(f"  {alt['reason']}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"完整结果已写入 {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
