"""多进程根节点切分 vs 单进程：固定深度对照基准（不改动 engine 代码）。

原理：
    根节点每个合法着法在子局面里相互独立，可以把它们分给多个 worker 进程，
    各自用自己的置换表搜索到同一深度，最后按分数汇总取最优。
    并行时无法共享 alpha 窗口，每个根着法只能用全窗口，因此节点总数会放大。

用法：
    python test/bench_mp_root.py                          # depth=5, workers=4,8, 前 4 张图
    python test/bench_mp_root.py --depth 6 --limit 2 --workers 4,8,12
    python test/bench_mp_root.py --seq-only               # 只看单进程基线

输出：每张图的单进程/并行耗时、节点数、加速比、最佳着法一致性，最后打印均值。
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# worker 侧全局状态（每张图重建一次进程池时由 initializer 注入）
_W = {}


def _init_worker(fen, depth):
    """进程池初始化：构造棋盘并缓存搜索深度。"""
    from engine.board import Board

    _W["board"] = Board.from_fen(fen)
    _W["depth"] = depth


def _search_root_move(payload):
    """worker 任务：搜索单个根着法，返回 (着法, 分数, 节点数)。

    payload = (move, alpha)：alpha=-INF 时是全窗口（精确分）；
    否则是 (alpha, +INF) 窗口，分数 <= alpha 表示该着法不够好（上界）。
    每个根着法在自己的进程内做 1..depth 浅层迭代预热置换表。
    """
    from engine.search import INF, Searcher

    move, alpha = payload
    board = _W["board"].clone()
    board.apply(move)
    depth = _W["depth"]
    searcher = Searcher(board, time_limit=1e9, max_depth=depth)
    score = -INF
    for d in range(1, depth + 1):
        # 父窗口 (alpha, +INF) 对应子局面窗口 (-INF, -alpha)
        score = -searcher.negamax(d - 1, -INF, -alpha, 1)
    return move, score, searcher.nodes


def build_board(grid_str, view_side):
    """按 decide_position 的截图约定构造棋盘（我方在图下方，先走）。"""
    from engine.board import Board
    from engine.constants import BLACK, RED
    from engine.decide import mirror_grid, parse_side

    orient = parse_side(view_side)
    board = Board.from_grid(
        mirror_grid(grid_str) if orient == BLACK else grid_str, RED)
    board.turn = orient
    return board


def _clear_tt():
    """清空模块级置换表，避免上一次测量污染节点数。"""
    import importlib

    S = importlib.import_module("engine.search")

    for lst, val in zip((S._TT_KEY, S._TT_DEPTH, S._TT_SCORE,
                         S._TT_FLAG, S._TT_MOVE), (0, -1, 0, 0, None)):
        lst[:] = [val] * S.TT_SIZE


def run_sequential(board, depth):
    """单进程基线：完整迭代加深 + 根节点 alpha-beta。"""
    from engine.search import search

    _clear_tt()
    t0 = time.perf_counter()
    outcome = search(board.clone(), time_limit=1e9, max_depth=depth,
                     top_n=0, fill_pv=False)
    elapsed = time.perf_counter() - t0
    return outcome.best_move, outcome.score, outcome.nodes, elapsed


def run_parallel(fen, moves, depth, workers, mode):
    """多进程根节点切分。

    mode="naive"：所有根着法用全窗口独立搜索（最坏情况，无 alpha 共享）。
    mode="alpha"：先并行搜前 workers 个着法拿到 alpha，其余着法用
                  (alpha, +INF) 窄窗口，失败即剪枝。
    mode="asp"  ：先在主进程做 depth-2 的浅层搜索得到评分估计，
                  再用 (估计-100, +INF) 窗口并行搜全部根着法。

    返回 (最佳着法, 分数, 节点总数, 搜索耗时, 启动耗时)。
    """
    from engine.board import Board
    from engine.search import INF, Searcher, search

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    pool = ctx.Pool(processes=workers, initializer=_init_worker,
                    initargs=(fen, depth))
    startup = time.perf_counter() - t0
    results = []
    est = None
    shallow_time = 0.0
    try:
        t1 = time.perf_counter()
        if mode == "asp":
            # 浅层估分：与顺序搜索的 1..depth 迭代加深同源，不算额外作弊
            _clear_tt()
            t_sh = time.perf_counter()
            shallow = search(Board.from_fen(fen), time_limit=1e9,
                             max_depth=max(1, depth - 2), top_n=0,
                             fill_pv=False)
            shallow_time = time.perf_counter() - t_sh
            est = shallow.score
            alpha0 = est - 100
            results = list(pool.imap_unordered(
                _search_root_move, [(mv, alpha0) for mv in moves], chunksize=1))
            if max(r[1] for r in results) <= alpha0:
                # 估计偏差导致全部失败，退回全窗口重搜
                results = list(pool.imap_unordered(
                    _search_root_move, [(mv, -INF) for mv in moves],
                    chunksize=1))
        elif mode == "naive":
            results = list(pool.imap_unordered(
                _search_root_move, [(mv, -INF) for mv in moves], chunksize=1))
        else:
            wave, rest = moves[:workers], moves[workers:]
            results = list(pool.imap_unordered(
                _search_root_move, [(mv, -INF) for mv in wave], chunksize=1))
            if rest:
                alpha = max(r[1] for r in results)
                results.extend(pool.imap_unordered(
                    _search_root_move, [(mv, alpha) for mv in rest],
                    chunksize=1))
                results = list(results)
        search_time = time.perf_counter() - t1
    finally:
        pool.close()
        pool.join()

    best_move, best_score, total_nodes = None, None, 0
    for move, score, nodes in results:
        total_nodes += nodes
        if best_score is None or score > best_score:
            best_move, best_score = move, score
    # 统计时把浅层估分耗时计入总搜索耗时（它属于本方案的固有成本）
    return (best_move, best_score, total_nodes,
            search_time + shallow_time, startup, est)


def main():
    parser = argparse.ArgumentParser(description="根节点并行加速比对照")
    parser.add_argument("--depth", type=int, default=5, help="固定搜索深度")
    parser.add_argument("--limit", type=int, default=4, help="最多取几张截图")
    parser.add_argument("--workers", default="4,8", help="并行进程数，逗号分隔")
    parser.add_argument("--mode", default="asp,naive",
                        choices=["asp", "alpha", "naive", "asp,naive"],
                        help="切分方式")
    parser.add_argument("--dir", default=os.path.join(_ROOT, "images"),
                        help="截图目录")
    parser.add_argument("--seq-only", action="store_true", help="只跑单进程基线")
    args = parser.parse_args()

    from engine.movegen import generate_legal
    from engine.notation import chinese_notation
    from engine.recognize import analyze_image
    from engine.search import Searcher

    worker_counts = [] if args.seq_only else [
        int(x) for x in args.workers.split(",") if x.strip()]
    modes = [] if args.seq_only else args.mode.split(",")

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(".png"))
    files = [f for f in files if f.startswith("Screenshot")][:args.limit]
    if not files:
        print(f"未在 {args.dir} 找到截图")
        return 1

    print(f"深度 {args.depth}  图片 {len(files)} 张  配置 "
          f"{[f'{m}/{w}进程' for m in modes for w in worker_counts]}")

    stats = {(m, w): ([], []) for m in modes for w in worker_counts}
    for fn in files:
        analysis = analyze_image(os.path.join(args.dir, fn))
        board = build_board(analysis.grid_str, analysis.view_side)
        moves = generate_legal(board, board.turn)
        if not moves:
            print(f"{fn}: 无合法着法，跳过")
            continue

        seq_move, seq_score, seq_nodes, seq_time = run_sequential(board, args.depth)
        seq_text = chinese_notation(board, seq_move) if seq_move else "-"
        print(f"\n{fn}  根着法 {len(moves)} 个")
        print(f"  单进程   {seq_time:6.2f}s  {seq_nodes:>9,d} 节点  最优 {seq_text} "
              f"({seq_score:+d})")

        # 用引擎自身的排序策略决定波次顺序，尽量让好着法先进波次 1
        ordered = Searcher(board.clone(), 1e9, args.depth)._order_moves(moves, 0)
        for mode in modes:
            for w in worker_counts:
                par_move, par_score, par_nodes, par_time, startup, est = run_parallel(
                    board.to_fen(), ordered, args.depth, w, mode)
                speedup = seq_time / par_time if par_time > 0 else 0.0
                ratio = par_nodes / seq_nodes if seq_nodes else 0.0
                same = "是" if (par_move == seq_move and par_score == seq_score) else "否"
                sp_list, ratio_list = stats[(mode, w)]
                sp_list.append(speedup)
                ratio_list.append(ratio)
                est_text = f" 估分{est:+d}" if est is not None else ""
                print(f"  {mode:<5}/{w:<2} {par_time:6.2f}s  {par_nodes:>9,d} 节点  "
                      f"加速 {speedup:5.2f}x  节点 {ratio:5.2f}x  一致 {same}  "
                      f"启动 {startup:.2f}s{est_text}")

    print("\n汇总（均值）")
    for mode in modes:
        for w in worker_counts:
            sp_list, ratio_list = stats[(mode, w)]
            if not sp_list:
                continue
            print(f"  {mode:<5}/{w:<2}: 平均加速 {sum(sp_list) / len(sp_list):5.2f}x  "
                  f"节点放大 {sum(ratio_list) / len(ratio_list):5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
