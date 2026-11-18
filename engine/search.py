"""迭代加深 alpha-beta 搜索（置换表、静态搜索、空着剪枝、LMR、aspiration）。

对外接口：search(board, time_limit, max_depth) -> SearchOutcome

若已编译 Cython 扩展 engine._xiangqi，search() 优先走原生核；
否则回退到本文件的纯 Python Searcher。
"""

import time

from .constants import BLACK, CANNON, CHARIOT, HORSE, PIECE_VALUE, RED
from .evaluate import evaluate, see_value
from .movegen import generate_captures, generate_legal, in_check

MATE = 100000
INF = 10 ** 7
MAX_PLY = 64
ASPIRATION = 60

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 定长置换表（模块级共享，避免每次搜索分配）：2^20 槽，depth-prefer
TT_SIZE = 1 << 20
TT_MASK = TT_SIZE - 1
_TT_KEY = [0] * TT_SIZE
_TT_DEPTH = [-1] * TT_SIZE
_TT_SCORE = [0] * TT_SIZE
_TT_FLAG = [0] * TT_SIZE
_TT_MOVE = [None] * TT_SIZE

try:
    from engine._xiangqi import search_cells as _native_search_cells
except ImportError:  # pragma: no cover - 未编译时回退
    _native_search_cells = None


def native_available() -> bool:
    """是否已加载 Cython 搜索核。"""
    return _native_search_cells is not None


def _outcome_from_native(raw) -> "SearchOutcome":
    return SearchOutcome(
        best_move=raw["best_move"],
        score=raw["score"],
        depth=raw["depth"],
        nodes=raw["nodes"],
        elapsed_ms=raw["elapsed_ms"],
        ranked=raw["ranked"],
        pv=raw["pv"],
        alt_depth=raw.get("alt_depth", 0),
    )


class TimeUp(Exception):
    """搜索超时，用于从递归中快速退出。"""


class SearchOutcome:
    """搜索结果容器。ranked 中每项为 (着法, 评分, 评分是否精确)。"""

    def __init__(self, best_move, score, depth, nodes, elapsed_ms, ranked, pv, alt_depth=0):
        self.best_move = best_move
        self.score = score
        self.depth = depth
        self.nodes = nodes
        self.elapsed_ms = elapsed_ms
        self.ranked = ranked
        self.pv = pv
        self.alt_depth = alt_depth

    @property
    def mate_in(self):
        """若评分是杀棋，返回我方取胜所需步数，否则 None。"""
        if abs(self.score) > MATE - 1000:
            return (MATE - abs(self.score) + 1) // 2
        return None


class Searcher:
    """一次搜索会话，内部维护置换表、杀手着法与历史启发表。"""

    def __init__(self, board, time_limit: float = 3.0, max_depth: int = 8):
        self.board = board
        self.start = time.perf_counter()
        self.time_limit = time_limit
        self.max_depth = max_depth
        self.deadline = self.start + time_limit
        self.nodes = 0
        self.tt_key = _TT_KEY
        self.tt_depth = _TT_DEPTH
        self.tt_score = _TT_SCORE
        self.tt_flag = _TT_FLAG
        self.tt_move = _TT_MOVE
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.root_tt_move = None

    # -- 辅助 -------------------------------------------------------------
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def _time_check(self) -> None:
        if (self.nodes & 511) == 0 and time.perf_counter() > self.deadline:
            raise TimeUp

    def _tt_probe(self, key):
        i = key & TT_MASK
        if self.tt_key[i] == key and self.tt_depth[i] >= 0:
            return (self.tt_depth[i], self.tt_score[i],
                    self.tt_flag[i], self.tt_move[i])
        return None

    def _tt_store(self, key, depth, score, flag, move):
        i = key & TT_MASK
        if self.tt_key[i] == key and self.tt_depth[i] > depth:
            return
        self.tt_key[i] = key
        self.tt_depth[i] = depth
        self.tt_score[i] = score
        self.tt_flag[i] = flag
        self.tt_move[i] = move

    def _order_moves(self, moves, ply: int, tt_move=None):
        """按 TT 着法、吃子价值（MVV-LVA+SEE）、杀手着法、历史启发排序。"""
        cells = self.board.cells
        killers = self.killers[ply] if ply < MAX_PLY else (None, None)
        history = self.history
        board = self.board
        is_open = board.is_opening()
        opp_rows = range(0, 3) if board.turn == RED else range(7, 10)

        scored = []
        for mv in moves:
            victim = cells[mv[1]]
            if mv == tt_move:
                s = 10 ** 7
            elif victim:
                attacker_ptype = cells[mv[0]] & 7
                # MVV-LVA；SEE 只作微调，避免打乱原有吃子优先级
                s = (100000
                     + PIECE_VALUE[victim & 7] * 10
                     - PIECE_VALUE[attacker_ptype]
                     + (see_value(board, mv) >> 2))
                if (is_open
                        and attacker_ptype in (CANNON, HORSE, CHARIOT)
                        and (mv[1] // 9) in opp_rows):
                    if attacker_ptype == CANNON:
                        s -= 60000
                    else:
                        s -= 40000
            elif mv == killers[0]:
                s = 90000
            elif mv == killers[1]:
                s = 89000
            else:
                s = history.get(mv, 0)
            scored.append((s, mv))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [mv for _, mv in scored]

    # -- 静态搜索 ---------------------------------------------------------
    def quiesce(self, alpha, beta, ply, qdepth):
        """只搜索吃子（被将军时搜索全部应着），消除水平线效应。"""
        self.nodes += 1
        self._time_check()
        board = self.board
        turn = board.turn
        stand = evaluate(board)
        if turn == BLACK:
            stand = -stand

        checked = in_check(board, turn)
        if checked:
            qcap = self.max_depth + 4
        else:
            qcap = self.max_depth + 2
        if qdepth >= qcap:
            return stand
        if not checked:
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            moves = generate_captures(board, turn, checked)
        else:
            moves = generate_legal(board, turn)
            if not moves:
                return -MATE + ply

        moves = self._order_moves(moves, ply)
        for mv in moves:
            if not checked:
                victim = board.cells[mv[1]]
                # 差值剪枝：即使白吃也不足以提高 alpha（保持原阈值，避免误剪战术吃子）
                if victim and stand + PIECE_VALUE[victim & 7] + 200 < alpha:
                    continue
            board.apply(mv)
            try:
                score = -self.quiesce(-beta, -alpha, ply + 1, qdepth + 1)
            finally:
                board.undo()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    # -- 主搜索 -----------------------------------------------------------
    def negamax(self, depth, alpha, beta, ply, allow_null=True):
        self.nodes += 1
        self._time_check()
        board = self.board

        if ply > 0 and board.is_repetition():
            return 0

        if depth <= 0:
            return self.quiesce(alpha, beta, ply, 0)

        key = board.hash
        entry = self._tt_probe(key)
        tt_move = None
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == TT_EXACT:
                    return e_score
                if e_flag == TT_LOWER and e_score > alpha:
                    alpha = e_score
                elif e_flag == TT_UPPER and e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score

        checked = in_check(board, board.turn)

        if (allow_null and depth >= 3 and ply > 0 and not checked
                and beta < INF // 2 and alpha > -INF // 2):
            R = 3 if depth >= 6 else 2
            board.apply_null()
            try:
                score = -self.negamax(depth - 1 - R, -beta, -beta + 1, ply + 1,
                                      allow_null=False)
            finally:
                board.undo()
            if score >= beta:
                return score

        alpha_orig = alpha
        moves = generate_legal(board, board.turn)
        if not moves:
            return -MATE + ply

        moves = self._order_moves(moves, ply, tt_move)
        best = -INF
        best_move = None
        for i, mv in enumerate(moves):
            is_capture = bool(board.cells[mv[1]])
            reduction = 0
            if (depth >= 3 and i >= 4 and not is_capture and not checked
                    and mv != tt_move):
                reduction = 1 if i < 8 else 2
                reduction = min(reduction, depth - 1)

            board.apply(mv)
            try:
                if reduction:
                    score = -self.negamax(depth - 1 - reduction, -alpha - 1, -alpha,
                                          ply + 1)
                    if score > alpha:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.undo()
            if score > best:
                best = score
                best_move = mv
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if not is_capture:
                        slot = self.killers[ply]
                        if slot[0] != mv:
                            slot[1] = slot[0]
                            slot[0] = mv
                        self.history[mv] = self.history.get(mv, 0) + depth * depth
                    break

        if best <= alpha_orig:
            flag = TT_UPPER
        elif best >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self._tt_store(key, depth, best, flag, best_move)
        return best

    # -- 根节点 -----------------------------------------------------------
    def root_search(self, depth: int, alpha=-INF, beta=INF):
        """搜索根节点，返回 (最佳着法, 最佳评分, [(着法, 评分, 是否精确), ...])。"""
        board = self.board
        moves = generate_legal(board, board.turn)
        if not moves:
            return None, -MATE, []
        moves = self._order_moves(moves, 0, self.root_tt_move)

        best_score = -INF
        best_move = None
        results = []
        for mv in moves:
            prev_alpha = alpha
            board.apply(mv)
            try:
                score = -self.negamax(depth - 1, -beta, -alpha, 1)
            finally:
                board.undo()
            results.append((mv, score, score > prev_alpha))
            if score > best_score:
                best_score = score
                best_move = mv
                self.root_tt_move = mv
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        return best_move, best_score, results

    # -- 主变提取 ---------------------------------------------------------
    def extract_pv(self, first_move, max_len: int = 6, fill: bool = True):
        """抽出长度尽量接近 max_len 的主变。

        优先跟置换表；fill=True 且表断了则浅层续搜（分析页）；
        fill=False 时只跟 TT（对弈页）。
        """
        pv = []
        board = self.board.clone()
        move = first_move
        if fill:
            self.deadline = max(self.deadline, time.perf_counter() + 0.8)
        for ply in range(max_len):
            legal = generate_legal(board, board.turn)
            if not legal:
                break
            if move is None or move not in legal:
                if not fill:
                    break
                origin = self.board
                self.board = board
                try:
                    remain = max_len - ply
                    bm, _, _ = self.root_search(max(1, min(remain, 3)))
                except TimeUp:
                    bm = None
                finally:
                    self.board = origin
                move = bm
                if move is None or move not in generate_legal(board, board.turn):
                    break
            pv.append(move)
            board.apply(move)
            entry = self._tt_probe(board.hash)
            move = entry[3] if entry else None
        return pv


def search(board, time_limit: float = 4.0, max_depth: int = 8,
           top_n: int = 2, fill_pv: bool = True) -> SearchOutcome:
    """迭代加深搜索入口。

    time_limit 为总时间预算（秒）。根据上一层实际耗时预估下一层，
    超出预算就不再加深，保证在几秒内返回可用结果。
    top_n<=0 时跳过备选着法复核；fill_pv=False 时 PV 只跟置换表。
    若存在 engine._xiangqi，优先走原生核。
    """
    if _native_search_cells is not None:
        raw = _native_search_cells(
            board.cells, board.turn, time_limit, max_depth, top_n, fill_pv)
        return _outcome_from_native(raw)
    return search_python(board, time_limit, max_depth, top_n, fill_pv)


def search_python(board, time_limit: float = 4.0, max_depth: int = 8,
                  top_n: int = 2, fill_pv: bool = True) -> SearchOutcome:
    """纯 Python 搜索（基准 / 回退路径）。"""
    searcher = Searcher(board, time_limit, max_depth)
    best = None
    last_duration = 0.0
    prev_score = 0

    for depth in range(1, max_depth + 1):
        if last_duration and searcher.elapsed() + last_duration * 3.0 > searcher.time_limit:
            break
        t0 = time.perf_counter()
        try:
            if depth >= 3 and best is not None:
                alpha = prev_score - ASPIRATION
                beta = prev_score + ASPIRATION
                delta = ASPIRATION
                while True:
                    bm, bs, results = searcher.root_search(depth, alpha, beta)
                    if bm is None:
                        break
                    if bs <= alpha:
                        alpha = max(-INF, alpha - delta)
                        delta *= 2
                        continue
                    if bs >= beta:
                        beta = min(INF, beta + delta)
                        delta *= 2
                        continue
                    break
            else:
                bm, bs, results = searcher.root_search(depth)
        except TimeUp:
            break
        last_duration = time.perf_counter() - t0
        if bm is None:
            return SearchOutcome(None, -MATE, depth, searcher.nodes,
                                 int(searcher.elapsed() * 1000), [], [])
        best = (depth, bm, bs, results)
        prev_score = bs
        if abs(bs) > MATE - 1000:
            break

    if best is None:
        legal = generate_legal(board)
        bm = legal[0] if legal else None
        return SearchOutcome(bm, 0, 0, searcher.nodes,
                             int(searcher.elapsed() * 1000),
                             [(bm, 0, True)] if bm else [], [bm] if bm else [])

    depth, best_move, best_score, results = best
    ranked = sorted(results, key=lambda item: item[1], reverse=True)
    alt_depth = 0

    if top_n > 0:
        searcher.deadline = searcher.start + searcher.time_limit * 1.15
        alt_depth = max(2, depth - 1)
        candidates = [mv for mv, _, _ in ranked if mv != best_move][:min(4, top_n)]
        alternatives = []
        for mv in candidates:
            if searcher.elapsed() > searcher.deadline - searcher.start:
                break
            scratch = board.clone()
            scratch.apply(mv)
            origin = searcher.board
            searcher.board = scratch
            try:
                score = -searcher.negamax(alt_depth, -INF, INF, 1)
            except TimeUp:
                break
            finally:
                searcher.board = origin
            alternatives.append((mv, score, True))
        alternatives.sort(key=lambda item: item[1], reverse=True)
        ranked = [(best_move, best_score, True)] + alternatives
    else:
        ranked = [(best_move, best_score, True)]

    # 软偏好只在备选已精确复核时启用。top_n=0 时根节点非最佳着法多为
    # alpha-beta 上界，用其替换最佳着会明显伤棋力（对弈路径的主要回归点）。
    _MARGIN = 50
    if top_n > 0 and best_move and board.cells[best_move[1]]:
        for mv, sc, exact in ranked:
            if mv == best_move or not exact:
                continue
            if (not board.cells[mv[1]]) and (best_score - sc) < _MARGIN:
                best_move = mv
                best_score = sc
                ranked = [(mv, sc, True)] + [r for r in ranked if r[0] != mv]
                break

    pv = searcher.extract_pv(best_move, max_len=depth, fill=fill_pv)
    return SearchOutcome(
        best_move=best_move,
        score=best_score,
        depth=depth,
        nodes=searcher.nodes,
        elapsed_ms=int(searcher.elapsed() * 1000),
        ranked=ranked,
        pv=pv,
        alt_depth=alt_depth,
    )
