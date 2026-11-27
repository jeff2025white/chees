"""迭代加深 alpha-beta 搜索（置换表、静态搜索、空着剪枝、LMR、aspiration）。

对外接口：search(board, time_limit, max_depth) -> SearchOutcome

若已编译 Cython 扩展 engine._xiangqi，search() 优先走原生核；
否则回退到本文件的纯 Python Searcher。
"""

import threading
import time

from .constants import BLACK, CANNON, CHARIOT, HORSE, PAWN, PIECE_VALUE, RED
from .evaluate import evaluate, see_value
from .movegen import generate_captures, generate_legal, in_check

MATE = 100000
INF = 10 ** 7
MAX_PLY = 64
ASPIRATION = 60
# 长将：比任何子力损失都重，又轻于真正的将死。按步数略减，避免和杀棋分混在一起。
PERP_CHECK = 30000
PERP_SLACK = 512

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 无效剪枝余量（厘兵），下标为深度。兵 100、马 400、车 1200。
_FUTILITY_MARGIN = (0, 300, 550, 900, 1400)
# 浅层每节点至少先搜这么多手安静着，再跳过后面的。
_LMP_QUIET = (0, 8, 12, 18)


def perp_magnitude(ply: int) -> int:
    """长将分的绝对值。越早被判，分越重，但仍落在杀棋分以下。"""
    if ply < 0:
        ply = 0
    if ply > PERP_SLACK:
        ply = PERP_SLACK
    return PERP_CHECK - ply


def is_perp_score(score: int) -> bool:
    """分数是否来自长将（与路径有关，不能写入置换表）。"""
    mag = abs(score)
    return PERP_CHECK - PERP_SLACK <= mag <= PERP_CHECK


def cycle_score(board, ply: int) -> int:
    """局面已经重复时，从行棋方视角给出分数。

    最近这一圈里只有一方步步将军：该方大负分。
    双方都是、或有一步不是将军、或圈里夹着空着：和棋 0。
    """
    n = len(board.hash_stack) - 1
    if len(board.gave_check) != n or len(board.null_move) != n:
        return 0
    h = board.hash_stack[-1]
    k = -1
    for i in range(n - 1, -1, -1):
        if board.hash_stack[i] == h:
            k = i
            break
    if k < 0:
        return 0
    span = n - k
    if span < 2 or span % 2:
        return 0
    us_all = True
    them_all = True
    for i in range(span):
        if board.null_move[k + i]:
            return 0
        gave = board.gave_check[k + i]
        if i % 2 == 0:
            if not gave:
                us_all = False
        elif not gave:
            them_all = False
    if us_all == them_all:
        return 0
    mag = perp_magnitude(ply)
    return -mag if us_all else mag

# 定长置换表（模块级共享，避免每次搜索分配）：2^20 槽，depth-prefer
TT_SIZE = 1 << 20
TT_MASK = TT_SIZE - 1
_TT_KEY = [0] * TT_SIZE
_TT_DEPTH = [-1] * TT_SIZE
_TT_SCORE = [0] * TT_SIZE
_TT_FLAG = [0] * TT_SIZE
_TT_MOVE = [None] * TT_SIZE

try:
    from engine._xiangqi import bump_search_gen as _bump_search_gen
    from engine._xiangqi import current_search_gen as _current_search_gen
    from engine._xiangqi import search_cells as _native_search_cells
except ImportError:  # pragma: no cover - 未编译时回退
    _native_search_cells = None
    _bump_search_gen = None
    _current_search_gen = None

_search_lock = threading.Lock()
_accept_local = threading.local()
_PY_GEN = 0


class SearchCancelled(Exception):
    """有更新的局面接上了，这一轮搜索作废。"""


def cancel_running_search() -> int:
    """打断正在跑的搜索，返回新世代。新线程必须 bind 这个值。"""
    global _PY_GEN
    if _bump_search_gen is not None:
        return int(_bump_search_gen())
    _PY_GEN += 1
    return _PY_GEN


def current_search_token() -> int:
    if _current_search_gen is not None:
        return int(_current_search_gen())
    return _PY_GEN


def bind_search_token(token) -> None:
    """本线程只认这个世代；世代已变就不再搜索。"""
    _accept_local.token = token


def _bound_token():
    return getattr(_accept_local, "token", None)


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
        threads=int(raw.get("threads") or 1),
    )


class TimeUp(Exception):
    """搜索超时，用于从递归中快速退出。"""


class SearchOutcome:
    """搜索结果容器。ranked 中每项为 (着法, 评分, 评分是否精确)。"""

    def __init__(self, best_move, score, depth, nodes, elapsed_ms, ranked, pv, alt_depth=0, threads=1):
        self.best_move = best_move
        self.score = score
        self.depth = depth
        self.nodes = nodes
        self.elapsed_ms = elapsed_ms
        self.ranked = ranked
        self.pv = pv
        self.alt_depth = alt_depth
        self.threads = threads

    @property
    def mate_in(self):
        """若评分是杀棋，返回我方取胜所需步数，否则 None。"""
        if abs(self.score) > MATE - 1000:
            return (MATE - abs(self.score) + 1) // 2
        return None


class Searcher:
    """一次搜索会话，内部维护置换表、杀手着法与历史启发表。"""

    def __init__(self, board, time_limit: float = 3.0, max_depth: int = 9):
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
        if (self.nodes & 511) != 0:
            return
        token = _bound_token()
        if token is not None and token != current_search_token():
            raise SearchCancelled
        if time.perf_counter() > self.deadline:
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
                victim_ptype = victim & 7
                # MVV-LVA；SEE 只作微调，避免打乱原有吃子优先级
                s = (100000
                     + PIECE_VALUE[victim_ptype] * 10
                     - PIECE_VALUE[attacker_ptype]
                     + (see_value(board, mv) >> 2))
                if is_open and attacker_ptype in (CANNON, HORSE, CHARIOT):
                    to_row = mv[1] // 9
                    # 深入对方底三行吃子降权
                    if to_row in opp_rows:
                        s -= 60000 if attacker_ptype == CANNON else 40000
                # 全程压低车马炮吃兵的排序，残局也不把贪兵放在车炮前面
                if (victim_ptype == PAWN
                        and attacker_ptype in (CANNON, HORSE, CHARIOT)):
                    if is_open:
                        s -= 55000 if attacker_ptype == CANNON else 35000
                    else:
                        s -= 30000 if attacker_ptype == CANNON else 20000
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
        board.stamp_check(checked)
        if ply > 0 and board.is_repetition():
            return cycle_score(board, ply)
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
            board.stamp_check(in_check(board, board.turn))
            return cycle_score(board, ply)

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
        board.stamp_check(checked)

        # 根以下才剪。吃子、将军、置换表着法不剪。
        # 反向无效：静态分已经高出 beta 一个子力档，直接判高。
        # 无效：静态分低过 alpha 一个子力档时，跳过安静着。
        # 后期：浅层里排在后面的安静着不再展开。
        futile = False
        if ply > 0 and not checked and depth <= 4 and beta < MATE - 1000:
            stand = evaluate(board)
            if board.turn == BLACK:
                stand = -stand
            margin = _FUTILITY_MARGIN[depth]
            if stand - margin >= beta:
                return beta
            if depth <= 2 and stand + margin <= alpha:
                futile = True

        if (allow_null and depth >= 3 and ply > 0 and not checked
                and beta < INF // 2 and alpha > -INF // 2):
            R = 3 if depth >= 6 else 2
            board.apply_null()
            try:
                score = -self.negamax(depth - 1 - R, -beta, -beta + 1, ply + 1,
                                      allow_null=False)
            finally:
                board.undo()
            if score >= beta and not is_perp_score(score):
                return score

        alpha_orig = alpha
        moves = generate_legal(board, board.turn)
        if not moves:
            return -MATE + ply

        moves = self._order_moves(moves, ply, tt_move)
        best = -INF
        best_move = None
        quiet_n = 0
        lmp_limit = _LMP_QUIET[depth] if depth <= 3 else 0
        for i, mv in enumerate(moves):
            is_capture = bool(board.cells[mv[1]])
            if ply > 0 and not checked and not is_capture and mv != tt_move:
                quiet_n += 1
                if futile or (depth <= 3 and quiet_n > lmp_limit):
                    board.apply(mv)
                    try:
                        gives_check = in_check(board, board.turn)
                    finally:
                        board.undo()
                    if not gives_check:
                        continue
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
        # 长将分依赖这条着法栈，不能留给别的路径复用。
        if not is_perp_score(best):
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
    @staticmethod
    def _causes_repetition(board, move) -> bool:
        """走该着后是否立刻形成重复局面（循环着法的搜索侧近似）。

        允许单次进退/调位；禁止把已经出现过的局面再走一遍（连续循环）。
        """
        board.apply(move)
        rep = board.is_repetition()
        board.undo()
        return rep

    def _pv_move_ok(self, board, move) -> bool:
        return move is not None and not self._causes_repetition(board, move)

    def _pick_pv_fill(self, board, remain: int, forbidden: set):
        """浅层续搜；若着法会重复局面则按排序改选其它着。"""
        origin = self.board
        self.board = board
        bm = None
        ordered = []
        try:
            try:
                bm, _, _ = self.root_search(max(1, min(remain, 3)))
            except TimeUp:
                bm = None
            legal = generate_legal(board, board.turn)
            if (bm is not None and bm not in forbidden and bm in legal
                    and self._pv_move_ok(board, bm)):
                return bm
            ordered = self._order_moves(legal, ply=0)
        finally:
            self.board = origin
        for cand in ordered:
            if cand in forbidden:
                continue
            if self._pv_move_ok(board, cand):
                return cand
        return None

    def extract_pv(self, first_move, max_len: int = 6, fill: bool = True):
        """抽出长度尽量接近 max_len 的主变。

        优先跟置换表；fill=True 且表断了则浅层续搜（分析页）；
        fill=False 时只跟 TT（对弈页）。
        允许合理进退，但遇重复局面则截断或改选，避免推演一直循环同一套着法。
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
            forbidden = set()
            chosen = None
            for _ in range(len(legal) + 1):
                if move is None or move not in legal or move in forbidden:
                    if not fill:
                        break
                    move = self._pick_pv_fill(board, max_len - ply, forbidden)
                    if move is None:
                        break
                if not self._pv_move_ok(board, move):
                    forbidden.add(move)
                    move = None
                    if not fill:
                        break
                    continue
                chosen = move
                break
            if chosen is None:
                break
            pv.append(chosen)
            board.apply(chosen)
            entry = self._tt_probe(board.hash)
            move = entry[3] if entry else None
        return pv


def search(board, time_limit: float = 4.0, max_depth: int = 9,
           top_n: int = 2, fill_pv: bool = True) -> SearchOutcome:
    """迭代加深搜索入口。

    time_limit 为总时间预算（秒）。根据上一层实际耗时预估下一层，
    超出预算就不再加深，保证在几秒内返回可用结果。
    top_n<=0 时跳过备选着法复核；fill_pv=False 时 PV 只跟置换表。
    若存在 engine._xiangqi，优先走原生核。
    """
    token = _bound_token()
    with _search_lock:
        if token is not None and token != current_search_token():
            raise SearchCancelled
        if _native_search_cells is not None:
            raw = _native_search_cells(
                board.cells, board.turn, time_limit, max_depth, top_n, fill_pv,
                -1 if token is None else int(token))
            if token is not None and token != current_search_token():
                raise SearchCancelled
            return _outcome_from_native(raw)
        return search_python(board, time_limit, max_depth, top_n, fill_pv)


def search_python(board, time_limit: float = 4.0, max_depth: int = 9,
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
