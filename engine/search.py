"""迭代加深 alpha-beta 搜索（含置换表、静态搜索、着法排序）。

对外接口：search(board, time_limit, max_depth) -> SearchOutcome
"""

import time

from .constants import BLACK, CANNON, CHARIOT, HORSE, PIECE_VALUE, RED
from .evaluate import evaluate
from .movegen import generate_captures, generate_legal, in_check

MATE = 100000
INF = 10 ** 7
MAX_PLY = 64

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2


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
        self.tt = {}
        self.killers = [[None, None] for _ in range(MAX_PLY)]
        self.history = {}
        self.root_tt_move = None

    # -- 辅助 -------------------------------------------------------------
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def _time_check(self) -> None:
        if (self.nodes & 511) == 0 and time.perf_counter() > self.deadline:
            raise TimeUp

    def _order_moves(self, moves, ply: int, tt_move=None):
        """按 TT 着法、吃子价值（MVV-LVA）、杀手着法、历史启发排序。

        开局阶段，进攻子（炮/马/车）深入对方底线附近的吃子着法大幅降级，
        让引擎优先搜索正常发展着法。
        """
        cells = self.board.cells
        killers = self.killers[ply] if ply < MAX_PLY else (None, None)
        history = self.history

        # 开局判断：双方帅原位 + 棋子多
        board = self.board
        is_open = (board.cells[85] == 1 and board.cells[4] == 9
                   and sum(1 for p in board.cells if p != 0) > 26)
        # 对方底线阈值
        opp_rows = range(0, 3) if board.turn == RED else range(7, 10)

        scored = []
        for mv in moves:
            victim = cells[mv[1]]
            if mv == tt_move:
                s = 10 ** 7
            elif victim:
                attacker_ptype = cells[mv[0]] & 7
                # MVV-LVA 基础分
                s = (100000
                     + PIECE_VALUE[victim & 7] * 10
                     - PIECE_VALUE[attacker_ptype])
                # 方案 D：开局 + 进攻子深入对方底线吃子 → 降级
                if (is_open
                        and attacker_ptype in (CANNON, HORSE, CHARIOT)
                        and (mv[1] // 9) in opp_rows):
                    if attacker_ptype == CANNON:
                        s -= 60000   # 炮深入吃子降级到杀手着法之后
                    else:
                        s -= 40000   # 马/车深入吃子降级到杀手着法之前
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
        """只搜索吃子（被将军时搜索全部应着），消除水平线效应。

        qdepth 上限与 max_depth 统一：正常情况 ≤ max_depth+2，
        被将军时 ≤ max_depth+4。UI 设 depth=4 时，quiesce 最多走 6/8 层，
        不再出现 depth=4 但 quiesce 偷偷跑 12 层的情况。
        """
        self.nodes += 1
        self._time_check()
        board = self.board
        turn = board.turn
        stand = evaluate(board)
        if turn == BLACK:
            stand = -stand

        checked = in_check(board, turn)
        # 动态上限：跟 max_depth 挂钩
        if checked:
            qcap = self.max_depth + 4      # 被将军多留余量，避免漏杀
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
                # 差值剪枝：即使白吃也不足以提高 alpha
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
    def negamax(self, depth, alpha, beta, ply):
        self.nodes += 1
        self._time_check()
        board = self.board

        # 重复局面按和棋处理（简化规则：不做长将判负）
        if ply > 0 and board.is_repetition():
            return 0

        if depth <= 0:
            return self.quiesce(alpha, beta, ply, 0)

        key = board.hash
        entry = self.tt.get(key)
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

        alpha_orig = alpha
        moves = generate_legal(board, board.turn)
        if not moves:
            # 无合法着法：被将死或困毙，均判负
            return -MATE + ply

        moves = self._order_moves(moves, ply, tt_move)
        best = -INF
        best_move = None
        for mv in moves:
            board.apply(mv)
            try:
                score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.undo()
            if score > best:
                best = score
                best_move = mv
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    # 记录杀手着法（仅非吃子着法）
                    if not board.cells[mv[1]]:
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
        self.tt[key] = (depth, best, flag, best_move)
        return best

    # -- 根节点 -----------------------------------------------------------
    def root_search(self, depth: int):
        """搜索根节点，返回 (最佳着法, 最佳评分, [(着法, 评分, 是否精确), ...])。"""
        board = self.board
        moves = generate_legal(board, board.turn)
        if not moves:
            return None, -MATE, []
        moves = self._order_moves(moves, 0, self.root_tt_move)

        alpha, beta = -INF, INF
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
            # 只有把 alpha 抬高的着法得到精确分；其余是对手回应的评分上界
            results.append((mv, score, score > prev_alpha))
            if score > best_score:
                best_score = score
                best_move = mv
                self.root_tt_move = mv
            if score > alpha:
                alpha = score
        return best_move, best_score, results

    # -- 主变提取 ---------------------------------------------------------
    def extract_pv(self, first_move, max_len: int = 6):
        pv = []
        board = self.board.clone()
        move = first_move
        for _ in range(max_len):
            if move is None:
                break
            if move not in generate_legal(board, board.turn):
                break
            pv.append(move)
            board.apply(move)
            entry = self.tt.get(board.hash)
            move = entry[3] if entry else None
        return pv


def search(board, time_limit: float = 4.0, max_depth: int = 8) -> SearchOutcome:
    """迭代加深搜索入口。

    time_limit 为总时间预算（秒）。根据上一层实际耗时预估下一层，
    超出预算就不再加深，保证在几秒内返回可用结果。
    """
    searcher = Searcher(board, time_limit, max_depth)
    best = None
    last_duration = 0.0

    for depth in range(1, max_depth + 1):
        # 预估下一层耗时（约 3 倍于上一层），超出预算就不再加深
        if last_duration and searcher.elapsed() + last_duration * 3.0 > searcher.time_limit:
            break
        t0 = time.perf_counter()
        try:
            bm, bs, results = searcher.root_search(depth)
        except TimeUp:
            break
        last_duration = time.perf_counter() - t0
        if bm is None:
            return SearchOutcome(None, -MATE, depth, searcher.nodes,
                                 int(searcher.elapsed() * 1000), [], [])
        best = (depth, bm, bs, results)
        if abs(bs) > MATE - 1000:
            break

    if best is None:
        # 极端情况：连第一层都没完成，退化为第一个合法着法
        legal = generate_legal(board)
        bm = legal[0] if legal else None
        return SearchOutcome(bm, 0, 0, searcher.nodes,
                             int(searcher.elapsed() * 1000),
                             [(bm, 0, True)] if bm else [], [bm] if bm else [])

    depth, best_move, best_score, results = best
    ranked = sorted(results, key=lambda item: item[1], reverse=True)

    # 备选着法：根节点的非最佳着法只有评分上界，直接用会并列、难以排序。
    # 这里用“浅一层、完整窗口”复核前几个候选，得到可比的真实评分。
    searcher.deadline = searcher.start + searcher.time_limit * 1.15
    alt_depth = max(2, depth - 1)
    candidates = [mv for mv, _, _ in ranked if mv != best_move][:4]
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

    # --- 根节点软偏好：吃子 vs 布局差距极小时，优先布局 ---
    # 只在差距 ≤ MARGIN 厘兵时干预；差距大（白吃车）时绝对不碰。
    _MARGIN = 50
    if best_move:
        bto = best_move[1]
        if board.cells[bto]:       # best 是吃子
            for mv, sc, _ in ranked:
                if mv == best_move:
                    continue
                if (not board.cells[mv[1]]) and (best_score - sc) < _MARGIN:
                    best_move = mv
                    best_score = sc
                    ranked = [(mv, sc, True)] + ranked[1:]
                    break

    pv = searcher.extract_pv(best_move, max_len=depth)
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
