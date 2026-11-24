# cython: language_level=3
"""Native Xiangqi search kernel (Board + movegen + evaluate + search)."""

include "tables.pxi"
include "board.pxi"
include "movegen.pxi"
include "evaluate.pxi"
include "search.pxi"

cdef bint _TABLES_READY = False


cdef void _ensure_init():
    global _TABLES_READY
    if not _TABLES_READY:
        _init_tables()
        _init_movegen_tables()
        tt_clear()
        _TABLES_READY = True


def native_available():
    return True


def perft_cells(cells, int turn, int depth):
    """Count leaves for the given cells (length 90) and side to move."""
    cdef CBoard b
    cdef int buf[90]
    cdef int i
    _ensure_init()
    if len(cells) != 90:
        raise ValueError("cells must have length 90")
    for i in range(90):
        buf[i] = cells[i]
    board_init_from_cells(&b, buf, turn)
    return perft_rec(&b, depth)


def evaluate_cells(cells, int turn):
    """Red-perspective static eval for cells."""
    cdef CBoard b
    cdef int buf[90]
    cdef int i
    _ensure_init()
    for i in range(90):
        buf[i] = cells[i]
    board_init_from_cells(&b, buf, turn)
    return evaluate_board(&b)


def search_cells(cells, int turn, double time_limit=4.0, int max_depth=9,
                 int top_n=2, bint fill_pv=True, int cancel_gen=-1):
    """Run iterative-deepening search; return a dict matching SearchOutcome fields.

    best_move / ranked / pv use (from_sq, to_sq) tuples or None.
    """
    cdef CBoard root
    cdef Searcher s
    cdef int buf[90]
    cdef int i, j, depth, bm_frm, bm_to, bm_score, ok
    cdef int alpha, beta, delta
    cdef int rfrms[MAX_MOVES], rtos[MAX_MOVES], rscores[MAX_MOVES], rexact[MAX_MOVES]
    cdef int rn, best_depth, best_frm, best_to, best_score
    cdef int have_best, alt_depth
    cdef double last_duration, t0, elapsed
    cdef int frms[MAX_MOVES], tos[MAX_MOVES], n
    cdef int pv_frms[64], pv_tos[64], pv_len
    cdef int cand_frm[8], cand_to[8], cand_n
    cdef CBoard scratch, saved
    cdef int score, margin
    cdef list ranked_list, pv_list, results_snap, alts
    cdef tuple best_move_t

    _ensure_init()
    if len(cells) != 90:
        raise ValueError("cells must have length 90")
    for i in range(90):
        buf[i] = cells[i]
    board_init_from_cells(&root, buf, turn)
    searcher_init(&s, &root, time_limit, max_depth)
    if cancel_gen >= 0:
        s.cancel_gen = cancel_gen

    have_best = 0
    best_depth = 0
    best_frm = -1
    best_to = -1
    best_score = 0
    last_duration = 0.0
    results_snap = []
    alt_depth = 0

    for depth in range(1, max_depth + 1):
        if last_duration and (now_sec() - s.start) + last_duration * 3.0 > s.time_limit:
            break
        t0 = now_sec()
        if s.cancel_gen != SEARCH_GEN:
            s.aborted = True
            break
        s.aborted = False
        if depth >= 3 and have_best:
            alpha = best_score - ASPIRATION
            beta = best_score + ASPIRATION
            delta = ASPIRATION
            while True:
                with nogil:
                    ok = root_search(&s, depth, alpha, beta,
                                     &bm_frm, &bm_to, &bm_score,
                                     rfrms, rtos, rscores, rexact, &rn)
                if s.aborted or not ok or bm_frm < 0:
                    break
                if bm_score <= alpha:
                    alpha = alpha - delta
                    if alpha < -INF_SCORE:
                        alpha = -INF_SCORE
                    delta *= 2
                    continue
                if bm_score >= beta:
                    beta = beta + delta
                    if beta > INF_SCORE:
                        beta = INF_SCORE
                    delta *= 2
                    continue
                break
        else:
            with nogil:
                ok = root_search(&s, depth, -INF_SCORE, INF_SCORE,
                                 &bm_frm, &bm_to, &bm_score,
                                 rfrms, rtos, rscores, rexact, &rn)
        last_duration = now_sec() - t0
        if s.aborted:
            break
        if bm_frm < 0:
            elapsed = now_sec() - s.start
            return {
                "best_move": None,
                "score": -MATE_SCORE,
                "depth": depth,
                "nodes": s.nodes,
                "elapsed_ms": int(elapsed * 1000),
                "ranked": [],
                "pv": [],
                "alt_depth": 0,
                "threads": s.root_threads,
            }
        have_best = 1
        best_depth = depth
        best_frm = bm_frm
        best_to = bm_to
        best_score = bm_score
        results_snap = []
        for i in range(rn):
            results_snap.append(((rfrms[i], rtos[i]), rscores[i], bool(rexact[i])))
        if abs(bm_score) > MATE_SCORE - 1000:
            break

    elapsed = now_sec() - s.start
    if not have_best:
        n = generate_legal(&root, root.turn, False, False, frms, tos)
        if n > 0:
            best_move_t = (frms[0], tos[0])
            ranked_list = [(best_move_t, 0, True)]
            pv_list = [best_move_t]
        else:
            best_move_t = None
            ranked_list = []
            pv_list = []
        return {
            "best_move": best_move_t,
            "score": 0,
            "depth": 0,
            "nodes": s.nodes,
            "elapsed_ms": int(elapsed * 1000),
            "ranked": ranked_list,
            "pv": pv_list,
            "alt_depth": 0,
            "threads": s.root_threads,
        }

    # sort results by score desc for ranked seed
    results_snap.sort(key=lambda item: item[1], reverse=True)

    if top_n > 0:
        s.deadline = s.start + s.time_limit * 1.15
        alt_depth = best_depth - 1
        if alt_depth < 2:
            alt_depth = 2
        cand_n = 0
        for mv, sc, exact in results_snap:
            if mv[0] == best_frm and mv[1] == best_to:
                continue
            cand_frm[cand_n] = mv[0]
            cand_to[cand_n] = mv[1]
            cand_n += 1
            if cand_n >= min(4, top_n):
                break
        ranked_list = [((best_frm, best_to), best_score, True)]
        alts = []
        for i in range(cand_n):
            if now_sec() > s.deadline:
                break
            board_clone(&scratch, &root)
            board_apply(&scratch, cand_frm[i], cand_to[i])
            saved = s.board
            s.board = scratch
            s.aborted = False
            score = -negamax(&s, alt_depth, -INF_SCORE, INF_SCORE, 1, True)
            s.board = saved
            if s.aborted:
                break
            alts.append(((cand_frm[i], cand_to[i]), score, True))
        alts.sort(key=lambda item: item[1], reverse=True)
        ranked_list.extend(alts)
    else:
        ranked_list = [((best_frm, best_to), best_score, True)]

    margin = 50
    if top_n > 0 and best_frm >= 0 and root.cells[best_to]:
        for mv, sc, exact in ranked_list:
            if mv[0] == best_frm and mv[1] == best_to:
                continue
            if not exact:
                continue
            if (not root.cells[mv[1]]) and (best_score - sc) < margin:
                best_frm, best_to = mv[0], mv[1]
                best_score = sc
                ranked_list = [(mv, sc, True)] + [r for r in ranked_list if r[0] != mv]
                break

    # 已被新一轮打断时不再续搜主变，尽快把锁让给新线程
    if s.cancel_gen != SEARCH_GEN:
        fill_pv = False
    # restore searcher board to root for PV
    board_clone(&s.board, &root)
    pv_len = extract_pv(&s, best_frm, best_to, best_depth, fill_pv, pv_frms, pv_tos)
    pv_list = [(pv_frms[i], pv_tos[i]) for i in range(pv_len)]

    return {
        "best_move": (best_frm, best_to),
        "score": best_score,
        "depth": best_depth,
        "nodes": s.nodes,
        "elapsed_ms": int((now_sec() - s.start) * 1000),
        "ranked": ranked_list,
        "pv": pv_list,
        "alt_depth": alt_depth,
        "threads": s.root_threads,
    }


def bump_search_gen():
    """作废正在跑的搜索。新搜索应记下返回值，世代不符就立刻退出。"""
    global SEARCH_GEN
    SEARCH_GEN += 1
    return SEARCH_GEN


def current_search_gen():
    return SEARCH_GEN
