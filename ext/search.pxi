# Alpha-beta search with TT, null move, LMR, aspiration, quiescence.
# 深度够时，根着法分到多个核；置换表按槽加锁。

from libc.stdlib cimport malloc, free
from cython.parallel cimport prange
cimport openmp

cdef extern from *:
    """
    #ifdef _MSC_VER
    #include <intrin.h>
    static void tt_acquire(volatile long* lock) {
        while (_InterlockedCompareExchange(lock, 1, 0) != 0) {}
    }
    static void tt_release(volatile long* lock) {
        _InterlockedExchange(lock, 0);
    }
    #else
    static void tt_acquire(volatile long* lock) {
        while (!__sync_bool_compare_and_swap(lock, 0, 1)) {}
    }
    static void tt_release(volatile long* lock) {
        __sync_lock_release(lock);
    }
    #endif
    """
    void tt_acquire(long* lock) nogil
    void tt_release(long* lock) nogil

cdef enum:
    MATE_SCORE = 100000
    INF_SCORE = 10000000
    MAX_PLY = 64
    ASPIRATION = 60
    TT_SIZE = 1 << 20
    TT_MASK = (1 << 20) - 1
    TT_EXACT = 0
    TT_LOWER = 1
    TT_UPPER = 2


cdef unsigned long long TT_KEY[1048576]
cdef int TT_DEPTH[1048576]
cdef int TT_SCORE[1048576]
cdef int TT_FLAG[1048576]
cdef int TT_FROM[1048576]
cdef int TT_TO[1048576]


cdef int SEARCH_GEN = 0
cdef long TT_SPIN[1024]
cdef int ROOT_THREADS_CAP = 12


cdef struct Searcher:
    CBoard board
    double start
    double time_limit
    double deadline
    int max_depth
    long long nodes
    bint aborted
    int cancel_gen
    int killers_frm[64][2]
    int killers_to[64][2]
    int history[8100]
    int root_tt_frm
    int root_tt_to
    int root_threads


cdef inline double now_sec():
    import time
    return time.perf_counter()


cdef void searcher_reset(Searcher* s, CBoard* b, int max_depth) noexcept nogil:
    cdef int i
    board_clone(&s.board, b)
    s.max_depth = max_depth
    s.nodes = 0
    s.aborted = False
    s.root_tt_frm = -1
    s.root_tt_to = -1
    s.root_threads = 1
    for i in range(MAX_PLY):
        s.killers_frm[i][0] = -1
        s.killers_to[i][0] = -1
        s.killers_frm[i][1] = -1
        s.killers_to[i][1] = -1
    for i in range(8100):
        s.history[i] = 0


cdef void searcher_init(Searcher* s, CBoard* b, double time_limit, int max_depth):
    searcher_reset(s, b, max_depth)
    s.start = now_sec()
    s.time_limit = time_limit
    s.deadline = s.start + time_limit
    s.cancel_gen = SEARCH_GEN


cdef inline void time_check(Searcher* s) noexcept nogil:
    # 每 512 节点短暂拿回 GIL，监听线程才能打断这一轮
    if (s.nodes & 511) == 0:
        with gil:
            if SEARCH_GEN != s.cancel_gen or now_sec() > s.deadline:
                s.aborted = True


cdef inline void tt_clear():
    cdef int i
    for i in range(TT_SIZE):
        TT_DEPTH[i] = -1


cdef inline bint tt_probe(unsigned long long key, int* depth, int* score, int* flag,
                          int* frm, int* to) noexcept nogil:
    cdef int i = <int>(key & TT_MASK)
    cdef int lane = i & 1023
    cdef bint hit = False
    tt_acquire(&TT_SPIN[lane])
    if TT_KEY[i] == key and TT_DEPTH[i] >= 0:
        depth[0] = TT_DEPTH[i]
        score[0] = TT_SCORE[i]
        flag[0] = TT_FLAG[i]
        frm[0] = TT_FROM[i]
        to[0] = TT_TO[i]
        hit = True
    tt_release(&TT_SPIN[lane])
    return hit


cdef inline void tt_store(unsigned long long key, int depth, int score, int flag,
                          int frm, int to) noexcept nogil:
    cdef int i = <int>(key & TT_MASK)
    cdef int lane = i & 1023
    tt_acquire(&TT_SPIN[lane])
    if not (TT_KEY[i] == key and TT_DEPTH[i] > depth):
        TT_KEY[i] = key
        TT_DEPTH[i] = depth
        TT_SCORE[i] = score
        TT_FLAG[i] = flag
        TT_FROM[i] = frm
        TT_TO[i] = to
    tt_release(&TT_SPIN[lane])


cdef void order_moves(Searcher* s, int* frms, int* tos, int n, int ply,
                      int tt_frm, int tt_to) noexcept nogil:
    cdef int scores[MAX_MOVES]
    cdef int i, j, victim, attacker_ptype, sc, best_i, tmp
    cdef bint is_open = board_is_opening(&s.board)
    cdef int turn = s.board.turn
    cdef int to_row

    for i in range(n):
        victim = s.board.cells[tos[i]]
        if frms[i] == tt_frm and tos[i] == tt_to and tt_frm >= 0:
            sc = 10000000
        elif victim:
            attacker_ptype = s.board.cells[frms[i]] & 7
            sc = (100000
                  + PIECE_VAL[victim & 7] * 10
                  - PIECE_VAL[attacker_ptype]
                  + (see_value(&s.board, frms[i], tos[i]) >> 2))
            to_row = tos[i] // 9
            if is_open and (attacker_ptype == CANNON or attacker_ptype == HORSE
                            or attacker_ptype == CHARIOT):
                if (turn == RED and to_row <= 2) or (turn == BLACK and to_row >= 7):
                    if attacker_ptype == CANNON:
                        sc -= 60000
                    else:
                        sc -= 40000
            # 全程压低车马炮吃兵的排序，残局也不把贪兵放在车炮前面
            if ((victim & 7) == PAWN
                    and (attacker_ptype == CANNON or attacker_ptype == HORSE
                         or attacker_ptype == CHARIOT)):
                if is_open:
                    sc -= 55000 if attacker_ptype == CANNON else 35000
                else:
                    sc -= 30000 if attacker_ptype == CANNON else 20000
        elif ply < MAX_PLY and frms[i] == s.killers_frm[ply][0] and tos[i] == s.killers_to[ply][0]:
            sc = 90000
        elif ply < MAX_PLY and frms[i] == s.killers_frm[ply][1] and tos[i] == s.killers_to[ply][1]:
            sc = 89000
        else:
            sc = s.history[frms[i] * 90 + tos[i]]
        scores[i] = sc

    # selection sort descending
    for i in range(n):
        best_i = i
        for j in range(i + 1, n):
            if scores[j] > scores[best_i]:
                best_i = j
        if best_i != i:
            tmp = scores[i]; scores[i] = scores[best_i]; scores[best_i] = tmp
            tmp = frms[i]; frms[i] = frms[best_i]; frms[best_i] = tmp
            tmp = tos[i]; tos[i] = tos[best_i]; tos[best_i] = tmp


cdef int quiesce(Searcher* s, int alpha, int beta, int ply, int qdepth) noexcept nogil:
    cdef int stand, checked, qcap, n, i, victim, score
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]
    s.nodes += 1
    time_check(s)
    if s.aborted:
        return 0

    stand = evaluate_board(&s.board)
    if s.board.turn == BLACK:
        stand = -stand

    checked = in_check(&s.board, s.board.turn)
    if checked:
        qcap = s.max_depth + 4
    else:
        qcap = s.max_depth + 2
    if qdepth >= qcap:
        return stand
    if not checked:
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        n = generate_captures(&s.board, s.board.turn, True, False, frms, tos)
    else:
        n = generate_legal(&s.board, s.board.turn, True, True, frms, tos)
        if n == 0:
            return -MATE_SCORE + ply

    order_moves(s, frms, tos, n, ply, -1, -1)
    for i in range(n):
        if not checked:
            victim = s.board.cells[tos[i]]
            if victim and stand + PIECE_VAL[victim & 7] + 200 < alpha:
                continue
        board_apply(&s.board, frms[i], tos[i])
        score = -quiesce(s, -beta, -alpha, ply + 1, qdepth + 1)
        board_undo(&s.board)
        if s.aborted:
            return 0
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


cdef int negamax(Searcher* s, int depth, int alpha, int beta, int ply, bint allow_null) noexcept nogil:
    cdef int key_depth, key_score, key_flag, tt_frm, tt_to
    cdef unsigned long long key
    cdef bint checked
    cdef int R, alpha_orig, n, i, is_capture, reduction, score, best, best_frm, best_to, flag
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]

    s.nodes += 1
    time_check(s)
    if s.aborted:
        return 0

    if ply > 0 and board_is_repetition(&s.board):
        return 0

    if depth <= 0:
        return quiesce(s, alpha, beta, ply, 0)

    key = board_hash(&s.board)
    tt_frm = -1
    tt_to = -1
    if tt_probe(key, &key_depth, &key_score, &key_flag, &tt_frm, &tt_to):
        if key_depth >= depth:
            if key_flag == TT_EXACT:
                return key_score
            if key_flag == TT_LOWER and key_score > alpha:
                alpha = key_score
            elif key_flag == TT_UPPER and key_score < beta:
                beta = key_score
            if alpha >= beta:
                return key_score

    checked = in_check(&s.board, s.board.turn)

    if (allow_null and depth >= 3 and ply > 0 and not checked
            and beta < INF_SCORE // 2 and alpha > -INF_SCORE // 2):
        R = 3 if depth >= 6 else 2
        board_apply_null(&s.board)
        score = -negamax(s, depth - 1 - R, -beta, -beta + 1, ply + 1, False)
        board_undo(&s.board)
        if s.aborted:
            return 0
        if score >= beta:
            return score

    alpha_orig = alpha
    n = generate_legal(&s.board, s.board.turn, True, checked, frms, tos)
    if n == 0:
        return -MATE_SCORE + ply

    order_moves(s, frms, tos, n, ply, tt_frm, tt_to)
    best = -INF_SCORE
    best_frm = -1
    best_to = -1
    for i in range(n):
        is_capture = 1 if s.board.cells[tos[i]] else 0
        reduction = 0
        if (depth >= 3 and i >= 4 and not is_capture and not checked
                and not (frms[i] == tt_frm and tos[i] == tt_to)):
            reduction = 1 if i < 8 else 2
            if reduction > depth - 1:
                reduction = depth - 1

        board_apply(&s.board, frms[i], tos[i])
        if reduction:
            score = -negamax(s, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, True)
            if score > alpha and not s.aborted:
                score = -negamax(s, depth - 1, -beta, -alpha, ply + 1, True)
        else:
            score = -negamax(s, depth - 1, -beta, -alpha, ply + 1, True)
        board_undo(&s.board)
        if s.aborted:
            return 0

        if score > best:
            best = score
            best_frm = frms[i]
            best_to = tos[i]
        if score > alpha:
            alpha = score
            if alpha >= beta:
                if not is_capture and ply < MAX_PLY:
                    if not (s.killers_frm[ply][0] == frms[i] and s.killers_to[ply][0] == tos[i]):
                        s.killers_frm[ply][1] = s.killers_frm[ply][0]
                        s.killers_to[ply][1] = s.killers_to[ply][0]
                        s.killers_frm[ply][0] = frms[i]
                        s.killers_to[ply][0] = tos[i]
                    s.history[frms[i] * 90 + tos[i]] += depth * depth
                break

    if best <= alpha_orig:
        flag = TT_UPPER
    elif best >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    tt_store(key, depth, best, flag, best_frm, best_to)
    return best


cdef bint root_search_parallel(Searcher* s, int depth, int alpha, int beta,
                               int* frms, int* tos, int n,
                               int* out_frm, int* out_to, int* out_score,
                               int* rfrms, int* rtos, int* rscores, int* rexact,
                               int* rn) noexcept nogil:
    """第 1 手定窗口，其余手零窗口分到多核，失败高的再单核重搜。"""
    cdef int procs, nthreads, i, score, prev_alpha, alpha_now
    cdef int best_score, best_frm, best_to
    cdef Searcher* pool
    cdef int* par_score
    cdef int* par_ok

    procs = openmp.omp_get_num_procs()
    nthreads = procs if procs < ROOT_THREADS_CAP else ROOT_THREADS_CAP
    if nthreads > n - 1:
        nthreads = n - 1
    if nthreads < 2:
        return False

    pool = <Searcher*>malloc(n * sizeof(Searcher))
    par_score = <int*>malloc(n * sizeof(int))
    par_ok = <int*>malloc(n * sizeof(int))
    if pool == NULL or par_score == NULL or par_ok == NULL:
        if pool != NULL:
            free(pool)
        if par_score != NULL:
            free(par_score)
        if par_ok != NULL:
            free(par_ok)
        return False

    s.root_threads = nthreads
    prev_alpha = alpha
    board_apply(&s.board, frms[0], tos[0])
    score = -negamax(s, depth - 1, -beta, -alpha, 1, True)
    board_undo(&s.board)
    if s.aborted:
        out_frm[0] = frms[0]
        out_to[0] = tos[0]
        out_score[0] = score
        rfrms[0] = frms[0]
        rtos[0] = tos[0]
        rscores[0] = score
        rexact[0] = 0
        rn[0] = 1
        free(pool)
        free(par_score)
        free(par_ok)
        return True

    best_score = score
    best_frm = frms[0]
    best_to = tos[0]
    rfrms[0] = frms[0]
    rtos[0] = tos[0]
    rscores[0] = score
    rexact[0] = 1 if score > prev_alpha else 0
    rn[0] = 1
    if score >= beta:
        out_frm[0] = best_frm
        out_to[0] = best_to
        out_score[0] = best_score
        s.root_tt_frm = best_frm
        s.root_tt_to = best_to
        free(pool)
        free(par_score)
        free(par_ok)
        return True
    alpha_now = alpha
    if score > alpha_now:
        alpha_now = score
        s.root_tt_frm = best_frm
        s.root_tt_to = best_to

    if alpha_now <= -INF_SCORE // 2:
        free(pool)
        free(par_score)
        free(par_ok)
        s.root_threads = 1
        return False

    for i in prange(1, n, num_threads=nthreads, schedule='dynamic'):
        searcher_reset(&pool[i], &s.board, s.max_depth)
        pool[i].time_limit = s.time_limit
        pool[i].deadline = s.deadline
        pool[i].start = s.start
        pool[i].cancel_gen = s.cancel_gen
        board_apply(&pool[i].board, frms[i], tos[i])
        par_score[i] = -negamax(&pool[i], depth - 1, -(alpha_now + 1), -alpha_now, 1, True)
        par_ok[i] = 0 if pool[i].aborted else 1

    for i in range(1, n):
        s.nodes += pool[i].nodes
        if not par_ok[i]:
            s.aborted = True
    if SEARCH_GEN != s.cancel_gen:
        s.aborted = True
    if not s.aborted:
        for i in range(1, n):
            if par_score[i] <= alpha_now:
                rfrms[rn[0]] = frms[i]
                rtos[rn[0]] = tos[i]
                rscores[rn[0]] = par_score[i]
                rexact[rn[0]] = 0
                rn[0] += 1
                continue
            prev_alpha = alpha_now
            board_apply(&s.board, frms[i], tos[i])
            score = -negamax(s, depth - 1, -beta, -alpha_now, 1, True)
            board_undo(&s.board)
            if s.aborted:
                break
            rfrms[rn[0]] = frms[i]
            rtos[rn[0]] = tos[i]
            rscores[rn[0]] = score
            rexact[rn[0]] = 1 if score > prev_alpha else 0
            rn[0] += 1
            if score > best_score:
                best_score = score
                best_frm = frms[i]
                best_to = tos[i]
                s.root_tt_frm = best_frm
                s.root_tt_to = best_to
            if score > alpha_now:
                alpha_now = score
                if alpha_now >= beta:
                    break

    out_frm[0] = best_frm
    out_to[0] = best_to
    out_score[0] = best_score
    free(pool)
    free(par_score)
    free(par_ok)
    return True


cdef int root_search(Searcher* s, int depth, int alpha, int beta,
                     int* out_frm, int* out_to, int* out_score,
                     int* rfrms, int* rtos, int* rscores, int* rexact, int* rn) noexcept nogil:
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]
    cdef int n, i, prev_alpha, score
    cdef int best_score = -INF_SCORE
    cdef int best_frm = -1, best_to = -1

    n = generate_legal(&s.board, s.board.turn, False, False, frms, tos)
    if n == 0:
        out_frm[0] = -1
        out_to[0] = -1
        out_score[0] = -MATE_SCORE
        rn[0] = 0
        return 0

    order_moves(s, frms, tos, n, 0, s.root_tt_frm, s.root_tt_to)
    rn[0] = 0
    if depth >= 4 and n >= 6 and root_search_parallel(
            s, depth, alpha, beta, frms, tos, n,
            out_frm, out_to, out_score, rfrms, rtos, rscores, rexact, rn):
        return 1
    rn[0] = 0
    s.root_threads = 1
    for i in range(n):
        prev_alpha = alpha
        board_apply(&s.board, frms[i], tos[i])
        score = -negamax(s, depth - 1, -beta, -alpha, 1, True)
        board_undo(&s.board)
        if s.aborted:
            break
        rfrms[rn[0]] = frms[i]
        rtos[rn[0]] = tos[i]
        rscores[rn[0]] = score
        rexact[rn[0]] = 1 if score > prev_alpha else 0
        rn[0] += 1
        if score > best_score:
            best_score = score
            best_frm = frms[i]
            best_to = tos[i]
            s.root_tt_frm = best_frm
            s.root_tt_to = best_to
        if score > alpha:
            alpha = score
            if alpha >= beta:
                break

    out_frm[0] = best_frm
    out_to[0] = best_to
    out_score[0] = best_score
    return 1


cdef inline bint pv_causes_repetition(CBoard* b, int move_frm, int move_to) noexcept nogil:
    """允许单次进退；禁止走成已出现过的局面（连续循环）。"""
    cdef bint rep
    board_apply(b, move_frm, move_to)
    rep = board_is_repetition(b)
    board_undo(b)
    return rep


cdef inline bint pv_move_ok(CBoard* b, int move_frm, int move_to) noexcept nogil:
    if move_frm < 0:
        return False
    return not pv_causes_repetition(b, move_frm, move_to)


cdef inline bint pv_is_forbidden(int move_frm, int move_to,
                                 int* bad_frms, int* bad_tos, int bad_n) noexcept nogil:
    cdef int i
    for i in range(bad_n):
        if bad_frms[i] == move_frm and bad_tos[i] == move_to:
            return True
    return False


cdef int extract_pv(Searcher* s, int first_frm, int first_to, int max_len, bint fill,
                    int* pv_frms, int* pv_tos):
    cdef CBoard scratch
    cdef int ply, n, i, found, move_frm, move_to, remain, pv_len
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]
    cdef int bm_frm, bm_to, bm_score, rn
    cdef int rfrms[MAX_MOVES], rtos[MAX_MOVES], rscores[MAX_MOVES], rexact[MAX_MOVES]
    cdef int e_depth, e_score, e_flag, e_frm, e_to
    cdef CBoard saved
    cdef int bad_frms[MAX_MOVES], bad_tos[MAX_MOVES], bad_n, attempt
    cdef int chosen_frm, chosen_to

    board_clone(&scratch, &s.board)
    move_frm = first_frm
    move_to = first_to
    pv_len = 0
    if fill:
        if s.deadline < now_sec() + 0.8:
            s.deadline = now_sec() + 0.8

    for ply in range(max_len):
        n = generate_legal(&scratch, scratch.turn, False, False, frms, tos)
        if n == 0:
            break
        bad_n = 0
        chosen_frm = -1
        chosen_to = -1
        for attempt in range(n + 1):
            found = 0
            if move_frm >= 0 and not pv_is_forbidden(move_frm, move_to, bad_frms, bad_tos, bad_n):
                for i in range(n):
                    if frms[i] == move_frm and tos[i] == move_to:
                        found = 1
                        break
            if not found:
                if not fill:
                    break
                saved = s.board
                s.board = scratch
                remain = max_len - ply
                if remain < 1:
                    remain = 1
                if remain > 3:
                    remain = 3
                root_search(s, remain, -INF_SCORE, INF_SCORE,
                            &bm_frm, &bm_to, &bm_score,
                            rfrms, rtos, rscores, rexact, &rn)
                s.board = saved
                if s.aborted:
                    s.aborted = False
                move_frm = bm_frm
                move_to = bm_to
                found = 0
                if (move_frm >= 0
                        and not pv_is_forbidden(move_frm, move_to, bad_frms, bad_tos, bad_n)
                        and pv_move_ok(&scratch, move_frm, move_to)):
                    n = generate_legal(&scratch, scratch.turn, False, False, frms, tos)
                    for i in range(n):
                        if frms[i] == move_frm and tos[i] == move_to:
                            found = 1
                            break
                if not found:
                    # 续搜着会重复局面：按着法排序改选
                    n = generate_legal(&scratch, scratch.turn, False, False, frms, tos)
                    saved = s.board
                    s.board = scratch
                    order_moves(s, frms, tos, n, 0, -1, -1)
                    s.board = saved
                    move_frm = -1
                    move_to = -1
                    for i in range(n):
                        if pv_is_forbidden(frms[i], tos[i], bad_frms, bad_tos, bad_n):
                            continue
                        if pv_move_ok(&scratch, frms[i], tos[i]):
                            move_frm = frms[i]
                            move_to = tos[i]
                            found = 1
                            break
                if not found:
                    break
            if not pv_move_ok(&scratch, move_frm, move_to):
                if bad_n < MAX_MOVES:
                    bad_frms[bad_n] = move_frm
                    bad_tos[bad_n] = move_to
                    bad_n += 1
                move_frm = -1
                move_to = -1
                if not fill:
                    break
                continue
            chosen_frm = move_frm
            chosen_to = move_to
            break
        if chosen_frm < 0:
            break
        pv_frms[pv_len] = chosen_frm
        pv_tos[pv_len] = chosen_to
        pv_len += 1
        board_apply(&scratch, chosen_frm, chosen_to)
        if tt_probe(board_hash(&scratch), &e_depth, &e_score, &e_flag, &e_frm, &e_to):
            move_frm = e_frm
            move_to = e_to
        else:
            move_frm = -1
            move_to = -1
    return pv_len
