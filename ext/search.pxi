# Alpha-beta search with TT, null move, LMR, aspiration, quiescence.

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


cdef struct Searcher:
    CBoard board
    double start
    double time_limit
    double deadline
    int max_depth
    long long nodes
    bint aborted
    int killers_frm[64][2]
    int killers_to[64][2]
    int history[8100]
    int root_tt_frm
    int root_tt_to


cdef inline double now_sec():
    import time
    return time.perf_counter()


cdef void searcher_init(Searcher* s, CBoard* b, double time_limit, int max_depth):
    cdef int i, j
    board_clone(&s.board, b)
    # Also copy hash history length 1 already done by clone; for search start we want full cells
    s.start = now_sec()
    s.time_limit = time_limit
    s.deadline = s.start + time_limit
    s.max_depth = max_depth
    s.nodes = 0
    s.aborted = False
    s.root_tt_frm = -1
    s.root_tt_to = -1
    for i in range(MAX_PLY):
        s.killers_frm[i][0] = -1
        s.killers_to[i][0] = -1
        s.killers_frm[i][1] = -1
        s.killers_to[i][1] = -1
    for i in range(8100):
        s.history[i] = 0


cdef inline void time_check(Searcher* s):
    if (s.nodes & 511) == 0 and now_sec() > s.deadline:
        s.aborted = True


cdef inline void tt_clear():
    cdef int i
    for i in range(TT_SIZE):
        TT_DEPTH[i] = -1


cdef inline bint tt_probe(unsigned long long key, int* depth, int* score, int* flag,
                          int* frm, int* to):
    cdef int i = <int>(key & TT_MASK)
    if TT_KEY[i] == key and TT_DEPTH[i] >= 0:
        depth[0] = TT_DEPTH[i]
        score[0] = TT_SCORE[i]
        flag[0] = TT_FLAG[i]
        frm[0] = TT_FROM[i]
        to[0] = TT_TO[i]
        return True
    return False


cdef inline void tt_store(unsigned long long key, int depth, int score, int flag,
                          int frm, int to):
    cdef int i = <int>(key & TT_MASK)
    if TT_KEY[i] == key and TT_DEPTH[i] > depth:
        return
    TT_KEY[i] = key
    TT_DEPTH[i] = depth
    TT_SCORE[i] = score
    TT_FLAG[i] = flag
    TT_FROM[i] = frm
    TT_TO[i] = to


cdef void order_moves(Searcher* s, int* frms, int* tos, int n, int ply,
                      int tt_frm, int tt_to):
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


cdef int quiesce(Searcher* s, int alpha, int beta, int ply, int qdepth):
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


cdef int negamax(Searcher* s, int depth, int alpha, int beta, int ply, bint allow_null):
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


cdef int root_search(Searcher* s, int depth, int alpha, int beta,
                     int* out_frm, int* out_to, int* out_score,
                     int* rfrms, int* rtos, int* rscores, int* rexact, int* rn):
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


cdef int extract_pv(Searcher* s, int first_frm, int first_to, int max_len, bint fill,
                    int* pv_frms, int* pv_tos):
    cdef CBoard scratch
    cdef int ply, n, i, found, move_frm, move_to, remain, pv_len
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]
    cdef int bm_frm, bm_to, bm_score, rn
    cdef int rfrms[MAX_MOVES], rtos[MAX_MOVES], rscores[MAX_MOVES], rexact[MAX_MOVES]
    cdef int e_depth, e_score, e_flag, e_frm, e_to
    cdef CBoard saved

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
        found = 0
        if move_frm >= 0:
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
            if move_frm >= 0:
                n = generate_legal(&scratch, scratch.turn, False, False, frms, tos)
                for i in range(n):
                    if frms[i] == move_frm and tos[i] == move_to:
                        found = 1
                        break
            if not found:
                break
        pv_frms[pv_len] = move_frm
        pv_tos[pv_len] = move_to
        pv_len += 1
        board_apply(&scratch, move_frm, move_to)
        if tt_probe(board_hash(&scratch), &e_depth, &e_score, &e_flag, &e_frm, &e_to):
            move_frm = e_frm
            move_to = e_to
        else:
            move_frm = -1
            move_to = -1
    return pv_len
