# Move generation, attack detection, legal filtering, perft.

cdef int HORSE_TO[90][8]
cdef int HORSE_LEG[90][8]
cdef int HORSE_N[90]
cdef int ELE_TO[90][4]
cdef int ELE_EYE[90][4]
cdef int ELE_NR[90][4]
cdef int ELE_N[90]

cdef int ORTHO_DR[4]
cdef int ORTHO_DC[4]
cdef int DIAG_DR[4]
cdef int DIAG_DC[4]


cdef void _init_movegen_tables() noexcept:
    cdef int sq, r, c, i, nr, nc, dr, dc, lr, lc
    cdef int horse_moves[8][4]
    cdef int ele_moves[4][4]

    ORTHO_DR[0] = 1; ORTHO_DC[0] = 0
    ORTHO_DR[1] = -1; ORTHO_DC[1] = 0
    ORTHO_DR[2] = 0; ORTHO_DC[2] = 1
    ORTHO_DR[3] = 0; ORTHO_DC[3] = -1

    DIAG_DR[0] = 1; DIAG_DC[0] = 1
    DIAG_DR[1] = 1; DIAG_DC[1] = -1
    DIAG_DR[2] = -1; DIAG_DC[2] = 1
    DIAG_DR[3] = -1; DIAG_DC[3] = -1

    horse_moves[0][0] = -2; horse_moves[0][1] = -1; horse_moves[0][2] = -1; horse_moves[0][3] = 0
    horse_moves[1][0] = -2; horse_moves[1][1] = 1; horse_moves[1][2] = -1; horse_moves[1][3] = 0
    horse_moves[2][0] = 2; horse_moves[2][1] = -1; horse_moves[2][2] = 1; horse_moves[2][3] = 0
    horse_moves[3][0] = 2; horse_moves[3][1] = 1; horse_moves[3][2] = 1; horse_moves[3][3] = 0
    horse_moves[4][0] = -1; horse_moves[4][1] = -2; horse_moves[4][2] = 0; horse_moves[4][3] = -1
    horse_moves[5][0] = 1; horse_moves[5][1] = -2; horse_moves[5][2] = 0; horse_moves[5][3] = -1
    horse_moves[6][0] = -1; horse_moves[6][1] = 2; horse_moves[6][2] = 0; horse_moves[6][3] = 1
    horse_moves[7][0] = 1; horse_moves[7][1] = 2; horse_moves[7][2] = 0; horse_moves[7][3] = 1

    ele_moves[0][0] = -2; ele_moves[0][1] = -2; ele_moves[0][2] = -1; ele_moves[0][3] = -1
    ele_moves[1][0] = -2; ele_moves[1][1] = 2; ele_moves[1][2] = -1; ele_moves[1][3] = 1
    ele_moves[2][0] = 2; ele_moves[2][1] = -2; ele_moves[2][2] = 1; ele_moves[2][3] = -1
    ele_moves[3][0] = 2; ele_moves[3][1] = 2; ele_moves[3][2] = 1; ele_moves[3][3] = 1

    for sq in range(SQUARES):
        HORSE_N[sq] = 0
        ELE_N[sq] = 0
        r = sq // COLS
        c = sq % COLS
        for i in range(8):
            dr = horse_moves[i][0]; dc = horse_moves[i][1]
            lr = horse_moves[i][2]; lc = horse_moves[i][3]
            nr = r + dr; nc = c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                HORSE_TO[sq][HORSE_N[sq]] = nr * COLS + nc
                HORSE_LEG[sq][HORSE_N[sq]] = (r + lr) * COLS + (c + lc)
                HORSE_N[sq] += 1
        for i in range(4):
            dr = ele_moves[i][0]; dc = ele_moves[i][1]
            lr = ele_moves[i][2]; lc = ele_moves[i][3]
            nr = r + dr; nc = c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                ELE_TO[sq][ELE_N[sq]] = nr * COLS + nc
                ELE_EYE[sq][ELE_N[sq]] = (r + lr) * COLS + (c + lc)
                ELE_NR[sq][ELE_N[sq]] = nr
                ELE_N[sq] += 1


cdef inline bint in_palace(int color, int row, int col) noexcept nogil:
    if col < 3 or col > 5:
        return False
    if color == RED:
        return 7 <= row <= 9
    return 0 <= row <= 2


cdef inline bint in_board_rc(int row, int col) noexcept nogil:
    return 0 <= row < ROWS and 0 <= col < COLS


cdef inline void _add_move(int* frms, int* tos, int* n, int frm, int to) noexcept nogil:
    if n[0] < MAX_MOVES:
        frms[n[0]] = frm
        tos[n[0]] = to
        n[0] += 1


cdef int generate_pseudo(CBoard* b, int color, int* frms, int* tos) noexcept nogil:
    cdef int n = 0
    cdef int frm, piece, ptype, r, c, i, to, t, nr, nc, dr, dc, eye, lo_r, hi_r
    cdef int enemy = 1 - color
    cdef bint is_red = color == RED
    cdef int* cells = b.cells

    for frm in range(SQUARES):
        piece = cells[frm]
        if piece == EMPTY or (piece >> 3) != color:
            continue
        ptype = piece & 7
        r = frm // COLS
        c = frm % COLS

        if ptype == KING:
            if is_red:
                lo_r, hi_r = 7, 9
            else:
                lo_r, hi_r = 0, 2
            for i in range(4):
                nr = r + ORTHO_DR[i]
                nc = c + ORTHO_DC[i]
                if lo_r <= nr <= hi_r and 3 <= nc <= 5:
                    to = nr * COLS + nc
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        _add_move(frms, tos, &n, frm, to)

        elif ptype == ADVISOR:
            if is_red:
                lo_r, hi_r = 7, 9
            else:
                lo_r, hi_r = 0, 2
            for i in range(4):
                nr = r + DIAG_DR[i]
                nc = c + DIAG_DC[i]
                if lo_r <= nr <= hi_r and 3 <= nc <= 5:
                    to = nr * COLS + nc
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        _add_move(frms, tos, &n, frm, to)

        elif ptype == ELEPHANT:
            for i in range(ELE_N[frm]):
                nr = ELE_NR[frm][i]
                if is_red:
                    if nr < 5:
                        continue
                elif nr > 4:
                    continue
                eye = ELE_EYE[frm][i]
                if cells[eye] != EMPTY:
                    continue
                to = ELE_TO[frm][i]
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    _add_move(frms, tos, &n, frm, to)

        elif ptype == HORSE:
            for i in range(HORSE_N[frm]):
                if cells[HORSE_LEG[frm][i]] != EMPTY:
                    continue
                to = HORSE_TO[frm][i]
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    _add_move(frms, tos, &n, frm, to)

        elif ptype == CHARIOT:
            for i in range(4):
                dr = ORTHO_DR[i]; dc = ORTHO_DC[i]
                nr = r + dr; nc = c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS:
                    to = nr * COLS + nc
                    t = cells[to]
                    if t == EMPTY:
                        _add_move(frms, tos, &n, frm, to)
                    else:
                        if (t >> 3) == enemy:
                            _add_move(frms, tos, &n, frm, to)
                        break
                    nr += dr
                    nc += dc

        elif ptype == CANNON:
            for i in range(4):
                dr = ORTHO_DR[i]; dc = ORTHO_DC[i]
                nr = r + dr; nc = c + dc
                while 0 <= nr < ROWS and 0 <= nc < COLS and cells[nr * COLS + nc] == EMPTY:
                    _add_move(frms, tos, &n, frm, nr * COLS + nc)
                    nr += dr
                    nc += dc
                nr += dr
                nc += dc
                while 0 <= nr < ROWS and 0 <= nc < COLS and cells[nr * COLS + nc] == EMPTY:
                    nr += dr
                    nc += dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if (cells[nr * COLS + nc] >> 3) == enemy:
                        _add_move(frms, tos, &n, frm, nr * COLS + nc)

        elif ptype == PAWN:
            nr = r - 1 if is_red else r + 1
            if 0 <= nr < ROWS:
                to = nr * COLS + c
                t = cells[to]
                if t == EMPTY or (t >> 3) == enemy:
                    _add_move(frms, tos, &n, frm, to)
            if (r <= 4) if is_red else (r >= 5):
                if c > 0:
                    to = r * COLS + c - 1
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        _add_move(frms, tos, &n, frm, to)
                if c + 1 < COLS:
                    to = r * COLS + c + 1
                    t = cells[to]
                    if t == EMPTY or (t >> 3) == enemy:
                        _add_move(frms, tos, &n, frm, to)
    return n


cdef bint is_attacked(CBoard* b, int sq, int by_color, bint king_face) noexcept nogil:
    cdef int* cells = b.cells
    cdef int r = sq // COLS
    cdef int c = sq % COLS
    cdef int chariot = (by_color << 3) | CHARIOT
    cdef int cannon = (by_color << 3) | CANNON
    cdef int king_piece = (by_color << 3) | KING
    cdef int horse = (by_color << 3) | HORSE
    cdef int advisor = (by_color << 3) | ADVISOR
    cdef int elephant = (by_color << 3) | ELEPHANT
    cdef int pawn = (by_color << 3) | PAWN
    cdef int i, first, hr, hc, leg_r, leg_c, ar, ac
    cdef int abs_dr, hdr0, hdc0, hi, e_dr, e_dc, e_er, e_ec, ei

    # vertical
    i = r - 1
    while i >= 0 and cells[i * COLS + c] == EMPTY:
        i -= 1
    if i >= 0:
        first = cells[i * COLS + c]
        if first == chariot:
            return True
        if first == king_piece and king_face:
            return True
        i -= 1
        while i >= 0 and cells[i * COLS + c] == EMPTY:
            i -= 1
        if i >= 0 and cells[i * COLS + c] == cannon:
            return True

    i = r + 1
    while i < ROWS and cells[i * COLS + c] == EMPTY:
        i += 1
    if i < ROWS:
        first = cells[i * COLS + c]
        if first == chariot:
            return True
        if first == king_piece and king_face:
            return True
        i += 1
        while i < ROWS and cells[i * COLS + c] == EMPTY:
            i += 1
        if i < ROWS and cells[i * COLS + c] == cannon:
            return True

    # horizontal
    i = c - 1
    while i >= 0 and cells[r * COLS + i] == EMPTY:
        i -= 1
    if i >= 0:
        first = cells[r * COLS + i]
        if first == chariot:
            return True
        i -= 1
        while i >= 0 and cells[r * COLS + i] == EMPTY:
            i -= 1
        if i >= 0 and cells[r * COLS + i] == cannon:
            return True

    i = c + 1
    while i < COLS and cells[r * COLS + i] == EMPTY:
        i += 1
    if i < COLS:
        first = cells[r * COLS + i]
        if first == chariot:
            return True
        i += 1
        while i < COLS and cells[r * COLS + i] == EMPTY:
            i += 1
        if i < COLS and cells[r * COLS + i] == cannon:
            return True

    # horse (reverse offsets matching Python HORSE_MOVES)
    for hi in range(8):
        if hi == 0: hdr0 = -2; hdc0 = -1
        elif hi == 1: hdr0 = -2; hdc0 = 1
        elif hi == 2: hdr0 = 2; hdc0 = -1
        elif hi == 3: hdr0 = 2; hdc0 = 1
        elif hi == 4: hdr0 = -1; hdc0 = -2
        elif hi == 5: hdr0 = 1; hdc0 = -2
        elif hi == 6: hdr0 = -1; hdc0 = 2
        else: hdr0 = 1; hdc0 = 2
        hr = r + hdr0
        hc = c + hdc0
        if hr < 0 or hr >= ROWS or hc < 0 or hc >= COLS:
            continue
        if cells[hr * COLS + hc] != horse:
            continue
        abs_dr = hdr0 if hdr0 >= 0 else -hdr0
        if abs_dr == 2:
            leg_r = hr - (1 if hdr0 > 0 else -1)
            leg_c = hc
        else:
            leg_r = hr
            leg_c = hc - (1 if hdc0 > 0 else -1)
        if cells[leg_r * COLS + leg_c] == EMPTY:
            return True

    # pawn
    if by_color == RED:
        if r + 1 < ROWS and cells[(r + 1) * COLS + c] == pawn:
            return True
        if r <= 4:
            if c > 0 and cells[r * COLS + c - 1] == pawn:
                return True
            if c + 1 < COLS and cells[r * COLS + c + 1] == pawn:
                return True
    else:
        if r - 1 >= 0 and cells[(r - 1) * COLS + c] == pawn:
            return True
        if r >= 5:
            if c > 0 and cells[r * COLS + c - 1] == pawn:
                return True
            if c + 1 < COLS and cells[r * COLS + c + 1] == pawn:
                return True

    # advisor
    if in_palace(by_color, r, c):
        for i in range(4):
            ar = r + DIAG_DR[i]
            ac = c + DIAG_DC[i]
            if in_palace(by_color, ar, ac) and cells[ar * COLS + ac] == advisor:
                return True

    # elephant
    for ei in range(4):
        if ei == 0: e_dr = -2; e_dc = -2; e_er = -1; e_ec = -1
        elif ei == 1: e_dr = -2; e_dc = 2; e_er = -1; e_ec = 1
        elif ei == 2: e_dr = 2; e_dc = -2; e_er = 1; e_ec = -1
        else: e_dr = 2; e_dc = 2; e_er = 1; e_ec = 1
        ar = r + e_dr
        ac = c + e_dc
        if not in_board_rc(ar, ac):
            continue
        if cells[ar * COLS + ac] != elephant:
            continue
        if (by_color == RED and ar < 5) or (by_color == BLACK and ar > 4):
            continue
        if cells[(r + e_er) * COLS + (c + e_ec)] == EMPTY:
            return True

    return False


cdef inline bint in_check(CBoard* b, int color) noexcept nogil:
    cdef int ks = b.kings[color]
    if ks < 0:
        return False
    return is_attacked(b, ks, 1 - color, True)


cdef void _sensitive(CBoard* b, int color, int ks, int* sens, int* sens_n) noexcept nogil:
    cdef int* cells = b.cells
    cdef int kr = ks // COLS
    cdef int kc = ks % COLS
    cdef int c, r, sq, p, i, hr, hc, leg, hdr0, hdc0
    cdef int horse = ((1 - color) << 3) | HORSE
    cdef int seen[90]
    for sq in range(SQUARES):
        seen[sq] = 0
    sens_n[0] = 0

    seen[ks] = 1
    sens[0] = ks
    sens_n[0] = 1

    for c in range(COLS):
        sq = kr * COLS + c
        p = cells[sq]
        if p and (p >> 3) == color and not seen[sq]:
            seen[sq] = 1
            sens[sens_n[0]] = sq
            sens_n[0] += 1
    for r in range(ROWS):
        sq = r * COLS + kc
        p = cells[sq]
        if p and (p >> 3) == color and not seen[sq]:
            seen[sq] = 1
            sens[sens_n[0]] = sq
            sens_n[0] += 1

    for i in range(8):
        if i == 0: hdr0 = -2; hdc0 = -1
        elif i == 1: hdr0 = -2; hdc0 = 1
        elif i == 2: hdr0 = 2; hdc0 = -1
        elif i == 3: hdr0 = 2; hdc0 = 1
        elif i == 4: hdr0 = -1; hdc0 = -2
        elif i == 5: hdr0 = 1; hdc0 = -2
        elif i == 6: hdr0 = -1; hdc0 = 2
        else: hdr0 = 1; hdc0 = 2
        hr = kr + hdr0
        hc = kc + hdc0
        if hr < 0 or hr >= ROWS or hc < 0 or hc >= COLS:
            continue
        if cells[hr * COLS + hc] != horse:
            continue
        if (hdr0 if hdr0 >= 0 else -hdr0) == 2:
            leg = (kr + hdr0 // 2) * COLS + (kc + hdc0)
        else:
            leg = (kr + hdr0) * COLS + (kc + hdc0 // 2)
        if 0 <= leg < SQUARES and not seen[leg]:
            seen[leg] = 1
            sens[sens_n[0]] = leg
            sens_n[0] += 1


cdef inline bint _in_sensitive(int* sens, int sens_n, int frm) noexcept nogil:
    cdef int i
    for i in range(sens_n):
        if sens[i] == frm:
            return True
    return False


cdef int generate_legal(CBoard* b, int color, bint checked_known, bint checked,
                        int* frms, int* tos) noexcept nogil:
    cdef int pf[MAX_MOVES], pt[MAX_MOVES]
    cdef int pn, i, frm, to, tr, tc, ksq, opp
    cdef int ks = b.kings[color]
    cdef int sens[90], sens_n
    cdef int kr, kc
    cdef int n = 0
    cdef bint use_sens

    pn = generate_pseudo(b, color, pf, pt)
    if ks < 0:
        for i in range(pn):
            frms[i] = pf[i]
            tos[i] = pt[i]
        return pn

    opp = 1 - color
    if not checked_known:
        checked = is_attacked(b, ks, opp, True)
    use_sens = not checked
    if use_sens:
        _sensitive(b, color, ks, sens, &sens_n)
    kr = ks // COLS
    kc = ks % COLS

    for i in range(pn):
        frm = pf[i]
        to = pt[i]
        if use_sens and not _in_sensitive(sens, sens_n, frm):
            tr = to // COLS
            tc = to % COLS
            if tr != kr and tc != kc:
                frms[n] = frm
                tos[n] = to
                n += 1
                continue
        board_apply(b, frm, to)
        ksq = to if frm == ks else ks
        if not is_attacked(b, ksq, opp, True):
            frms[n] = frm
            tos[n] = to
            n += 1
        board_undo(b)
    return n


cdef int generate_captures(CBoard* b, int color, bint checked_known, bint checked,
                           int* frms, int* tos) noexcept nogil:
    cdef int pf[MAX_MOVES], pt[MAX_MOVES]
    cdef int pn, i, frm, to, tr, tc, ksq, opp
    cdef int ks = b.kings[color]
    cdef int sens[90], sens_n
    cdef int kr, kc
    cdef int n = 0
    cdef bint use_sens

    pn = generate_pseudo(b, color, pf, pt)
    if ks < 0:
        for i in range(pn):
            if b.cells[pt[i]]:
                frms[n] = pf[i]
                tos[n] = pt[i]
                n += 1
        return n

    opp = 1 - color
    if not checked_known:
        checked = is_attacked(b, ks, opp, True)
    use_sens = not checked
    if use_sens:
        _sensitive(b, color, ks, sens, &sens_n)
    kr = ks // COLS
    kc = ks % COLS

    for i in range(pn):
        frm = pf[i]
        to = pt[i]
        if b.cells[to] == EMPTY:
            continue
        if use_sens and not _in_sensitive(sens, sens_n, frm):
            tr = to // COLS
            tc = to % COLS
            if tr != kr and tc != kc:
                frms[n] = frm
                tos[n] = to
                n += 1
                continue
        board_apply(b, frm, to)
        ksq = to if frm == ks else ks
        if not is_attacked(b, ksq, opp, True):
            frms[n] = frm
            tos[n] = to
            n += 1
        board_undo(b)
    return n


cdef long long perft_rec(CBoard* b, int depth) noexcept nogil:
    cdef int frms[MAX_MOVES], tos[MAX_MOVES]
    cdef int n, i
    cdef long long total
    if depth <= 0:
        return 1
    n = generate_legal(b, b.turn, False, False, frms, tos)
    if depth == 1:
        return n
    total = 0
    for i in range(n):
        board_apply(b, frms[i], tos[i])
        total += perft_rec(b, depth - 1)
        board_undo(b)
    return total
