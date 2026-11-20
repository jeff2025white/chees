# Static evaluation (red perspective) + SEE helper.


cdef int evaluate_board(CBoard* b) noexcept nogil:
    cdef int* cells = b.cells
    cdef int score = b.base_score
    cdef bint opening = board_is_opening(b)
    cdef int red_a = 0, red_e = 0, black_a = 0, black_e = 0, devel = 0
    cdef int file_red_ch[9], file_red_ca[9], file_black_ch[9], file_black_ca[9]
    cdef int file_blockers[9]
    cdef int sq, p, ptype, r, c, open_bonus, center, i

    for c in range(COLS):
        file_red_ch[c] = 0
        file_red_ca[c] = 0
        file_black_ch[c] = 0
        file_black_ca[c] = 0
        file_blockers[c] = 0

    if opening:
        if cells[82] == HORSE:
            devel -= 16
        if cells[88] == HORSE:
            devel -= 16
        if cells[1] == ((BLACK << 3) | HORSE):
            devel += 16
        if cells[7] == ((BLACK << 3) | HORSE):
            devel += 16

    for sq in range(SQUARES):
        p = cells[sq]
        if p == EMPTY:
            continue
        ptype = p & 7
        r = sq // COLS
        c = sq % COLS

        if ptype == CHARIOT or ptype == CANNON:
            if p < 8:
                if ptype == CHARIOT:
                    file_red_ch[c] += 1
                else:
                    file_red_ca[c] += 1
            else:
                if ptype == CHARIOT:
                    file_black_ch[c] += 1
                else:
                    file_black_ca[c] += 1
        else:
            file_blockers[c] += 1

        if opening:
            score += DEEP_ARR[p * SQUARES + sq]

        if p < 8:
            if ptype == ADVISOR and r >= 7:
                red_a += 1
            elif ptype == ELEPHANT and r >= 5:
                red_e += 1
            if opening:
                if ptype == HORSE and sq != 82 and sq != 88:
                    devel += 22
                elif ptype == CANNON:
                    if r <= 3:
                        devel -= 80
                    elif c == 4:
                        devel += 26
                    elif c == 2 or c == 6:
                        devel += 10
                elif ptype == PAWN and r == 5 and (c == 0 or c == 2 or c == 4 or c == 6 or c == 8):
                    devel += 14
                elif ptype == CHARIOT:
                    if sq != 81 and sq != 89:
                        devel += 18
                        if r < 9:
                            devel += 12
                        if r <= 7:
                            devel += 6
        else:
            if ptype == ADVISOR and r <= 2:
                black_a += 1
            elif ptype == ELEPHANT and r <= 4:
                black_e += 1
            if opening:
                if ptype == HORSE and sq != 1 and sq != 7:
                    devel -= 22
                elif ptype == CANNON:
                    if r >= 6:
                        devel += 80
                    elif c == 4:
                        devel -= 26
                    elif c == 2 or c == 6:
                        devel -= 10
                elif ptype == PAWN and r == 4 and (c == 0 or c == 2 or c == 4 or c == 6 or c == 8):
                    devel -= 14
                elif ptype == CHARIOT:
                    if sq != 0 and sq != 8:
                        devel -= 18
                        if r > 0:
                            devel -= 12
                        if r >= 2:
                            devel -= 6

    score += devel
    score += red_a * 12 + red_e * 10
    score -= black_a * 12 + black_e * 10
    if red_a == 0:
        score -= 18
    if red_e == 0:
        score -= 14
    if black_a == 0:
        score += 18
    if black_e == 0:
        score += 14

    for c in range(COLS):
        open_bonus = 14 if file_blockers[c] == 0 else (8 if file_blockers[c] <= 2 else 0)
        center = 6 if c == 4 else (3 if c == 3 or c == 5 else 0)
        for i in range(file_red_ch[c]):
            score += open_bonus + center + 4
        for i in range(file_red_ca[c]):
            score += open_bonus + center + 2
        for i in range(file_black_ch[c]):
            score -= open_bonus + center + 4
        for i in range(file_black_ca[c]):
            score -= open_bonus + center + 2
    return score


cdef inline int see_value(CBoard* b, int frm, int to) noexcept nogil:
    cdef int victim = b.cells[to]
    cdef int attacker
    cdef int gain
    cdef bint defended
    if not victim:
        return 0
    attacker = b.cells[frm]
    gain = PIECE_VAL[victim & 7]
    board_apply(b, frm, to)
    # apply 后 turn 为对方；落点可被吃回则扣满攻击子
    defended = is_attacked(b, to, b.turn, False)
    board_undo(b)
    if defended:
        return gain - PIECE_VAL[attacker & 7]
    return gain
