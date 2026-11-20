# Board state machine: apply / undo / null / repetition / hash.

cdef enum:
    COLS = 9
    ROWS = 10
    SQUARES = 90
    RED = 0
    BLACK = 1
    EMPTY = 0
    KING = 1
    ADVISOR = 2
    ELEPHANT = 3
    HORSE = 4
    CHARIOT = 5
    CANNON = 6
    PAWN = 7
    MAX_UNDO = 256
    MAX_MOVES = 256


cdef struct UndoEntry:
    int frm
    int to
    int captured
    int is_null
    unsigned long long hash_after


cdef struct CBoard:
    int cells[90]
    int turn
    int kings[2]
    int piece_count
    int base_score
    int undo_len
    int hash_len
    UndoEntry undo_stack[MAX_UNDO]
    unsigned long long hash_stack[MAX_UNDO + 1]


cdef inline int piece_base_term(int piece, int sq) noexcept nogil:
    cdef int ptype
    if piece == 0:
        return 0
    ptype = piece & 7
    # 子力+PST；深入惩罚仅在 evaluate_board 开局时叠加
    if piece < 8:
        return PIECE_VAL[ptype] + PST_FLAT_ARR[ptype][sq]
    return -(PIECE_VAL[ptype] + PST_FLAT_ARR[ptype][MIRROR_ARR[sq]])


cdef inline void board_init_from_cells(CBoard* b, int* cells, int turn) noexcept nogil:
    cdef int sq, piece
    cdef unsigned long long h
    for sq in range(SQUARES):
        b.cells[sq] = cells[sq]
    b.turn = turn
    b.undo_len = 0
    b.kings[0] = -1
    b.kings[1] = -1
    b.piece_count = 0
    b.base_score = 0
    h = 0
    for sq in range(SQUARES):
        piece = b.cells[sq]
        if piece:
            b.piece_count += 1
            b.base_score += piece_base_term(piece, sq)
            if (piece & 7) == KING:
                b.kings[piece >> 3] = sq
            h ^= ZOBRIST[piece][sq]
    if turn == BLACK:
        h ^= ZOBRIST_SIDE
    b.hash_stack[0] = h
    b.hash_len = 1


cdef inline unsigned long long board_hash(CBoard* b) noexcept nogil:
    return b.hash_stack[b.hash_len - 1]


cdef inline bint board_is_opening(CBoard* b) noexcept nogil:
    return b.piece_count > 26 and b.cells[85] == 1 and b.cells[4] == 9


cdef inline int board_apply(CBoard* b, int frm, int to) noexcept nogil:
    cdef int piece = b.cells[frm]
    cdef int captured = b.cells[to]
    cdef unsigned long long h = board_hash(b) ^ ZOBRIST[piece][frm] ^ ZOBRIST[piece][to]
    if captured:
        h ^= ZOBRIST[captured][to]
        b.base_score -= piece_base_term(captured, to)
        b.piece_count -= 1
    h ^= ZOBRIST_SIDE
    b.base_score -= piece_base_term(piece, frm)
    b.base_score += piece_base_term(piece, to)
    b.cells[frm] = EMPTY
    b.cells[to] = piece
    if (piece & 7) == KING:
        b.kings[piece >> 3] = to
    b.turn = 1 - b.turn
    b.undo_stack[b.undo_len].frm = frm
    b.undo_stack[b.undo_len].to = to
    b.undo_stack[b.undo_len].captured = captured
    b.undo_stack[b.undo_len].is_null = 0
    b.undo_stack[b.undo_len].hash_after = h
    b.undo_len += 1
    b.hash_stack[b.hash_len] = h
    b.hash_len += 1
    return captured


cdef inline void board_apply_null(CBoard* b) noexcept nogil:
    cdef unsigned long long h = board_hash(b) ^ ZOBRIST_SIDE
    b.turn = 1 - b.turn
    b.undo_stack[b.undo_len].frm = -1
    b.undo_stack[b.undo_len].to = -1
    b.undo_stack[b.undo_len].captured = 0
    b.undo_stack[b.undo_len].is_null = 1
    b.undo_stack[b.undo_len].hash_after = h
    b.undo_len += 1
    b.hash_stack[b.hash_len] = h
    b.hash_len += 1


cdef inline void board_undo(CBoard* b) noexcept nogil:
    cdef UndoEntry* u
    cdef int frm, to, captured, piece
    b.undo_len -= 1
    u = &b.undo_stack[b.undo_len]
    b.hash_len -= 1
    if u.is_null:
        b.turn = 1 - b.turn
        return
    frm = u.frm
    to = u.to
    captured = u.captured
    piece = b.cells[to]
    b.base_score -= piece_base_term(piece, to)
    b.base_score += piece_base_term(piece, frm)
    if captured:
        b.base_score += piece_base_term(captured, to)
        b.piece_count += 1
    b.cells[frm] = piece
    b.cells[to] = captured
    if (piece & 7) == KING:
        b.kings[piece >> 3] = frm
    b.turn = 1 - b.turn


cdef inline bint board_is_repetition(CBoard* b) noexcept nogil:
    cdef unsigned long long h = board_hash(b)
    cdef int i
    cdef int n = b.hash_len - 1
    for i in range(n):
        if b.hash_stack[i] == h:
            return True
    return False


cdef inline void board_clone(CBoard* dst, CBoard* src) noexcept nogil:
    cdef int sq
    for sq in range(SQUARES):
        dst.cells[sq] = src.cells[sq]
    dst.turn = src.turn
    dst.kings[0] = src.kings[0]
    dst.kings[1] = src.kings[1]
    dst.piece_count = src.piece_count
    dst.base_score = src.base_score
    dst.undo_len = 0
    dst.hash_stack[0] = board_hash(src)
    dst.hash_len = 1
