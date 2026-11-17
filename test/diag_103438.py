"""诊断截图 Screenshot_20261117-103438 的识别问题。"""
import sys
sys.path.insert(0, r'g:\chessCn')
import cv2
from engine.recognize import analyze_image, detect_pieces, find_grid, classify
from engine.decide import decide_position

PATH = r'g:\chessCn\images\Screenshot_20261117-103438.png'

img = cv2.imread(PATH)
h, w = img.shape[:2]
y0, y1 = int(101.0/1280.0 * h), int(1264.0/1280.0 * h)
crop = img[y0:y1, 0:w]

pieces = detect_pieces(crop)
print(f"共检测到 {len(pieces)} 个棋子:")
for cx, cy, color, area in sorted(pieces, key=lambda p: (round(p[1]/35.5), p[0])):
    print(f"  cx={cx:6.1f} cy={cy:6.1f}  color={color}  area={area:.0f}")

row0, col0, cell, score, hit = find_grid(crop, pieces)
print(f"\n网格: row0={row0:.1f} col0={col0:.1f} cell={cell:.1f}  hit={hit}")

grid_raw, centers = classify(crop, row0, col0, cell)
print("\nclassify grid (原始 @/#):")
for r in range(10):
    print(f"row{r}: {''.join(grid_raw[r])}")

print("\n每个格点附近匹配到的棋子:")
for r in range(10):
    row_parts = []
    for c in range(9):
        cx, cy = centers[r*9 + c]
        found = None
        for px, py, color, area in pieces:
            dx, dy = px - cx, py - cy
            if (dx*dx + dy*dy) ** 0.5 < cell * 0.35:
                found = f"{color}({area:.0f})"
        row_parts.append((found or ".").ljust(12))
    print(f"row{r}: {''.join(row_parts)}")

# 现在用引擎算一下，走炮5进4后对方应着
print("\n====== 引擎分析 ======")
a = analyze_image(PATH)
print("\n识别推断后的网格 (红方在上):")
print(a.grid_str)
print("view_side (图下方)=", a.view_side)

d = decide_position(grid=a.grid_str, side="red", view_side=a.view_side, max_depth=8, time_limit=5)
print("\nBEST:", d['best_move']['chinese'], "score=", d['best_move']['score_text'])
print("REASON:", d['best_move']['reason'])
print("PV:", " -> ".join(d['pv']['chinese']))
print("PV scores:", d['pv']['scores'])

# 手动走炮五进四，看引擎怎么评估
print("\n====== 手动推演：炮五进四 ======")
from engine.board import Board
from engine.search import search as search_engine
from engine.constants import RED

board = Board.from_grid(a.grid_str, RED)
print("走前 FEN:", board.to_fen())

# 找到炮五进四对应的 move
from engine.movegen import generate_legal
moves = generate_legal(board, RED)
from engine.notation import chinese_notation
for mv in moves:
    cn = chinese_notation(board, mv)
    if "炮五进四" in cn or "炮5进4" in cn:
        print(f"找到 move: {mv} = {cn}")
        board.apply(mv)
        print("走后 FEN:", board.to_fen())
        print("轮到:", "黑方")
        outcome = search_engine(board, time_limit=4.0, max_depth=6)
        print("黑方最佳应着 PV:", outcome.pv)
        # 记录黑方会吃炮吗
        from engine.notation import sq_to_iccs, piece_name
        bm = outcome.best_move
        if bm:
            print(f"黑方最佳应着: {chinese_notation(board, bm)}  score={outcome.score}")
        break
