# 着法计算逻辑文档

代码位置：`engine/`（board / movegen / evaluate / search / decide / explain / notation）

## 1. 整体架构

### 1.1 调用链

```
截图识别模块
  │
  ▼  grid_str (10×9 文本) + view_side + side
  │
  ▼  decide_position(grid=..., side=..., view_side=..., time_limit=4.0, max_depth=8, top_n=2)
  │
  ├─ mirror_grid          ← 黑方在下时把 grid 翻转
  ├─ Board.from_grid      ← 解析成 Board 对象
  ├─ game_status          ← 开局先判断是否已死
  │
  ▼  search(board, time_limit, max_depth)
  │
  ├─ Searcher 初始化       ← 置换表 / 杀手着法 / 历史启发
  ├─ 迭代加深 depth=1..max_depth
  │   └─ root_search(d)   ← 根节点全着法扫描
  │       └─ negamax      ← Alpha-Beta 递归
  │           ├─ 置换表查/存
  │           ├─ is_repetition → 和棋(0)
  │           ├─ depth<=0 → quiesce
  │           └─ generate_legal → _order_moves → apply/undo
  ├─ 备选着法浅搜索复核
  └─ extract_pv            ← 从 TT 串里提取主变
  │
  ▼  decide_position 包装返回
  │
  ├─ _move_payload         ← best_move + alternatives
  ├─ _pv_payload           ← 主变序列每步坐标 + 中文记谱 + 静态评分
  └─ 返回 dict             ← 含 coordinate_contract、search 统计、message
```

### 1.2 入口函数签名

```python
decide_position(
    fen: str = None,          # FEN（与 grid 二选一）
    grid: str = None,         # 10×9 文本行（界面方向：row0=图最上）
    side=None,                # 轮到谁走："red"/"black"，缺省读 FEN
    view_side=None,           # 图下方是谁：缺省=side
    time_limit: float = 4.0,  # 总时间预算（秒）
    max_depth: int = 8,       # 最大搜索深度
    top_n: int = 2,           # 备选着法数量
) -> dict
```

### 1.3 输出结构概览

```json
{
  "ok": true,
  "input": { "fen", "grid", "side_to_move", "in_check", "legal_move_count" },
  "coordinate_contract": { "cols": 9, "rows": 10, ... },
  "status": "playing" | "check" | "checkmate" | "stalemate",
  "best_move": {
    "from": {"col": 0, "row": 0}, "to": {"col": 0, "row": 0},
    "iccs": "h3e3", "ucci": "h2e2", "chinese": "炮二平五",
    "piece": "炮", "color": "red",
    "captured": null, "is_capture": false,
    "gives_check": false, "is_mate": false,
    "mate_in": null, "score_cp": 35, "score_text": "+0.35",
    "reason": "出动炮，抢占要道、增强进攻子力。；占据中路要点。",
    "tags": ["development"]
  },
  "alternatives": [ /* 同 best_move 结构 */ ],
  "pv": {
    "iccs": ["h3e3", "h8e8"],
    "chinese": ["炮二平五", "炮8平5"],
    "sides": ["red", "black"],
    "scores": [25, -15],          // 走完该步后的我方视角静态评分
    "steps": [
      {"from": {"row": 2, "col": 7}, "to": {"row": 2, "col": 4}},
      {"from": {"row": 7, "col": 7}, "to": {"row": 7, "col": 4}}
    ]
  },
  "search": {
    "depth": 6, "nodes": 123456, "time_ms": 3820,
    "score_cp": 35, "mate_in": null
  },
  "message": "出动炮，抢占要道、增强进攻子力。；占据中路要点。"
}
```

---

## 2. 棋盘状态 Board

代码位置：`engine/board.py` + `engine/constants.py`

### 2.1 坐标与编码

**内部坐标（全引擎统一）**

- 一维下标 `sq = row × 9 + col`
- `col: 0..8` 从左到右（红方视角）
- `row: 0..9` 从上到下，**row 0 = 黑方底线，row 9 = 红方底线**

**棋子编码**

```python
piece = (color << 3) | type
color: RED=0, BLACK=1
type : KING=1, ADVISOR=2, ELEPHANT=3, HORSE=4, CHARIOT=5, CANNON=6, PAWN=7
```

因此红方棋子取值 `1..7`，黑方棋子取值 `9..15`，`0` 表示空。

**子力价值（单位：厘兵 centipawn，兵 = 100）**

| 兵种 | value | 说明 |
|------|------:|------|
| KING 帅/将 | 60000 | 杀棋阈值 MATE=100000，远大于任何价值 |
| CHARIOT 车 | 900 | 最强进攻子 |
| CANNON 炮 | 450 | |
| HORSE 马 | 400 | |
| ADVISOR 仕/士 | 200 | |
| ELEPHANT 相/象 | 200 | |
| PAWN 兵/卒 | 100 | |

### 2.2 Board 核心数据结构

```python
class Board:
    __slots__ = ("cells", "turn", "undo_stack", "hash_stack", "kings")
    cells: List[int]       # 长度 90 的一维数组
    turn: int              # RED=0 或 BLACK=1
    undo_stack: List       # [(move, captured, hash_after), ...]
    hash_stack: List       # Zobrist 哈希历史，用于重复局面检测
    kings: List[int]       # [红帅sq, 黑将sq]，-1 表示不在棋盘上
```

### 2.3 Zobrist 哈希

固定种子 `random.Random(20261117)`，保证可复现：

```python
ZOBRIST = [[randbits(64) for _ in range(90)] for _ in range(16)]  # 16 种棋子 × 90 格
_ZOBRIST_SIDE = randbits(64)                                      # 谁走
```

**增量更新**（apply 走子时）：

```python
h = hash_stack[-1]                              # 上一步哈希
h ^= ZOBRIST[piece][frm]                        # 移除原位置
h ^= ZOBRIST[piece][to]                         # 添加新位置
if captured:
    h ^= ZOBRIST[captured][to]                  # 移除被吃子
h ^= _ZOBRIST_SIDE                              # 切换行棋方
```

undo 不需要重新算哈希——直接从 `hash_stack.pop()` 恢复。

### 2.4 apply / undo

```python
apply(move) → captured_piece
  # 1. cells[to] 吃走 captured
  # 2. cells[frm] → EMPTY, cells[to] → piece
  # 3. 若 piece 是将/帅 → 更新 kings 缓存
  # 4. turn 翻转
  # 5. push undo_stack + hash_stack

undo()
  # 1. cells[to] → cells[frm]（棋子退回）
  # 2. cells[frm] → captured（被吃子复活）
  # 3. 若退回的是将/帅 → 更新 kings
  # 4. turn 翻转
  # 5. pop undo_stack + hash_stack
```

### 2.5 FEN / Grid 解析

**FEN 行序**：row 0（黑方底线）→ row 9（红方底线），每行 9 格。

```python
Board.from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1")
```

**Grid 行序**：自定义文本，大写红、小写黑，`.` 或空格为空。`from_grid(text, turn)` 同样是 row 0 在 top。

### 2.6 重复局面检测

```python
is_repetition() → bool
  # 线性扫描 hash_stack[:-1]，看当前哈希是否出现过
  # 搜索中若 ply>0 且重复 → 直接返回 0（和棋）
```

**局限**：不做长将/长捉判负，全部按和棋处理。

---

## 3. 着法生成 movegen

代码位置：`engine/movegen.py`

### 3.1 总体分层

```
generate_pseudo_moves      # 伪合法：不检查"走后己方是否被将军"
  │
  ▼  apply(move) → 检查 ksq 是否被攻击 → undo()
  │
generate_legal_moves       # 合法着法：过滤掉送将 / 飞将
generate_captures          # 吃子着法子集（静态搜索用）
```

### 3.2 七种兵种的走法规则

**帅/将 KING**

- 九宫内直走一步（row 7-9 / col 3-5 for RED；row 0-2 / col 3-5 for BLACK）
- ORTHO 四方向：上下左右
- 飞将规则**不在生成时检查**，留到 `in_check` 的 `king_face=True`

**仕/士 ADVISOR**

- 九宫内斜走一步（DIAG 四方向）
- 同 KING，九宫外不可去

**相/象 ELEPHANT**

- 田字走法（ELEPHANT_MOVES 4 个方向，(dr, dc, er, ec)）
- **塞象眼**：`cells[(r+er)*9 + (c+ec)] == EMPTY` 才能走
- **不过河**：红象不能去 row<5，黑象不能去 row>4

**马 HORSE**

- 8 个方向（HORSE_MOVES），每个含 (dr, dc, lr, lc)
- **蹩马腿**：腿在长边方向的相邻格
  - 若 `|dr|==2` → 腿在 `(r+lr, c)`
  - 若 `|dc|==2` → 腿在 `(r, c+lc)`
- 棋盘内 + 腿空 + 目标为己方可去

**车 CHARIOT**

- 四方向无限直走
- 遇敌方子可吃后停止，遇己方子停止

**炮 CANNON**

- 未过炮架前：跟车一样直走空位
- 过炮架（遇到第一个棋子）后：继续走，遇到**第一个敌子**可吃
- 即吃子规则 = 跳过恰好一个棋子后吃

**兵/卒 PAWN**

- 红方往上（row-1），黑方往下（row+1）：走一步
- **过河后**（红 row≤4，黑 row≥5）：额外可以横走 ±1

### 3.3 合法着法过滤

伪合法着法生成后，需要过滤掉"走后己方将/帅被攻击"的着法。直接方法是 `apply → in_check → undo`，但有**敏感格优化**：

```python
_sensitive_squares(board, color, ks) → set
  # 只需验证三类起点：
  # 1) 将/帅自身所在格
  # 2) 将/帅同一行或同一列的己方子（挡住车/炮/飞将）
  # 3) 位于敌方马腿上的己方子（移开后被马将军）
```

**非敏感格着法**可直接接受，无需完整验证，只有当**落点恰好也在将/帅行列上**时才补做完整验证（因为可能成为敌方炮的炮架）。

### 3.4 攻击判定 is_attacked

判断某格是否被指定方攻击，是搜索中最热的函数，采用反向扫描：

```
is_attacked(sq, by_color, king_face=False)
  1) 竖直方向：
     - 找到 sq 上方/下方第一个棋子 → 若是 CHARIOT 则被攻击
     - 若 king_face=True 且第一个是对方 KING → 飞将攻击
     - 跳过炮架 → 再找一个棋子 → 若是 CANNON 则被攻击
  2) 水平方向：同上
  3) 马：从目标格反向找 8 个马位，每个检查马腿
  4) 兵/卒：根据方向判断（红兵从下方吃，黑兵从上方吃）
  5) 士/象：在九宫内/田字+不过河+象眼空
```

`king_face=True` 只在判断**将/帅本身**是否被攻击时打开（飞将规则），其他时候保持 False。

### 3.5 局面状态 game_status

```python
game_status(board, color) → (status, legal_moves)
  status:
    "playing"   正常
    "check"     被将军但有应着
    "checkmate" 被将死（无合法着法 + in_check）
    "stalemate" 困毙（无合法着法 + 未将军）
```

---

## 4. 局面评估 evaluate

代码位置：`engine/evaluate.py`

### 4.1 评估框架

- **红方视角**：`evaluate(board) → int`，正数 = 红方占优，单位厘兵
- **指定方视角**：`evaluate_for(board, color)`，非红方时取负
- 搜索中 `turn` 是 BLACK 时把 `stand = -evaluate(board)`

**评估公式**：

```python
红方棋子 sq 处：+ PIECE_VALUE[ptype] + PST_FLAT[ptype][sq]
黑方棋子 sq 处：- PIECE_VALUE[ptype] - PST_FLAT[ptype][mirror[sq]]
```

`mirror[sq]` = `(9 - r) * 9 + c`，即把黑方上下翻转后复用红方位置表。

### 4.2 位置表设计原则

- 7 张 10×9 表，数值量级 **±60 以内**（保证子力价值主导，位置只是微调）
- 表行序同内部坐标：row 0 = 黑方底线，row 9 = 红方底线

### 4.3 各兵种 PST 要点

**PAWN_PST 兵/卒**

- row 6-9 全 0：未过河不加分，鼓励过河
- row 5（河界己方侧）：中间列（col 3-5）加分最多 25
- row 0-4（过河后）：从 60 递减，中间列最高，鼓励兵深入 + 占中路

**HORSE_PST 马**

- 中心列（col 3-5）加分高于边线
- row 3-4 加分最高（活跃区域），row 0/9 最低（底线）
- 对称表，中间 30、边线 4

**CANNON_PST 炮**

- 结构同马但整体偏低（最大 18 vs 30）
- 也是鼓励居中 + 活跃区域

**CHARIOT_PST 车**

- row 0（黑方底线）中间列加 22，row 9 底线只加 6
- 鼓励车过河，压制对方底线

**ADVISOR_PST 仕/士**

- row 8 col 3/5 扣 6（离开原位不好）
- row 9 col 4 加 6（原位）
- 其余全 0，**极小的微调表**——士主要是防守，位置影响不大

**ELEPHANT_PST 相/象**

- row 5 col 2/6 加 4（田字点，中路附近）
- row 9 col 2/6 加 6（原位田字点）
- 其余 0，同样微调表

**KING_PST 帅/将**

- row 7 col 3/4/5 扣 8（在底线和河界之间的过渡行不好）
- row 8 col 3/4/5 扣 4
- row 9 col 4 加 10（居中在原位最好）
- 鼓励帅守在原位，不要瞎跑

### 4.4 预展开优化

```python
PST_FLAT = { ptype: [table[r][c] for r in range(10) for c in range(9)] }
MIRROR   = [(9 - r) * 9 + c for r in range(10) for c in range(9)]
```

评估时用一维表查，避免双重下标开销。

---

## 5. 搜索算法 search

代码位置：`engine/search.py`

### 5.1 总体框架

```
search(board, time_limit=4.0, max_depth=8)
  │
  ├─ for depth in range(1, max_depth+1):   ← 迭代加深
  │   ├─ 预估耗时（上一层 × 3），超出预算就停
  │   ├─ root_search(d)                    ← 根节点全着法搜索
  │   ├─ 杀棋则提前 break
  │   └─ TimeUp 异常捕获 → break
  │
  ├─ 备选着法浅搜索（完整窗口复核前 4 个）
  ├─ extract_pv(best_move)                ← 从 TT 串里提取主变
  └─ SearchOutcome(best_move, score, depth, nodes, time_ms, ranked, pv)
```

### 5.2 Negamax 主搜索

```python
negamax(depth, alpha, beta, ply):
  1) 重复局面 → return 0（简化为和棋）
  2) depth <= 0 → quiesce(alpha, beta, ply, 0)
  3) 查置换表：
     - 若 TT_EXACT 且 e_depth >= depth → return e_score
     - 若 TT_LOWER 且 e_score > alpha → alpha = e_score
     - 若 TT_UPPER 且 e_score < beta  → beta  = e_score
     - alpha >= beta → return e_score
  4) generate_legal → _order_moves(moves, ply, tt_move)
  5) 无合法着法 → return -MATE + ply（被将死/困毙均判负）
  6) for mv in moves:
       apply(mv)
       score = -negamax(depth-1, -beta, -alpha, ply+1)
       undo()
       更新 best / alpha / beta
       若 alpha >= beta：更新杀手着法 + 历史启发，break
  7) 按 alpha/beta 边界定 TT flag：
     best <= alpha_orig → TT_UPPER（上界）
     best >= beta       → TT_LOWER（下界）
     其余               → TT_EXACT
  8) store TT[key] = (depth, best, flag, best_move)
  return best
```

**Negamax 交替**：`score = -negamax(depth-1, -beta, -alpha)`，双方都最大化自己的分数，天然对称。

### 5.3 静态搜索 quiesce

消除**水平线效应**（搜索边界上的吃子导致评估失真）：

```python
quiesce(alpha, beta, ply, qdepth):
  stand = evaluate(board)      # 红方视角
  if turn == BLACK: stand = -stand

  checked = in_check(board, turn)

  if not checked:
    if qdepth >= 6: return stand          # 未被将军，最多搜 6 层静态
    if stand >= beta: return stand        # 截断
    if stand > alpha: alpha = stand
    moves = generate_captures(...)        # 只搜吃子
  else:
    if qdepth >= 12: return stand         # 被将军，最多搜 12 层
    moves = generate_legal(...)           # 搜全部应着
    if not moves: return -MATE + ply      # 将死

  moves = _order_moves(moves, ply)
  for mv in moves:
    if not checked:
      # 差值剪枝：白吃价值也不足以提高 alpha
      victim = board.cells[mv[1]]
      if victim and stand + PIECE_VALUE[victim] + 200 < alpha:
        continue
    apply(mv)
    score = -quiesce(-beta, -alpha, ply+1, qdepth+1)
    undo()
    if score >= beta: return score
    if score > alpha: alpha = score
  return alpha
```

**关键剪枝**：被将军时不能只搜吃子（可能存在不吃子的解将着法），必须搜全部合法着法。

### 5.4 置换表

```python
TT[key] = (e_depth, e_score, e_flag, e_move)
e_flag ∈ {TT_EXACT=0, TT_LOWER=1, TT_UPPER=2}
```

**命中条件**：`e_depth >= depth` 且边界区间相交。

**三种 flag 含义**：

| flag | 条件 | 解释 |
|------|------|------|
| TT_EXACT | `best` 在 `(alpha_orig, beta)` 之间 | 精确值 |
| TT_LOWER | `best >= beta`（抬高了 alpha） | 真实值至少是 best |
| TT_UPPER | `best <= alpha_orig`（没抬高 alpha） | 真实值至多是 best |

TT 着法在 `_order_moves` 里排第一，给 `10^7` 的排序分。

### 5.5 着法排序

```python
_order_moves(moves, ply, tt_move):
  for mv in moves:
    victim = cells[mv[1]]
    if mv == tt_move:           score = 10_000_000
    elif victim:                score = 100_000 + PIECE_VALUE[victim]*10 - PIECE_VALUE[attacker]
    elif mv == killers[0]:      score = 90_000
    elif mv == killers[1]:      score = 89_000
    else:                       score = history.get(mv, 0)   # depth² 累加
  sorted → 降序
```

**排序优先级**：TT 着法 > MVV-LVA 吃子 > 杀手着法 0 > 杀手着法 1 > 历史启发

**杀手着法**（Killer Heuristic）：每层记录**两个**引起 beta 截断的着法（只记非吃子着法），新入的挤掉旧的。

**历史启发**（History Heuristic）：对弈分支节点的着法，`history[mv] += depth²`。

### 5.6 迭代加深与时间控制

```python
for depth in range(1, max_depth + 1):
    # 预估下一层耗时（约 3 倍上一层）
    if last_duration and elapsed + last_duration * 3 > time_limit:
        break
    root_search(d)
    last_duration = actual_duration
    if abs(score) > MATE - 1000: break   # 杀棋提前终止
```

**超时机制**：

- 每 512 节点检查一次 `time.perf_counter() > deadline`，抛出 `TimeUp` 异常
- 主循环捕获后立即返回上一次完整的搜索结果
- 极端情况（连 depth=1 都没完成）：退化为第一个合法着法

### 5.7 根节点处理

根节点 `root_search(d)` 不做 alpha-beta 截断，而是**完整扫描所有着法**：

```python
root_search(d):
  for mv in moves:
    score = -negamax(d-1, -INF, -alpha, 1)
    # 只有真正抬高了 alpha 的着法，其评分才是精确的
    # 其余着法的评分是"对手回应的上界"
    results.append((mv, score, score > prev_alpha))
```

**备选着法**：根节点的非最佳着法只有评分上界，难以排序。所以搜索完后用**浅一层、完整窗口 `(-INF, INF)`** 复核前 4 个候选，得到可比的真实评分。

### 5.8 主变提取 extract_pv

从 TT 里走一条链：

```python
extract_pv(first_move, max_len=6):
  board = clone()
  move = first_move
  for _ in range(max_len):
    pv.append(move)
    board.apply(move)
    entry = TT[board.hash]          # 取 TT 中记录的最佳着法
    move = entry[3] if entry else None
  return pv
```

### 5.9 评分约定

```python
MATE = 100000
INF  = 10_000_000
```

**杀棋换算**：

- 被将死 / 困毙 → 返回 `-MATE + ply`（越到深层死，分数越接近 0）
- 胜方杀棋 → `format_score` 把 `MATE - |score| + 1 // 2` 换算成步数

```python
format_score(+99995) → "绝杀(3步)"     # 99995 = 100000 - 3*2 + 1
format_score(-99989) → "被绝杀(6步)"
format_score(35)     → "+0.35"         # 普通局面厘兵
```

---

## 6. 着法解释 explain

代码位置：`engine/explain.py`

### 6.1 战术特征分析 analyze_move

在走子前的局面 board 上，对 move 做完整推演：

```python
analyze_move(board, move) → dict:
  was_check          # 走前是否已经被将军
  captured           # 被吃子编码（0 表示没吃）
  captured_defended  # 被吃子是否有防守（吃子是否兑子）
  gives_check        # 走后是否将军对方
  is_mate            # 走后对方无合法着法
  attacked_after     # 我方着法落点是否被敌方攻击
  defended_after     # 我方着法落点是否有己方子保护
  threat             # 走后下一步我方可白吃的最大敌方子
  threat_defended    # threat 是否有防守（无根才真的威胁）
  crossed_river      # 兵/卒是否过河
  to_col             # 落点列号
  forward_gain       # 前进格数
```

**最大威胁搜索**：遍历敌方所有棋子（不含帅/将），对每个无子防守的敌方棋子计算我方是否能攻击到，取子力价值最大的那个。

### 6.2 中文理由生成 build_reason

按优先级拼接 clauses，用 `；` 分隔：

| 优先级 | 条件 | 输出示例 |
|--------|------|---------|
| 1 | `is_mate` | "直接绝杀，对手已无合法应着" |
| 2 | `was_check` | "先解掉对方的将军" |
| 3 | `captured` + 有防守 | "吃掉对方一马" |
| 3 | `captured` + 无防守 | "白吃对方一炮" |
| 4 | `gives_check` | "形成将军，逼迫对方应付" |
| 5 | 以上都不满足（走闲） | 按兵种分类：兵过河/推进 / 车马炮出动 / 士象补防守 / 帅调整 |
| 6 | 落点列在 (3,4,5) | "占据中路要点" |
| 7 | 落点被攻击 | "落点有己方子保护" 或 "注意落点暂无保护" |
| 8 | 有无根 threat | "下一步可白吃对方马" |

返回 `(text, tags)`，tags 包含 `mate / escape_check / capture / free_capture / check / development / defense / supported / risky / threat` 等。

---

## 7. 着法记谱 notation

代码位置：`engine/notation.py`

### 7.1 三种记谱格式

| 格式 | 示例 | 规则 |
|------|------|------|
| **ICCS** | `h3e3` | 列 a-i（红方视角），行 1-10 从红方底线往上数 |
| **UCCI** | `h2e2` | 列同上，行 0-9 从红方底线往上数（引擎协议） |
| **中文** | `炮二平五` / `马8进7` | 红方用中文数字（一二三四五六七八九），黑方用阿拉伯数字（0-9） |

### 7.2 中文记谱规则

```python
file_number(col, color) → 红方 9-col+1，黑方 col+1  # 每方从自己的右往左数
```

**字串结构**：

```
[前缀+兵种][路数][动词][目标路数 或 格数]

动词：
  "平" → 同行（tr == fr），后接目标路数
  "进"/"退" → 不同行
    斜行子（马/士/象）：后接目标路数
    直行子（车/炮/兵）：后接移动格数 abs(tr - fr)

前缀（消歧）：
  同一列同类子 ≥2 个时，用"前/中/后/后/前中后/一二三四五"消歧
  按"前进方向越靠前的越前"排序
```

### 7.3 坐标换算

ICCS/UCCI 始终以**红方为基准**，不随 view_side 改变。界面镜像变换由 `decide._view_coords` 负责。

---

## 8. 坐标契约（不能错）

代码位置：`engine/decide.py`

### 8.1 三套坐标系

```
内部 sq          → 一维 row*9+col，row 0=黑底线，row 9=红底线
图像 view_side   → row 0=图最上，row 9=图最下；view_side='black' 时镜像
ICCS/UCCI        → 始终以红方为基准，列 a-i 红方视角左→右
```

### 8.2 view_coords 映射

```python
_view_coords(sq, view_side):
  r, c = divmod(sq, 9)
  if view_side == RED:   return (c, r)        # 红方在下：直接用
  if view_side == BLACK: return (8-c, 9-r)    # 黑方在下：上下左右同时镜像
```

### 8.3 mirror_grid

当 `view_side == BLACK`（图下方是黑方），识别模块输出的 grid 行序与内部方向**上下颠倒 + 列也反向**，需要先镜像：

```python
mirror_grid(grid_text):
  reversed(lines)       # 行反转
  reversed(chars) per line  # 列反转
```

等价于把图像棋盘旋转 180° 后再解析。

### 8.4 输出坐标到像素

```python
pixel_x = margin_x + col * cell_w     # col 来自 _view_coords 的返回
pixel_y = margin_y + row * cell_h
```

---

## 9. 已知局限

| # | 局限 | 影响 | 可接受性 |
|---|------|------|---------|
| 1 | 重复局面一律判和（返回 0） | 长将/长捉应判负但现在按和棋 | 中残局长将时评分偏乐观，但不会严重误导决策 |
| 2 | 搜索深度无 futility pruning | 残局浅搜时仍会走一些明显没用的着法 | 时间限制 4 秒 + 迭代加深 6-8 层基本够用 |
| 3 | PST 表单一，无开局/中局/残局切换 | 开局时马在底线和活跃区只有 ±26 差异 | 开局主要靠子力价值主导，PST 只是微调 |
| 4 | 静态搜索差值剪枝固定 +200 | 子力差接近边界时可能剪掉重要着法 | 实际测试中很少触发 |
| 5 | 主变提取依赖 TT 链 | 若某节点的 TT 条目缺失 → PV 断链 | `extract_pv` 有 max_len 兜底，断链就停 |
| 6 | 着法排序无 SEE（静态交换评估） | 复杂吃子序列的排序可能不准 | MVV-LVA + 历史启发在大多数场景足够 |

---

## 10. 关键常数速查

```python
# constants.py
COLS = 9; ROWS = 10; SQUARES = 90
RED = 0; BLACK = 1
EMPTY = 0
KING=1, ADVISOR=2, ELEPHANT=3, HORSE=4, CHARIOT=5, CANNON=6, PAWN=7
PIECE_VALUE = {KING:60000, CHARIOT:900, CANNON:450, HORSE:400, ADVISOR:200, ELEPHANT:200, PAWN:100}

# search.py
MATE = 100000
INF  = 10_000_000
MAX_PLY = 64
```

---

## 11. 模块依赖图

```
constants.py        ← 所有模块依赖
    ↑
board.py            ← movegen / evaluate / decide 依赖
    ↑
movegen.py           ← search / decide 依赖（generate_legal, in_check, game_status）
evaluate.py          ← search 依赖（evaluate）
notation.py          ← decide 依赖（chinese_notation, move_to_iccs, move_to_ucci）
explain.py           ← decide 依赖（analyze_move, build_reason）
    ↑
search.py            ← decide 依赖（search 函数, MATE 常量）
    ↑
decide.py            ← 顶层入口，对外暴露 decide_position
```
