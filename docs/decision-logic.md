# 着法计算逻辑文档

代码位置：`engine/`（board / movegen / evaluate / search / book / decide / explain / notation）

人机对弈界面 `ui/play.py` 也走同一 `decide_position`（`top_n=0, fill_pv=False`）。

## 1. 整体架构

### 1.1 调用链

```
截图识别 / 对弈 FEN
  │
  ▼  decide_position(..., time_limit, max_depth, top_n, fill_pv)
  │
  ├─ mirror_grid / Board.from_fen|from_grid
  ├─ game_status
  ├─ lookup_book              ← 开局库命中则第 1 手用库
  │     └─ fill_pv 时对后续局面 search 续满主变
  │
  ▼  search(board, time_limit, max_depth, top_n, fill_pv)   # 未命中库
  │
  ├─ 定长 TT（2^20）/ 杀手 / 历史启发
  ├─ 迭代加深 + aspiration；negamax 含空着剪枝、LMR
  ├─ top_n>0 时备选着法浅搜复核（对弈 top_n=0 跳过）
  └─ extract_pv(fill=fill_pv)  ← fill=False 只跟 TT（对弈）
  │
  ▼  _move_payload / _pv_payload → 返回 dict（search.book 标记是否开局库）
```

### 1.2 入口函数签名

```python
decide_position(
    fen: str = None,          # FEN（与 grid 二选一）
    grid: str = None,         # 10×9 文本行（界面方向：row0=图最上）
    side=None,                # 轮到谁走："red"/"black"，缺省读 FEN
    view_side=None,           # 图下方是谁：缺省=side；等于「我方」
    time_limit: float = 4.0,  # 总时间预算（秒）
    max_depth: int = 8,       # 最大搜索深度
    top_n: int = 2,           # 备选数量；<=0 跳过备选复核
    fill_pv: bool = True,     # True=TT 断了浅层续搜（分析）；False=对弈
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
    "score_cp": 35, "mate_in": null, "book": false
  },
  "message": "出动炮，抢占要道、增强进攻子力。；占据中路要点。"
}
```

`input` 另含 `our_side` / `view_side`（均为图下方「我方」）与 `side_to_move`。

---

## 2. 棋盘状态 Board

代码位置：`engine/board.py` + `engine/constants.py`

`Board` 增量维护 `piece_count` 与 `base_score`（子力+PST+深入惩罚），`apply`/`undo` 同步更新，供 `evaluate()` 使用。`is_opening()` 为 O(1)。

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
    __slots__ = ("cells", "turn", "undo_stack", "hash_stack", "kings",
                 "piece_count", "base_score")
    cells: List[int]       # 长度 90 的一维数组
    turn: int              # RED=0 或 BLACK=1
    undo_stack: List       # [(move, captured, hash_after), ...]；空着 move=None
    hash_stack: List       # Zobrist 哈希历史，用于重复局面检测
    kings: List[int]       # [红帅sq, 黑将sq]，-1 表示不在棋盘上
    piece_count: int       # 当前棋子数，走子/撤销时增量维护
    base_score: int        # 子力+PST+深入惩罚，走子/撤销时增量维护
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
  # 1. 增量更新 Zobrist、base_score；有吃子时更新 piece_count
  # 2. cells[frm] → EMPTY, cells[to] → piece
  # 3. 若 piece 是将/帅 → 更新 kings 缓存
  # 4. turn 翻转
  # 5. push undo_stack + hash_stack

apply_null()
  # 空着剪枝专用：棋子不动，只翻转 turn、更新哈希并保存撤销信息

undo()
  # 1. cells[frm] → piece（棋子退回），cells[to] → captured（被吃子复活）
  # 2. 恢复 base_score / piece_count；若退回的是将/帅 → 更新 kings
  # 3. turn 翻转，pop undo_stack + hash_stack
  # 4. 若撤销的是空着，只恢复 turn 与哈希栈
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
- **增量**：`Board.base_score` = 子力 + PST + 深入惩罚；`evaluate` 在其上单次扫盘补算出子 / 将帅安全 / 车炮控线

**子力+PST 公式**：

```python
红方棋子 sq 处：+ PIECE_VALUE[ptype] + PST_FLAT[ptype][sq] + deep(piece,sq)
黑方棋子 sq 处：- PIECE_VALUE[ptype] - PST_FLAT[ptype][mirror[sq]] + deep(piece,sq)
```

`mirror[sq]` = `(9 - r) * 9 + c`，黑方上下翻转后复用红方位置表。另有开局出子、控线、将帅安全项。

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
search(board, time_limit=4.0, max_depth=8, top_n=2, fill_pv=True)
  │
  ├─ for depth in range(1, max_depth+1):   ← 迭代加深
  │   ├─ 预估耗时（上一层 × 3），超出预算就停
  │   ├─ depth≥3 时 aspiration 窗口；失败则加宽重搜
  │   ├─ root_search(d)
  │   ├─ 杀棋则提前 break
  │   └─ TimeUp → break
  │
  ├─ top_n>0：备选浅搜复核（完整窗口）；top_n<=0 跳过
  ├─ 软偏好（仅 top_n>0 且备选 exact）：吃子与布局分差极小时优先布局
  ├─ extract_pv(best, fill=fill_pv)
  └─ SearchOutcome(...)
```

### 5.2 Negamax 主搜索

```python
negamax(depth, alpha, beta, ply, allow_null=True):
  1) 重复局面 → 0
  2) depth <= 0 → quiesce(...)
  3) 定长 TT probe（depth-prefer 存储）
  4) 空着剪枝：未将军且 depth≥3 时 R=2/3，失败高则截断
  5) generate_legal → _order_moves
  6) LMR：靠后安静着法先减层，失败高则完整重搜
  7) 存 TT flag（EXACT / LOWER / UPPER）
```

**Negamax 交替**：`score = -negamax(depth-1, -beta, -alpha)`。

### 5.3 静态搜索 quiesce

消除水平线效应。`qdepth` 上限与 `max_depth` 挂钩：正常 ≤ `max_depth+2`，被将军 ≤ `max_depth+4`。

- 未将军：只搜吃子；`stand + PIECE_VALUE[victim] + 200 < alpha` 差值剪枝
- 被将军：搜全部合法应着（不可只搜吃子）

### 5.4 置换表

模块级定长数组，`TT_SIZE = 2^20`，下标 `key & (TT_SIZE-1)`，存完整 key 防误命中；**depth-prefer** 替换。

`e_flag ∈ {TT_EXACT, TT_LOWER, TT_UPPER}`，含义同标准 alpha-beta TT。

### 5.5 着法排序

优先级：TT 着法 > MVV-LVA（+ 轻微 SEE 微调）吃子 > 杀手 > 历史启发。

开局时进攻子深入对方底线的吃子会降级，避免乱吃。

**杀手 / 历史**：beta 截断的非吃子写入杀手；`history[mv] += depth²`。

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
| 5 | 主变提取依赖 TT 链 | TT 断链则 PV 缩短 | 分析页 `fill_pv=True` 可浅层续搜；对弈 `fill_pv=False` 只跟 TT |
| 6 | SEE 仅为粗近似 | 复杂交换排序不准 | 排序以 MVV-LVA 为主，SEE 只作微调 |

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
TT_SIZE = 1 << 20
ASPIRATION = 60
```

---

## 11. 模块依赖图

```
constants.py
    ↑
board.py  ←→ evaluate.py（增量 base_score / piece_base_term）
    ↑
movegen.py / book.py / notation.py / explain.py
    ↑
search.py
    ↑
decide.py   ← 顶层入口 decide_position
    ↑
ui/app.py（截图分析）  ui/play.py（人机对弈）
```

---

## 12. 当前人机对弈计算链与约 10 秒耗时分析（2026-11-18）

### 12.1 结论先行

1. **人机对弈默认“深度 6”是最大迭代深度，不是保证完成 6 层。** 引擎从 1 层开始逐层加深，并根据剩余时间和上一层耗时决定是否继续。
2. `ui/play.py` 在深度 6 时传入的搜索预算是 **7.5 秒**，不是固定每次计算 10 秒。实际墙钟时间还包含合法性检查、开局库、结果解释、坐标转换、线程排队与界面刷新，并受超时检测粒度和系统调度影响。
3. 2026-11-18 已编译 `engine/_xiangqi.cp310-win_amd64.pyd`，复测 `native_available=True`。默认 `search()` 现在优先走 **`engine._xiangqi` Cython 原生核**；只有扩展不存在或无法导入的环境才回退到 `search_python()`。已经运行的旧界面需要重启后才能加载新扩展。
4. 对弈路径明确设置 `top_n=0, fill_pv=False`，因此**不做备选着法复核、不启用吃子/布局软偏好，也不为补全长主变额外续搜**。主要耗时在 Negamax + 静态搜索的节点展开，而不是界面绘图。
5. 开局库命中时对弈路径直接返回库着法，可能接近瞬时；脱库后的复杂中局更可能用满搜索预算。因此“每次约 10 秒”应理解为当前一批局面的体感范围，不是所有局面的固定耗时。

### 12.2 从玩家落子到电脑落子的完整流程

```text
玩家在 ui/play.py 点击合法落点
  │
  ├─ generate_legal() 校验玩家着法
  ├─ Board.apply() 执行玩家着法
  ├─ game_status() 检查将军、绝杀、困毙
  └─ _start_engine() 保存当前 FEN 和深度，启动后台线程
          │
          ▼
      _engine_worker(fen, max_depth)
          │
          ├─ time_limit = min(30, max(2.5, 1.2 × 2.5^(max_depth-4)))
          └─ decide_position(
                 fen=fen, side="black", view_side="red",
                 max_depth=max_depth, time_limit=time_limit,
                 top_n=0, fill_pv=False)
                  │
                  ├─ Board.from_fen() 重建局面
                  ├─ game_status() 生成当前合法着法并判断状态
                  ├─ lookup_book() 查开局库
                  │    └─ 命中且 fill_pv=False：直接用库，不续搜
                  └─ 未命中：search()
                           │
                           ├─ 有 engine._xiangqi：Cython 原生核（当前实际路径）
                           └─ 无扩展：search_python()（自动回退）
                                    │
                                    ├─ 迭代加深 + aspiration
                                    ├─ Negamax + alpha-beta
                                    ├─ TT / 空着剪枝 / LMR
                                    ├─ 静态搜索与评估
                                    └─ 返回最后一个完整深度的最佳着法
                  │
                  ├─ extract_pv(fill=False)：只沿 TT 取已有主变
                  ├─ _move_payload()：中文记谱、战术解释、评分与坐标
                  └─ _pv_payload()：校验并转换已有 PV
          │
          ▼
      queue.Queue → Tk 主线程每 80ms 轮询
          │
          ├─ 再次校验电脑着法合法
          ├─ Board.apply() 执行黑方着法
          └─ 重画棋盘并检查游戏状态
```

后台线程只负责计算，Tk 控件仍在主线程更新。因此计算期间界面不至于因搜索递归直接卡死，但电脑要等结果入队后才会落子。

### 12.3 难度、时间预算与实际深度的关系

对弈页难度范围为 2～12，默认 6。时间预算公式位于 `ui/play.py::_engine_worker`：

| 设置深度 | 搜索时间预算 | 说明 |
|---:|---:|---|
| 2～4 | 2.5 秒 | 下限保护 |
| 5 | 3.0 秒 | `1.2 × 2.5` |
| 6 | 7.5 秒 | 当前默认值 |
| 7 | 18.75 秒 | 预算明显增长 |
| 8～12 | 30 秒 | 达到封顶 |

每一层开始前还有独立的“是否值得继续加深”判断：

```python
if last_duration and elapsed + last_duration * 3.0 > time_limit:
    break
```

这表示：

- 预算只是允许搜索使用的上限，不保证一定用完；
- 如果上一层已经较贵，引擎可能提前停止，最终深度小于界面设置；
- 如果下一层已经开始，之后主要靠递归内的超时检查退出；
- 超时后丢弃未完成层，使用**上一个完整深度**的最佳着法；连 1 层都未完成时才退化为第一个合法着法。

### 12.4 一次搜索内部具体算了什么

#### A. 迭代加深与 aspiration

从深度 1 搜到 `max_depth`。上一轮最佳着法会进入下一轮根节点排序，置换表也会保留较浅层结果，虽然浅层需要重复计算，但通常能显著提高深层剪枝效率。

深度 3 起使用上一轮评分上下各 60 厘兵的 aspiration 窗口。若失败低/失败高，窗口按 60、120、240……扩宽并重搜；评分波动大时，这部分重搜会增加节点数。

#### B. Negamax + alpha-beta

双方轮流行棋，通过 `-negamax(...)` 交替取负统一评分视角。每个节点的主要工作为：

1. 重复局面检测，重复则按 0 分处理；
2. 查询 2²⁰ 槽的定长置换表，满足深度和边界条件时直接复用；
3. 判断是否被将军；
4. 合适时做空着剪枝：深度至少 3，削减 2 层，深度至少 6 时削减 3 层；
5. 生成并排序合法着法；
6. 逐个 `apply → 递归 → undo`；
7. 对靠后的安静着法执行 LMR，若减层搜索反而提高 alpha，再按原深度重搜；
8. 把评分边界和最佳着法写回置换表。

根节点会扫描当前窗口下的根着法并保存本轮结果；`top_n=0` 时最终只保留最佳着法，不对其他根着法做完整窗口复核。

#### C. 着法排序

当前优先级为：

```text
TT 最佳着法
  > 吃子（MVV-LVA + 轻量 SEE）
  > 两个杀手着法
  > 历史启发表
```

开局阶段，车/马/炮深入对方底三行吃子会被降低排序分，其中炮额外降 60000，其他两者降 40000。该规则只改变搜索顺序；对弈路径 `top_n=0` 不会在搜索结束后再用软偏好替换最佳着法。

#### D. 静态搜索与评估

主搜索深度归零后不是立刻采用静态分，而是进入 `quiesce()`：

- 普通局面只展开吃子，并使用差值剪枝；
- 被将军时展开全部合法应着，没有应着则按杀棋计分；
- 普通局面的递归上限为 `max_depth+2`，被将军时为 `max_depth+4`。

所以界面显示深度 6 时，战术分支可能继续向下展开，实际节点数不只由“6 层”决定。

静态评估由两部分组成：

1. `Board.base_score` 增量维护的子力、位置表和深入惩罚；
2. 每次评估扫描棋盘补算的开局出子、士象防守、车炮控线等着法环境分。

### 12.5 时间主要花在哪里

| 阶段 | 当前是否执行 | 成本特征 |
|---|---|---|
| FEN 解析、合法着法与状态检查 | 每次执行 | 通常远小于主搜索 |
| 开局库查询 | 每次执行 | 字典查询；命中时对弈路径可省掉整个搜索 |
| 迭代加深主搜索 | 脱库后执行 | **主要耗时**，节点数随局面和剪枝效率变化 |
| 合法着法生成与攻击检测 | 每个搜索节点反复执行 | 高频热点；native 可显著降低单节点成本 |
| `apply/undo`、哈希、增量评分 | 每个展开节点执行 | 高频小操作累积 |
| 静态评估与吃子排序 | 叶子及吃子节点执行 | 战术局面会明显放大 |
| 备选着法复核 | 对弈不执行 | `top_n=0` 已跳过 |
| PV 断断续搜 | 对弈不执行 | `fill_pv=False` 已跳过，也无额外 0.8 秒续搜窗口 |
| 已有 PV 提取、中文解释和记谱 | 每次执行 | 通常很小，但包含合法性检查与棋盘推演 |
| 队列轮询与界面刷新 | 每次执行 | 轮询间隔 80ms，绘图通常不是主要成本 |

### 12.6 本机独立测量：Python 基线与 native 复测

为隔离开局库影响，2026-11-18 在当前工作区直接调用 `search()`，使用与对弈默认档一致的 `time_limit=7.5, max_depth=6, top_n=0, fill_pv=False`，局面为红方炮二平五后轮到黑方。

编译扩展前的纯 Python 基线：

```text
native_available = False
wall_s             = 3.410
engine_ms          = 3409
最终完整深度        = 5
nodes              = 37,990
平均速度            = 约 11,142 nodes/s
PV 长度            = 5
```

同一局面、同一入口和同一参数，编译扩展后的 native 复测：

```text
native_available = True
wall_s             = 0.113
engine_ms          = 111
最终完整深度        = 6
nodes              = 92,872
平均速度            = 约 818,354 nodes/s
PV 长度            = 6
```

两次完成的深度和节点数不同，不能直接把 `3.410 / 0.113` 当作等深度加速比。固定深度 4 的 `test/bench_native.py` 对照结果如下：

```text
Python    2.754s   nodes=21,789   move=(64,65)   score=0
Native    0.030s   nodes=21,084   move=(64,65)   score=0
加速约 92.9 倍，最佳着法与评分一致
```

这些结果说明：

1. **设置深度 6 不等于一定完成深度 6**；纯 Python 基线因继续加深预估超预算而停在深度 5。
2. **7.5 秒是预算而不是固定等待时间**；native 在本例完成深度 6 后便立即返回，没有继续等待到预算上限。
3. 当前默认 native 路径已显著降低该样例的计算时间，同时保持固定深度对照中的最佳着法和评分一致。

### 12.7 如何理解此前“一次约 10 秒”

在纯 Python 路径下，默认档搜索预算为 7.5 秒，复杂中局可能接近或用满预算；此外还有以下非固定差额：

- 每 512 个节点才检查一次时钟，超时后还要沿递归栈退出；
- 搜索前的状态检查，以及搜索后的解释、PV 校验、坐标和记谱转换，发生在 `Searcher.start` 的纯搜索计时之外；
- 后台线程结果最多等待下一次 80ms UI 轮询；
- 搜索进程与模拟器等其他进程同时运行时，CPU 调度会改变单位时间节点数和超时检测粒度对应的墙钟时间，纯 Python 路径尤其明显。

启用 native 后，上述默认深度样例已不再需要用完预算。**如果重启界面后仍稳定耗时约 10 秒，就不能继续笼统归因于搜索速度**，应记录以下阶段时间戳：

```text
玩家落子完成时刻 → worker 进入时刻 → decide_position 返回时刻
→ UI 收到结果时刻 → 电脑落子绘制完成时刻
```

并与返回值 `decision["search"]` 中的 `time_ms / depth / nodes / book` 对照。`time_ms` 主要反映搜索会话，不等于玩家感受到的全部墙钟时间；更深难度或更复杂局面仍可能合法地使用更多预算。

### 12.8 native 运行与后续调速注意事项

- 当前扩展已编译，`native_available()` 和 `engine._xiangqi.native_available()` 均返回 `True`，默认 `search()` 自动进入原生核。
- 已打开的 `ui/play.py` / `ui/app.py` 在编译前就完成了模块导入，需要**重启界面**才能加载新扩展。
- `.pyd` 是本地构建产物且被 `.gitignore` 排除；更换机器、Python 版本或拉取全新代码目录后，需要重新执行 `python setup.py build_ext --inplace`。
- 想直接缩短高难度的最长等待：降低对弈页难度，或调整 `ui/play.py` 的 `time_limit` 公式；深度 5 当前预算为 3 秒。
- 修改 Cython 源码后，应重新编译，并运行 `python test/bench_native.py --depth 4` 与 `python test/test_xiangqi_engine.py`，检查加速效果、结果一致性和规则回归。
