# 中国象棋「局面 → 该走哪一步」决策内核 · 设计说明

版本：2026-11-17　代码：`g:\chessCn\engine\`　测试：`g:\chessCn\test\test_xiangqi_engine.py`

---

## 1. 目标与范围

本次交付**决策内核**：输入一个局面（FEN 或 10×9 网格）与行棋方，输出

- 最佳着法的结构化坐标（从哪个格子到哪个格子）、ICCS、UCCI、中文记谱；
- 该着法的评分、搜索深度、主变（PV）；
- 1–2 个备选着法；
- 一句人话理由（吃子 / 解将 / 将军 / 绝杀 / 出动 / 过河 / 护子 / 落点安全）。

不在本次范围：棋盘截图识别、红圈绘制。为此保留的接口见 §5。

---

## 2. 整体流水线

```
     截图                     局面重建                引擎决策               标注
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐   ┌────────────────┐
│ 棋盘图        │──▶│ 网格/棋子在格判定   │──▶│ decide_position  │──▶│ 按 from/to 画红圈│
│ (用户上传)    │   │ → 10×9 网格文本    │   │ 规则+评估+搜索    │   │ 我方子 + 目标格  │
│              │   │ + 行棋方/我方执色   │   │ 结构化 JSON      │   │ + 方案说明       │
└──────────────┘   └───────────────────┘   └──────────────────┘   └────────────────┘
   【待接】             【待接】                【已完成】              【待接】
```

边界约定：识别模块**只需产出**与 `decide_position` 入参同构的数据（网格 + 行棋方 + 我方执色），
决策模块**只返回**带格子下标的结构化结果。两边不共享任何像素/模型细节。

---

## 3. 坐标契约（三套，互不混淆）

| 名称 | 形式 | 约定 |
|---|---|---|
| 内部格号 `sq` | `0..89`，`sq = row*9+col` | `row 0 = 黑方底线`，`row 9 = 红方底线`；`col 0 = 红方视角最左` |
| 界面格 `(col,row)` | `col 0..8`，`row 0..9` | **图下方固定为我方**：`row 9` 是我方底线，`row 0` 是对方底线；`col 0` 是图最左。我方执黑时已整体镜像（`col' = 8-col`, `row' = 9-row`） |
| ICCS | 如 `h3e3` / `i9i10` | `a..i` 列、`1..10` 行，**以红方视角**为基准（红方底线为第 1 行）。中文记谱示例「炮二平五」= `h3e3` |
| UCCI | 如 `h2e2` | 列 `a..i`、行 `0..9`，0 基，与多数引擎/棋谱工具一致 |

给识别/标注模块用的是**界面格 `(col,row)`**：

```
x = margin_x + col * cell_w
y = margin_y + row * cell_h
```

`decide_position` 返回 JSON 中的 `coordinate_contract` 字段会原样再声明一遍这套约定，
即使调用方忘了，也能从返回体里读到。

> 注意：`from_sq/to_sq/iccs/ucci` 是红方基准的棋谱坐标，与界面格是两套；画圈请用 `from/to` 的 `col,row`。

---

## 4. 输入格式

### 4.1 FEN（推荐，无歧义）

10 行以 `/` 分隔，每行从**黑方底线（row 0）到红方底线（row 9）**，大写为红、小写为黑：

```
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
K=帅 k=将  A/a=仕士  B/b=相象  N/n=马（也接受 H/h）  R/r=车  C/c=炮  P/p=兵卒
第二段 w/red/r 表示红先，b/black 表示黑先；后续字段忽略
```

### 4.2 自定义网格（识别模块的天然输出）

10 行 × 9 个字符，`.` `-` `_` `0` 表示空位，字符含义同 FEN；允许空格与 `|` 作为分隔符。

```
...ak....
........R
.....a...
...N.....
.........
.........
.........
.........
.........
...K.....
```

**网格方向 = 界面方向**（第 0 行是图最上方）。若 `view_side=black`，引擎内部会自动旋转 180° 再解析，
因此识别模块永远按图像行序输出即可，不必自己翻转。

---

## 5. 模块与 I/O 契约

### 5.1 文件结构

| 文件 | 职责 | 对外主要符号 |
|---|---|---|
| `engine/constants.py` | 棋子编码、子力价值、FEN 字符表、中文数字 | `PIECE_VALUE`、`START_FEN` |
| `engine/board.py` | 局面状态、FEN/网格解析与导出、走子撤销、Zobrist 哈希、将位缓存 | `Board.from_fen` / `from_grid` / `to_fen` / `to_grid` / `apply` / `undo` |
| `engine/movegen.py` | 规则与着法生成、攻击判定、合法着法过滤、perft、局面状态 | `generate_pseudo_moves`、`generate_legal_moves`、`in_check`、`is_attacked`、`game_status`、`perft` |
| `engine/evaluate.py` | 评估：子力价值 + 各兵种位置分 | `evaluate(board)`（红方视角）、`evaluate_for` |
| `engine/search.py` | 迭代加深 alpha-beta、置换表、静态搜索、着法排序、备选复核、PV | `search(board, time_limit, max_depth) -> SearchOutcome` |
| `engine/notation.py` | ICCS/UCCI/中文记谱、格子与棋子名 | `move_to_iccs`、`move_to_ucci`、`chinese_notation` |
| `engine/explain.py` | 战术事实提取与人话理由生成 | `analyze_move`、`build_reason` |
| `engine/decide.py` | **对外统一入口** | `decide_position(...)`、`explain_position(fen)`、`mirror_grid` |
| `engine/cli.py` | 命令行 | `python -m engine.cli` |

### 5.2 核心接口

```python
from engine import decide_position

result = decide_position(
    fen=None,            # 标准 FEN，与 grid 二选一
    grid=None,           # 10×9 网格文本（界面方向）
    side=None,           # "red"/"black"/"w"/"b"/"红"/"黑"；缺省取 FEN 行棋方
    view_side=None,      # 图下方是哪一方；缺省 = side
    time_limit=4.0,      # 搜索时间预算（秒）
    max_depth=8,         # 迭代加深上限（实际通常 4–6 层）
    top_n=2,             # 备选着法数量
) -> dict
```

返回（节选，完整字段见 `outputs/example_run.md`）：

```jsonc
{
  "ok": true,
  "input": {"fen": "...", "grid": "...", "side_to_move": "red", "view_side": "red",
            "in_check": false, "legal_move_count": 26},
  "coordinate_contract": {"cols": 9, "rows": 10, /* 同 §3 */},
  "status": "playing",            // playing | check | checkmate | stalemate
  "best_move": {
    "from": {"col": 8, "row": 1}, "to": {"col": 8, "row": 0},
    "from_sq": "i9", "to_sq": "i10", "iccs": "i9i10", "ucci": "i8i9",
    "chinese": "车一进一", "piece": "车", "color": "red",
    "captured": null, "is_capture": false,
    "gives_check": true, "is_mate": true, "mate_in": 1,
    "score_cp": 99999, "score_text": "绝杀(1步)", "score_exact": true, "score_depth": null,
    "reason": "直接绝杀，对手已无合法应着。",
    "tags": ["mate"]              // mate/escape_check/capture/free_capture/check/defense/development/threat/supported/risky
  },
  "alternatives": [ /* 同 best_move 结构，score_depth 标注复核层数 */ ],
  "pv": {"iccs": [...], "chinese": [...]},
  "search": {"depth": 1, "nodes": 305, "time_ms": 11, "score_cp": 99999, "mate_in": 1},
  "message": "直接绝杀，对手已无合法应着。"
}
```

分数约定：`score_cp` 为**厘兵**，且**从行棋方（side_to_move）视角**，正数表示行棋方占优；
有杀棋时进入 ±100000 区间，`mate_in` 给出还需几步。

调用失败不抛业务异常：无合法着法时 `best_move=null`、`status` 为 `checkmate/stalemate`、`message` 说明原因。

---

## 6. 规则实现（已完成并测试）

- 兵种走法：车、炮（炮架吃子）、马（蹩马腿）、象/相（塞象眼、不过河）、士（九宫斜走）、帅/将（九宫直走）、兵/卒（过河后可横走）。
- **飞将**：将帅同列且中间无子，视为被攻击，走子后出现即不合法（用例 `test_flying_general_pins_rook`）。
- 将军判定与**应将过滤**：所有伪合法着法走后重算己方是否被攻击，不合法则丢弃。
  性能优化：未将军时只对「敏感格」（将帅自身、同行/同列己方子、敌方马腿格）上的起点，
  以及**落点落在将帅行列上**的走法做完整验证——后者若漏掉，会出现「自己把子走到将帅与敌炮之间当炮架」
  而未被判非法的错误（已修复并有回归用例）。
- **绝杀**（无合法着法且被将军）与**困毙**（无合法着法且未被将军）**均判负**，`game_status` 分别返回 `checkmate` / `stalemate`。
- 正确性基线：perft 从初始局面的 1/2/3/4 层分别得到 **44 / 1920 / 79666 / 3290240**，与公认值一致。

---

## 7. 评估与搜索

**评估**（`evaluate.py`，红方视角，厘兵）

- 子力：帅 60000、车 900、炮 450、马 400、士/象 200、兵/卒 100（将帅仅在杀棋时体现，不参与交换）。
- 位置分：每个兵种一张 10×9 位置表（`PST`），黑方镜像取分；鼓励马炮占中路、车占开放线、兵过河、士象归位。

**搜索**（`search.py`）

- 迭代加深（depth 1..8），每层完整根节点搜索；用上一层耗时预估下一层（×3），超预算即停。
- alpha-beta + 置换表（Zobrist，EXACT/LOWER/UPPER）+ 杀手着法 + 历史启发。
- 着法排序：TT 着法 → 吃子按 MVV-LVA → 杀手 → 历史分。
- 静态搜索（quiescence）：只搜吃子，消除水平线效应；被将军时改搜全部应着；含差值剪枝。
- 重复局面按和棋处理（`0`），避免搜索中互相兜圈。
- **备选着法**：根节点非最佳着法只有 alpha-beta 上界，不具可比性，故用「最佳深度 −1、完整窗口」对前 4 个候选复核，
  输出 `score_depth` 标明复核层数。
- 主变 PV 由置换表回溯提取，最多 6 步。

时间预算 4 s 时，通常完成 3–5 层、5–7 万节点（Python 纯标准库，约 8–15k nodes/s，具体随机器与局面波动）。

---

## 8. 中文记谱

- 红方用中文数字、黑方用阿拉伯数字；路数：红方 `9-col`，黑方 `col+1`（均从各自右侧数起）。
- 同一路上有同类子时用 前/中/后（或多子时 前二/前三/后二…）消歧。
- 车/炮/兵/将的「进/退」写**步数**；马/仕/相的「进/退」写**目的路数**；横走一律「平」。
- 示例（已测，同一着法的三种写法）：

| 中文 | ICCS | UCCI |
|---|---|---|
| 炮二平五 | `h3e3` | `h2e2` |
| 马8进7（黑） | `h10g8` | `h9g7` |
| 兵六平五 | `d6e6` | `d5e5` |

---

## 9. 已完成 / 后续待接

**已完成**

1. 规则与合法着法生成（含飞将、应将过滤、绝杀/困毙），perft 校验通过。
2. 评估 + 迭代加深 alpha-beta + 静态搜索 + 置换表 + 着法排序，输出最佳/备选/评分/PV。
3. 结构化输出：界面格 `(col,row)`、ICCS、UCCI、中文记谱、评分、理由、tags；JSON 可序列化。
4. 三种输入形态：FEN、界面方向网格（自动按 `view_side` 镜像）、`--side/--view-side` 覆盖。
5. CLI、17 个单元测试、可复现示例（见 `outputs/example_run.md`）。

**后续待接（本次未做）**

1. 截图识别：棋盘四角/网格检测 → 90 格中心 → 棋子分类 → 生成 §4.2 网格；输出同时给出行棋方与「我方执色」。
2. 红圈标注：用 `best_move.from/to` 的 `(col,row)` 结合格宽/边距换算像素，对**起点棋子描边**、**目标格画圈**；
   若 `best_move` 为 `null`（已被将死），改为展示 `message`。
3. 交互闭环：识别不确定时让用户点击修正单个格子；决策结果可回写为「局面 + 着法」记录用于复盘。
4. 与产品 UI 对接：`reason/tags` 直接作为「方案说明」，`alternatives` 作为「换一步」选项。

---

## 10. 已知局限

- **未实现长将/长捉判负**、60 回合自然限着：重复局面一律按和棋（0 分）处理。
- **无开局库、无残局库**：开局阶段着法合理但不够“谱”，残局杀法依赖搜索深度。
- 评估较简单（子力 + 位置分，无兵型结构、无王安全专项、无子力协同项），因此中局计划性偏弱。
- 搜索深度 3–5 层（4 s 预算），战术视野有限；`time_limit` 是软预算，超时即返回上一层结果。
- 备选着法评分来自「最佳深度 −1」复核，与最佳着法的分数不完全同级（已在 `score_depth` 标注）。
- 中文记谱的「前/中/后」消歧已覆盖同路同类子，但极端的同路三兵以上局面仅做简化处理。
- 未校验输入局面的“可达性”（例如双方各缺将帅、兵数超限），只校验格式与基本合法性。

---

## 11. 运行与测试

```powershell
# 初始局面（4 秒预算）
python -m engine.cli --demo --time 4

# 指定 FEN 与视角
python -m engine.cli --fen "3ak4/8R/5a3/3N5/9/9/9/9/9/3K5 w - - 0 1" --time 3
python -m engine.cli --fen "<FEN>" --view-side black --json out.json

# 网格输入（识别模块对接形态）
python -m engine.cli --grid-file board.txt --side black --view-side black

# 单元测试
python -m unittest test.test_xiangqi_engine -v
```
