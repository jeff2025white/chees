# 可复现运行示例（真实输出）

环境：Windows / Python 3.10.11，纯标准库，无第三方依赖、不调用任何外部引擎。
所有命令均在仓库根目录 `g:\chessCn` 执行；耗时会随机器负载波动，着法结论以本机 4 秒预算为准。

---

## 示例 1：一步杀（主例）——从局面到「该走哪一步」

输入局面（红先）：`3ak4/8R/5a3/3N5/9/9/9/9/9/3K5 w - - 0 1`
（红车在 i9，红马在 d7，黑将在 e10，黑士在 d10、f8）

```powershell
python -m engine.cli --fen "3ak4/8R/5a3/3N5/9/9/9/9/9/3K5 w - - 0 1" --time 4
```

真实输出：

```
    0 1 2 3 4 5 6 7 8
 0  . . . a k . . . .
 1  . . . . . . . . R
 2  . . . . . a . . .
 3  . . . N . . . . .
 4  . . . . . . . . .
 5  . . . . . . . . .
 6  . . . . . . . . .
 7  . . . . . . . . .
 8  . . . . . . . . .
 9  . . . K . . . . .

轮到: red
局面状态: playing（合法着法 26 个）
搜索: 深度 1，节点 305，用时 11 ms
最佳着法: 车一进一  (i9i10 / i8i9)  评分 绝杀(1步)，将军
  坐标: 从 (col=8, row=1) 到 (col=8, row=0)
  理由: 直接绝杀，对手已无合法应着。
  预计后续: 车一进一
备选1: 车一平四  (i9f9 / i8f8)  评分 绝杀(2步)，2层复核
  出动车，抢占要道、增强进攻子力；占据中路要点；下一步可白吃对方士。
备选2: 马六进四  (d7f8 / d6f7)  评分 +11.48，吃士，将军，2层复核
  白吃对方一士；并同时将军。
```

要点：

- 决策结果直接给出**界面格坐标** `(col=8,row=1) → (col=8,row=0)`，标注模块据此画两个红圈即可。
- `i9i10` 是 ICCS（红方基准、行号 1..10），`i8i9` 是 UCCI（0 基）；中文记谱为「车一进一」。
- 评分 `绝杀(1步)` 表示一步内将死；`is_mate=true`、`mate_in=1`。
- 备选里 `车一平四` 也是杀（两步），`马六进四` 是吃士带将的物质收益。

同一棋局的结构化输出（`--json` 写出，节选）：

```jsonc
{
  "ok": true,
  "input": {
    "fen": "3ak4/8R/5a3/3N5/9/9/9/9/9/3K5 w - - 0 1",
    "grid": "...ak....\n........R\n.....a...\n...N.....\n.........\n.........\n.........\n.........\n.........\n...K.....",
    "side_to_move": "red", "our_side": "red", "view_side": "red",
    "in_check": false, "legal_move_count": 26
  },
  "coordinate_contract": {
    "cols": 9, "rows": 10,
    "col_range": "0..8，从左到右（图左为 0）",
    "row_range": "0..9，从上到下（图上方为 0）",
    "orientation": "row 9 为我方（图下方）底线；我方执黑时坐标已整体镜像",
    "view_side": "red",
    "pixel_mapping": "x = margin_x + col * cell_w, y = margin_y + row * cell_h"
  },
  "status": "playing",
  "best_move": {
    "from": {"col": 8, "row": 1},
    "to":   {"col": 8, "row": 0},
    "from_sq": "i9", "to_sq": "i10",
    "iccs": "i9i10", "ucci": "i8i9", "chinese": "车一进一",
    "piece": "车", "color": "red",
    "captured": null, "is_capture": false,
    "gives_check": true, "is_mate": true, "mate_in": 1,
    "score_cp": 99999, "score_text": "绝杀(1步)", "score_exact": true, "score_depth": null,
    "reason": "直接绝杀，对手已无合法应着。",
    "tags": ["mate"]
  },
  "alternatives": [
    {"chinese": "车一平四", "iccs": "i9f9", "score_text": "绝杀(2步)", "score_depth": 2,
     "reason": "出动车，抢占要道、增强进攻子力；占据中路要点；下一步可白吃对方士。",
     "tags": ["development", "threat"]},
    {"chinese": "马六进四", "iccs": "d7f8", "score_text": "+11.48", "score_depth": 2,
     "captured": "士", "gives_check": true,
     "reason": "白吃对方一士；并同时将军。",
     "tags": ["capture", "free_capture", "check"]}
  ],
  "pv": {"iccs": ["i9i10"], "chinese": ["车一进一"]},
  "search": {"depth": 1, "nodes": 305, "time_ms": 11, "score_cp": 99999, "mate_in": 1},
  "message": "直接绝杀，对手已无合法应着。"
}
```

---

## 示例 2：初始局面（4 秒预算）

```powershell
python -m engine.cli --demo --time 4
```

真实输出：

```
    0 1 2 3 4 5 6 7 8
 0  r n b a k a b n r
 1  . . . . . . . . .
 2  . c . . . . . c .
 3  p . p . p . p . p
 4  . . . . . . . . .
 5  . . . . . . . . .
 6  P . P . P . P . P
 7  . C . . . . . C .
 8  . . . . . . . . .
 9  R N B A K A B N R

轮到: red
局面状态: playing（合法着法 44 个）
搜索: 深度 4，节点 53778，用时 4227 ms
最佳着法: 炮八平七  (b3c3 / b2c2)  评分 +0.00
  坐标: 从 (col=1, row=7) 到 (col=2, row=7)
  理由: 出动炮，抢占要道、增强进攻子力。
  预计后续: 炮八平七 炮2平3 炮七进四 炮3进4
备选1: 炮八平五  (b3e3 / b2e2)  评分 +0.00，3层复核
  出动炮，抢占要道、增强进攻子力；占据中路要点。
备选2: 兵七进一  (c4c5 / c3c4)  评分 +0.00，3层复核
  推进兵线，为后续进攻做准备。
```

说明：开局四秒只能搜 4 层，评估为均势（+0.00），选的是出动炮抢要道；这类局面没有明显战术，
不同运行可能在同一集团里换着法，属于预期行为（无开局库，见设计说明“已知局限”）。

---

## 示例 3：中局 + 我方执黑（坐标镜像）

局面来自中炮过河车 vs 屏风马（真实走法生成：炮二平五 马8进7 马二进三 车9平8 车一平二 卒7进1 车二进六），黑先：

```powershell
python -m engine.cli --fen "rnbakabr1/9/1c4nc1/p1p1p2Rp/6p2/9/P1P1P1P1P/1C2C1N2/9/RNBAKAB2 b - - 0 1" --view-side black --time 4
```

真实输出：

```
    0 1 2 3 4 5 6 7 8
 9  . . B A K A B N R
 8  . . . . . . . . .
 7  . . N . C . . C .
 6  P . P . P . P . P
 5  . . . . . . . . .
 4  . . p . . . . . .
 3  p R . . p . p . p
 2  . c n . . . . c .
 1  . . . . . . . . .
 0  . r b a k a b n r

轮到: black
局面状态: playing（合法着法 34 个）
搜索: 深度 3，节点 48386，用时 4237 ms
最佳着法: 炮2平3  (b8c8 / b7c7)  评分 -0.20
  坐标: 从 (col=7, row=7) 到 (col=6, row=7)
  理由: 出动炮，抢占要道、增强进攻子力；下一步可白吃对方兵。
  预计后续: 炮2平3 炮八进五 炮8平9 车二进三
备选1: 马2进3  (b10c8 / b9c7)  评分 -0.87，2层复核
  出动马，抢占要道、增强进攻子力。
备选2: 马2进1  (b10a8 / b9a7)  评分 -0.95，2层复核
  出动马，抢占要道、增强进攻子力。
```

要点：

- 棋盘按「黑方在下」打印；黑炮实际在内部格 `(row 1, col 2)`，换算成界面格是
  `(col 7, row 7)`（`col' = 8-2`、`row' = 9-1`），所以标注模块用返回的 `col/row` 直接画圈即可。
- ICCS 始终以红方为基准：`b8c8`；UCCI 为 `b7c7`。
- 评分 `-0.20` 是**黑方（行棋方）视角**：黑方略亏，因为红车已过河。

---

## 示例 4：网格输入——识别模块的对接形态

识别模块只需产出「界面方向」的 10×9 网格（第 0 行 = 图最上方）：

`g:\chessCn\2026-11-17\xiangqi-move-logic\work\board_mate1.txt`

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

```powershell
python -m engine.cli --grid-file "g:\chessCn\2026-11-17\xiangqi-move-logic\work\board_mate1.txt" --side red --view-side red --time 2 --quiet
```

真实输出：

```
轮到: red
局面状态: playing（合法着法 26 个）
搜索: 深度 1，节点 305，用时 12 ms
最佳着法: 车一进一  (i9i10 / i8i9)  评分 绝杀(1步)，将军
  坐标: 从 (col=8, row=1) 到 (col=8, row=0)
  理由: 直接绝杀，对手已无合法应着。
  预计后续: 车一进一
备选1: 车一平四  (i9f9 / i8f8)  评分 绝杀(2步)，2层复核
  出动车，抢占要道、增强进攻子力；占据中路要点；下一步可白吃对方士。
备选2: 马六进四  (d7f8 / d6f7)  评分 +11.48，吃士，将军，2层复核
  白吃对方一士；并同时将军。
```

与示例 1 的 FEN 输入结果完全一致，说明「网格 → 局面」这条链路可用。
若截图里我方是黑方，调用时传 `--view-side black`，引擎会把界面网格旋转 180° 后再解析，
识别模块无需自己翻转（单元测试 `test_grid_input_and_view_mirror` 覆盖）。

---

## 复现与自检

```powershell
# 规则/记谱/接口共 18 个用例
python -m unittest test.test_xiangqi_engine -v
```

真实结果：

```
Ran 18 tests in 1.218s
OK
```

其中包含 perft 校验：初始局面 1/2/3 层 = 44 / 1920 / 79666；另外单独跑过 4 层 = **3290240**
（与公认值一致。过程中发现并修复了一个真实 bug：己方棋子**走到**将/帅所在的行/列上会变成敌方炮的炮架，
原快速合法性过滤只检查起点、漏掉了这种“自己走出一个将军”的走法，导致 perft(4) 多算 250 个节点；
现由用例 `test_cannon_screen_created_by_own_move` 锁定）。

中间产物（可删）：`work\demo.json`、`work\demo.txt`、`work\mate1.json`、`work\mate1.txt`、
`work\midgame_black.json`、`work\midgame_black.txt`、`work\grid_red.txt`、`work\board_mate1.txt`、
`work\make_position.py`、`work\check_legality.py`、`work\refgen.py`（独立实现的朴素生成器，用于交叉校验 perft）。
