# 中国象棋助手

两个界面共用同一套自研引擎：

1. **截图分析** `ui/app.py`：识别截图 → 搜索最佳着法 → 在图上标注推演
2. **人机对弈** `ui/play.py`：红方（下方）对电脑，右侧难度拉杆 = 迭代深度

两侧底部有互跳按钮；对弈进行中若无显式放行，直接启动分析窗会被 `.play_session.lock` 拦住（避免其它进程抢开窗口）。

## 运行

```bash
python ui/app.py          # 截图分析（1024×768）
python ui/play.py         # 人机对弈
python test/batch_run.py  # 批量处理 images/，输出到 images/annotated/
python -m engine          # 引擎命令行
```

依赖：`opencv-python`、`numpy`、`pillow`、`tkinter`。

## 数据流

```
截图 PNG
  └─ engine/recognize.py : analyze_image()  → Analysis
        └─ engine/decide.py : decide_position() → dict（着法 + PV）
              └─ engine/recognize.py : annotate() → 标注 BGR 图

对弈（无识别）
  └─ Board / 人机走子
        └─ decide_position(fen=..., side="black", view_side="red",
                           top_n=0, fill_pv=False)
```

`ui/app.py` / `ui/play.py` 均在后台线程算棋，结果经 `queue.Queue` 回主线程。

## 模块职责

| 文件 | 职责 | 改动时注意 |
|---|---|---|
| `engine/constants.py` | 棋盘尺寸、棋子编码、`PIECE_VALUE` | 大写=红、小写=黑 |
| `engine/board.py` | 棋盘、`apply`/`undo`、Zobrist；增量 `base_score` / `piece_count` | `sq = row*9 + col` |
| `engine/movegen.py` | 走子规则、合法性；马/象落点预计算 | 飞将、炮架在这里 |
| `engine/evaluate.py` | 子力+PST+出子/控线/将帅；单次扫盘 | 改棋力主要改 PST |
| `engine/search.py` | Negamax、空着/LMR/aspiration、定长 TT、静态搜索 | `top_n`/`fill_pv` 控制复核与 PV |
| `engine/book.py` | 开局库 | 命中后第 1 手用库，分析路径再续搜补 PV |
| `engine/notation.py` | ICCS / 中文记谱 | |
| `engine/explain.py` | 着法理由、标签 | 分析页 `reason` |
| `engine/decide.py` | **对外唯一决策入口** | 改返回结构同步改 UI |
| `engine/recognize.py` | 识别与标注 | HSV 阈值在文件顶部 |
| `ui/app.py` | 截图分析界面 | 默认深度 6；`time_limit≈2.5·3^(d-4)` |
| `ui/play.py` | 人机对弈界面 | 默认深度 6；更短预算；`fill_pv=False` |
| `test/test_xiangqi_engine.py` | 引擎回归 | 改引擎后必跑 |
| `test/test_ui_smoke.py` | 分析界面冒烟 | 改 UI 后建议跑 |
| `test/batch_run.py` | 批量识别+决策 | 改识别后主验证 |

`test/` 下 `diag_*` / `recognize_v*` 等为调试遗留，可忽略。

## 防回退契约（曾反复出错）

完整条文见 `.cursor/rules/xiangqi-side-pv-annotate.mdc`。

1. **我方 = 棋盘下半部** = `view_side`，不是固定红色，也不是「行棋方=我方」。
2. **分析页推演步数 ≈ 迭代深度**：开局库定第 1 手后必须续搜补满；`extract_pv` 在 TT 断开时可浅层续搜。对弈页 `fill_pv=False`，只要最佳一手。
3. **标注必须同时画前 2 手**：我方红线圆、对方绿/黄框；先画对方再画我方。

截图分析固定：

```python
decide_position(grid=analysis.grid_str,
                side=analysis.view_side,
                view_side=analysis.view_side)
```

人机对弈（电脑执黑）：

```python
decide_position(fen=board.to_fen(), side="black", view_side="red",
                top_n=0, fill_pv=False, max_depth=..., time_limit=...)
```

## 坐标契约

1. **内部**：`row 0 = 黑底`，`row 9 = 红底`，`sq = row*9 + col`
2. **图像 / grid**：`row 0 = 图最上`；黑在下时 `decide` 会先 `mirror_grid`
3. **输出 from/to、pv.steps**：已按 `view_side` 转成图像方向，标注不要再镜像

## 关键接口

```python
analysis = analyze_image(path)
analysis.grid_str / view_side / red_count / black_count / hit

d = decide_position(grid=..., side="red", view_side="red",
                    time_limit=6.0, max_depth=8, top_n=2, fill_pv=True)
# top_n<=0 → 跳过备选复核；fill_pv=False → PV 只跟置换表

d["best_move"]   # chinese, iccs, from/to, score_text, reason, tags, ...
d["pv"]          # iccs/chinese/sides/scores/steps 等长
d["search"]      # depth, nodes, time_ms, score_cp, mate_in, book
d["input"]       # fen, grid, our_side, view_side, side_to_move, in_check

vis = annotate(analysis, d)   # 新图，不改原图
```

`search(board, time_limit, max_depth, top_n=2, fill_pv=True)` 为底层搜索；一般走 `decide_position`。

## 时间预算（界面）

| 界面 | 公式（封顶） | 深度 6 约 |
|---|---|---|
| 截图分析 | `min(90, max(4, 2.5·3^(d-4)))` | 22.5s |
| 人机对弈 | `min(30, max(2.5, 1.2·2.5^(d-4)))` | ~7.5s |

引擎按「上一层 ×3」预估是否继续加深；界面显示的是实际达到深度。

## 标注样式

`annotate()`（两手都要画）：

- 我方 `pv.steps[0]`：红线 + 红圈起点 + 蓝圈终点
- 对方 `pv.steps[1]`：绿线 + 绿框起点 + 黄框终点
- 先对方后我方；颜色是走位样式，与棋子红黑无关（BGR）

## 常见修改

| 想改什么 | 去哪里 |
|---|---|
| 识别颜色 / 网格 | `recognize.py` 顶部常量、`find_grid()` |
| 兵种推断 | `infer_pieces()`（仍偏标准开局） |
| 棋力 | `evaluate.py` PST；`search.py` 深度/剪枝；`book.py` |
| 分析 UI | `ui/app.py` |
| 对弈 UI / 难度 | `ui/play.py`（默认深度 6） |
| 对弈响应速度 | `ui/play.py` 的 `time_limit` 公式 |

## 测试

```bash
python test/test_xiangqi_engine.py   # 引擎回归
python test/test_ui_smoke.py         # 分析界面冒烟 → images/annotated/ui_window.png
python test/batch_run.py             # 批量识别+决策
```

## 已知限制

- 兵种靠开局位置推断，中局区分车马炮兵能力弱。
- 棋盘靠棋子聚类拟合，大面积遮挡可能失败。
- 纯 Python 搜索；深度 ≥8 时分析页预算很长，对弈页用更短预算保响应。
- 截图按钮未实现。
