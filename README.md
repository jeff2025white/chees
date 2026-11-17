# 中国象棋截图助手

从手机/模拟器截图识别棋盘 → 引擎搜索最佳着法 → 把走位线画回原图。

## 运行

```python
python ui/app.py          # 图形界面（1024x768）
python test\batch_run.py  # 命令行批量处理 images\ 下所有图，输出到 images\annotated\
python -m engine          # 引擎命令行
```

依赖：`opencv-python`、`numpy`、`pillow`、`tkinter`（均已可用）。

## 数据流

```
截图 PNG
  └─ engine/recognize.py : analyze_image()  → Analysis（网格、视角、格点）
        └─ engine/decide.py : decide_position() → dict（最佳着法 + 主变 PV）
              └─ engine/recognize.py : annotate() → 标注后的 BGR 图
```

界面 `ui/app.py` 就是把这三步串起来，并负责显示和翻页；计算跑在后台线程，结果经 `queue.Queue` 回主线程。

## 模块职责

| 文件 | 职责 | 改动时注意 |
|---|---|---|
| `engine/constants.py` | 棋盘尺寸、棋子字符、子力价值 `PIECE_VALUE` | 棋子用大写=红、小写=黑 |
| `engine/board.py` | 棋盘状态、`from_grid`/`to_fen`/`apply` | 内部坐标是 `sq = row*9 + col` |
| `engine/movegen.py` | 所有走子规则、合法性过滤 | 飞将、将帅照面、炮架都在这里 |
| `engine/evaluate.py` | 静态评估：子力 + 位置表 | 改棋力主要改这里的位置表 |
| `engine/search.py` | Negamax + Alpha-Beta、迭代加深、置换表、静态搜索 | 时间限制靠 `time_limit` 截断 |
| `engine/notation.py` | ICCS 与中文记谱 | |
| `engine/explain.py` | 着法理由、标签（吃子/将军/绝杀） | 界面右侧 `reason` 来自这里 |
| `engine/decide.py` | **对外唯一决策入口**，坐标视角转换、PV 打包 | 改返回结构要同步改 `ui/app.py` |
| `engine/recognize.py` | **识别与标注**：棋子检测、网格拟合、兵种推断、画线 | 改颜色阈值改文件顶部常量 |
| `ui/app.py` | Tkinter 界面 | 颜色/布局常量在文件顶部 |
| `test/test_xiangqi_engine.py` | 引擎回归测试（perft、规则、杀棋） | 改动引擎后必须跑 |
| `test/test_ui_smoke.py` | 界面冒烟测试，会保存窗口截图 | 改界面后跑这个看效果 |
| `test/batch_run.py` | 批量跑图并打印校验结果 | 识别改动后的主要验证手段 |

`test/` 下其余 `diag_*.py`、`find_grid*.py`、`recognize_v*.py` 是调试历史遗留，不是入口，可以忽略。

## 防回退契约（曾反复出错，改代码前先看）

更完整条文见 `.cursor/rules/xiangqi-side-pv-annotate.mdc`（Cursor 会自动带上）。

1. **我方 = 棋盘下半部**，由 `view_side` 决定，**不是**固定红色，也不是「谁在走谁就是我方」。
2. **推演序列步数 ≈ 迭代深度**，禁止只返回 / 只显示 1 手（开局库第 1 手后必须续搜补满）。
3. **棋盘标注必须同时画前 2 手**：第 1 手我方（红线圆）、第 2 手对方（绿/黄框）；先画对方再画我方。

截图分析固定调用：

```python
decide_position(grid=analysis.grid_str,
                side=analysis.view_side,        # 我方先走 = 下半边
                view_side=analysis.view_side)
```

## 坐标契约（最容易出错的地方）

三套坐标，**不要在中间层随手翻转**：

1. **棋盘内部坐标**：`row 0 = 黑方底线`，`row 9 = 红方底线`，`sq = row*9 + col`。引擎全部用这套。
2. **图像坐标**：`row 0 = 图片最上方`，`col 0 = 最左`。`grid` 参数字符串就是这套，
   `Analysis.grid_str` 直接喂给 `decide_position(grid=...)`。
3. **界面坐标**：`decide_position` 的输出（`best_move.from/to`、`pv.steps`）**已经**按
   `view_side` 转成图像方向，取来即可直接画，**不要再镜像一次**。

`view_side` = 图片下方是哪一方（`Analysis.view_side`），也等于“我方”。

棋盘格中心用 `Analysis.cell_center(row, col)`，内部是“相邻两条网格线中点”，不是线 + cell/2。

## 关键接口

```python
# 识别
analysis = analyze_image(path)
analysis.grid          # list[list[str]]，已推断兵种（大写红/小写黑/'.' 空）
analysis.grid_str      # 10 行字符串，直接给 decide_position
analysis.view_side     # "red" / "black"，图片下方一方
analysis.red_count / black_count / hit   # 识别数量与格点命中数

# 决策
d = decide_position(grid=..., side="red", view_side="red",
                    time_limit=6.0, max_depth=8, top_n=2)
d["best_move"]   # chinese, iccs, score, score_text, from{row,col}, to{row,col},
                 # gives_check, is_mate, mate_in, reason, tags
d["pv"]          # iccs/chinese/sides/scores 等长；steps[i] = {from:{row,col}, to:{row,col}}
d["search"]      # depth, nodes, time_ms, score_cp, mate_in
d["input"]       # fen, grid, our_side, view_side, in_check

# 标注（返回新图，不修改原图）
vis = annotate(analysis, d)
```

## 标注样式

在 `engine/recognize.py` 的 `annotate()` 内（**两手都要画，缺一即回退**）：

- 我方（`pv.steps[0]`）：红色连线 + 起点红圈白内圈 + 终点蓝色空心圆
- 对方（`pv.steps[1]`）：绿色连线 + 绿色方框起点 + 黄色方框终点
- 先画对方、后画我方，保证我方在最上层
- 顶部有半透明底的中文着法与评分

颜色是 BGR 顺序，例如 `(0, 0, 255)` 是红色。标注里的「红/绿」是**走位样式**，与棋子红黑无关。

## 常见修改

| 想改什么 | 去哪里 |
|---|---|
| 识别不到棋子 / 颜色判错 | `recognize.py` 顶部 HSV 阈值常量 |
| 网格偏移 | `recognize.py` 的 `find_grid()`、`_ROI_Y0/_ROI_Y1` |
| 兵种认错 | `recognize.py` 的 `infer_pieces()`（目前按标准开局位置推断） |
| 棋力太弱 | `evaluate.py` 位置表、`search.py` 深度与剪枝 |
| 界面布局/控件 | `ui/app.py`，右侧面板宽度、拉杆范围在构建区 |
| 默认搜索深度 | `ui/app.py` 中 `tk.IntVar(value=5)` 与标签文案 |

## 测试

```python
python test\test_xiangqi_engine.py   # 引擎规则/搜索回归
python test\test_ui_smoke.py         # 起界面、算一张、存截图到 images\annotated\ui_window.png
python test\batch_run.py             # 8 张图批量识别+决策+校验
```

改完引擎必须跑第一个；改完识别或界面建议跑后两个。

## 已知限制

- 兵种靠“标准开局位置”推断，中局走了子之后无法区分车/马/炮/兵，需要 OCR 或模板匹配补足。
- 每张截图的棋盘位置靠棋子聚类拟合，±10px 内可容忍，棋盘被大面积遮挡时可能失败。
- 单次搜索有 `time_limit`（界面 6 秒）上限，深度拉到 8 层以上可能搜不完，界面显示的是实际达到深度。
- 截图按钮未实现。
