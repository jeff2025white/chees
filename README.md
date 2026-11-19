# 中国象棋助手

两个界面共用同一套自研引擎：

1. **截图分析** `ui/app.py`：打开图片或 **自动监听** → 识别 → 搜索最佳着法 → 在图上标注推演
2. **人机对弈** `ui/play.py`：红方（下方）对电脑，右侧难度拉杆 = 迭代深度

两侧底部有互跳按钮；对弈进行中若无显式放行，直接启动分析窗会被 `.play_session.lock` 拦住（避免其它进程抢开窗口）。

### 自动监听（截图分析 · Windows）

右侧「自动监听」按钮：找到标题含「手游助手」的窗口并置前，后台约每秒截客户区一帧：

1. `detect_board_fx`：中部墨云/吃/将军等特效帧直接跳过（不识别、不落盘）
2. `analyze_array` 内存识别（不先落盘）
3. 棋盘判定：`grid_str` 同时含红帅 `K` 与黑将 `k` 才算有棋盘（首页/结算页跳过）
4. 帧差：与上一有效局面 `grid_str` 不同，或首次有棋盘 → 触发计算
5. 计算期间暂停截图；算完用 `annotate` 覆盖预览，日志追加本次推演
6. 默认**不落盘**；右侧「自动截图」拉杆 >0 时写入 `images/auto/`，只保留最近 N 张（上限 50）

手动「打开图片」+「计算」仍可用；监听中会禁用打开/计算，点「停止监听」退出。特效扫描：`python test/detect_fx_overlay.py images/`。

## 运行

```bash
python ui/app.py          # 截图分析（1024×768）
python ui/play.py         # 人机对弈
python test/batch_run.py  # 批量处理 images/，输出到 images/annotated/
python -m engine          # 引擎命令行
```

依赖：`opencv-python`、`numpy`、`pillow`、`tkinter`。兵种 CNN（推荐）：`pip install torch`（见 `[project.optional-dependencies] ml`）。

### 兵种 CNN（可选，准确率更高）

```bash
pip install torch
python tools/export_piece_dataset.py --clear
python tools/train_piece_classifier.py --epochs 25
# 权重写入 engine/models/piece_cls.pt；未安装 torch 或无权重时自动回退模板 OCR
```

**macOS 注意**：系统自带 `/usr/bin/python3` 的 Tcl/Tk 8.5 在新系统上会**整窗白屏**。请安装 Homebrew 的 `python-tk`（如 `brew install python@3.14 python-tk@3.14`）。`ui/play.py` / `ui/app.py` 检测到旧 Tk 时会自动切到 `/opt/homebrew/bin/python3*`。

### Cython 搜索核（编译后默认优先，大幅加速算棋）

编译后 `search()` 默认优先走 `engine._xiangqi`，`native_available()` 返回 `True`；仅在扩展未编译或无法导入时自动回退纯 Python，功能不变。已经运行的界面需重启才能加载新编译的扩展。

```bash
pip install 'Cython>=3.0'
python setup.py build_ext --inplace   # 生成 engine/_xiangqi*.so（macOS/Linux）或 .pyd（Windows）
python test/bench_native.py --depth 4
```

- **macOS / Linux**：需系统 C 编译器（Xcode CLT / gcc）
- **Windows**：需 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（MSVC）
- 对照：`from engine import native_available` → `True` 表示已加载原生核

## 数据流

```
截图 PNG / 自动监听帧
  └─ analyze_image(path) 或 analyze_array(bgr)  → Analysis
        └─ decide_position(grid, side=view_side, view_side=view_side, …)
              └─ annotate() → 标注 BGR 图

对弈（无识别）
  └─ Board / 人机走子
        └─ decide_position(fen=..., side="black", view_side="red",
                           top_n=0, fill_pv=False)
```

`ui/app.py` / `ui/play.py` 均在后台线程算棋，结果经 `queue.Queue` 回主线程。自动监听另有截图循环线程；计算中用 `_computing` 暂停截帧。

## 模块职责

| 文件 | 职责 | 改动时注意 |
|---|---|---|
| `engine/constants.py` | 棋盘尺寸、棋子编码、`PIECE_VALUE` | 大写=红、小写=黑 |
| `engine/board.py` | 棋盘、`apply`/`undo`、Zobrist；增量 `base_score` / `piece_count` | `sq = row*9 + col` |
| `engine/movegen.py` | 走子规则、合法性；马/象落点预计算 | 飞将、炮架在这里 |
| `engine/evaluate.py` | 子力+PST+出子/控线/将帅；单次扫盘 | 改棋力主要改 PST |
| `engine/search.py` | Negamax、空着/LMR/aspiration、定长 TT、静态搜索；优先 `_xiangqi` | `top_n`/`fill_pv` 控制复核与 PV |
| `ext/_xiangqi.pyx` | Cython 搜索核（Board/着法/评估/搜索） | `setup.py build_ext --inplace` |
| `engine/book.py` | 开局库 | 命中后第 1 手用库，分析路径再续搜补 PV |
| `engine/notation.py` | ICCS / 中文记谱 | |
| `engine/explain.py` | 着法理由、标签 | 分析页 `reason` |
| `engine/decide.py` | **对外唯一决策入口** | 改返回结构同步改 UI |
| `engine/recognize.py` | 识别与标注 | `_BOARD_*` 固定格网；`annotate` 含棋盘轮廓 |
| `engine/piece_model.py` | 兵种 TinyCNN（Torch）加载/批量推理 | 权重 `engine/models/piece_cls.pt` |
| `engine/piece_ocr.py` | 兵种分配：CNN 优先，模板回退；配额/校验修补 | 禁止残留 `@` 兜成车 |
| `engine/fx_overlay.py` | 中部特效脏帧检测 | 自动监听跳过；`test/detect_fx_overlay.py` |
| `ui/app.py` | 截图分析 / 自动监听 | 默认深度 7；`time_limit≈2.5·3^(d-4)`；监听仅 Windows |
| `ui/play.py` | 人机对弈界面 | 默认深度 7；更短预算；`fill_pv=False` |
| `test/test_xiangqi_engine.py` | 引擎回归 | 改引擎后必跑 |
| `test/test_piece_ocr.py` | 兵种 CNN / 开局 GT | 改识别或重训后必跑 |
| `test/test_ui_smoke.py` | 分析界面冒烟 | 改 UI 后建议跑 |
| `test/batch_run.py` | 批量识别+决策 | 改识别后主验证 |

`test/` 下 `diag_*` / `recognize_v*` 等为调试遗留，可忽略。

## 防回退契约（曾反复出错）

完整条文见 `.cursor/rules/xiangqi-side-pv-annotate.mdc`。

1. **我方 = 棋盘下半部** = `view_side`，不是固定红色，也不是「行棋方=我方」。
2. **分析页推演步数 ≈ 迭代深度**：开局库定第 1 手后必须续搜补满；`extract_pv` 在 TT 断开时可浅层续搜。对弈页 `fill_pv=False`，只要最佳一手。
3. **标注必须同时画前 2 手**：我方红线圆、对方绿/黄框；先画对方再画我方。另画棋盘轮廓（青黄外框）便于目检格网。

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

| 界面 | 公式（封顶） | 默认深度 7 约 | 深度 6 约 |
|---|---|---|---|
| 截图分析 | `min(90, max(4, 2.5·3^(d-4)))` | 67.5s | 22.5s |
| 人机对弈 | `min(30, max(2.5, 1.2·2.5^(d-4)))` | ~18.75s | ~7.5s |

引擎按「上一层 ×3」预估是否继续加深；界面显示的是实际达到深度。native 下默认档通常远早于预算返回。

## 标注样式

`annotate()`（两手都要画）：

- 棋盘轮廓：青黄粗外框 + 四角角标（识别格网半格外扩）
- 我方 `pv.steps[0]`：红线 + 红圈起点 + 蓝圈终点
- 对方 `pv.steps[1]`：绿线 + 绿框起点 + 黄框终点
- 先对方后我方；颜色是走位样式，与棋子红黑无关（BGR）

## 常见修改

| 想改什么 | 去哪里 |
|---|---|
| 识别颜色 / 网格 | `recognize.py` 顶部 `_BOARD_*` 固定布局、`classify` 格心采样 |
| 兵种推断 | `infer_pieces()` + `piece_ocr`（**Torch CNN 优先**，回退模板）；`tools/export_piece_dataset.py` + `tools/train_piece_classifier.py` → `engine/models/piece_cls.pt` |
| 棋力 | `evaluate.py` PST；`search.py` 深度/剪枝；`book.py` |
| 分析 UI | `ui/app.py` |
| 对弈 UI / 难度 | `ui/play.py`（默认深度 7） |
| 对弈响应速度 | `ui/play.py` 的 `time_limit` 公式 |

## 测试

```bash
python test/test_xiangqi_engine.py   # 引擎回归
python test/test_piece_ocr.py        # 兵种 CNN / 开局 GT（需 torch + piece_cls.pt）
python test/test_ui_smoke.py         # 分析界面冒烟 → images/annotated/ui_window.png
python test/batch_run.py             # 批量识别+决策
```

## 已知限制

- 兵种：优先 Torch CNN（`piece_cls.pt`）；无 torch/无权重回退墨迹+彩色模板。仅近开局（双方 ≥14 子且底线完整）才位置种子；中局禁种子、禁 `@→车`。中局准确率依赖 `test/gt_boards/` 标注量，换皮需重导数据重训。
- 亮棕黑子与木纹难分；底线马位可用间隙修复补漏（已跳出则跳过），中局仍可能漏检占用。
- 棋盘用 `_BOARD_*` 相对固定布局（当前标定自 **782×1332** 竖屏）；其它分辨率/带叠加线的图会错位。识别在**原图像素**上做，界面缩放只影响显示。
- 搜索默认优先使用已编译的 Cython 原生核，未编译时回退纯 Python；深度 ≥8 时分析页预算很长，对弈页用更短预算保响应。
- 「自动监听」仅支持 Windows：置前标题含「手游助手」的窗口，按客户区屏幕截取（DirectX 窗口用 PrintWindow 常黑屏）；监听态占用不变时投票，合法一着时迁移兵种身份。
