# 棋子棋盘识别逻辑文档

代码位置：`engine/recognize.py`

## 1. 整体流水线

```
截图 (BGR；当前 images/* 多为 782×1332)
  │
  ▼  [ROI 裁剪]
crop（上下留缓冲，不全贴棋盘）
  │
  ▼  [find_grid = 固定相对布局]
row0/col0/cell ← 整图比例常量 × 宽高（可选 pieces 微调原点）
  │
  ▼  [classify 格心采样]
90 格 → grid_raw (@/#/.)；空格快路径跳过
  │
  ▼  [detect_our_side + infer_pieces]
下方九宫主将 → 我方；近开局种子 + 墨迹/彩色 OCR + 校验重判
  │
  ▼  [可选时序投票]（analyze_array / 自动监听）
占用不变时兵种最多 3 帧多数投票
  │
  ▼  [Analysis 封装]
对外 Analysis；annotate 画棋盘轮廓 + 网格 + 前两手
```

`detect_pieces`（连通域）仍保留，供诊断或 `find_grid(..., pieces=...)` 微调；主识别路径不再调用。当前 `images/` 根目录截图均为 **782×1332**。

---

## 2. ROI 裁剪

参考分辨率 `_REF_W=720, _REF_H=1280`，按比例从原图裁取棋盘区域：

| 参数 | 值 | 说明 |
|------|------|------|
| `_ROI_Y0` | 101/1280 ≈ 0.079 | 裁剪顶部起点 |
| `_ROI_Y1` | 1264/1280 ≈ 0.987 | 裁剪底部终点 |
| `_TOP_UI` | 0.075 | 可选 refine 时屏蔽顶部杂点 |
| `_BOTTOM_UI` | 0.92 | 可选 refine 时屏蔽底部杂点 |
| `_BOARD_ROW0_ABS` | 0.25443 | 首格中心 Y / 图高（images/* 标定） |
| `_BOARD_COL0_ABS` | 0.07344 | 首格中心 X / 图宽 |
| `_BOARD_CELL_W` | 0.09782 | 格宽 / 图宽（替换旧 76/720） |

裁剪后只保留 Y ∈ `[_ROI_Y0*H, _ROI_Y1*H]`、X ∈ `[0, W]` 的区域。

---

## 3. 棋子检测 `detect_pieces`（诊断 / 可选 refine）

主路径已改为格心 `classify`，本函数仅在调试或 `find_grid(..., pieces=...)` 时使用。

HSV 仍分红 / 暗黑两路连通域（亮棕不进 mask，避免整盘木纹）。过滤：面积、边长、圆形度；中心窗口比红黑 mask 定色。亮棕漏检靠 `classify` 后的 `_repair_backrow_horse_gaps`（底线马位间隙修复；马已跳出则禁止补假马）。

---

## 4. 网格定位 `find_grid`（固定相对布局）

### 4.1 核心思路

棋盘在 App 截图中的相对位置固定，按整图宽高缩放：

```python
cell  = _BOARD_CELL_W * full_w
col0  = _BOARD_COL0_ABS * full_w
row0  = _BOARD_ROW0_ABS * full_h - crop_y0   # 转到 ROI crop 坐标
```

标定来源：`images/` 根目录（不含子目录）782×1332 截图、原搜索路径 `hit≥20` 的中位数。换皮肤或改棋盘边距后需重测这三项。

### 4.2 可选原点微调

若传入 `pieces`（连通域中心），用 `_refine_grid_origin` 按最近格点反推 `row0/col0`；主路径不传 pieces，直接用固定值。

### 4.3 与旧聚类搜索的关系

旧「棋子聚类 + 轴向暴力搜 origin」已移除。残局侧栏导致 `76/720` 格宽偏大、网格平移的问题随之消失。

---

## 5. 格子颜色分类 `classify`（主占用检测）

### 5.1 逐格扫描

固定格网下扫描 10×9 格心，窗口半径 `rad = 0.22 × cell`：

```python
for r in range(10):
    cy = row0 + r * cell
    for c in range(9):
        cx = col0 + c * cell
```

### 5.2 判定逻辑

| 条件 | 结果 |
|------|------|
| `rr < 0.015 AND br < 0.015` | `.`（空格快路径） |
| `rr > 0.04 AND rr > br × 1.2` | `@`（红） |
| `br > 0.05 AND br > rr × 1.2` | `#`（黑） |
| `rr > br × 0.8 AND rr > 0.02` | `@`（红） |
| `br > rr × 0.8 AND br > 0.02` | `#`（黑） |
| 其余 | `.`（空） |

其中 `rr`/`br` 为窗口内红/黑 mask 均值。随后 `_repair_backrow_horse_gaps` 补底线亮棕马漏检。

### 5.3 输出

- `grid_raw`：10×9 的 `@/#/.`
- `centers`：90 个格心（crop 像素）
- `Analysis.hit`：非空格数（占用计数）

### 5.4 典型格心数值（历史样例）

| 格子 | rr | br | 判定 |
|------|------|------|------|
| 红车 | 0.451 | 0.000 | `@` |
| 暗底黑 | 0.000 | 0.292 | `#` |
| 空格 | 0.000 | 0.000 | `.` |
| 亮棕黑 | 0.000 | 0.000 | `.`（靠底线补漏） |

---

## 6. 我方颜色 `detect_our_side`（≠ 固定红色）

**契约**：图下方 = 我方 = `view_side` / `Analysis.view_side`，由下方九宫主将颜色决定，可能是红也可能是黑。

```python
detect_both_sides(grid_raw) → (our_side, opp_side)
  # 下半九宫 row 7-9, col 3-5 找主将：@ → red，# → black
  # 下半漏检则用上半九宫对立面；两边都没有则用半场 row5-9 的 @/# 多数兜底
detect_our_side(grid_raw) → "red" | "black" | None
detect_red_bottom(grid_raw) → bool   # 兼容旧调用：our_side == "red"
```

`analyze_array` / `analyze_image` 在推断兵种前调用 `detect_our_side`，写入 `Analysis.side` / `view_side` / `red_bottom`。

残局只要九宫主将还在就可靠；不要再按「上下半场红黑子数」判方向（旧逻辑已废弃）。

---

## 7. 兵种推断 `infer_pieces`

### 7.1 定位底线行

`pick_backrow(counts, prefer_top)` 找到红 / 黑方底线：
1. 优先在前 3 行（或后 3 行）里找有 ≥7 枚棋子的
2. 找不到就扫描前 5 行（或后 5 行），取棋子最多且 ≥4 枚的
3. 再找不到 → 该方底线返回 `None`

### 7.2 底线兵种硬编码

```python
RED_BACKROW   = ["R", "N", "B", "A", "K", "A", "B", "N", "R"]
BLACK_BACKROW = ["r", "n", "b", "a", "k", "a", "b", "n", "r"]
```
底线行上所有 `@` / `#` 按位置映射到底线字母。

### 7.3 炮 / 兵行推断

找到底线后，往对方方向：
- 底线 + 2 行 → 炮行（`C/c`）
- 底线 + 3 行 → 兵行（`P/p`）
- 其余中局残留 `@/#` → 兜底填 `R/r`

`step = 1 if backrow <= 4 else -1`，即底线在上方向下推，底线在下方向上推。

### 7.4 单格兵种 OCR / 模板（`engine/piece_ocr.py`）

近开局可位置种子；其余残留 `@/#` 走 **墨迹 + 彩色** 融合匹配：

1. `extract_cell_ink`：``0.32×cell`` 裁格 → 圆盘 → 正方形补边 → 40×40 二值  
2. `extract_cell_color`：``0.36×cell`` 圆盘彩色块 → 40×40 BGR  
3. `score_cell_types`：墨迹 NCC×0.55 + 彩色相关×0.45（无彩色模板则纯墨迹）；每兵种最多较新 8 份；車馬炮可跨色  
4. `assign_by_scores`：位置先验；top2 分差 &lt;0.08 时先验 ×1.6  
5. `repair_by_validation`：`validate_grid` 失败时（如 R 超配额、士出九宫）按候选次优重判  
6. 仍剩 `@/#` 兜底 `R/r`

近开局种子：完整底线（两端车 + ≥8）全套；或真底线 row0/9 且 ≥7 时只种已占格 + 经典炮/兵位。

自动监听：`analyze_array` 在占用不变时对兵种做最多 3 帧多数投票。

模板目录：
- 墨迹 `piece_templates/{red,black}/`
- 彩色 `piece_templates/color/{red,black}/`

采集：`python tools/harvest_piece_templates.py [--color-only|--ink-only] [--clear]`

### 7.5 已知缺陷

- 模板绑定当前 App 字形与 **782×1332** 布局；换皮/换分辨率需重采  
- 亮棕漏检仍会导致占位缺失  
- 彩色模板过多时匹配变慢（已限制每类 8 份）

---

## 8. 对外接口

### 8.1 `analyze_image(path)` → Analysis

读盘后走完整管线。返回 `Analysis` 对象，常用属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `grid_raw` | `List[List[str]]` | 原始 `@/#/.` 占位 |
| `grid` | `List[List[str]]` | 推断后的兵种字母 |
| `grid_str` | `str` | 9 列连续、10 行换行的字符串，直接给引擎 |
| `red_count` / `black_count` | `int` | 红 / 黑棋子数量 |
| `side` | `str` | 我方颜色（下半九宫主将） |
| `red_bottom` | `bool` | 兼容字段：`side == "red"` |
| `view_side` | `str` | 图下方是谁 = 我方（与 `side` 相同） |
| `row0, col0, cell` | `float` | 网格参数（crop 坐标系） |
| `cell_center(row, col)` | `(float, float)` | 整图坐标系的格子中心 |

### 8.2 `analyze_array(img, path="<memory>")` → Analysis

与 `analyze_image` 同一套 **ROI → 固定格网 → classify → infer → 校验**；入参是 **BGR 原图像素**（不读盘，也不使用界面缩放图）。

`path` 为 `"<auto>"` / `"<memory>"`（或路径含 `auto`）时启用兵种时序投票。`ui/app.py` 自动监听每秒一帧走这条路径，避免频繁磁盘 I/O。默认**不落盘**；「自动截图」拉杆 >0 时写入 `images/auto/auto_*.png` 并环形保留。标注用内存里的 `Analysis.image`。

### 8.3 `annotate(analysis, decision)` → BGR ndarray

在原图上标注（**前两手都要画**；契约前提：我方先走）：
1. **所有检出棋子**：黑色矩形描边（BGR `0,0,0`，边长 `0.92×cell`）
2. 兵种文字（格角，字色随红/黑）
3. 合法性违规格：红 X
4. **棋盘轮廓**：青黄色粗外框 + 四角短角标（半格外扩，便于目检）
5. 网格线（灰色，可关）；楚河汉界加橙色中线
6. **第 2 手（对方）** `pv.steps[1]`：绿线 + 绿框起点 + 黄框终点（先画）
7. **第 1 手（我方）** `pv.steps[0]`：红线 + 红圈起点 + 蓝圈终点（后画，盖上层）
8. 着法文字说明（左上角半透明黑底）

走位颜色是标注样式，与棋子红黑无关。`from`/`to` 已是图像方向，不要再镜像。

---

## 9. 识别失败模式总结

| 现象 | 根因 | 出现条件 | 影响 |
|------|------|------|------|
| 亮棕底色黑子漏检 | 与木纹 HSV 不可分 | 屏幕偏亮 / 压缩 | 底线马位可用 `_repair_backrow_horse_gaps` 补；中局仍可能漏 |
| 底线兵种推断错位 | 底线漏检导致底线行没被识别出来 | 底线漏检 ≥3 子 | 兵种位置全乱 |
| 我方颜色判反 | 双方九宫主将都漏检或同色误检 | 极端残缺识别 | `view_side` 错 → 决策/标注「我方」反了 |
| 固定格网整体偏移 | App 改了棋盘边距/分辨率布局 | 换皮或新模拟器边框 | 重测 `_BOARD_*` 三项 |
| 棋子数量远超 32 | 底线补漏等启发式逻辑错误补空格 | 已删除，但中局漏检仍会被 `infer_pieces` 兜底填 `R/r` 放大 | 引擎局面信息扭曲 |

---

## 10. 本次迭代踩坑记录

| 尝试 | 结论 | 原因 |
|------|------|------|
| HoughCircles 直接在 detect_pieces 里加补漏 | ❌ 废弃 | 产生假圆；主路径已不用连通域定位 |
| 底线补漏 "底线 ≥7 枚就把空格全补上" | ❌ 废弃 | 象棋底线并不总是满 9 枚（残局跳马走后真的有空格），Vstd 阈值 15-17 和确定空格无法区分 |
| 霍夫圆 + 底线位置先验组合 | ❌ 废弃 | 同上，底线不是满的就不该补 |
| KMeans 极坐标展开检测棋子轮廓 | ❌ 废弃 | 计算量大，对亮棕底色依然无效 |
| **纯 HSV mask + 底线位置先验（infer_pieces 层）** | ✅ 保留 | 占用检测仍靠格心 HSV；兵种另有模板 |
| **棋子聚类搜网格** | ❌ 已移除 | 改为 `_BOARD_*` 固定相对布局 |
| **墨迹 + 彩色模板 + 校验重判** | ✅ 当前 | 专治马/车易混与 R 超配额 |

**结论**：固定分辨率下格网可写死；兵种靠模板+先验+校验回退。亮棕漏检仍可能发生——与其假阳性补子，不如接受个别漏检，由引擎在真实占用上搜索。换皮/换分辨率需重测 `_BOARD_*` 并重采模板。

---

## 11. 坐标系契约（不可打破）

```
图像 / grid：row 0 = 图最上，row 9 = 图最下；col 0 = 图左
我方 = view_side = 图下方（下半九宫主将颜色）
内部引擎：row 0 = 黑底，row 9 = 红底

decide_position(grid=..., view_side=...)：
  view_side == black 时先 mirror_grid（旋转 180°）再解析
  输出 best_move.from/to、pv.steps 已转回图像方向

截图分析调用：
  side=analysis.view_side, view_side=analysis.view_side  # 我方先走
```

`annotate` / UI 推演列表直接用输出坐标与 `pv.sides`；`sides[i] == view_side` 标「我方」，否则「对方」。
