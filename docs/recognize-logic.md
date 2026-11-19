# 棋子棋盘识别逻辑文档

代码位置：`engine/recognize.py`、`engine/piece_ocr.py`、`engine/piece_model.py`

## 1. 整体流水线

```
截图 (BGR；标定与主样张为 782×1332 竖屏)
  │
  ▼  [ROI 裁剪]
crop（上下留缓冲，不全贴棋盘）
  │
  ▼  [find_grid = 固定相对布局]
row0/col0/cell ← 整图比例常量 × 宽高（棋盘边线固定）
  │
  ▼  [detect_pieces 圆检]
连通域 → [(cx,cy,R|B,area)]
  │
  ▼  [occupy_from_pieces 吸附格点]
有圆 → @/#；无圆 → '.'（不做马位规则补漏）
  │
  ▼  [enrich_occupancy_cell_masks 格心二次确认]
圆检漏的子：格心红/暗黑 mask 明确有墨迹则补 @/#（不用亮棕，避免木纹）
  │
  ▼  [detect_our_side + infer_pieces]
下方九宫主将 → 我方；
近开局仅底线 @/# 种子（禁止炮/兵行硬覆盖）；
其余 → Torch CNN（优先）/ 模板 OCR → 配额分配 → 超配额清空
  │
  ▼  [监听稳定化]（analyze_array 且 path 含 auto/memory）
占用不变：最多 3 帧多数投票；
占用差分为合法一着：迁移兵种身份
  │
  ▼  [Analysis 封装]
对外 Analysis；annotate 画棋盘轮廓 + 网格 + 前两手
```

主路径占用 = **圆检吸附 + 格心二次确认**；`classify`（全盘格心采样）仅作诊断对照。

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

## 3. 棋子检测 `detect_pieces`（主占用入口）

HSV 红 / 暗黑两路连通域（亮棕不进 mask，避免整盘木纹）。过滤：面积、边长、圆形度；中心窗口比红黑 mask 定色。同色近距 NMS 合并碎裂域。

亮底黑子（如士）连通域常碎：不靠亮棕进 mask，而由后续 `enrich_occupancy_cell_masks` 用格心暗黑墨迹补占用。禁止马位规则假补。

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

## 5. 占用 `occupy_from_pieces`（主路径）/ `classify`（诊断）

### 5.1 圆 → 格吸附

每个圆吸附到最近格点（距离 ≤ `0.42 × cell`）；同格冲突保留面积更大者。无圆的格为 `.`。

### 5.1b 格心二次确认 `enrich_occupancy_cell_masks`

圆检漏子（常见：黑士底亮、连通域碎）时，对仍为空的格用心点红/暗黑 mask 再判一次；**不用**亮棕 mask。不做马位规则补漏。

### 5.2 `classify`（诊断对照）

固定格网下扫描 10×9 格心，窗口半径 `rad = 0.22 × cell`，按红/黑 mask 均值判 `@/#/.`。**不做**马位间隙修复。

### 5.3 输出

- `grid_raw`：10×9 的 `@/#/.`
- `centers`：90 个格心（crop 像素）
- `Analysis.hit`：非空格数（占用计数）

### 5.4 历史格心数值样例（classify）

| 格子 | rr | br | 判定 |
|------|------|------|------|
| 红车 | 0.451 | 0.000 | `@` |
| 暗底黑 | 0.000 | 0.292 | `#` |
| 空格 | 0.000 | 0.000 | `.` |
| 亮底黑子墨迹 | 0.000 | ≥0.05 | `#`（enrich 格心可补；整圆 CC 可能仍漏） |

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

### 7.1 近开局门控（位置种子）

仅当 **双方占用 ≥14** 且 **至少一方底线完整**（两端车位在、同色 ≥8）时，才对仍为 `@/#` 的底线格写字母；**中局禁止**位置种子。

底线定位仍用 `pick_backrow`（前/后 3 行优先 ≥7，再放宽到半场 ≥4）。

完整底线字母：

```python
RED_BACKROW   = ["R", "N", "B", "A", "K", "A", "B", "N", "R"]
BLACK_BACKROW = ["r", "n", "b", "a", "k", "a", "b", "n", "r"]
```

**已废除**炮行/兵行硬编码覆盖（旧 `fill_cannon_pawn` 会把中局真马写成兵）。缺角底线不套整行模板。超 `TYPE_QUOTA` 时多余格清空为 `.`，不改判成其他兵种。

### 7.2 单格兵种：Torch CNN（优先）+ 模板回退

代码：`engine/piece_model.py`、`engine/piece_ocr.py`。

残留 `@/#`：

1. `extract_cell_color` → 圆盘彩色块（模型内 resize 到 **48×48**）+ 第 4 通道 `is_red`
2. **优先** `engine/models/piece_cls.pt`（TinyCNN，7 类 `RNBAKCP`）**批量** softmax；按占用色写成大小写
3. 无 torch / 无权重 → 墨迹 NCC×0.55 + 彩色相关×0.45 模板回退（每类最多 8 份；車馬炮可跨色）
4. **车/马纠偏** `_refine_nr_with_templates`：CNN 顶为马（或车马分接近）且墨迹/彩色模板更像车时，抬高 `R/r`、压低 `N/n`（专治左侧车被认成马）
5. `assign_by_scores`：高置信锁格 + 配额贪心；将/士九宫、象田字硬约束  
   - 近开局（已多种子或占位近满）才叠加正「家位」先验  
   - 中局正家位 bonus=0，避免把中局底线强行判成 `RNBAK…`
6. `repair_by_ocr_conflict` / `repair_by_validation`（超配额无合法次优则清空为 `.`）
7. 仍剩 `@/#`：**弱默认兵/卒**（有配额），否则清空；最后 `enforce_type_quota`，**禁止兜成车/改判其它兵种**

### 7.3 自动监听稳定化（`analyze_array`）

路径为 `"<auto>"` / `"<memory>"` 或含 `auto` 时：

| 情况 | 行为 |
|------|------|
| 占用不变 | 最多 3 帧兵种多数投票 |
| 占用差分 = 同色一格消失 + 一格出现 | 把旧格兵种**迁移**到新格 |
| 吃子（异色替换） | 不迁移，信任新格识别 |
| 特效帧 | `fx_overlay` 已跳过，跟踪状态不更新 |

### 7.4 训练与数据

```bash
pip install torch          # 或 pip install -e ".[ml]" / ".[train]"
python tools/export_piece_dataset.py --clear
python tools/train_piece_classifier.py --epochs 25
# → engine/models/piece_cls.pt
```

数据来源（`export_piece_dataset`）：

- `engine/piece_templates/color/{red,black}/` 彩色模板
- `test/gt_boards/*.json` 人工全盘/局部标注
- 开局图位置先验自动切格（占用完整时）

模板回退 / 旧采集仍可用：`tools/harvest_piece_templates.py`。

### 7.5 已知缺陷

- CNN/模板均绑定当前 App 字形与 **782×1332** 布局；换皮或换分辨率需重测 `_BOARD_*` 并重导数据重训
- 圆检仍可能漏亮底黑子；`enrich` 靠格心暗黑墨迹补，墨迹极弱时仍可能漏占用
- 中局若缺 GT，模型易偏开局字形；往 `test/gt_boards/` 加标注后重训即可抬准
- 无 torch 时回退模板，中局易混对（马/车/炮/卒）仍存在

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
| `valid` / `violations` | | 合法性校验结果 |

### 8.2 `analyze_array(img, path="<memory>")` → Analysis

与 `analyze_image` 同一套 **ROI → 固定格网 → detect_pieces → occupy_from_pieces → enrich → infer → 校验**；入参是 **BGR 原图像素**（不读盘，也不使用界面缩放图）。

`path` 为 `"<auto>"` / `"<memory>"`（或路径含 `auto`）时启用 §7.3 稳定化。`ui/app.py` 自动监听每秒一帧走这条路径。默认**不落盘**；「自动截图」拉杆 >0 时写入 `images/auto/` 并环形保留。

### 8.3 `annotate(analysis, decision)` → BGR ndarray

在原图上标注（**前两手都要画**；契约前提：我方先走）：
1. 兵种文字（格角，字色随红/黑；不再画棋子外包黑框）
2. 合法性违规格：红 X
3. **棋盘轮廓**：青黄色粗外框 + 四角短角标（半格外扩，便于目检）
4. 网格线（灰色，可关）；楚河汉界加橙色中线
5. **第 2 手（对方）** `pv.steps[1]`：绿线 + 绿框起点 + 黄框终点（先画）
6. **第 1 手（我方）** `pv.steps[0]`：红线 + 红圈起点 + 蓝圈终点（后画，盖上层）
7. 着法文字说明（左上角半透明黑底）

走位颜色是标注样式，与棋子红黑无关。`from`/`to` 已是图像方向，不要再镜像。

---

## 9. 识别失败模式总结

| 现象 | 根因 | 出现条件 | 影响 |
|------|------|------|------|
| 亮底黑子圆检漏 | 底亮不进 black CC；墨迹碎 | 士等亮底黑子 | 靠 `enrich` 格心暗黑墨迹补；仍可能漏 |
| 中局兵种错（马↔车↔炮） | CNN 缺中局 GT 或模板回退 | 未装 torch / 中局样本少 | 局面扭曲；补 GT 重训 |
| 我方颜色判反 | 双方九宫主将都漏检或同色误检 | 极端残缺识别 | `view_side` 错 → 决策/标注「我方」反了 |
| 固定格网整体偏移 | App 改了棋盘边距/分辨率布局 | 换皮、非 782×1332、叠加线截图 | 重测 `_BOARD_*`；错误分辨率图勿当 GT |
| 开局家位先验污染中局 | 旧逻辑；现已按门控关闭正 bonus | — | 已修：中局 `opening_prior=False` |
| 空位被规则补成子 | 旧马位间隙 / 炮兵行硬种子 | 中局马已跳出等 | 已废：无圆不造子；禁止 `fill_cannon_pawn` |

---

## 10. 迭代踩坑记录

| 尝试 | 结论 | 原因 |
|------|------|------|
| HoughCircles 直接在 detect_pieces 里加补漏 | ❌ 废弃 | 产生假圆，污染占用 |
| 底线补漏「底线 ≥7 枚就把空格全补上」 | ❌ 废弃 | 残局底线本可空；假阳性不可接受 |
| 马位间隙修复 `_repair_backrow_horse_gaps` 主路径 | ❌ 移出主路径 | 空马位假占位 → 配额满 → 真马被改判 |
| 近开局 `fill_cannon_pawn` 硬覆盖 | ❌ 已废除 | 中局真马格被写成兵，OCR 看不到 |
| 霍夫圆 + 底线位置先验 | ❌ 废弃 | 假阳性 |
| KMeans 极坐标展开 | ❌ 废弃 | 算力大，亮棕仍无效 |
| 棋子聚类搜网格 | ❌ 已移除 | 改为 `_BOARD_*` 固定相对布局 |
| 中局仍套底线/炮兵种子 + `@→R` 兜底 | ❌ 已废弃 | 中局兵种全乱、车配额爆炸 |
| 纯 HSV 格心占用作主路径 | ❌ 已替换 | 易误占空格；改为圆检优先 |
| **圆检吸附 + 格心 enrich** | ✅ 当前占用主路径 | 无圆不造子；enrich 只认红/暗黑墨迹 |
| **近开局仅底线 @/# 种子** | ✅ 保留（门控） | 不覆盖已有字母；无炮/兵行硬写 |
| **墨迹 + 彩色模板 + 校验重判** | ✅ 回退路径 | 无 torch 时使用 |
| **Torch TinyCNN + 配额分配 + 监听迁移** | ✅ 兵种主路径 | 权重 `engine/models/piece_cls.pt` |

**结论**：固定竖屏分辨率下格网可写死；占用先圆后格心墨迹确认；兵种以 CNN 为主、模板为辅；规则层只做近开局底线种子与配额/合法性/时序稳定。换皮/换分辨率：重测 `_BOARD_*` + 重采/重训。

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
