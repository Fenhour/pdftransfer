# 当前功能概况（2026-10-29整理版）

本副本保留阅读提取器与网页编辑器；共享V3默认采用本机Python常驻子进程，CPU6线程、MKLDNN开启。本仓库包含核心源码；依赖环境及模型需自行准备。

| 路线 | 行为 | 主要代码 |
| --- | --- | --- |
| 阅读HTML/MD | 可靠原生文字/表格直接提取，普通原生图片保图；扫描页V3；保留原图参考 | `engine.py`、`structure_reading.py`、`tables.py`、`export.py` |
| OCR画布 | 导入、编辑、保存、PDF/图片/JSON等同源导出；旧JSON安全迁入 | `app.py`、`parser.py`、`models.py`、`exporter.py`、`legacy*.py` |
| 当前功能四 | plus-L与BlockLayout分类/区域；共享det/rec、表格恢复；唯一可见归属；源行与格内锚点测量；可靠Region内有界适配 | `rebuild_pipeline.py`、`adaptive_reconstruction.py`、`source_*.py`、`spatial_relations.py`、`adaptive_layout.py`、`regional_layout.py`、`adaptive_export.py` |
| 05检测预览 | 只运行选定的det、plus-L或BlockLayout，生成编辑器场景 | `backend/detection_preview.py`、`services/tmp/detection_preview.py` |

当前运行版本：`v2.2026.10.28-internal-structure`；关系`07-spatial-v2`；布局`08-regional-v4-internal-structure`；内部锚点`source-lines-and-cell-anchors-v1`。2026-10-29是整理日期，不代表业务算法新版本。

功能四保留源页一对一，原位无冲突不移动；可靠Region内局部调整，必要时按有限比例整体缩放。旧保存项目仍可能走单栏兼容分支。图片内部OCR作为转录元数据，保持原图，不再可见叠写。复杂表、图片/结构歧义保图并提示复核；表格恢复必须同时保留原结构和原图证据。

`legacy*.py`、`text_pdf.py`、`rebuild_*.py`等不能按名称直接删除：应用导入、导出、旧项目迁移、显式legacy或测试仍依赖。第三项和第四项的数据模型不同，不能把重建项目JSON当画布JSON直接导入。

本副本没有TaCo、DINOv2+LoRA字体分类器，也没有新增样式模型。字体近似、OCR空格/符号、复杂表、公式和WPS编辑保存重开仍需人工验收。结构校验、测试通过与准确率是不同结论。

目录、安装、输入输出约定见 [根说明](../README.md)，具体源布局规则见 [重建规则](REBUILD_RULES.md)。


## 后续方向（尚未实施）

未来研究见 [位置、样式与分层残差对齐](FUTURE_DIRECTION_2026-10-29.md)：补充OCR几何/样式、稳定ID和父级局部坐标，研究顺序规则及块内/Region两层残差修正；字体模型和整体平移/缩放/释放留白顺序均为待验证候选。当前运行算法和模型不变。
