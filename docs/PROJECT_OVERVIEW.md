# 项目介绍

## 目标与现状

pdftransfer 是一个中文界面的本地 PDF 内容提取器，当前代码版本为 `1.0.0`（`transfer_kenel/pdf_extractor/__init__.py`）。面向从 PDF 中提取、分类、阅读及进一步编辑内容的工作流；不是通用 PDF 编辑器，也没有承诺无损重建 PDF 版式。

## 已有能力

- 分类识别正文、标题、表格、图片、页码、页眉页脚、OCR 文本。
- 生成 Markdown 和 HTML 两套输出，包含分类文档与顺序汇总；仅为识别出的类型创建目录。
- HTML 支持文字、表格编辑并下载修改后的文件；修改不会自动同步到其他分类文档或 Markdown。
- 输出包含图片资源和源 PDF 的相对链接；重复运行创建新名称，不覆盖旧结果。
- 基于文字坐标处理多栏顺序、章节层级、段落、代码符号与缩进；组合图提取尽量合并图像及图题。
- 支持自动 OCR、所有图片 OCR 或关闭 OCR；中英文识别在本机执行。
- 支持物理页码筛选、密码交互输入、表格策略、渲染 DPI、阅读顺序及取消。

## 技术组成

- Python 3.10+，建议使用 Python 3.12。
- PyMuPDF：PDF 文字、图像、图形、表格及页面处理。
- rapidocr-onnxruntime：本地 OCR。
- Tkinter：中文桌面界面。
- unittest：回归测试。

依赖范围以 `transfer_kenel/requirements.txt` 为准；本机环境与转换结果不纳入版本管理。

## 目录与职责

| 路径 | 职责 |
| --- | --- |
| `transfer_kenel/pdf_extractor/__main__.py` | 命令行入口；不传 PDF 时启动界面 |
| `transfer_kenel/pdf_extractor/gui.py` | 桌面界面与任务交互 |
| `transfer_kenel/pdf_extractor/engine.py` | 提取和导出的流程组织 |
| `transfer_kenel/pdf_extractor/model.py` | 数据模型、选项与页码解析 |
| `transfer_kenel/pdf_extractor/layout.py` | 类型分类、版面与阅读顺序 |
| `transfer_kenel/pdf_extractor/text.py` | 文字与代码排版处理 |
| `transfer_kenel/pdf_extractor/figures.py` | 图片与组合图提取 |
| `transfer_kenel/pdf_extractor/ocr.py` | OCR 和扫描表格处理 |
| `transfer_kenel/pdf_extractor/export.py` | Markdown、HTML 与资源导出 |
| `transfer_kenel/tests/test_extractor.py` | 自动生成测试输入的回归测试 |
| `docs/` | 项目介绍、开发日志、会话提醒 skill |
| `output/` | 本地生成结果，不提交 Git |

保留现有目录名称 `transfer_kenel`，本次归档不做重命名或转换逻辑变更。

## 已知限制与后续方向

原始问题记录见根目录 `待改进.txt`：

1. 图片内文字识别准确性、重复转录与文字顺序需要继续改进。
2. 探索将图片内图案提取为可编辑对象；当前输出仍为 PNG 位图，不支持软件原型/PPT 式对象编辑。
3. 梳理常见 PDF 数据类型及提取结果的编辑加工方式。
4. 原记录指出案例章节 2.1、2.2 的格式问题，以及第五页图片提取文本重复；本次未针对这些案例重新验证或修复。

复杂多栏、跨页表格、公式、手写体、低清扫描及无图题组合图仍需校对。OCR 可能误识符号，不能保证原样恢复。
