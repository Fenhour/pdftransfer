# pdftransfer

本地 PDF 阅读提取、OCR 页面编辑与结构化重建。当前版本见 VERSION.json。

- 阅读提取器：`transfer_kenel/pdf_extractor/`，输出 HTML / Markdown。
- 网页编辑器：`transfer_kernel_ocr/backend/` 与 `frontend/`。
- 共享本机 V3 推理：`services/structure_v3/`；检测预览还依赖 `services/tmp/detection_preview.py`。

## 安装与运行

建议 Python 3.12。Windows 建议将仓库及模型放在纯 ASCII 路径，避免 Paddle 读取中文路径失败。

1. 阅读功能运行 `transfer_kenel/setup.bat`。
2. 网页功能运行 `transfer_kernel_ocr/setup.bat`。
3. 扫描识别环境从仓库根执行 `powershell -ExecutionPolicy Bypass -File services/structure_v3/setup-native.ps1`。
4. 按 [模型准备](docs/MODELS.md) 配置本地模型。
5. 网页执行 `transfer_kernel_ocr/start.bat`，阅读界面执行 `transfer_kenel/start.bat`。

安装依赖需要网络；默认模型推理在本机 CPU 运行，不依赖云服务或大语言模型。模型权重、输入样例、输出、日志、环境及私人历史不随 Git 提供。

## 协作

先阅读 [项目概况](docs/PROJECT_OVERVIEW.md)、[重建规则](docs/REBUILD_RULES.md)、[未来方向](docs/FUTURE_DIRECTION_2026-10-29.md) 和 [验证说明](docs/VALIDATION.md)。

复杂表格、字体、公式和 OCR 仍需人工验收；保图回退不代表结构恢复。输入输出默认由应用在本地创建。提交前检查暂存区，勿提交真实文档、识别结果、日志、个人路径或密钥。
