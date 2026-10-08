# pdftransfer

本地 PDF 内容提取工具：将 PDF 中的正文、标题、表格、图片、页码、页眉页脚和 OCR 文字分类导出为 Markdown 与 HTML，并生成按阅读顺序排列的汇总文档。HTML 支持编辑文字和表格并下载保存。

## 快速开始

Windows 上安装 Python 3.10 或以上版本（建议 3.12），然后：

1. 首次使用运行 `transfer_kenel/setup.bat` 安装依赖。
2. 运行 `transfer_kenel/start.bat` 打开中文图形界面。
3. 选择本地 PDF；默认结果保存在项目根目录 `output/`。

命令行示例（在项目根目录执行，先完成依赖安装）：

```powershell
Set-Location transfer_kenel
.\.venv\Scripts\python.exe -m pdf_extractor "C:\path\sample.pdf"
.\.venv\Scripts\python.exe -m pdf_extractor "C:\path\sample.pdf" --pages "1-3,5" --ocr off
```

依赖安装需要网络；PDF 提取和 OCR 在本机执行。虚拟环境不可跨电脑复制，如路径失效，应使用本机 Python 重新创建环境并安装依赖。

## 文档与代码

- [文档索引](docs/README.md)
- [项目介绍](docs/PROJECT_OVERVIEW.md)
- [开发日志](docs/DEVELOPMENT_LOG.md)
- [上下文压缩会话提醒 skill](docs/skills/context-handoff/SKILL.md)
- [详细使用说明](transfer_kenel/README.md)
- `transfer_kenel/pdf_extractor/`：转换核心、OCR、导出和图形界面。
- `transfer_kenel/tests/`：自动化回归测试。
- `待改进.txt`：已有问题记录，保留原始内容。

## 测试

在项目根目录执行：

```powershell
Set-Location transfer_kenel
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

## 版本管理范围

仓库提交源码、依赖声明、启动脚本、测试和文档；不提交本地虚拟环境、缓存、转换结果、临时文件、PDF 样例或密钥。测试会自行生成所需的 PDF，无需仓库中的本地样例。

图片导出为 PNG 位图，图中文字可提供 OCR/原生文字转录；当前不支持把图形拆成可编辑的矢量对象。复杂排版、扫描件与 OCR 结果仍需人工校对。
