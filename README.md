# pdftransfer

基于 PP-StructureV3 的 PDF 转换器，支持扫描件与文本层 PDF 的版面分析、文字识别、表格结构化提取与 Markdown/PDF 重建，并针对多栏、标题层级、文本对齐与重叠框去重做了后处理优化。

| 功能 | 输出 |
| --- | --- |
| 可编辑 PDF → HTML | HTML、图片资源与源 PDF 的 ZIP 包 |
| 可编辑 PDF → Markdown | Markdown、图片资源与源 PDF 的 ZIP 包 |
| 图片 / 扫描 PDF → 可编辑 PDF | 可搜索、可选择和可编辑的文字，保留版面与原图参考 |

## 安装

支持 Windows 10 / 11，使用 Python 3.12（64位）和 Git。首次安装需要联网下载依赖与官方模型；转换在本机 CPU 运行，无需 Docker、云服务或 API 密钥。

如果尚未安装 Python 和 Git，在 PowerShell 执行下面两行，完成后关闭并重新打开 PowerShell：

```powershell
winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
winget install --id Git.Git -e --accept-package-agreements --accept-source-agreements
```

复制下面整段到 PowerShell。安装目录使用纯英文路径，避免 Windows 下模型读取中文路径失败；已有同名目录时请换一个空目录。

```powershell
git clone --branch codex/core-collaboration-20261029 https://github.com/Fenhour/pdftransfer.git C:\pdftransfer
Set-Location C:\pdftransfer
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

安装脚本会创建应用与 OCR 两个独立环境，下载版面、文字与表格模型，并生成独立的 CPU 优化权重。重复执行会复用已完成的环境和模型；原始权重保留。

## 启动与使用

安装后每次启动只需：

```powershell
Set-Location C:\pdftransfer
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1
```

浏览器会自动打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)。选择转换方式、上传文件并下载结果。每次最多30个文件、总计256 MB，可指定物理页码；扫描转换最多1000页/帧，并提供后台队列与逐页预览。支持 PDF、PNG、JPEG、多页 TIFF、WebP 和 BMP。

HTML / Markdown 转换读取 PDF 原生文字，不额外执行扫描 OCR。图片或扫描 PDF 请选择第三项。扫描重建结果中的文字、复杂表格、公式和字体仍需校对；无法可靠恢复的区域保留原图。遇到未解决的重叠布局会保留当前文字和位置、标记待复核，并继续其它页。

在服务终端按 `Ctrl+C` 停止；停止前请等待队列完成。若8765端口已被占用，可复制：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1 -Port 8766
```

输入保存在 `transfer_kernel_ocr/input/`，转换结果保存在 `transfer_kernel_ocr/output/`。这些文件、模型、环境和日志不纳入 Git。部署失败时修正终端中的报错后重新执行安装命令；模型完整性可单独检查：

```powershell
.\services\structure_v3\.venv\Scripts\python.exe -X utf8 services/structure_v3/prepare_models.py --check
```

[转换流程](docs/ARCHITECTURE.md) · [开发与验证](docs/DEVELOPMENT.md)
