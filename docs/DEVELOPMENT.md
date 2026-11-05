# 开发与验证

先在项目根目录安装运行依赖与模型，再安装测试依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

从项目根目录执行：

```powershell
Push-Location transfer_kenel
..\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
Pop-Location
Push-Location transfer_kernel_ocr
..\.venv\Scripts\python.exe -X utf8 -m pytest -q
Pop-Location
```

前端语法与进度行为检查需要 Node.js：

```powershell
node --check transfer_kernel_ocr/frontend/app.js
node --check transfer_kernel_ocr/frontend/queue.js
node --test transfer_kernel_ocr/tests/test_progress_frontend.cjs
```

API 文档位于运行服务的 `/api/docs`。公开转换为 `/api/convert/reading` 的 HTML / Markdown，以及 `/api/conversion-jobs` 的可编辑 PDF 队列；`/api/convert/rebuild` 提供同一 PDF 转换的同步调用。队列的页面预览和结果只通过对应任务下载地址发布。

测试使用临时生成的文字、图片和表格。运行通过、字符数量与缓存回放都不等同于真实 OCR 准确率、复杂表格验收或办公软件编辑验收。改变识别模型或后处理时，应另建人工真值并记录覆盖范围。

提交前执行 `git diff --check`，检查暂存区文件名与内容；不要提交输入、转换结果、用户文档、个人路径、密钥、权重、虚拟环境或缓存。
