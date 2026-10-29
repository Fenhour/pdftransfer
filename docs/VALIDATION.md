# 开发验证

安装依赖后，在 transfer_kenel 运行：

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

编辑器先安装 requirements-dev.txt，在 transfer_kernel_ocr 运行：

```powershell
$env:PDFTRANSFER_RECOGNITION='legacy'
.venv\Scripts\python.exe -X utf8 -m pytest -q
Remove-Item Env:PDFTRANSFER_RECOGNITION
```

全套兼容测试使用 legacy，V3 专项自行显式选择或 mock；部分环境边界测试仍要求安装共享 V3 解释器。正常应用默认 V3。进度组件可运行 `node --test transfer_kernel_ocr/tests/test_progress_frontend.cjs`。

真实模型推理、人工准确率、字体与编辑器兼容性需分别验收。私有样例不随仓库提供，依赖本地样例的测试可能跳过。历史运行结果不构成当前环境的通过证据。
