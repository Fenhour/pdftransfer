# 本机 V3 运行时

从根目录执行 setup-native.ps1 安装独立环境。默认 stdio 常驻子进程，无需 HTTP 或 Docker。配置见 native_config.json，模型需单独准备，见 ../../docs/MODELS.md。可用 PDFTRANSFER_V3_MODEL_DIR、PDFTRANSFER_V3_CONFIG、PDFTRANSFER_V3_PYTHON、PDFTRANSFER_V3_RESULTS 覆盖本机设置。性能日志仅本地生成，不提交。
