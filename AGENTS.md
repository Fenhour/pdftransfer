# pdftransfer 开发协作规则

- 开始项目开发前阅读 `docs/PROJECT_OVERVIEW.md`、`docs/DEVELOPMENT_LOG.md` 和 `transfer_kenel/README.md`；先检查 Git 状态，保留用户已有改动。
- 项目介绍和开发日志维护在 `docs/`。完成有意义的改动后记录实际日期、改动内容、验证结果与未完成事项，不虚构测试或上传状态。
- 本项目会话提醒规则见 `docs/skills/context-handoff/SKILL.md`，开始会话时读取。用户要求：能确认即将第三次上下文压缩时提醒更换会话，并整理交接信息。无运行时计数/预警时，不猜测或保证自动精确触发。
- 转换器源代码位于 `transfer_kenel/pdf_extractor/`，测试位于 `transfer_kenel/tests/`。保留现有 `transfer_kenel` 目录名，非任务需要不重命名。
- 从 `transfer_kenel` 执行 `python -X utf8 -m unittest discover -s tests -v`，优先使用安装依赖后的项目虚拟环境。环境不可用时报告原因，不将环境失败等同于业务测试失败。
- 不提交 `.venv`、缓存、`output`、临时文件、PDF 输入或密钥；本地样例不是公开发布授权。不要为了测试、日志或交接自动提交/推送，除非用户明确要求。
