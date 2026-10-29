# 开发协作规则

- 开始工作先检查 Git 状态，保留已有修改；阅读 README.md、docs/PROJECT_OVERVIEW.md、docs/REBUILD_RULES.md。
- 阅读提取器位于 transfer_kenel/pdf_extractor/，不要改动现有目录名。
- 在 transfer_kenel 执行 `python -X utf8 -m unittest discover -s tests -v`，优先使用项目虚拟环境。
- 编辑器验证方法见 docs/VALIDATION.md；环境失败与业务失败分开说明，不虚构测试结果。
- 不提交本地输入、输出、日志、历史记录、个人绝对路径、模型权重、环境或密钥。
- 模型引入前建立人工真值；运行成功和字符数不代表准确率。
- 表格保留原生结构与原图参考；复杂表或图表回退必须如实说明。
- 普通本地提取不得无故变成必须调用云服务或大语言模型。
