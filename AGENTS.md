# 开发协作规则

- 先检查 Git 状态并保留已有改动；阅读 README.md、docs/ARCHITECTURE.md 和 docs/DEVELOPMENT.md。
- 阅读提取代码位于 transfer_kenel/pdf_extractor/，保留既有 transfer_kenel 目录名。
- 转换只公开原生 PDF 到 HTML / Markdown、图片及扫描 PDF 到可编辑 PDF；修改入口时同时核对前端、API和测试。
- 按 docs/DEVELOPMENT.md 执行相关测试；环境失败与业务失败分开报告，不虚构验证或上传状态。
- 不提交输入、输出、日志、私人历史、个人路径、模型权重、环境或密钥。未经用户明确要求不提交或推送。
- 保留表格结构与原图参考；未解决的几何问题如实记录。模型或识别策略变更须以人工真值验证，运行成功不代表准确率。
- 保持本地运行路径，不引入不必要的云服务依赖。
