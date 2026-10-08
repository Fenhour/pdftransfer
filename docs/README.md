# 项目文档

这里维护项目介绍、开发日志和开发协作 skill。

- [项目介绍](PROJECT_OVERVIEW.md)：目标、现有能力、代码结构与限制。
- [开发日志](DEVELOPMENT_LOG.md)：按实际开发日期记录改动、验证结果和后续事项，不补造历史。
- [会话提醒 skill](skills/context-handoff/SKILL.md)：在能确认即将发生第三次上下文压缩时，提醒更换会话并准备交接。

仓库根目录 `AGENTS.md` 接入协作规则；`.agents/skills/context-handoff/SKILL.md` 是 skill 的发现入口，具体规则维护在本目录，避免重复维护。

开发完成后更新开发日志。更换会话前，将当前进度、未完成事项、测试结论和相关提交记录到开发日志；新会话从这些文件继续。
