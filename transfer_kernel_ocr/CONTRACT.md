# 当前接口与数据边界（2026-10-29）

以 `backend/app.py`、`models.py` 和真实 `/api/docs` 为接口定义；本文件只保留两类当前模型及主要入口，避免历史策略互相覆盖。

| 接口 | 作用 |
| --- | --- |
| `GET /api/health` | 健康、数据根、实际版本和识别后端 |
| `POST /api/convert/reading` | PDF阅读ZIP，format=html/md |
| `POST /api/documents` | 画布导入并持久化场景 |
| `POST /api/convert/rebuild` | 当前功能四，format=pdf/html/md |
| `POST /api/detection-preview` | PDF单模型检测，detector=det/layout/region，所选1–100页 |
| `POST /api/progress`、`GET /api/progress/{id}` | 实际操作阶段进度；请求头X-Progress-ID关联 |

页码为从1开始的物理页/帧，支持 `1-3,5`。第三项画布上限100页；功能四所选页/帧合计上限1000，逐页处理。具体上传、格式、导出参数以代码/OpenAPI为准。模型默认native，失败不自动回退legacy/HTTP。

画布模型：文档 `schemaVersion=1`，含 `id/name/pages`；页含 `id/index/width/height/sourcePage/elements`；单位pt、左上角原点。元素当前几何是 `x/y/width/height/rotation/zIndex`；原始 `polygon/glyphs/direction` 等来源证据不随编辑改写。类型含text/image/vector/region/background/shape；原始外观与编辑后的文字分开。`summary/document.json`是当前保存文件；旧JSON通过 `legacy*.py`迁入新副本。

功能四模型：`schema=feature4-rebuild/1`，含 `source_pages/blocks/relations/layout_result/report/limitations`。块含 `id/kind/source/content/style/local_layout/quality`；来源坐标和目标测量分存。表格保留行列跨度与原图；图片转录为元数据，不重复绘字。关系为 `07-spatial-v2`，布局为 `08-regional-v4-internal-structure`；源行与格内锚点的可靠性门控见 [重建规则](../docs/REBUILD_RULES.md)。源页一对一，旧未标记策略的保存项目保留单栏兼容分支。

功能四项目不等于画布文档，不承诺可直接导入旧画布。PDF/ZIP携带项目、报告与参考资源，校验通过不等于OCR准确率或WPS逻辑文本框/表格验收。
