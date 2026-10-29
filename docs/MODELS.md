# 本地模型准备

Git 不包含模型权重。通过 PaddleOCR / PaddleX 官方渠道单独获取下表推理模型，并遵守各自许可证。放置于 V3/models 或以 PDFTRANSFER_V3_MODEL_DIR 指定 ASCII 路径。

| 逻辑模型 | 副本目录（V3/models下） | 功能 |
| --- | --- | --- |
| PP-DocLayout_plus-L | `PP-DocLayout_plus-L` | 正文、标题、图片、表格等版面类别和位置 |
| PP-DocBlockLayout | `PP-DocBlockLayout` | Region区域检测；07/08布局与05预览使用 |
| PP-OCRv5_server_det | `PP-OCRv5_server_det_cpu_fp32_zero_subnormals_v1_183146fe9d99` | 文字行位置；使用当前配置的FP32 CPU优化权重 |
| PP-OCRv5_server_rec | `PP-OCRv5_server_rec_cpu_fp32_zero_subnormals_v1_63853f062a5f` | 文字识别；使用当前配置的FP32 CPU优化权重 |
| PP-LCNet_x1_0_table_cls | `PP-LCNet_x1_0_table_cls` | 有线/无线表格分类 |
| SLANeXt_wired | `SLANeXt_wired` | 有线表格结构 |
| SLANet_plus | `SLANet_plus` | 无线表格结构 |
| RT-DETR-L_wired_table_cell_det | `RT-DETR-L_wired_table_cell_det` | 有线表格单元格检测 |
| RT-DETR-L_wireless_table_cell_det | `RT-DETR-L_wireless_table_cell_det` | 无线表格单元格检测 |
| PP-LCNet_x1_0_doc_ori | `PP-LCNet_x1_0_doc_ori` | 表格局部方向分类；不能因整页方向关闭就删除 |
| PP-LCNet_x1_0_textline_ori | `PP-LCNet_x1_0_textline_ori` | 表格内部惰性OCR分支的文字行方向；主OCR关闭不等于所有子分支关闭 |

每个模型需要 inference.json、inference.pdiparams 和 inference.yml。当前配置的 det/rec 使用 FP32 派生权重；首次准备时使用 services/structure_v3/prepare_cpu_det.py 和 prepare_cpu_rec.py（先查看 --help），从官方原始模型生成配置所列目录；也可将 native_config.json 的 model_paths 显式改为原始模型目录。缺模型会报错，不自动下载。

固定依赖见 requirements-native.txt。方向模型仍可能被表格内部流程使用，不能仅因整页方向关闭而省略。字体使用本机环境，未打包系统/WPS字体。未来候选字体模型尚未接入。
