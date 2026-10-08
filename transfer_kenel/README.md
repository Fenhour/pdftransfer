# PDF 内容提取器

中文本地工具。分类提取正文、标题、表格、图片、页码和 OCR 文字，同时生成按原文页序排列的可编辑汇总。

## 启动

双击本目录下的 **start.bat**。当前电脑的依赖已经安装。选择 PDF 后，保存位置默认是项目根目录的 `output`，不受启动时所在目录影响。

首次在其他电脑使用：安装 Python 3.10 或更新版本（建议 3.12），先运行 `setup.bat`，再运行 `start.bat`。安装依赖需要网络；提取及中英文 OCR 在本机运行，不上传文件。

## 目录结构

```text
pdfTransfer/
├── transfer_kenel/                 转换器、启动文件、依赖和测试
│   ├── start.bat
│   ├── setup.bat
│   ├── pdf_extractor/
│   ├── tests/
│   └── .venv/
└── output/
    ├── output_md/
    │   ├── source/
    │   │   └── voyager.pdf
    │   └── voyager/
    │       ├── graph/              图片文档及 PNG 图片资源
    │       │   ├── graph.md
    │       │   └── page_0001_image_001.png
    │       ├── table/table.md
    │       ├── text/text.md
    │       ├── heading/heading.md
    │       ├── page_number/page_number.md
    │       ├── header_footer/header_footer.md
    │       ├── ocr/ocr.md
    │       └── summary/summary.md
    └── output_html/
        ├── source/
        │   └── voyager.pdf
        └── voyager/
            ├── graph/             图片文档及 PNG 图片资源
            ├── table/table.html
            ├── text/text.html
            ├── heading/heading.html
            ├── page_number/page_number.html
            ├── header_footer/header_footer.html
            ├── ocr/ocr.html
            └── summary/summary.html
```

只为实际识别出的类型创建目录。文档只生成 Markdown 和 HTML，不再生成 TXT、JSON、CSV。`source` 中保留源 PDF；PNG 是显示原图所需的资源文件，不是额外文档格式。

每次运行两套输出使用同一个名称；重复运行会依次生成 `voyager_2` 等新目录，对应源文件保存为 `source/voyager_2.pdf`，不覆盖以前的结果。分享时发送整个 `output_md` 或 `output_html`，以保留图片和源 PDF 的相对链接。

## 本版排版改进

- 根据文字坐标合并同一行的字体片段，清理正文重复空格，连接普通段落的排版换行。
- 将代码片段合并成连续代码块，并按位置恢复缩进；保留 `[]`、`{}`、`()`、下划线和其他原文字层符号。代码的空格和换行单独保留，不按普通正文压缩。
- Markdown 引用标记如 `[1, 2]` 保留为可读文字；可能被当成链接的原文语法会进行必要转义。
- 通过图题、嵌入图片和矢量图形位置，尽量重建完整组合图，避免把一张图拆成许多图标穿插在正文中。
- 图题与图片一起显示；图内原生文字、OCR 转录放入可展开区域，仍然可编辑。OCR 全文也单独放入 `ocr`。
- 目录使用章节层级。页码跳转单列，正文不再反复显示内容块编号。
- 每份 HTML（包括分类文档）都可编辑文字和表格。修改后点击“保存修改后的 HTML”，下载文件放回当前文件所在目录。修改不会自动同步到其他文档。

## 参数

- 页码留空表示全部，或者填写 `1-3,5,8-10`；页码是 PDF 的物理页码，从 1 开始。
- 自动 OCR 识别扫描页和包含嵌入图片的图形区域；“识别所有图片”还会识别纯矢量组合图的渲染结果。
- 默认按多栏阅读区域排序，也可切换为逐行从上到下排序。
- 默认识别有边框表格；无边框模式是实验模式，文字对齐相似时可能把正文误判为表格。
- 加密 PDF 可在界面中输入密码；密码不写入输出。
- 取消操作会等待当前 OCR 调用结束，再清理本次临时文件。

## 命令行

在 `transfer_kenel` 目录运行：

```powershell
.venv\Scripts\python.exe -m pdf_extractor "..\voyager.pdf"
.venv\Scripts\python.exe -m pdf_extractor "..\output\output_md\source\voyager.pdf" --pages "1-3,5"
.venv\Scripts\python.exe -m pdf_extractor "sample.pdf" --ocr off --order rows
.venv\Scripts\python.exe -m pdf_extractor "sample.pdf" --tables text --dpi 220
.venv\Scripts\python.exe -m pdf_extractor "locked.pdf" --ask-password
```

可用 `-o` 指定其他输出父目录，其下仍生成 `output_md` 和 `output_html`。不传 PDF 路径时打开中文界面。

## 校对范围

PDF 缺少统一的段落语义，章节、表格和页内顺序仍有推断成分。请核对复杂多栏、跨页表格、合并单元格、公式、手写体、低清扫描和没有图题的组合图。扫描页保留整页参考图，并输出可编辑的 OCR 内容；扫描表格仅支持清晰规则网格。

转换器不擅自改写代码，也不会凭语义猜测缺失字符。原生符号按文字层提取；OCR 可能误认符号，图内转录会标注置信度。原文行尾普通连字符会保留，避免把真实复合词误改。图片本身为位图，图中文字提供可编辑转录。

## 测试

```powershell
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

覆盖：分类、表格去重、多栏顺序、标题层级、代码符号和缩进、组合图合并、HTML/图片/源 PDF 相对链接、双格式目录、防覆盖、密码、取消，以及真实中英文 OCR 与扫描表格。
