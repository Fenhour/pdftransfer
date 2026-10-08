import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="本地 PDF 内容提取器：分类导出与有序汇总")
    parser.add_argument("pdf", nargs="?", help="PDF 文件；不传则打开中文界面")
    parser.add_argument("-o", "--output", default=str(Path(__file__).resolve().parents[2] / "output"), help="结果父目录，默认项目 output")
    parser.add_argument("--pages", default="", help="页码，例如 1-3,5；默认全部")
    parser.add_argument("--ocr", choices=["auto", "all", "off"], default="auto")
    parser.add_argument("--order", choices=["columns", "rows"], default="columns")
    parser.add_argument("--tables", choices=["lines_strict", "lines", "text"], default="lines_strict")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--ask-password", action="store_true", help="安全地交互输入 PDF 密码")
    args = parser.parse_args()
    try:
        if not args.pdf:
            from .gui import main as gui_main
            gui_main()
            return 0
        from .engine import run
        from .model import Options
        import getpass
        password = getpass.getpass("PDF 密码：") if args.ask_password else ""
        destination, result = run(Path(args.pdf), Path(args.output),
            Options(args.pages, args.ocr, password, args.dpi, args.tables, args.order),
            lambda done, total, message: print(f"[{done}/{total}] {message}", flush=True))
        print(f"\n完成：{destination.parent.parent}\nHTML：{destination / 'summary' / 'summary.html'}\nMarkdown：{destination.parent.parent / 'output_md' / destination.name / 'summary' / 'summary.md'}")
        for warning in result.warnings:
            print(f"提示：{warning}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"提取失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
