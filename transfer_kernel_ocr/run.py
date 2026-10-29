"""Project launcher; one shared port configuration for every entry point."""
import argparse
import json
from pathlib import Path
import threading
import time
import urllib.request
import webbrowser
import uvicorn

ROOT = Path(__file__).resolve().parent


def configured_port():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    port = config["port"]
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("config.json port must be an integer between 1024 and 65535.")
    return port


def open_when_ready(url, output):
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1) as response:
                health = json.load(response)
            if (health.get("status") == "ok"
                    and Path(health.get("storage", {}).get("output", "")).resolve() == output):
                webbrowser.open(url)
                return
        except (OSError, ValueError):
            pass
        time.sleep(.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description="transfer_kernel_ocr · PDF OCR 解析与编辑")
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--port", type=int, default=configured_port())
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--pages", default="", help="物理页码，例如 1-3,5；对每个输入应用")
    parser.add_argument("--out", type=Path, help="输出父目录")
    parser.add_argument("--mode", choices=["auto"], default="auto", help="新解析器统一采用 auto 流程")
    parser.add_argument("--dpi", type=int, choices=[144], default=144)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    if args.files:
        from backend.cli import extract
        print(extract(args.files, ocr=not args.no_ocr, pages=args.pages, output=args.out))
        return
    if args.out or args.pages or args.no_ocr:
        parser.error("--out / --pages / --no-ocr require input files")
    from backend.app import OUTPUT
    if not args.no_browser:
        url = f"http://127.0.0.1:{args.port}"
        threading.Thread(target=open_when_ready, args=(url, OUTPUT), daemon=True).start()
    uvicorn.run("backend.app:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
