import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser

from .engine import Cancelled, run
from .model import Options
from .paths import INPUT_PDF, OUTPUT


def open_folder(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF 内容提取器")
        self.geometry("900x780")
        self.minsize(780, 720)
        self.configure(bg="#eff4f5")
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.destination = None
        self.source = tk.StringVar()
        self.output = tk.StringVar(value=str(OUTPUT))
        self.pages = tk.StringVar()
        self.password = tk.StringVar()
        self.ocr = tk.StringVar(value="自动识别扫描页与无文字层图片")
        self.order = tk.StringVar(value="自动多栏阅读顺序")
        self.tables = tk.StringVar(value="自动识别表格（推荐）")
        self.format = tk.StringVar(value="HTML")
        self.backend = tk.StringVar(value="V3 扫描版面增强（本地）")
        self.status = tk.StringVar(value="准备就绪 · 文件仅在本机处理")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#eff4f5")
        style.configure("TLabel", background="#eff4f5", foreground="#173540", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=9)
        style.configure("Accent.TButton", background="#157f8c", foreground="white", padding=12)
        style.map("Accent.TButton", background=[("active", "#126773"), ("disabled", "#aac4c9")])
        style.configure("TEntry", padding=7)
        style.configure("TCombobox", padding=6)
        style.configure("Horizontal.TProgressbar", background="#157f8c", troughcolor="#dbe7ea")
        head = tk.Frame(self, bg="#173540", padx=30, pady=22)
        head.pack(fill="x")
        tk.Label(head, text="PDF / CONTENT EXTRACTOR", fg="#83ccd1", bg="#173540", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(head, text="让 PDF 变成可编辑的内容", fg="white", bg="#173540", font=("Microsoft YaHei UI", 22, "bold")).pack(anchor="w", pady=(8, 4))
        tk.Label(head, text="文字 · 表格 · 图片 · 标题结构 · 页码   /   分类归档，按顺序汇总", fg="#ccdde1", bg="#173540", font=("Microsoft YaHei UI", 10)).pack(anchor="w")
        form = ttk.Frame(self, padding=(30, 20))
        form.pack(fill="both", expand=True)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="PDF 文件").grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Entry(form, textvariable=self.source).grid(row=0, column=1, sticky="ew")
        self.pick = ttk.Button(form, text="选择文件", command=self.choose)
        self.pick.grid(row=0, column=2, padx=(10, 0))
        ttk.Label(form, text="保存位置").grid(row=1, column=0, sticky="w", pady=10)
        ttk.Entry(form, textvariable=self.output).grid(row=1, column=1, sticky="ew")
        ttk.Button(form, text="选择目录", command=self.choose_output).grid(row=1, column=2, padx=(10, 0))
        options = ttk.Frame(form)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 10))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text="提取页码").grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Entry(options, textvariable=self.pages, width=24).grid(row=0, column=1, sticky="ew")
        ttk.Label(options, text="文档密码").grid(row=0, column=2, padx=(20, 12))
        ttk.Entry(options, textvariable=self.password, show="●", width=18).grid(row=0, column=3, sticky="ew")
        ttk.Label(options, text="留空为全部，例如 1-3,5", foreground="#6a7f86").grid(row=1, column=1, sticky="w", pady=(4, 12))
        for row, label, variable, choices in [
            (2, "OCR 模式", self.ocr, ["自动识别扫描页与无文字层图片", "识别所有图片", "关闭 OCR"]),
            (3, "阅读顺序", self.order, ["自动多栏阅读顺序", "按从上到下、从左到右排列"]),
            (4, "表格检测", self.tables, ["自动识别表格（推荐）", "有边框表格", "宽松边框检测", "无边框表格（实验，可能误判正文）"]),
            (5, "输出格式", self.format, ["HTML", "Markdown"]),
            (6, "扫描识别", self.backend, ["V3 扫描版面增强（本地）", "轻量 RapidOCR"]),
        ]:
            ttk.Label(options, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Combobox(options, textvariable=variable, values=choices, state="readonly").grid(row=row, column=1, columnspan=3, sticky="ew", pady=5)
        ttk.Label(form, text="仅生成所选 HTML 或 Markdown；原生页直接提取，扫描页按所选后端识别。源文件保留在 source。", wraplength=800, foreground="#5b737c").grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 12))
        controls = ttk.Frame(form)
        controls.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.start_button = ttk.Button(controls, text="开始提取", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="取消", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.preview_button = ttk.Button(controls, text="查看汇总", command=self.preview, state="disabled")
        self.preview_button.pack(side="right")
        self.folder_button = ttk.Button(controls, text="打开结果目录", command=lambda: open_folder(self.destination), state="disabled")
        self.folder_button.pack(side="right", padx=8)
        self.progress = ttk.Progressbar(form, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(18, 8))
        ttk.Label(form, textvariable=self.status).grid(row=6, column=0, columnspan=3, sticky="w")
        self.log = tk.Text(form, height=6, bg="white", fg="#37545f", relief="flat", padx=12, pady=10, font=("Microsoft YaHei UI", 9), wrap="word", state="disabled")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        form.rowconfigure(7, weight=1)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(100, self.poll)

    def choose(self):
        current = Path(self.source.get()) if self.source.get() else None
        initial = current.parent if current and current.is_file() else INPUT_PDF
        value = filedialog.askopenfilename(initialdir=str(initial), filetypes=[("PDF 文档", "*.pdf")])
        if value:
            self.source.set(value)

    def choose_output(self):
        value = filedialog.askdirectory()
        if value:
            self.output.set(value)

    def append(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.busy:
            return
        source, output = self.source.get().strip(), self.output.get().strip()
        if not source or not output:
            messagebox.showerror("缺少路径", "请选择 PDF 文件和输出目录。")
            return
        options = Options(pages=self.pages.get(), password=self.password.get(),
                          ocr={"关闭 OCR": "off", "识别所有图片": "all"}.get(self.ocr.get(), "auto"),
                          order="columns" if self.order.get() == "自动多栏阅读顺序" else "rows",
                          table_strategy={"自动识别表格（推荐）": "auto", "有边框表格": "lines_strict", "宽松边框检测": "lines"}.get(self.tables.get(), "text"),
                          output_format="html" if self.format.get() == "HTML" else "md",
                          recognition_backend="v3" if self.backend.get().startswith("V3") else "legacy")
        self.busy = True
        self.cancel.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.preview_button.configure(state="disabled")
        self.folder_button.configure(state="disabled")
        self.status.set("正在读取 PDF…")
        self.progress["value"] = 0
        self.append(f"开始提取：{source}")

        def worker():
            try:
                destination, result = run(source, output, options,
                    lambda done, total, message: self.events.put(("progress", (done, total, message))), self.cancel)
                self.events.put(("success", (destination, result)))
            except Cancelled:
                self.events.put(("cancelled", None))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        self.cancel.set()
        self.status.set("正在取消，等待当前页面或 OCR 操作结束…")
        self.stop_button.configure(state="disabled")

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "progress":
                    done, total, message = data
                    self.progress["value"] = done * 100 / max(1, total)
                    if not self.cancel.is_set():
                        self.status.set(message)
                else:
                    self.busy = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if kind == "success":
                        self.destination, result = data
                        self.progress["value"] = 100
                        self.status.set(f"完成 · {len(result.selected_pages)} 页 · {len(result.blocks)} 个内容块 · {len(result.warnings)} 条提示")
                        self.append(f"已保存：{self.destination}")
                        for warning in result.warnings:
                            self.append(warning)
                        self.preview_button.configure(state="normal")
                        self.folder_button.configure(state="normal")
                    elif kind == "cancelled":
                        self.status.set("已取消，临时输出已清理。")
                    else:
                        self.status.set("提取未完成，请查看错误信息。")
                        self.append(data)
                        messagebox.showerror("提取失败", data)
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def preview(self):
        if self.destination:
            extension = "html" if self.destination.parent.name == "output_html" else "md"
            webbrowser.open((self.destination / "summary" / f"summary.{extension}").as_uri())

    def close(self):
        if self.busy:
            self.stop()
            self.after(200, self.close_when_finished)
        else:
            self.destroy()

    def close_when_finished(self):
        if self.busy:
            self.after(200, self.close_when_finished)
        else:
            self.destroy()


def main():
    Application().mainloop()
