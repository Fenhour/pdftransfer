"""Keep the former command-line entry point on the new import/save pipeline."""
from contextlib import ExitStack
from pathlib import Path
from fastapi import UploadFile


def extract(files, *, ocr=True, pages="", output=None):
    from . import app
    previous_output = app.OUTPUT
    try:
        if output is not None:
            app.OUTPUT = Path(output).resolve()
            app.OUTPUT.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            uploads = []
            for path in files:
                path = Path(path).resolve(strict=True)
                stream = stack.enter_context(path.open("rb"))
                uploads.append(UploadFile(filename=path.name, file=stream))
            document = app.upload(uploads, ocr=ocr, pages=pages, mode='editable')
        return app.OUTPUT / document["id"] / "summary" / "index.html"
    finally:
        app.OUTPUT = previous_output
