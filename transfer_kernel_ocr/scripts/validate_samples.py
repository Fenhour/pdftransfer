"""End-to-end compatibility smoke test with a user-selected local sample folder.

This measures successful workflows, not OCR accuracy (no human ground truth).
Reports and downloads stay in the ignored output directory.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
import zipfile

import httpx
from PIL import Image
import pymupdf as fitz


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def page_count(path):
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            return len(document)
    with Image.open(path) as image:
        return getattr(image, "n_frames", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parents[2]/"output/ocr_replacement_validation")
    parser.add_argument("--resume", action="store_true", help="reuse completed jobs and interrupted uploaded scenes")
    args = parser.parse_args()
    root = args.samples.resolve(strict=True)
    output = args.report.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"})
    before = {str(path): sha(path) for path in files}
    groups = [[p] for p in files if p.suffix.lower() == ".pdf"]
    images = [p for p in files if p.suffix.lower() != ".pdf"]
    groups += [images[i:i+30] for i in range(0, len(images), 30)]
    report = {"samples": str(root), "files": len(files), "jobs": [], "accuracy": "not evaluated; no human ground truth"}
    state_path = output / "validation-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if args.resume and state_path.exists() else {}
    if args.resume and (output / "report.json").exists():
        previous = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if previous.get("samples") != str(root):
            raise ValueError("Cannot resume a different sample directory")
        report["jobs"] = previous["jobs"]
    with httpx.Client(base_url=args.url, timeout=1800, trust_env=False) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        assert health.json()["ocr"]["available"], health.text
        report["storage"] = health.json()["storage"]
        for number, paths in enumerate(groups, 1):
            names = [str(p.relative_to(root)) for p in paths]
            if any(job["files"] == names for job in report["jobs"]):
                continue
            started = time.monotonic()
            key = "|".join(str(p) + ":" + before[str(p)] for p in paths)
            cached_id = state.get(key)
            if cached_id:
                response = client.get(f"/api/documents/{cached_id}")
            else:
                response = None
            with ExitStack() as stack:
                upload = [("files", (p.name, stack.enter_context(p.open("rb")), "application/octet-stream")) for p in paths]
                if response is None or response.status_code != 200:
                    response = client.post("/api/documents", files=upload, data={"ocr": "true"})
            response.raise_for_status()
            doc = response.json()
            expected = sum(page_count(p) for p in paths)
            assert len(doc["pages"]) == expected
            assert all(p["elements"] for p in doc["pages"])
            original_id = doc["id"]
            state[key] = original_id
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            # Exercise edit/save from the exact scene rendered on screen.
            text = next((e for p in doc["pages"] for e in p["elements"] if e["type"] == "text"), None)
            if text and not cached_id:
                text["x"] += .25
            saved = client.put(f"/api/documents/{original_id}", json=doc)
            saved.raise_for_status()
            doc = saved.json()
            downloads = {}
            for format in ("json", "html", "pdf", "zip"):
                target = output / f"sample-{number:02d}.{format}"
                if not (args.resume and cached_id and target.exists()):
                    response = client.post(f"/api/documents/{original_id}/export/{format}", json=doc)
                    response.raise_for_status()
                    target.write_bytes(response.content)
                downloads[format] = {"path": str(target), "bytes": target.stat().st_size}
                if format == "json":
                    portable = json.loads(target.read_text(encoding="utf-8"))
                    assert portable["id"] == original_id
                    assert len(portable["pages"]) == expected
                elif format == "html":
                    assert "<!doctype html>" in target.read_text(encoding="utf-8").lower()
                elif format == "pdf":
                    with fitz.open(target) as pdf:
                        assert len(pdf) == expected
                        assert "scene-document.json" in pdf.embfile_names()
                else:
                    with zipfile.ZipFile(target) as archive:
                        assert "summary/document.json" in archive.namelist()
                        assert "summary/document.pdf" in archive.namelist()
                        assert not any(name.startswith("input/") for name in archive.namelist())
            # PDF carries the edit model: re-import must restore the scene,
            # rather than flattening it into a fresh OCR result.
            with (output / f"sample-{number:02d}.pdf").open("rb") as stream:
                imported = client.post("/api/documents", files={"files": ("roundtrip.pdf", stream, "application/pdf")})
            imported.raise_for_status()
            restored = imported.json()
            assert restored["id"] != original_id
            assert len(restored["pages"]) == expected
            assert [len(p["elements"]) for p in restored["pages"]] == [len(p["elements"]) for p in doc["pages"]]
            if text:
                restored_text = next(e for p in restored["pages"] for e in p["elements"] if e["id"] == text["id"])
                assert restored_text["text"] == text["text"] and restored_text["x"] == text["x"]
            result = {"files": [str(p.relative_to(root)) for p in paths], "id": original_id,
                      "roundtripId": restored["id"], "pages": expected,
                      "elements": sum(len(p["elements"]) for p in doc["pages"]),
                      "textElements": sum(e["type"] == "text" for p in doc["pages"] for e in p["elements"]),
                      "seconds": round(time.monotonic() - started, 2), "downloads": downloads}
            report["jobs"].append(result)
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in result.items() if k != "downloads"}, ensure_ascii=False), flush=True)
    after = {str(path): sha(path) for path in files}
    assert before == after, "A source sample was modified"
    report["sourceHashesUnchanged"] = True
    report["pages"] = sum(job["pages"] for job in report["jobs"])
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(files)} files / {report['pages']} pages. Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
