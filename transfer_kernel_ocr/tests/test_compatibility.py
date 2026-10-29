"""Compatibility gates for old launchers, saved projects, and page selection."""
import base64
import copy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image
from fastapi.testclient import TestClient
import pymupdf as fitz

from backend.legacy import convert_legacy
from backend.options import selected_pages


def legacy_scene():
    buffer = io.BytesIO()
    Image.new("RGBA", (16, 12), (220, 30, 20, 128)).save(buffer, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    base = {"x": 12, "y": 18, "width": 32, "height": 24, "base_width": 16, "base_height": 12,
            "z": 5, "rotation": 20, "opacity": .7, "visible": False, "locked": True, "style": {},
            "appearance": {"kind": "image", "data": uri}}
    return {"schema": "transfer_kernel_ocr/1", "id": "job_old", "name": "旧项目 中文", "pages": [
        {"id": "page_old", "width": 200, "height": 160, "paper_color": "#faf7f0", "defs": "", "reference": uri,
         "elements": [dict(base, id="text_old", type="text", text="原文字", original_text="原文字", render_mode="original",
                           style={"font_size": 10, "font_family": "Arial", "color": "#123456"}),
                      dict(base, id="image_old", type="image"), dict(base, id="line_old", type="line")]}]}


def test_legacy_mapping_preserves_geometry_text_alpha_visibility_and_paper():
    original = legacy_scene()
    before = copy.deepcopy(original)
    result = convert_legacy(original)
    assert original == before
    assert result["schemaVersion"] == 1
    page = result["pages"][0]
    assert page["sourceSrc"] == original["pages"][0]["reference"]
    mapped = {e["id"]: e for e in page["elements"]}
    text = mapped["text_old"]
    assert text["text"] == "原文字"
    assert text["fontSize"] == 20
    assert text["rotation"] == 20 and text["zIndex"] == 5
    assert (text["x"], text["y"], text["width"], text["height"]) == (12, 18, 32, 24)
    assert text["visible"] is False and text["locked"] is True and text["opacity"] == .7
    with Image.open(io.BytesIO(base64.b64decode(text["appearanceSrc"].split(",")[1]))) as image:
        assert image.getpixel((0, 0))[3] == 128
    assert mapped["line_old"]["type"] == "region"
    assert page["elements"][0]["name"] == "旧版纸色"


@pytest.mark.parametrize("fragment", ['<script>alert(1)</script>', '<image href="file:///secret.png"/>',
                                       '<path style="fill:red"/>', '<image href="https://example.invalid/x.png"/>'])
def test_legacy_import_cannot_bypass_svg_security(fragment):
    data = legacy_scene()
    data["pages"][0]["defs"] = fragment
    with pytest.raises(ValueError):
        convert_legacy(data)


def test_legacy_shared_svg_definitions_are_rendered_locally():
    data = legacy_scene()
    data["pages"][0]["defs"] = '<path id="red" d="M0 0H16V12H0Z" fill="#ff0000"/>'
    old = data["pages"][0]["elements"][1]
    old["type"] = "vector"
    old["appearance"] = {"kind": "svg", "svg": '<use href="#red"/>'}
    converted = convert_legacy(data)
    element = next(e for e in converted["pages"][0]["elements"] if e["id"] == "image_old")
    with Image.open(io.BytesIO(base64.b64decode(element["src"].split(",")[1]))) as image:
        assert image.getpixel((8, 8))[:3] == (255, 0, 0)


def test_import_legacy_json_edit_save_export_roundtrip(tmp_path, monkeypatch):
    from backend import app
    monkeypatch.setattr(app, "INPUT", tmp_path / "input")
    monkeypatch.setattr(app, "OUTPUT", tmp_path / "output")
    with TestClient(app.app) as client:
        response = client.post("/api/documents", files={"files": ("旧项目.json", json.dumps(legacy_scene()).encode(), "application/json")})
        assert response.status_code == 200, response.text
        doc = response.json()
        text = next(e for e in doc["pages"][0]["elements"] if e["type"] == "text")
        text.update(text="迁移后编辑", x=42, visible=True, locked=False)
        text.pop("appearanceSrc", None)
        saved = client.put(f"/api/documents/{doc['id']}", json=doc)
        assert saved.status_code == 200, saved.text
        exported = client.post(f"/api/documents/{doc['id']}/export/json", json=saved.json())
        assert exported.status_code == 200
        assert next(e for e in exported.json()["pages"][0]["elements"] if e["type"] == "text")["text"] == "迁移后编辑"


@pytest.mark.parametrize("spec, total, expected", [("", 3, [0,1,2]), ("1, 3-4,3", 5, [0,2,3]), ("2", 3, [1])])
def test_page_selection(spec, total, expected):
    assert selected_pages(spec, total) == expected


@pytest.mark.parametrize("spec", ["0", "3-1", "1,,2", "4", "1-99", "x", "1-2-3"])
def test_bad_page_selection_is_not_silently_ignored(spec):
    with pytest.raises(ValueError):
        selected_pages(spec, 3)


def test_cli_compatibility_entrypoint_and_pages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    pdf = tmp_path / "source with spaces.pdf"
    with fitz.open() as source:
        for text in ("Page ONE", "Page TWO"):
            source.new_page().insert_text((20,40), text)
        source.save(pdf)
    output = tmp_path / "exports"
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "ocr_canvas", str(pdf), "--pages", "2", "--no-ocr", "--out", str(output)],
                            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert result.returncode == 0, result.stderr
    files = list(output.glob("*/summary/document.json"))
    assert len(files) == 1
    document = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(document["pages"]) == 1
    assert document["pages"][0]["sourcePage"] == 2
    assert any("Page TWO" in e.get("text", "") for e in document["pages"][0]["elements"])
    assert not any("Page ONE" in e.get("text", "") for e in document["pages"][0]["elements"])


def test_unsupported_old_modes_fail_clearly():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "ocr_canvas", "--mode", "native"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_configured_upload_limit_is_bounded_and_applies_at_request_boundary():
    from backend import app
    assert 16 <= app.MAX_UPLOAD_MB <= 512
    assert app.MAX_UPLOAD_MB == json.loads((Path(app.ROOT)/"config.json").read_text(encoding="utf-8"))["max_upload_mb"]
    with TestClient(app.app) as client:
        response = client.post("/api/documents", content=b"", headers={"content-length": str(app.MAX_UPLOAD + 2*1024*1024)})
        assert response.status_code == 413
        assert f"{app.MAX_UPLOAD_MB} MB" in response.json()["detail"]


def test_legacy_json_page_selection_uses_the_same_import_boundary(tmp_path, monkeypatch):
    from backend import app
    monkeypatch.setattr(app, "INPUT", tmp_path / "input")
    monkeypatch.setattr(app, "OUTPUT", tmp_path / "output")
    data = legacy_scene()
    other = copy.deepcopy(data["pages"][0])
    other["id"] = "second_page"
    for element in other["elements"]:
        element["id"] += "_second"
    data["pages"].append(other)
    with TestClient(app.app) as client:
        response = client.post("/api/documents", data={"pages":"2"}, files={"files":("old.json",json.dumps(data).encode(),"application/json")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["pages"]) == 1
    assert any(e["id"] == "text_old_second" for e in result["pages"][0]["elements"])
