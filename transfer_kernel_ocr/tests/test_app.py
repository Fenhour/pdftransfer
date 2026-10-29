"""Exercise the actual upload/edit/export boundary without requiring OCR models."""
from __future__ import annotations

import base64
import copy
import importlib
import io
import json
import re
import zipfile
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.demo import create_demo
from backend.exporter import render_page


def png_bytes(color="white", size=(180, 120)):
    image = Image.new("RGB", size, color)
    if color == "white":
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((18, 20, 63, 63), fill="#d52b35")
        drawing.line((90, 20, 150, 80), fill="#167c73", width=4)
        drawing.text((18, 86), "Scan 42", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    # Import the real parser; no substitute implementation hides integration failures.
    module = importlib.import_module("backend.app")
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(module, "INPUT", input_dir)
    monkeypatch.setattr(module, "OUTPUT", output_dir)
    with TestClient(module.app, raise_server_exceptions=False) as client:
        yield client, input_dir, output_dir


@pytest.fixture
def image_document(local_app):
    client, _, _ = local_app
    response = client.post(
        "/api/documents",
        files=[("files", ("scan.png", png_bytes(), "image/png"))],
        data={"ocr": "false"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def editable_text(text="Edited text"):
    return {
        "id": "manual-review-text", "name": "Editable text", "type": "text",
        "x": 42, "y": 38, "width": 250, "height": 45, "rotation": 7,
        "zIndex": 10000, "fontSize": 18, "fontFamily": "Arial",
        "color": "#123456", "text": text, "align": "center",
        "source": "manual", "visible": True, "opacity": .9,
    }


def assert_portable(document):
    for page in document["pages"]:
        if page.get("sourceSrc"):
            assert page["sourceSrc"].startswith("data:image/")
        for element in page["elements"]:
            for key in ("src", "appearanceSrc", "originalAppearanceSrc"):
                if element.get(key):
                    assert element[key].startswith("data:image/"), (key, element[key])


def test_mixed_upload_edit_save_exports_and_pdf_roundtrip(local_app, tmp_path):
    client, input_dir, output_dir = local_app
    sample = tmp_path / "mixed.pdf"
    create_demo(sample)
    original_pdf = sample.read_bytes()
    original_image = png_bytes()
    uploaded = client.post(
        "/api/documents",
        files=[
            ("files", ("mixed.pdf", original_pdf, "application/pdf")),
            ("files", ("scan.png", original_image, "image/png")),
        ],
        data={"ocr": "false"},
    )
    assert uploaded.status_code == 200, uploaded.text
    document = uploaded.json()
    document_id = document["id"]
    assert len(document["pages"]) == 3
    assert any(e["type"] == "text" for e in document["pages"][0]["elements"])
    assert any(e["type"] == "vector" for e in document["pages"][0]["elements"])
    assert document["pages"][0]["width"] == pytest.approx(595)
    assert document["pages"][0]["height"] == pytest.approx(842)
    assert (input_dir / document_id / "01_mixed.pdf").read_bytes() == original_pdf
    assert (input_dir / document_id / "02_scan.png").read_bytes() == original_image
    assert not list((output_dir / document_id).glob("*.pdf"))

    # Save a change to native text, a deletion, and a manual element together.
    page = document["pages"][0]
    native = next(e for e in page["elements"] if e["type"] == "text")
    native.update(text="Changed native text", x=native["x"] + 12, rotation=11)
    native.pop("appearanceSrc", None)
    removed = next(e for e in page["elements"] if e["type"] == "vector")
    removed_id = removed["id"]
    page["elements"].remove(removed)
    page["elements"].append(editable_text())
    saved = client.put(f"/api/documents/{document_id}", json=document)
    assert saved.status_code == 200, saved.text
    current = saved.json()
    reloaded = client.get(f"/api/documents/{document_id}").json()
    assert reloaded == current
    assert all(e["id"] != removed_id for e in current["pages"][0]["elements"])
    assert current["stats"]["elementCount"] == sum(len(p["elements"]) for p in current["pages"])
    out = output_dir / document_id
    assert json.loads((out / "summary" / "document.json").read_text("utf-8")) == current
    text_manifest = json.loads((out / "text" / "text.json").read_text("utf-8"))
    assert text_manifest[0]["elements"] == [e for e in current["pages"][0]["elements"] if e["type"] == "text"]
    vectors = json.loads((out / "vectors" / "vector.json").read_text("utf-8"))
    assert all(e["id"] != removed_id for p in vectors for e in p["elements"])
    for directory in ("images", "vectors", "text", "backgrounds", "pages", "summary", "exports"):
        assert (out / directory).is_dir()

    exported_json = client.post(f"/api/documents/{document_id}/export/json", json=current)
    assert exported_json.status_code == 200, exported_json.text
    portable = exported_json.json()
    assert_portable(portable)
    assert any(e.get("text") == "Changed native text" for e in portable["pages"][0]["elements"])

    exported_pdf = client.post(f"/api/documents/{document_id}/export/pdf", json=current)
    assert exported_pdf.status_code == 200, exported_pdf.text
    assert exported_pdf.headers["content-type"] == "application/pdf"
    with fitz.open(stream=exported_pdf.content, filetype="pdf") as pdf:
        assert len(pdf) == 3
        assert pdf[0].rect.width == pytest.approx(595)
        assert pdf[0].rect.height == pytest.approx(842)
        assert "scene-document.json" in pdf.embfile_names()
        attached = json.loads(pdf.embfile_get("scene-document.json"))
        assert attached["pages"] == portable["pages"]
        assert_portable(attached)

    # The PDF attachment restores edited elements, not OCR of the rasterized page.
    imported_pdf = client.post(
        "/api/documents",
        files=[("files", ("edited.pdf", exported_pdf.content, "application/pdf"))],
        data={"ocr": "false"},
    )
    assert imported_pdf.status_code == 200, imported_pdf.text
    restored = imported_pdf.json()
    assert restored["id"] != document_id
    restored_text = next(e for e in restored["pages"][0]["elements"] if e["id"] == "manual-review-text")
    assert restored_text["text"] == "Edited text"
    assert restored_text["rotation"] == 7
    assert len(restored["pages"][0]["elements"]) == len(current["pages"][0]["elements"])
    assert restored["pages"][0]["sourceSrc"].startswith(f"/files/{restored['id']}/")

    imported_json = client.post(
        "/api/documents",
        files=[("files", ("edited.json", exported_json.content, "application/json"))],
        data={"ocr": "false"},
    )
    assert imported_json.status_code == 200, imported_json.text
    assert imported_json.json()["pages"][0]["elements"][-1]["text"] == "Edited text"

    archive_response = client.post(f"/api/documents/{document_id}/export/zip", json=current)
    assert archive_response.status_code == 200, archive_response.text
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        names = [n.replace("\\", "/") for n in archive.namelist()]
        assert {"summary/document.json", "summary/index.html", "summary/document.pdf"}.issubset(names)
        assert all(not n.startswith(("input/", "exports/")) for n in names)
        assert all("mixed.pdf" not in n and "01_" not in n for n in names)
        assert_portable(json.loads(archive.read("summary/document.json")))
    assert (input_dir / document_id / "01_mixed.pdf").read_bytes() == original_pdf


def test_html_exports_escape_content_and_embed_assets(local_app, image_document):
    client, _, output_dir = local_app
    document = copy.deepcopy(image_document)
    payload = '</script><script>alert("xss")</script>&<div>文字</div>'
    document["name"] = 'Report <img src=x onerror="alert(1)">'
    document["pages"][0]["elements"].append(editable_text(payload))
    response = client.post(f"/api/documents/{document['id']}/export/html", json=document)
    assert response.status_code == 200, response.text
    exported = response.text
    assert payload not in exported
    assert '<img src=x onerror="alert(1)">' not in exported
    match = re.search(r'<script id="scene-data" type="application/json">(.*?)</script>', exported, flags=re.S)
    assert match
    embedded = json.loads(match.group(1))
    assert embedded["pages"][0]["elements"][-1]["text"] == payload
    assert_portable(embedded)
    assert response.content == (output_dir / document["id"] / "summary" / "index.html").read_bytes()


@pytest.mark.parametrize("field,source", [
    ("src", "https://example.invalid/private.png"),
    ("appearanceSrc", "http://localhost:9999/private.png"),
    ("originalAppearanceSrc", "data:image/svg+xml;base64,PHN2Zy8+"),
    ("src", "/files/{id}/../../input/private.png"),
    ("src", "/files/{id}/%2e%2e/%2e%2e/input/private.png"),
    ("src", "/files/another-document/images/private.png"),
])
def test_asset_rejection_preserves_saved_document(local_app, image_document, field, source):
    client, _, _ = local_app
    before = copy.deepcopy(image_document)
    changed = copy.deepcopy(before)
    changed["pages"][0]["elements"][0][field] = source.replace("{id}", changed["id"])
    response = client.put(f"/api/documents/{changed['id']}", json=changed)
    assert response.status_code == 422, response.text
    assert client.get(f"/api/documents/{changed['id']}").json() == before


@pytest.mark.parametrize("svg", [
    '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><rect width="5" height="5"/></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.invalid/x.png"/></svg>',
    '<svg xmlns="http://www.w3.org/2000/svg"><rect style="fill:url(https://example.invalid/x)"/></svg>',
])
def test_svg_script_and_external_resource_rejection(local_app, image_document, svg):
    client, _, _ = local_app
    changed = copy.deepcopy(image_document)
    changed["pages"][0]["elements"].append({
        "id": "unsafe-vector", "type": "vector", "x": 0, "y": 0,
        "width": 20, "height": 20, "svg": svg,
    })
    response = client.put(f"/api/documents/{changed['id']}", json=changed)
    assert response.status_code == 422, response.text
    assert client.get(f"/api/documents/{changed['id']}").json() == image_document


def test_file_endpoint_cannot_read_input_or_another_document(local_app, image_document):
    client, input_dir, output_dir = local_app
    document_id = image_document["id"]
    secret = input_dir / "private.png"
    secret.write_bytes(png_bytes("blue"))
    other = output_dir / ("f" * 32) / "images"
    other.mkdir(parents=True)
    (other / "private.png").write_bytes(png_bytes("blue"))
    for path in (
        "%2e%2e/%2e%2e/input/private.png",
        f"%2e%2e/{'f' * 32}/images/private.png",
        "summary/document.json",
    ):
        response = client.get(f"/files/{document_id}/{path}")
        assert response.status_code == 404, response.text
    source_response = client.get(image_document["pages"][0]["sourceSrc"])
    assert source_response.status_code == 200
    assert source_response.headers["x-content-type-options"] == "nosniff"
    assert Image.open(io.BytesIO(source_response.content)).width > 0


def test_invalid_upload_is_reported_and_original_is_preserved(local_app):
    client, input_dir, _ = local_app
    broken = b"This is not an image"
    response = client.post(
        "/api/documents", files=[("files", ("broken.png", broken, "image/png"))],
        data={"ocr": "false"},
    )
    assert response.status_code == 422, response.text
    originals = list(input_dir.glob("*/01_broken.png"))
    assert len(originals) == 1
    assert originals[0].read_bytes() == broken
    assert client.get("/api/documents").json() == []


def test_renderer_applies_center_rotation_layer_opacity_and_removal(tmp_path):
    red = "data:image/png;base64," + base64.b64encode(png_bytes("red", (24, 10))).decode()
    element = {
        "id": "rotated", "type": "image", "src": red, "x": 20, "y": 30,
        "width": 24, "height": 10, "rotation": 90, "zIndex": 1,
    }
    page = {"width": 80, "height": 80, "elements": [element]}
    rendered = render_page(page, "demo", tmp_path, scale=1)
    assert rendered.getpixel((32, 25)) == (255, 0, 0)
    assert rendered.getpixel((22, 35)) == (255, 255, 255)

    element.update(rotation=0, x=45)
    moved = render_page(page, "demo", tmp_path, scale=1)
    assert moved.getpixel((32, 35)) == (255, 255, 255), "Moving an element must not leave its original pixels behind"
    assert moved.getpixel((50, 35)) == (255, 0, 0)
    page["elements"] = []
    assert render_page(page, "demo", tmp_path, scale=1).getpixel((50, 35)) == (255, 255, 255)

    shape = {"id": "blue", "type": "shape", "shape": "rect", "x": 10, "y": 10,
             "width": 40, "height": 40, "fill": "#0000ff", "zIndex": 0}
    top = dict(shape, id="red", fill="#ff0000", opacity=.5, zIndex=1)
    hidden = dict(shape, id="hidden", fill="#00ff00", visible=False, zIndex=2)
    page["elements"] = [hidden, top, shape]
    assert render_page(page, "demo", tmp_path, scale=1).getpixel((30, 30)) == (128, 0, 127)


def test_cross_origin_mutation_rejected(local_app, image_document):
    client, _, _ = local_app
    response = client.put(
        f"/api/documents/{image_document['id']}", json=image_document,
        headers={"Origin": "https://unrelated.example"},
    )
    assert response.status_code == 403
