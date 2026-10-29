"""Offline review output must preserve inputs and clearly remain a draft."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts import inspect_boundaries as review


def candidate(text_id, text, box):
    x0, y0, x1, y1 = box
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return dict(element=dict(id=text_id, text=text, height=y1-y0, polygon=polygon, confidence=.99),
                bbox=box, quad=polygon)


def inputs(tmp_path, candidates):
    image = tmp_path / "page.png"
    frozen = tmp_path / "frozen.json"
    rgb = np.full((120, 150, 3), 255, np.uint8)
    rgb[20:100, 20:130] = [20, 80, 140]
    Image.fromarray(rgb).save(image)
    frozen.write_text(json.dumps(candidates), encoding="utf-8")
    return image, frozen, rgb


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_writes_crop_alpha_and_explicit_unreviewed_metadata(tmp_path, monkeypatch):
    separate = candidate("body", "Next paragraph", [40, 40, 65, 50])
    label = candidate("label", "Figure", [25, 25, 38, 35])
    image, frozen, rgb = inputs(tmp_path, [separate, label])
    before = digest(image), digest(frozen)

    def geometry(source, candidates, scale):
        assert scale == 1
        assert np.array_equal(source, rgb)
        region = dict(bbox=[20, 20, 130, 100], pixelBox=[20, 20, 130, 100],
                      retainedText=[dict(text="Figure", bbox=label["bbox"], polygon=label["quad"],
                                         confidence=.99, reason="graphic-overlap")],
                      excludedText=[dict(textId="body", bbox=separate["bbox"], polygon=separate["quad"])],
                      contentType="mixed", reviewRequired=True,
                      boundary=dict(method="foreground-barriers-v2", exact=False, clip="alpha-text-exclusion"))
        return [candidates[0]], [region], "#ffffff"

    monkeypatch.setattr(review, "image_regions", geometry)
    output = tmp_path / "review"
    report = review.inspect(image, frozen, 1, output)
    assert {path.name for path in output.iterdir()} == {"analysis.json", "overlay.png", "region-001.png"}
    persisted = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert report["reviewRequired"] is True
    assert report["humanReviewed"] is False
    assert report["accuracyEvaluated"] is False
    assert report["source"]["sourceUnchanged"] is True
    assert report["ownershipIssues"] == []
    assert report["regions"][0]["retainedText"][0]["candidateIds"] == ["label"]
    assert report["candidateOwnership"] == [
        dict(textId="body", visibleOwners=["independent"], reviewRequired=True),
        dict(textId="label", visibleOwners=["region-001"], reviewRequired=True),
    ]
    with Image.open(output / "region-001.png") as crop:
        crop_rgba = np.asarray(crop)
        assert crop.mode == "RGBA"
        assert np.array_equal(crop_rgba[:, :, :3], rgb[20:100, 20:130])
        assert crop_rgba[25, 25, 3] == 0
        assert crop_rgba[0, 0, 3] == 255
    with Image.open(output / "overlay.png") as overlay:
        assert overlay.height > rgb.shape[0]
    assert (digest(image), digest(frozen)) == before


def test_review_refuses_existing_output_without_overwriting(tmp_path):
    image, frozen, _ = inputs(tmp_path, [])
    output = tmp_path / "review"
    output.mkdir()
    sentinel = output / "analysis.json"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        review.inspect(image, frozen, 1, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert len(list(output.iterdir())) == 1


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_review_rejects_invalid_scale_before_output(tmp_path, scale):
    image, frozen, _ = inputs(tmp_path, [])
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="positive finite"):
        review.inspect(image, frozen, scale, output)
    assert not output.exists()


def test_review_rejects_non_list_candidates_before_output(tmp_path):
    image, frozen, _ = inputs(tmp_path, [])
    frozen.write_text('{"candidates": []}', encoding="utf-8")
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="_ocr_candidates list"):
        review.inspect(image, frozen, 1, output)
    assert not output.exists()


def test_review_cli_works_outside_repository_without_ocr(tmp_path):
    image, frozen, _ = inputs(tmp_path, [])
    output = tmp_path / "review"
    script = Path(review.__file__).resolve()
    result = subprocess.run([sys.executable, "-X", "utf8", str(script), "--image", str(image),
                             "--candidates", str(frozen), "--scale", "1", "--output", str(output)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["reviewRequired"] is True
    assert summary["candidates"] == 0
    assert summary["regions"] == 1
    assert (output / "region-001.png").is_file()
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert report["method"] == "foreground-barriers-v2"
    assert report["source"]["sourceUnchanged"] is True
