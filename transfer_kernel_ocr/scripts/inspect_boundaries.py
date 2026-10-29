"""Review frozen OCR geometry offline without changing the page or running OCR.

Example (paths may be absolute and the working directory is irrelevant)::

    python inspect_boundaries.py --image page.png --candidates frozen.json \
        --scale 2 --output boundary-review

The candidate JSON must be the list returned by ``parser._ocr_candidates``.
The output directory must not exist. Results are a review draft, not an
accuracy measurement or a record of human review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import only the geometry path. Importing parser would unnecessarily expose
# this offline diagnostic to OCR initialization and unrelated export code.
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
from backend.rebuild_layout import image_regions, region_pixels


COLORS = {
    "image": "#ed8a20",
    "independent": "#16964b",
    "retained": "#d73535",
    "excluded": "#2563d9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _polygon(value, label: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must contain four [x, y] points")
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label} must contain four [x, y] points")
        for axis in point:
            _number(axis, label)


def load_candidates(path: Path, page_size: tuple[int, int], scale: float) -> list[dict]:
    candidates = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(candidates, list):
        raise ValueError("candidate JSON must be an _ocr_candidates list")
    ids = set()
    for index, candidate in enumerate(candidates):
        label = f"candidate {index}"
        if not isinstance(candidate, dict) or not isinstance(candidate.get("element"), dict):
            raise ValueError(f"{label} must contain an element object")
        element = candidate["element"]
        text_id = element.get("id")
        if not isinstance(text_id, str) or not text_id or text_id in ids:
            raise ValueError(f"{label} must have a unique nonempty element.id")
        ids.add(text_id)
        if not isinstance(element.get("text"), str):
            raise ValueError(f"{label}.element.text must be a string")
        if _number(element.get("height"), f"{label}.element.height") <= 0:
            raise ValueError(f"{label}.element.height must be positive")
        if "confidence" in element:
            _number(element["confidence"], f"{label}.element.confidence")
        _polygon(candidate.get("quad"), f"{label}.quad")
        _polygon(element.get("polygon"), f"{label}.element.polygon")
        box = candidate.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{label}.bbox must contain [x0, y0, x1, y1]")
        x0, y0, x1, y1 = [_number(v, f"{label}.bbox") for v in box]
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{label}.bbox must have positive area")
        if x1 <= 0 or y1 <= 0 or x0 >= page_size[0] / scale or y0 >= page_size[1] / scale:
            raise ValueError(f"{label}.bbox does not intersect the page at the requested scale")
    return candidates


def _line_key(text, box, polygon):
    return text, tuple(box), tuple(tuple(point) for point in polygon)


def _draw_polygon(draw, polygon, scale, color, width):
    points = [(round(x * scale), round(y * scale)) for x, y in polygon]
    draw.line(points + points[:1], fill=color, width=width, joint="curve")


def make_overlay(rgb, independent, regions, scale):
    """Append a legend below the unscaled page; never paint over the input."""
    height, width = rgb.shape[:2]
    legend_height = 98
    overlay = Image.new("RGB", (max(width, 470), height + legend_height), "#ffffff")
    overlay.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(2, round(max(width, height) / 700))
    for index, region in enumerate(regions, 1):
        x0, y0, x1, y1 = region["pixelBox"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=COLORS["image"], width=line_width)
        draw.text((x0 + 3, y0 + 3), f"R{index:03d}", font=font, fill=COLORS["image"], stroke_width=1, stroke_fill="white")
        for line in region.get("retainedText", []):
            _draw_polygon(draw, line["polygon"], scale, COLORS["retained"], line_width)
        for line in region.get("excludedText", []):
            _draw_polygon(draw, line["polygon"], scale, COLORS["excluded"], line_width + 2)
    for candidate in independent:
        _draw_polygon(draw, candidate["quad"], scale, COLORS["independent"], line_width)
    draw.line((0, height, overlay.width, height), fill="#888888")
    legend = [
        ("image", "Image boundary (inferred)"),
        ("independent", "Independent text"),
        ("retained", "Text retained in image"),
        ("excluded", "Alpha text exclusion (outer blue)"),
    ]
    for row, (kind, label) in enumerate(legend):
        y = height + 7 + row * 17
        draw.rectangle((8, y + 1, 21, y + 10), outline=COLORS[kind], width=2)
        draw.text((28, y), label, font=font, fill="#222222")
    draw.text((8, height + 79), "REVIEW DRAFT - no human review or accuracy score", font=font, fill="#222222")
    return overlay


def inspect(image_path: Path, candidates_path: Path, scale: float, output: Path) -> dict:
    image_path, candidates_path, output = image_path.resolve(), candidates_path.resolve(), output.resolve()
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("--scale must be a positive finite number")
    if output.exists():
        raise FileExistsError(f"output directory already exists; choose a new directory: {output}")
    image_hash, candidates_hash = sha256(image_path), sha256(candidates_path)
    with Image.open(image_path) as page:
        rgb = np.asarray(page.convert("RGB")).copy()
    if min(rgb.shape[:2]) < 2:
        raise ValueError("page image must be at least 2 x 2 pixels")
    original = rgb.copy()
    candidates = load_candidates(candidates_path, (rgb.shape[1], rgb.shape[0]), scale)
    independent, regions, paper = image_regions(rgb, candidates, scale)
    # Fail before writing if future geometry code starts altering page pixels.
    if not np.array_equal(rgb, original):
        raise RuntimeError("boundary analysis changed source pixels")
    output.mkdir(parents=True, exist_ok=False)
    matches = defaultdict(list)
    owners = {c["element"]["id"]: [] for c in candidates}
    for candidate in candidates:
        element = candidate["element"]
        matches[_line_key(element["text"], candidate["bbox"], element["polygon"])].append(element["id"])
    independent_report = []
    for candidate in independent:
        element = candidate["element"]
        owners[element["id"]].append("independent")
        independent_report.append(dict(textId=element["id"], text=element["text"], bbox=candidate["bbox"],
                                       polygon=candidate["quad"], confidence=element.get("confidence"), reviewRequired=True))
    region_report = []
    for index, region in enumerate(regions, 1):
        region_id = f"region-{index:03d}"
        filename = f"{region_id}.png"
        Image.fromarray(region_pixels(rgb, region, scale)).save(output / filename)
        entry = dict(region, id=region_id, file=filename)
        entry["retainedText"] = []
        for line in region.get("retainedText", []):
            matched = matches[_line_key(line["text"], line["bbox"], line["polygon"])]
            entry["retainedText"].append(dict(line, candidateIds=matched))
            for text_id in matched:
                owners[text_id].append(region_id)
        region_report.append(entry)
    make_overlay(rgb, independent, regions, scale).save(output / "overlay.png")
    unchanged = (np.array_equal(rgb, original) and sha256(image_path) == image_hash
                 and sha256(candidates_path) == candidates_hash)
    if not unchanged:
        raise RuntimeError("input files or source pixels changed during inspection")
    ownership = [dict(textId=text_id, visibleOwners=sorted(set(items)), reviewRequired=True)
                 for text_id, items in owners.items()]
    report = dict(
        schemaVersion="boundary-review-v1", generatedAt=datetime.now(timezone.utc).isoformat(),
        method="foreground-barriers-v2", reviewRequired=True, humanReviewed=False, accuracyEvaluated=False,
        notice="Geometry review draft only; no OCR was run and no accuracy is claimed.",
        source=dict(image=str(image_path), imageSha256=image_hash, candidates=str(candidates_path),
                    candidatesSha256=candidates_hash, sourceUnchanged=unchanged, scale=scale,
                    pixelSize=[rgb.shape[1], rgb.shape[0]], coordinateSpace="top-left PDF points"),
        paperColor=paper, overlay="overlay.png", candidateCount=len(candidates),
        independentTextCount=len(independent), regionCount=len(regions), independentText=independent_report,
        regions=region_report, candidateOwnership=ownership,
        ownershipIssues=[item for item in ownership if len(item["visibleOwners"]) != 1],
        reviewChecklist=["Compare image boundaries with the original page.",
                         "Check next-paragraph text is independent, not retained in a figure.",
                         "Check alpha exclusions do not remove artwork.",
                         "Record human corrections separately before treating this as ground truth."],
    )
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="original page raster; read only")
    parser.add_argument("--candidates", required=True, type=Path, help="frozen _ocr_candidates JSON list")
    parser.add_argument("--scale", required=True, type=float, help="image pixels per PDF point")
    parser.add_argument("--output", required=True, type=Path, help="new, exclusive output directory")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.image, args.candidates, args.scale, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Boundary inspection failed: {exc}\n")
    print(json.dumps(dict(output=str(args.output.resolve()), candidates=report["candidateCount"],
                          independentText=report["independentTextCount"], regions=report["regionCount"],
                          ownershipIssues=len(report["ownershipIssues"]), reviewRequired=True), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
