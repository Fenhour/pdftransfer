"""Keep composite figures intact instead of scattering embedded image tiles."""
import re
import pymupdf as fitz

from .layout import inside

CAPTION = re.compile(r"^(?:Figure|Fig\.?|图)\s*(?:[A-Z][.\-])?\s*\d+\s*[:.：]?", re.I)


def figure_regions(page, native, tables):
    images = []
    for info in page.get_image_info():
        rect = fitz.Rect(info["bbox"]) & page.rect
        if rect.width >= 8 and rect.height >= 8 and not any(inside(rect, r, .97) and inside(r, rect, .97) for r in images):
            images.append(rect)
    drawings = [r for r in page.cluster_drawings() if r.width > 35 and r.height > 25 and r.get_area() < page.rect.get_area() * .9]
    captions = sorted([b for b in native if CAPTION.match(b.text)], key=lambda b: b.bbox[1])
    regions = []
    used_images = set()
    for caption in captions:
        bottom = caption.bbox[1] - 3
        candidates = list(drawings)
        # Transparent image canvases can extend into the caption even though
        # their visible artwork ends above it. Clip only modest overflow.
        for rect in images:
            if rect.y0 < bottom and 0 < rect.y1 - bottom <= max(28, rect.height * .2):
                candidates.append(fitz.Rect(rect.x0, rect.y0, rect.x1, bottom))
            else:
                candidates.append(rect)
        # Prefer nearby vector groups and image tiles immediately above a caption.
        seeds = [r for r in candidates if r.y1 <= bottom + 4 and bottom - r.y1 < 85
                 and r.x1 > caption.bbox[0] and r.x0 < caption.bbox[2]
                 and not any(inside(r, t.bbox, .4) for t in tables)]
        if not seeds:
            continue
        region = fitz.Rect(seeds[0])
        for seed in seeds[1:]:
            region |= seed
        # Expand through connected drawing groups and image tiles, but never
        # jump across a substantial paragraph or the preceding figure caption.
        for _ in range(6):
            prior = tuple(region)
            for candidate in candidates:
                if candidate.y1 > bottom + 4 or any(inside(candidate, t.bbox, .4) for t in tables):
                    continue
                expanded = fitz.Rect(region.x0 - 12, region.y0 - 28, region.x1 + 12, region.y1 + 12)
                if candidate.intersects(expanded):
                    region |= candidate
            if tuple(region) == prior:
                break
        if region.width < 70 or region.height < 30:
            continue
        # Include labels aligned with the visual, including those beyond a tile.
        for b in native:
            if b is caption or b.meta.get("vertical"):
                continue
            if b.bbox[1] >= region.y0 - 18 and b.bbox[3] <= bottom and b.bbox[0] >= caption.bbox[0] - 8 and b.bbox[2] <= caption.bbox[2] + 8:
                if b.bbox[3] - b.bbox[1] < region.height * .85:
                    region |= fitz.Rect(b.bbox)
        region = fitz.Rect(region.x0 - 2, region.y0 - 2, region.x1 + 2, min(bottom, region.y1 + 2)) & page.rect
        if any(inside(region, previous[0], .8) for previous in regions):
            continue
        regions.append((region, caption))
        used_images.update(i for i, r in enumerate(images) if inside(r, region, .65))
    for i, rect in enumerate(images):
        if i not in used_images:
            regions.append((rect, None))
    return sorted(regions, key=lambda item: (item[0].y0, item[0].x0))
