"""Font matching and visible text PDF behavior, using local synthetic scans.

No OCR, downloaded fonts, private samples, or exact font-name accuracy scores.
Matcher-dependent cases skip only while backend.font_library does not exist;
once present, missing APIs and behavioral regressions fail normally.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import math
import os
from pathlib import Path
import unicodedata

import pymupdf as fitz
import pytest
from PIL import Image
from fontTools.ttLib import TTFont

from backend.exporter import write_text_pdf
from backend.models import validate_document


@pytest.fixture(scope="module")
def system_faces():
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    names = {
        "arial": "arial.ttf", "arial-bold": "arialbd.ttf", "arial-italic": "ariali.ttf",
        "times": "times.ttf", "times-bold": "timesbd.ttf", "times-italic": "timesi.ttf",
    }
    missing = [name for name in names.values() if not (folder / name).is_file()]
    if missing:
        pytest.skip("Synthetic scan requires installed Arial/Times faces: " + ", ".join(missing))
    return {key: fitz.Font(fontfile=str(folder / name)) for key, name in names.items()}


@pytest.fixture(scope="module")
def font_api():
    if importlib.util.find_spec("backend.font_library") is None:
        pytest.skip("Font library interface is pending; fixture/export-only cases still run")
    return importlib.import_module("backend.font_library")


@pytest.fixture(scope="module")
def library(font_api):
    return font_api.get_library()


def _traits(font):
    return tuple(bool(font.flags.get(key)) for key in ("serif", "bold", "italic"))


def _plain(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _face_entry(library, system_faces, face):
    # Select a controlled template for exporter tests, not a matcher prediction.
    # Family aliases/subset prefixes are fine; style comes from the font program.
    family = face.split("-")[0]
    expected = _traits(system_faces[face])
    for entry in library.public_fonts():
        if family not in entry["family"].casefold():
            continue
        if any(word in entry['style'].casefold() for word in ('narrow','condensed','expanded','black','light')):
            continue
        font = library.font_for_key(entry["key"])
        if _traits(font) == expected:
            return entry
    pytest.fail(f"Installed {face} face is absent from the template library")


@pytest.fixture
def scene_factory(tmp_path, system_faces):
    """Render actual font programs, then discard all native text from the scan.

    Scene appearance PNGs use the parser's transparent tight-crop convention.
    Input fontFamily/style are deliberately generic so matching must inspect
    appearanceSrc. Truth offsets are kept outside the scene document.
    """
    def make(lines, *, size=20):
        document_id = "font-fixture"
        root = tmp_path / f"scene-{len(list(tmp_path.glob('scene-*')))}"
        (root / "text").mkdir(parents=True)
        (root / "pages").mkdir()
        scale = 3
        page_width, page_height = 720, max(180, 60 + len(lines) * 48)
        scan = Image.new("RGB", (page_width * scale, page_height * scale), "white")
        elements, truth = [], []
        for index, parts in enumerate(lines):
            fonts = [system_faces[face] for _, face in parts]
            advances = [font.text_length(text, fontsize=size) for (text, _), font in zip(parts, fonts)]
            ascender = max(font.ascender for font in fonts)
            descender = min(font.descender for font in fonts)
            baseline = 6 + ascender * size
            with fitz.open() as original:
                page = original.new_page(width=sum(advances) + 12,
                                         height=(ascender - descender) * size + 12)
                writer = fitz.TextWriter(page.rect)
                cursor = 6
                for (text, _), font, advance in zip(parts, fonts, advances):
                    writer.append((cursor, baseline), text, font=font, fontsize=size)
                    cursor += advance
                writer.write_text(page)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
                line_width, line_height = page.rect.width, page.rect.height
                # Isolate each painted run on a same-size transparent page.
                # Actual ink bounds (including italic overhangs) differ from
                # advance widths; keep the shared baseline as fixture evidence.
                ink_bounds = []
                cursor = 6
                for (text, _), font, advance in zip(parts, fonts, advances):
                    run_page = original.new_page(width=line_width, height=line_height)
                    run_writer = fitz.TextWriter(run_page.rect)
                    run_writer.append((cursor, baseline), text, font=font, fontsize=size)
                    run_writer.write_text(run_page)
                    run_pix = run_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
                    run_image = Image.frombytes("RGBA", (run_pix.width, run_pix.height), run_pix.samples)
                    ink_bounds.append(run_image.getchannel("A").getbbox())
                    cursor += advance
            image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
            bounds = image.getchannel("A").getbbox()
            assert bounds, "Fixture must contain visible ink"
            left, top, right, bottom = bounds
            crop_left, crop_top = max(0, left - 2), max(0, top - 2)
            crop = image.crop((crop_left, crop_top, min(image.width, right + 2), min(image.height, bottom + 2)))
            relative = f"text/line-{index}.png"
            crop.save(root / relative)
            x, y = 24, 24 + index * 48
            scan.paste(crop, (x * scale, y * scale), crop)
            text = "".join(text for text, _ in parts)
            elements.append({
                "id": f"line-{index}", "type": "text", "source": "ocr", "textRole": "independent",
                "text": text, "x": x, "y": y, "width": crop.width / scale, "height": crop.height / scale,
                "fontFamily": "sans-serif", "fontSize": size, "fontWeight": "normal", "fontStyle": "normal",
                "color": "#000000",
                "rotation": 0, "zIndex": index, "appearanceSrc": f"/files/{document_id}/{relative}",
                "originalAppearanceSrc": f"/files/{document_id}/{relative}",
            })
            start, runs = 0, []
            for (value, face), box in zip(parts, ink_bounds):
                assert box, "Each truth run must contain visible ink"
                ink_left, ink_top, ink_right, ink_bottom = box
                runs.append({"text": value, "face": face, "start": start, "end": start + len(value),
                             "x": (ink_left - crop_left) / scale, "y": (ink_top - crop_top) / scale,
                             "width": (ink_right - ink_left) / scale, "height": (ink_bottom - ink_top) / scale,
                             "baseline": baseline - crop_top / scale, "fontSize": size})
                start += len(value)
            truth.append(runs)
        scan.save(root / "pages/scan.png")
        with fitz.open() as raster:
            raster.new_page(width=page_width, height=page_height).insert_image(
                fitz.Rect(0, 0, page_width, page_height), filename=str(root / "pages/scan.png"))
            raster.save(root / "scan.pdf")
        document = validate_document({
            "id": document_id, "name": "Synthetic font behavior", "pages": [{
                "id": "page-1", "index": 0, "width": page_width, "height": page_height,
                "sourceSrc": f"/files/{document_id}/pages/scan.png", "elements": elements,
            }],
        })
        return document, root, truth
    return make


def _install_runs(document, truth, library, system_faces):
    """Populate the documented fontRuns contract to isolate PDF rendering."""
    for element, expected in zip(document["pages"][0]["elements"], truth):
        runs = []
        for source in expected:
            entry = _face_entry(library, system_faces, source["face"])
            _, bold, italic = _traits(system_faces[source["face"]])
            runs.append({key: source[key] for key in ("text", "start", "end", "x", "y", "width", "height", "fontSize")})
            runs[-1].update(fontKey=entry["key"], fontFamily=entry["family"],
                            fontWeight="bold" if bold else "normal", fontStyle="italic" if italic else "normal")
        element["fontRuns"] = runs
        element["fontMatch"] = dict(_face_entry(library, system_faces, expected[0]["face"]))
        element["fontMatch"].update(
            text=element["text"], geometry=[element["width"], element["height"]],
            inputStyle=[element.get(key) for key in ("fontFamily", "fontWeight", "fontStyle", "fontSize")],
        )
        assert importlib.import_module("backend.font_library").valid_runs(element), "Fixture must exercise the matched-run path"


def _assert_runs(element, library):
    match = element["fontMatch"]
    assert all(match.get(key) for key in ("key", "family", "style"))
    assert match["text"] == element["text"]
    assert match["geometry"] == [element["width"], element["height"]]
    assert match["inputStyle"] == [element.get(key) for key in ("fontFamily", "fontWeight", "fontStyle", "fontSize")]
    assert isinstance(library.font_for_key(match["key"]), fitz.Font)
    runs = element["fontRuns"]
    assert runs and "".join(run["text"] for run in runs) == element["text"]
    cursor = 0
    for run in runs:
        assert run["start"] == cursor and run["end"] > cursor
        assert element["text"][run["start"]:run["end"]] == run["text"]
        assert all(math.isfinite(run[key]) for key in ("x", "y", "width", "height", "fontSize"))
        assert run["width"] > 0 and run["height"] > 0 and run["fontSize"] > 0
        assert run["fontFamily"] and run["fontWeight"] is not None
        assert run["fontStyle"] in {"normal", "italic", "oblique"}
        assert isinstance(library.font_for_key(run["fontKey"]), fitz.Font)
        cursor = run["end"]
    assert cursor == len(element["text"])
    return runs


def _at(element, token):
    offset = element["text"].index(token) + len(token) // 2
    return next(run for run in element["fontRuns"] if run["start"] <= offset < run["end"])


def _spans(page):
    return [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
            for line in block["lines"] for span in line["spans"]]


def _embedded(pdf, page):
    result = []
    for xref in {row[0] for row in page.get_fonts()}:
        _, extension, _, program = pdf.extract_font(xref)
        assert extension != "n/a" and program, "Visible text font must be embedded"
        result.append(fitz.Font(fontbuffer=program))
    assert result, "A bitmap or attachment alone is not a text PDF"
    return result


def test_scan_fixture_really_has_only_pixels_and_distinct_font_styles(scene_factory, system_faces):
    document, root, _ = scene_factory([[('Serif body ', 'times'), ('bold emphasis', 'arial-bold')],
                                      [('A continuous italic line', 'times-italic')]])
    with fitz.open(root / "scan.pdf") as source:
        assert source[0].get_text() == "" and len(source[0].get_images()) == 1
    assert _traits(system_faces["times"]) == (True, False, False)
    assert _traits(system_faces["arial-bold"]) == (False, True, False)
    assert _traits(system_faces["times-italic"]) == (True, False, True)
    assert all(element["appearanceSrc"] for element in document["pages"][0]["elements"])


def test_template_catalog_has_stable_keys_and_loadable_real_font_variants(library, system_faces):
    entries = library.public_fonts()
    assert entries and len({entry["key"] for entry in entries}) == len(entries)
    assert [entry["key"] for entry in entries] == [entry["key"] for entry in library.public_fonts()]
    assert all(entry.get("family") and entry.get("style") for entry in entries)
    for face in system_faces:
        entry = _face_entry(library, system_faces, face)
        assert _traits(library.font_for_key(entry["key"])) == _traits(system_faces[face])


def test_font_catalog_rejects_invalid_notdef_metrics(tmp_path,system_faces,font_api):
    folder=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    with TTFont(folder/'arial.ttf') as font:
        font['hmtx'].metrics['.notdef']=(65535,0)
        font.save(tmp_path/'invalid.ttf')
    assert not font_api.FontLibrary([tmp_path]).faces


@pytest.mark.parametrize("face", ["arial", "times-italic"])
def test_matching_keeps_a_continuous_paragraph_in_one_style(scene_factory, system_faces, font_api, library, face):
    document, root, _ = scene_factory([
        [("Continuous reading keeps the same clear style.", face)],
        [("Follow the paragraph without changing its typeface.", face)],
        [("The final line belongs to the same paragraph.", face)],
    ])
    font_api.match_document_fonts(document, root)
    keys = set()
    for element in document["pages"][0]["elements"]:
        for run in _assert_runs(element, library):
            if run["text"].strip():
                assert _traits(library.font_for_key(run["fontKey"])) == _traits(system_faces[face])
                keys.add(run["fontKey"])
    assert len(keys) == 1, "One paragraph must not drift between unrelated template faces"


def test_same_sentence_can_distinguish_serif_body_and_sans_bold(scene_factory, font_api, library):
    document, root, _ = scene_factory([[
        ("Classical reading ", "times"), ("bold emphasis", "arial-bold"), (" returns to prose.", "times"),
    ]])
    font_api.match_document_fonts(document, root)
    element = document["pages"][0]["elements"][0]
    _assert_runs(element, library)
    first, middle, last = [_at(element, token) for token in ("Classical", "emphasis", "prose")]
    assert _traits(library.font_for_key(first["fontKey"])) == (True, False, False)
    assert _traits(library.font_for_key(middle["fontKey"])) == (False, True, False)
    assert first["fontKey"] == last["fontKey"] != middle["fontKey"]
    assert middle["x"] > first["x"] and last["x"] > middle["x"], "Run x is an element-relative point offset"


@pytest.mark.parametrize('emphasis',['times-bold','times-italic'])
def test_same_family_emphasis_preserves_bold_and_italic(scene_factory, system_faces, font_api, library, emphasis):
    document,root,_=scene_factory([[('Classical reading ', 'times'),
                                  ('distinct emphasis ',emphasis),('returns to prose.','times')]])
    font_api.match_document_fonts(document,root)
    element=document['pages'][0]['elements'][0]
    _assert_runs(element,library)
    first,middle,last=[_at(element,token) for token in ('Classical','emphasis','prose')]
    assert _traits(library.font_for_key(first['fontKey']))==(True,False,False)
    assert _traits(library.font_for_key(middle['fontKey']))==_traits(system_faces[emphasis])
    assert first['fontKey']==last['fontKey']!=middle['fontKey']


def test_missing_ocr_space_keeps_the_two_observed_word_positions(scene_factory, font_api, library):
    document,root,_=scene_factory([[('Test ', 'arial'),('1','arial')]],size=26)
    element=document['pages'][0]['elements'][0]
    element['text']='Test1'
    font_api.match_document_fonts(document,root)
    runs=_assert_runs(element,library)
    first,last=_at(element,'Test'),_at(element,'1')
    assert last['fontSize']==pytest.approx(first['fontSize'],rel=.1)
    assert last['x']>first['x']+first['width'], 'Missed OCR whitespace must not absorb a letter into the digit image'


@pytest.mark.parametrize('field,value',[('fontMatch',[]),('fontRuns',{}),('fontRuns',[None])])
def test_malformed_font_metadata_is_ignored(scene_factory,font_api,field,value):
    document,_,_=scene_factory([[('Safe current text','arial')]])
    element=document['pages'][0]['elements'][0]
    element[field]=value
    assert not font_api.valid_runs(element)


def test_pdf_uses_distinct_embedded_fonts_and_visible_selectable_spans(scene_factory, system_faces, library):
    document, root, truth = scene_factory([[
        ("Classical reading ", "times"), ("bold emphasis", "arial-bold"), (" italic ending", "times-italic"),
    ]])
    _install_runs(document, truth, library, system_faces)
    destination = root / "visible-text.pdf"
    write_text_pdf(document, root, destination)
    with fitz.open(destination) as pdf:
        page = pdf[0]
        assert _plain(page.get_text()) == _plain(document["pages"][0]["elements"][0]["text"])
        assert page.search_for("emphasis"), "PDF selection/search must use actual page text"
        assert all(trace["type"] != 3 for trace in page.get_texttrace())
        assert not page.get_images(), "Text appearance crops must not replace semantic text"
        fonts = _embedded(pdf, page)
        assert {(True, False, False), (False, True, False), (True, False, True)} <= {_traits(font) for font in fonts}
        spans = _spans(page)
        assert len({span["font"] for span in spans}) >= 3
        assert len(spans) >= 3, "A mixed sentence must not collapse to one replacement font"
        assert page.get_pixmap().samples.count(0) > 0, "Visible text must actually paint ink"


@pytest.mark.parametrize("with_runs", [False, True])
def test_ocr_pdf_does_not_stretch_glyphs_to_fill_a_wide_box(scene_factory, system_faces, request, with_runs):
    text = "MMMMiiiiWW00"
    document, root, truth = scene_factory([[(text, "arial")]])
    element = document["pages"][0]["elements"][0]
    element.update(fontFamily="Arial", width=truth[0][0]["width"] * 1.8, height=60)
    if with_runs:
        _install_runs(document, truth, request.getfixturevalue("library"), system_faces)
    else:
        # No current appearance means this exercises fallback, not a fresh
        # automatic match interpreting the larger box as a resized scan.
        element.pop("appearanceSrc", None)
    destination = root / "natural-advances.pdf"
    write_text_pdf(document, root, destination)
    with fitz.open(destination) as pdf:
        chars = [char for trace in pdf[0].get_texttrace() for char in trace["chars"]]
        assert "".join(chr(char[0]) for char in chars) == text
        advance = chars[-1][2][0] - chars[0][2][0]
        _embedded(pdf, pdf[0])
        resources = pdf[0].get_fonts()
        assert len(resources) == 1
        # Fallback may select a different installed face, such as a narrow
        # variant. Measure against the actual embedded outline, not an exact
        # font-name/metric guess. An anisotropic fit still changes this ratio.
        # Use trace glyph IDs with the embedded program's advance table:
        # subset programs need not retain their original Unicode cmap.
        with TTFont(io.BytesIO(pdf.extract_font(resources[0][0])[3])) as program:
            order = program.getGlyphOrder()
            advances = program["hmtx"].metrics
            expected = sum(advances[order[char[1]]][0] for char in chars[:-1]) / program["head"].unitsPerEm * 20
        assert advance == pytest.approx(expected, rel=.05, abs=.5), "OCR box width must not distort glyph advances"


@pytest.mark.parametrize("edit", ["shorter", "same-length", "longer"])
def test_current_text_invalidates_old_runs_and_remains_selectable(scene_factory, system_faces, library, edit):
    document, root, truth = scene_factory([[("Original serif ", "times"), ("bold words", "arial-bold")]])
    _install_runs(document, truth, library, system_faces)
    element = document["pages"][0]["elements"][0]
    original = element["text"]
    new_text = {"shorter": "New text 99", "same-length": "Edited " + "X" * (len(original) - 7),
                "longer": "New wording replaces the complete prior sentence with current text 2026"}[edit]
    # Simulate editing the selected element, while an older client leaves stale
    # fontRuns in the scene. Never let old run text overwrite current text.
    element["text"] = new_text
    assert not importlib.import_module("backend.font_library").valid_runs(element), "Text-only edits must invalidate the snapshot"
    element.update(x=65, y=70, width=620, height=45)
    element.pop("appearanceSrc", None)
    destination = root / "edited.pdf"
    write_text_pdf(document, root, destination)
    with fitz.open(destination) as pdf:
        page = pdf[0]
        assert _plain(page.get_text()) == _plain(new_text)
        assert not page.search_for("Original") and page.search_for(new_text.split()[0])
        assert all(trace["type"] != 3 for trace in page.get_texttrace())
        for word in page.get_text("words"):
            assert word[0] >= element["x"] - 2 and word[1] >= element["y"] - 2


def test_missing_template_glyph_falls_back_to_visible_embedded_unicode(scene_factory, system_faces, library):
    font = system_faces["arial"]
    fallback = fitz.Font("cjk")
    glyph = next((value for value in "中文漢" if not font.has_glyph(ord(value), fallback=False)
                  and fallback.has_glyph(ord(value), fallback=False)), None)
    if glyph is None:
        pytest.skip("Requires a CJK glyph absent from installed Arial but available in the PDF fallback")
    document, root, truth = scene_factory([[(f"Before {glyph} After", "arial")]])
    element = document["pages"][0]["elements"][0]
    element.update(width=350, height=45)
    _install_runs(document, truth, library, system_faces)
    destination = root / "glyph-fallback.pdf"
    write_text_pdf(document, root, destination)
    with fitz.open(destination) as pdf:
        page = pdf[0]
        assert _plain(page.get_text()) == _plain(element["text"])
        assert page.search_for(glyph)
        glyphs = [char for trace in page.get_texttrace() if trace["type"] != 3
                  for char in trace["chars"] if char[0] == ord(glyph)]
        assert glyphs and all(char[1] != 0 for char in glyphs), "Fallback must paint a real glyph, not .notdef"
        fonts = _embedded(pdf, page)
        assert len(fonts) >= 2, "The unavailable Arial glyph needs an embedded fallback face"
        # Subset CID programs can omit a Unicode cmap; has_glyph(unicode) on
        # the extracted program is not evidence about PDF CID/ToUnicode use.
        # Require actual painted ink at the selectable fallback glyph instead.
        clip = page.search_for(glyph)[0]
        pixels = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip, colorspace=fitz.csGRAY)
        assert min(pixels.samples) < 150, "Fallback text must visibly paint its selected region"
