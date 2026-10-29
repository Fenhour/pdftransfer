"""Bounded typed-rebuild fitting, using synthetic pixels and installed fonts only."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from backend import font_library, rebuild_fonts


@pytest.fixture(scope='module')
def controlled_library():
    folder = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    names = ('arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf',
             'times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf')
    if not all((folder/name).is_file() for name in names):
        pytest.skip('Synthetic font fitting requires installed Arial and Times variants')
    installed = font_library.get_library()
    lib = font_library.FontLibrary([])
    lib.faces = {key:face for key,face in installed.faces.items()
                 if font_library._normal(face.family) in {'arial','timesnewroman'}
                 and not any(s in face.style.lower() for s in ('narrow','black','light'))}
    assert len(lib.faces) == 8
    return lib


@pytest.fixture
def library(monkeypatch,controlled_library):
    monkeypatch.setattr(rebuild_fonts,'get_library',lambda:controlled_library)
    monkeypatch.setattr(font_library,'get_library',lambda:controlled_library)
    return controlled_library


@pytest.fixture
def scene(tmp_path,library):
    def make(lines, *, size=24, shared_block=True, underline=False):
        docid = 'bounded-fonts'
        root = tmp_path/f'scene-{len(list(tmp_path.iterdir()))}'
        (root/'text').mkdir(parents=True)
        elements = []
        for index,parts in enumerate(lines):
            # parts are (text, family substring, bold, italic[, actual size]).
            resolved = []
            for part in parts:
                text,family,bold,italic,*sizes = part
                face = next(f for f in library.faces.values()
                            if family.lower() in f.family.lower() and f.bold==bold and f.italic==italic)
                actual = sizes[0] if sizes else size
                resolved.append((text,face,actual,library.pil_font(face.key,round(actual*3))))
            width = sum(font.getlength(text) for text,_,_,font in resolved)+18
            image = Image.new('RGBA',(round(width),round(size*6+24)),(0,0,0,0))
            draw = ImageDraw.Draw(image)
            x,baseline = 9,round(size*3.5+8)
            for text,_,_,font in resolved:
                draw.text((x,baseline),text,font=font,fill=(0,0,0,255),anchor='ls')
                x += font.getlength(text)
            if underline:
                draw.line((9,baseline+round(size*.8),image.width-9,baseline+round(size*.8)),fill='black',width=2)
            inkbox = image.getchannel('A').getbbox()
            assert inkbox
            image = image.crop((max(0,inkbox[0]-3),max(0,inkbox[1]-3),inkbox[2]+3,inkbox[3]+3))
            relative = f'text/line-{index}.png'
            image.save(root/relative)
            elements.append(dict(id=f'e-{index}',type='text',source='ocr',text=''.join(p[0] for p in parts),
                                 x=20.,y=20.+index*45,width=image.width/3,height=image.height/3,
                                 fontFamily='sans-serif',fontWeight='normal',fontStyle='normal',fontSize=size,
                                 appearanceSrc=f'/files/{docid}/{relative}',
                                 regionId='body',blockId='paragraph' if shared_block else f'b-{index}',lineId=f'l-{index}'))
        return dict(id=docid,pages=[dict(id='page-1',elements=elements)]),root
    return make


def _run_at(element, token):
    at = element['text'].index(token)+len(token)//2
    return next(r for r in element['fontRuns'] if r['start'] <= at < r['end'])


def _assert_contract(document):
    for element in document['pages'][0]['elements']:
        assert font_library.valid_runs(element)
        assert ''.join(r['text'] for r in element['fontRuns']) == element['text']
        assert all(r['start'] < r['end'] for r in element['fontRuns'])


def test_sampling_budget_scales_by_block_and_never_scans_all_faces(scene):
    document,root = scene([[(f'Continuous reading line number {i} remains ordinary prose.', 'Arial',False,False)] for i in range(35)])
    text = [e['text'] for e in document['pages'][0]['elements']]
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert [e['text'] for e in document['pages'][0]['elements']] == text
    assert stats['sampleCount'] <= rebuild_fonts.MAX_BLOCK_SAMPLES
    assert stats['faceComparisons'] <= stats['sampleCount']*rebuild_fonts.MAX_SAMPLE_WORDS*rebuild_fonts.MAX_FACES
    assert stats['wordComparisons'] <= rebuild_fonts.MAX_PAGE_STYLE_WORDS*rebuild_fonts.MAX_STYLE_FACES
    assert stats['matchedLines'] == 35
    assert stats['unreviewedStyleLines'] == 0
    sizes = [e['fontSize'] for e in document['pages'][0]['elements']]
    assert max(sizes)/min(sizes) < 1.08


@pytest.mark.parametrize('bold,italic',[(True,False),(False,True)])
def test_local_emphasis_keeps_style_with_shared_line_scale(scene,library,bold,italic):
    document,root = scene([[('Ordinary reading ', 'Times',False,False),
                            ('distinct emphasis ', 'Times',bold,italic),
                            ('returns to prose.', 'Times',False,False)]])
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    element = document['pages'][0]['elements'][0]
    first,middle,last = [_run_at(element,t) for t in ('Ordinary','emphasis','prose')]
    assert (library.faces[first['fontKey']].bold,library.faces[first['fontKey']].italic) == (False,False)
    assert (library.faces[middle['fontKey']].bold,library.faces[middle['fontKey']].italic) == (bold,italic)
    assert first['fontKey'] == last['fontKey']
    assert middle['fontSize'] == pytest.approx(first['fontSize'],rel=.08)
    assert len({round(r['baseline'],5) for r in element['fontRuns']}) == 1
    assert stats['wordComparisons'] > 0


def test_single_digit_fragment_uses_its_physical_line_scale(scene):
    document,root = scene([[('The presentation lasts ', 'Arial',False,False)],
                           [('1','Arial',False,False)]],size=24)
    a,b = document['pages'][0]['elements']
    b['lineId'] = a['lineId']
    b['x'] = a['x']+a['width']+5
    b['y'] = a['y']+3
    # OCR geometry around one narrow digit must not become a new font size.
    b['height'] *= 1.6
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert b['fontRuns'][0]['fontSize'] == pytest.approx(a['fontSize'],rel=.01)
    assert a['y']+a['fontRuns'][0]['baseline'] == pytest.approx(b['y']+b['fontRuns'][0]['baseline'])
    assert b['fontRuns'][0]['fontSize'] < 28


def test_rule_ink_does_not_inflate_line_size_or_change_source(scene):
    clean,root = scene([[('Reliable printed words anchor this line.', 'Arial',False,False)]])
    ruled,other = scene([[('Reliable printed words anchor this line.', 'Arial',False,False)]],underline=True)
    source = next((other/'text').glob('*.png'))
    before = source.read_bytes()
    rebuild_fonts.match_rebuild_fonts(clean,root)
    rebuild_fonts.match_rebuild_fonts(ruled,other)
    assert source.read_bytes() == before
    assert ruled['pages'][0]['elements'][0]['fontSize'] == pytest.approx(clean['pages'][0]['elements'][0]['fontSize'],rel=.06)


def test_supported_large_words_are_not_flattened_to_body_size(scene):
    document,root = scene([[('Ordinary reading ', 'Arial',False,False,24),
                            ('LARGER WORDS ', 'Arial',False,False,34),
                            ('returns to prose.', 'Arial',False,False,24)]])
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    element = document['pages'][0]['elements'][0]
    assert _run_at(element,'LARGER')['fontSize'] > _run_at(element,'Ordinary')['fontSize']*1.25
    assert stats['supportedSizeDifferences'] >= 1


def test_regions_do_not_share_font_decisions_and_native_manual_are_untouched(scene,library):
    document,root = scene([[('An independent serif paragraph.', 'Times',False,False)],
                           [('An independent sans paragraph.', 'Arial',False,False)]])
    elements = document['pages'][0]['elements']
    elements[1]['regionId'] = 'another-column'
    # Identical block IDs cannot make a region borrow a different region's style.
    native = dict(elements[0],id='native',source='native',text='Preserved native',fontSize=77)
    manual = dict(elements[0],id='manual',source='manual',text='User edited',fontSize=19)
    before = copy.deepcopy([native,manual])
    elements.extend([native,manual])
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    assert [native,manual] == before
    assert library.faces[elements[0]['fontRuns'][0]['fontKey']].generic == 'serif'
    assert library.faces[elements[1]['fontRuns'][0]['fontKey']].generic == 'sans-serif'
    assert stats['sampleCount'] == 2


def test_unsafe_faces_never_reach_geometry_and_missing_coverage_is_not_invented(scene,library,monkeypatch):
    document,root = scene([[('Real words need safe metrics.', 'Arial',False,False)]])
    from dataclasses import replace
    first = next(iter(library.faces.values()))
    unsafe = replace(first,key='unsafe-face',outline='cff')
    monkeypatch.setitem(library.faces,unsafe.key,unsafe)
    original = library.pil_font
    def guarded(key,*args,**kwargs):
        assert key != unsafe.key
        return original(key,*args,**kwargs)
    monkeypatch.setattr(library,'pil_font',guarded)
    rebuild_fonts.match_rebuild_fonts(document,root)
    assert all(library.faces[r['fontKey']].outline == 'truetype' for r in document['pages'][0]['elements'][0]['fontRuns'])
    element = document['pages'][0]['elements'][0]
    element['fontRuns'][0]['fontKey'] = unsafe.key
    assert font_library.valid_runs(element)
    rebuild_fonts.match_rebuild_fonts(document,root)
    assert all(r['fontKey'] != unsafe.key for r in element['fontRuns'])
    unknown = copy.deepcopy(document)
    unknown['pages'][0]['elements'][0].pop('fontRuns')
    unknown['pages'][0]['elements'][0].pop('fontMatch')
    unknown['pages'][0]['elements'][0]['text'] += '\U0010ffff'
    stats = rebuild_fonts.match_rebuild_fonts(unknown,root)
    assert unknown['pages'][0]['elements'][0]['text'].endswith('\U0010ffff')
    assert 'fontRuns' not in unknown['pages'][0]['elements'][0]
    assert stats['skippedLines'] == 1


def test_missing_ocr_spaces_do_not_trigger_beam_or_replace_current_characters(scene,monkeypatch):
    document,root = scene([[('No spaces were recognized', 'Arial',False,False)]])
    element = document['pages'][0]['elements'][0]
    element['text'] = element['text'].replace(' ','')
    monkeypatch.setattr(font_library,'_word_regions',lambda *args:pytest.fail('legacy beam matcher called'))
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert ''.join(r['text'] for r in element['fontRuns']) == 'Nospaceswererecognized'
    assert stats['matchedLines'] == 1


def test_cache_and_existing_runs_are_bounded_and_do_not_overwrite_edits(scene):
    document,root = scene([[('Reusable evidence for a paragraph.', 'Arial',False,False)]])
    cache = {}
    rebuild_fonts.match_rebuild_fonts(document,root,cache=cache)
    before = copy.deepcopy(document)
    stats = rebuild_fonts.match_rebuild_fonts(document,root,cache=cache)
    assert stats['faceComparisons'] == stats['wordComparisons'] == 0
    assert document == before
    assert len(cache['templates']) <= rebuild_fonts.MAX_TEMPLATE_CACHE
    element = document['pages'][0]['elements'][0]
    element['text'] = 'User current wording'
    after_edit = copy.deepcopy(element)
    stats = rebuild_fonts.match_rebuild_fonts(document,root,cache=cache)
    assert element == after_edit
    assert stats['skippedLines'] == 1


def test_many_blocks_report_sampling_limit_and_keep_every_character(scene,library):
    document,root = scene([[(f'An independent paragraph number {i} remains editable.', 'Arial',False,False)]
                           for i in range(40)],shared_block=False)
    text = [e['text'] for e in document['pages'][0]['elements']]
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert stats['sampleCount'] == rebuild_fonts.MAX_PAGE_SAMPLES
    assert stats['unsampledBlocks'] > 0
    assert stats['faceComparisons'] <= rebuild_fonts.MAX_PAGE_SAMPLES*rebuild_fonts.MAX_SAMPLE_WORDS*rebuild_fonts.MAX_FACES
    assert [e['text'] for e in document['pages'][0]['elements']] == text
    assert all(library.faces[e['fontRuns'][0]['fontKey']].generic == 'sans-serif'
               for e in document['pages'][0]['elements'])


def test_fitted_runs_pass_actual_visible_pdf_export_and_glyph_gate(scene):
    import pymupdf as fitz
    from backend.exporter import append_text_pages
    from backend.pdf_safety import character_inventory, prepare_fonts, validate_pdf
    document,root = scene([[('Actual visible ', 'Arial',False,False),
                            ('bold emphasis ', 'Arial',True,False),
                            ('has 15 minutes.', 'Arial',False,False)]])
    document['pages'][0].update(width=1200,height=200)
    rebuild_fonts.match_rebuild_fonts(document,root)
    prepare_fonts(document)
    _assert_contract(document)
    with fitz.open() as pdf:
        assert append_text_pages(pdf,document,root) == 1
        checked = validate_pdf(pdf.tobytes(),[character_inventory(document['pages'][0])])
        assert checked['referencedGlyphOutlines']
        assert checked['hiddenText'] is False


def test_missing_some_spaces_rejects_shifted_word_alignment(scene):
    document,root = scene([[('Several separate words provide reliable geometry before the final phrase.',
                            'Arial',False,False)]])
    element = document['pages'][0]['elements'][0]
    element['text'] = 'Severalseparatewords provide reliable geometrybeforethefinalphrase.'
    rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert len(element['fontRuns']) == 1
    assert element['fontRuns'][0]['width'] > element['width']*.9
    assert element['fontRuns'][0]['fontSize'] < 33


def test_neighboring_line_fragments_do_not_create_words_or_inflate_size(scene,library):
    text = 'A civil engineer contacted several reliable building experts.'
    clean,root = scene([[(text,'Arial',False,False)]])
    dirty,other = scene([[(text,'Arial',False,False)]])
    e = dirty['pages'][0]['elements'][0]
    path = other/e['appearanceSrc'].split('/files/bounded-fonts/')[1]
    with Image.open(path) as opened:
        original = opened.convert('RGBA')
    image = Image.new('RGBA',(original.width,original.height+24),(0,0,0,0))
    image.alpha_composite(original,(0,12))
    # Short partial ascenders/descenders distributed over an entire neighbor
    # row mimic dense OCR quads; their ink must never become a fake tiny word.
    font = library.pil_font(next(f.key for f in library.faces.values()
                                if f.family=='Arial' and not f.bold and not f.italic),72)
    neighbor = Image.new('RGBA',(original.width,100),(0,0,0,0))
    draw = ImageDraw.Draw(neighbor)
    draw.text((3,75),'neighboring printed descenders jumping quietly',font=font,fill='black',anchor='ls')
    image.alpha_composite(neighbor.crop((0,73,neighbor.width,80)),(0,0))
    image.alpha_composite(neighbor.crop((0,24,neighbor.width,31)),(0,image.height-7))
    image.save(path)
    e['height'] = image.height/3
    rebuild_fonts.match_rebuild_fonts(clean,root)
    rebuild_fonts.match_rebuild_fonts(dirty,other)
    _assert_contract(dirty)
    assert e['fontSize'] == pytest.approx(clean['pages'][0]['elements'][0]['fontSize'],rel=.06)
    assert all(r['width'] > r['fontSize']*.16 for r in e['fontRuns'] if len(r['text'].strip())>=3)


def test_long_intact_line_is_actually_sampled_and_keeps_regular_style(scene,library):
    document,root = scene([[('Long ordinary reading content repeated for a bounded visual comparison. '*4,
                            'Arial',False,False)]])
    element = document['pages'][0]['elements'][0]
    element['text'] = element['text'].replace(' ','')
    assert len(element['text']) > 160
    cache = {}
    stats = rebuild_fonts.match_rebuild_fonts(document,root,cache=cache)
    _assert_contract(document)
    assert stats['faceComparisons'] > 0
    face = library.faces[element['fontRuns'][0]['fontKey']]
    assert not face.bold and not face.italic
    assert all(mask.shape[1] <= 2048 for mask in cache['templates'].values())


def test_missing_spaces_line_uses_verified_same_block_scale(scene):
    lines = ['Ordinary printed reading contains several separate words.',
             'Another printed reading line provides reliable geometry.',
             'This printed reading line loses spaces but retains its size.']
    document,root = scene([[(text,'Arial',False,False)] for text in lines])
    element = document['pages'][0]['elements'][2]
    element['text'] = element['text'].replace(' ','')
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    sizes = [e['fontSize'] for e in document['pages'][0]['elements']]
    assert sizes[2] == pytest.approx(sum(sizes[:2])/2,rel=.08)
    assert stats['blockScaleAdjustments']+stats['intactLineSizeAdjustments'] >= 1


def test_missed_numeric_leaders_do_not_enlarge_the_recognized_digits(scene,library):
    document,root = scene([[('- 39 ....','Arial',False,False)]])
    element = document['pages'][0]['elements'][0]
    # The OCR string omits source marks but the appearance retains them.
    element['text'] = '39...'
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    face = library.faces[element['fontRuns'][0]['fontKey']]
    assert element['fontSize'] == pytest.approx(24,rel=.08)
    assert face.generic == 'sans-serif' and not face.bold and not face.italic
    assert element['text'] == '39...'
    assert stats['shortLabelSizeAdjustments'] == 1


def test_real_large_numeric_label_and_punctuation_keep_their_observed_scale(scene):
    document,root = scene([[('Ordinary body reading remains smaller.','Arial',False,False)],
                           [('39...','Arial',False,False,36)],
                           [('...','Arial',False,False,36)]],shared_block=False)
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    body,label,punctuation = document['pages'][0]['elements']
    assert label['fontSize'] == pytest.approx(36,rel=.08)
    assert punctuation['fontSize'] == pytest.approx(36,rel=.12)
    assert label['fontSize'] > body['fontSize']*1.35
    assert stats['shortLabelSizeAdjustments'] == 0


@pytest.mark.parametrize('family,bold,italic',[('Arial',False,False),('Times',True,True)])
def test_missing_interior_spaces_preserve_family_and_real_emphasis(scene,library,family,bold,italic):
    text = 'Important separate printed words retain the original family and emphasis.'
    document,root = scene([[(text,family,bold,italic)]])
    element = document['pages'][0]['elements'][0]
    element['text'] = 'Important separateprintedwordsretaintheoriginalfamilyand emphasis.'
    # Match a low resolution scan, while leaving the physical OCR geometry.
    path = root/element['appearanceSrc'].split('/files/bounded-fonts/')[1]
    with Image.open(path) as opened:
        image = opened.convert('RGBA')
    image.resize((round(image.width*.6),round(image.height*.6)),Image.Resampling.LANCZOS).save(path)
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert len(element['fontRuns']) == 1
    face = library.faces[element['fontRuns'][0]['fontKey']]
    assert family.lower() in face.family.lower()
    assert (face.bold,face.italic) == (bold,italic)
    assert stats['faceComparisons'] <= rebuild_fonts.MAX_SAMPLE_WORDS*rebuild_fonts.MAX_FACES


def test_capital_heading_does_not_borrow_the_following_italic_prose_style(scene,library):
    document,root = scene([[('ORDINARY PRINTED HEADING','Times',True,False)],
                           [('An ordinary italic introduction follows this heading.','Arial',False,True)]])
    heading,prose = document['pages'][0]['elements']
    # Equal OCR crop heights and a shared layout block still leave independent
    # typographic evidence for a capital heading and mixed-case prose.
    heading['height'] = prose['height']
    rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    a,b = [library.faces[e['fontRuns'][0]['fontKey']] for e in (heading,prose)]
    assert a.generic == 'serif' and a.bold and not a.italic
    assert b.generic == 'sans-serif' and not b.bold and b.italic


def test_missing_spaces_in_tracked_capitals_do_not_enlarge_character_height(scene):
    document,root = scene([[('READING      PRINTED      HEADING','Arial',False,False)]])
    element = document['pages'][0]['elements'][0]
    element['text'] = 'READINGPRINTEDHEADING'
    stats = rebuild_fonts.match_rebuild_fonts(document,root)
    _assert_contract(document)
    assert element['fontSize'] == pytest.approx(24,rel=.10)
    assert stats['intactLineSizeAdjustments'] == 1
