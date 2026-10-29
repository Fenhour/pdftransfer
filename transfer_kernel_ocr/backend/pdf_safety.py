"""Reconstruction export gates, not an OCR or visual accuracy measurement."""

from .usage import tracked
from collections import Counter
import io
import logging
import statistics

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont
import pymupdf as fitz

from .font_library import get_library, valid_runs
from .progress import progress_span, report


@tracked('pdf.prepare_fonts')
def prepare_fonts(document):
    """Use matching static TrueType substitutes until unsafe formats are fixed."""
    report(0,'准备安全字体','正在读取本机字体模板')
    library=get_library()
    for index,page in enumerate(document['pages']):
        report(100*index/max(1,len(document['pages'])),'准备安全字体',current=index+1,total=len(document['pages']))
        for e in page['elements']:
            if (e['type']!='text' or not e.get('visible',True) or e.get('opacity',1)<=0
                    or not e.get('text','').strip()):
                continue
            e['safePdfFonts']=True
            replacements=[]
            if valid_runs(e):
                for run in e['fontRuns']:
                    face=library.faces[run['fontKey']]
                    if face.outline=='truetype' and not face.variable:
                        continue
                    substitute=library.resolve(run['text'],face.family,face.bold,face.italic,safe_only=True)
                    if not substitute:
                        raise ValueError('没有覆盖该文字的安全字体模板，未生成可能损坏的文本 PDF')
                    run['fontKey']=substitute.key
                    run['family']=substitute.family
                    replacements.append(dict(fromFamily=face.family,toFamily=substitute.family,
                                             reason='unverified-cff-or-variable'))
                runs=e['fontRuns']
                if (e.get('tableId') and 1<len(e['text'])<=4 and e['text'].isalnum() and
                        len(runs)>1 and len({r['fontKey'] for r in runs})==1):
                    # Ink-height differences between A and 1 must not split a
                    # short same-style cell value into independently-sized
                    # runs, which some PDF readers copy as "A 1".
                    left=min(r['x'] for r in runs)
                    top=min(r['y'] for r in runs)
                    right=max(r['x']+r['width'] for r in runs)
                    bottom=max(r['y']+r['height'] for r in runs)
                    joined=dict(runs[0],text=e['text'],start=0,end=len(e['text']),x=left,y=top,
                                width=right-left,height=bottom-top,
                                fontSize=statistics.median(r['fontSize'] for r in runs))
                    baselines=[r['baseline'] for r in runs if 'baseline' in r]
                    if baselines:
                        joined['baseline']=statistics.median(baselines)
                    e['fontRuns']=[joined]
            else:
                bold=str(e.get('fontWeight')) in {'bold','600','700','800','900'}
                face=library.resolve(e['text'],e.get('fontFamily',''),bold,e.get('fontStyle')=='italic',safe_only=True)
                if not face:
                    raise ValueError('没有覆盖该文字的安全字体模板，未生成可能损坏的文本 PDF')
                e['resolvedFontFamily']=face.family
            e['fontFallbacks']=replacements


def character_inventory(page):
    return Counter(c for e in page['elements'] if e['type']=='text' and e.get('visible',True)
                   and e.get('opacity',1)>0 for c in e.get('text','') if not c.isspace())


@tracked('pdf.validate')
def validate_pdf(payload, expected):
    """Check saved Unicode inventory AND actual referenced glyph outlines.

    Whitespace/order and visual similarity are separate benchmark dimensions.
    These structural gates cannot certify recognition or style correctness.
    """
    fonts={}
    outlines={}
    try:
        with fitz.open(stream=payload,filetype='pdf') as pdf:
            if len(pdf)!=len(expected):
                raise ValueError('重建 PDF 页数不一致')
            for index,(page,wanted) in enumerate(zip(pdf,expected)):
                report(100*index/max(1,len(expected)),'检查文字与字形',current=index+1,total=len(expected))
                resources={}
                for item in page.get_fonts(full=True):
                    xref,_,_,name,*_=item
                    if xref not in fonts:
                        data=pdf.extract_font(xref)[3]
                        if not data:
                            raise ValueError('重建文字字体未嵌入，无法验证字形')
                        fonts[xref]=TTFont(io.BytesIO(data),lazy=True)
                    resources[name.split('+')[-1]]=xref
                    # MuPDF may append an abbreviated style to /BaseFont,
                    # while texttrace uses the embedded PostScript name.
                    for record in fonts[xref]['name'].names:
                        if record.nameID in {1,4,6,16}:
                            resources[record.toUnicode().split('+')[-1]]=xref
                found=Counter()
                for span in page.get_texttrace():
                    if span.get('type')==3:
                        raise ValueError('重建正文出现隐藏文字层')
                    for code,gid,*_ in span['chars']:
                        char=chr(code)
                        if char.isspace():
                            continue
                        found[char]+=1
                        trace_name=span['font'].split('+')[-1]
                        xref=resources.get(trace_name)
                        if xref is None and len(trace_name)>=20:
                            # Native subsetting can truncate MuPDF's display
                            # name (e.g. Times New Roman Bold Italic). Resolve
                            # only a unique embedded font, never an ambiguous
                            # family prefix; glyphs still need full validation.
                            matches={ref for name,ref in resources.items() if name.startswith(trace_name)}
                            if len(matches)==1:
                                xref=matches.pop()
                        if xref is None:
                            raise ValueError('无法验证 PDF 文字对应的嵌入字体')
                        key=(xref,gid)
                        if key not in outlines:
                            font=fonts[xref]
                            order=font.getGlyphOrder()
                            if not 0<gid<len(order):
                                raise ValueError('PDF 文字引用无效字形')
                            glyphs=font.getGlyphSet()
                            pen=BoundsPen(glyphs)
                            glyphs[order[gid]].draw(pen)
                            outlines[key]=pen.bounds is not None
                        if not outlines[key]:
                            raise ValueError('PDF 文字引用空字形，已阻止损坏结果下载')
                if found!=wanted:
                    raise ValueError('PDF 保存后的文字映射与重建内容不一致，已阻止下载')
        return dict(characterInventory=True,referencedGlyphOutlines=True,hiddenText=False)
    finally:
        for font in fonts.values():
            font.close()


@tracked('pdf.save_verified')
def save_verified(pdf,destination,expected):
    report(0,'保存并验证 PDF','正在检查压缩前的文字完整性')
    before=pdf.tobytes(deflate=True,garbage=3)
    with progress_span(0,30):
        checks={'beforeSubset':validate_pdf(before,expected)}
    try:
        # The fontTools fallback writes temporary font files which can stay
        # locked on Windows after a variable-font failure. MuPDF subsets in
        # memory; verify its result before making it downloadable.
        report(30,'压缩 PDF 字体','压缩失败时将保留通过校验的完整字体')
        pdf.subset_fonts(fallback=False)
        after=pdf.tobytes(deflate=True,garbage=3)
        with progress_span(40,70):
            checks['afterSubset']=validate_pdf(after,expected)
    except Exception:
        # Subsetting only reduces file size. Keep the verified original bytes
        # if this optional step fails or changes the text / glyph outlines.
        logging.exception('PDF font subsetting failed; preserving verified full fonts')
        after=before
        checks['subsetFallback']='verified-full-fonts'
        report(70,'保留完整字体','字体压缩检查未通过，使用已验证的完整字体版本')
    # Only a checked byte stream becomes a downloadable result.
    report(75,'写出 PDF 文件')
    destination.write_bytes(after)
    with progress_span(80,100):
        checks['reopened']=validate_pdf(destination.read_bytes(),expected)
    report(100,'PDF 已生成并通过检查')
    return checks
