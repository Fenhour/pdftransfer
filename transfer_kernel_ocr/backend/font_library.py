"""Local font templates and conservative matching of continuous OCR style runs.

Font files stay in their installed locations. PDF exports embed the selected
faces; a page's OCR coordinates remain authoritative, never the font's layout.
Matching scores are visual distances, not recognition accuracy measurements.
"""
from __future__ import annotations

from .usage import tracked

from collections import Counter
import copy
from dataclasses import dataclass
from functools import lru_cache
import base64
import hashlib
import io
import math
import os
from pathlib import Path
import re

import cv2
from fontTools.ttLib import TTCollection, TTFont, TTLibError
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz

from .progress import progress_span, report


def _normal(name):
    return re.sub(r'[\s_\-]', '', name).casefold()


def _generic(name):
    name=_normal(name)
    if any(s in name for s in ('mono','courier','consolas')):
        return 'monospace'
    if 'sans' not in name and any(s in name for s in ('serif','times','cambria','georgia','simsun','宋','明朝')):
        return 'serif'
    return 'sans-serif'


def _unique_unicode_glyphs(font):
    """Disambiguate cmap aliases before constructing a PDF CID/Unicode map.

    E.g. Times New Roman shares one outline for hyphen and soft hyphen.
    TextWriter otherwise chooses a single Unicode for that glyph. Clone only
    the in-memory outline/metrics, preserving both source codepoints instead
    of deleting a cmap entry or repairing a finished PDF's ToUnicode by guess.
    """
    groups={}
    for code,glyph in (font.getBestCmap() or {}).items():
        groups.setdefault(glyph,[]).append(code)
    # glyf.__setitem__ also appends to its own glyphOrder. Work on a copy so
    # that the new glyph is not appended twice through a shared list.
    order=list(font.getGlyphOrder())
    for glyph,codes in groups.items():
        for code in sorted(codes)[1:]:
            alias=f'ptuni{code:06X}'
            while alias in font['glyf']:
                alias+='x'
            if len(order)>=65535:
                raise ValueError('字体别名字形超过 TrueType 上限，未生成不安全字体')
            font['glyf'][alias]=copy.deepcopy(font['glyf'][glyph])
            font['hmtx'].metrics[alias]=font['hmtx'].metrics[glyph]
            if 'vmtx' in font:
                font['vmtx'].metrics[alias]=font['vmtx'].metrics[glyph]
            order.append(alias)
            for table in font['cmap'].tables:
                if table.isUnicode() and code in table.cmap:
                    table.cmap[code]=alias
    font.setGlyphOrder(order)


@dataclass(frozen=True)
class FontFace:
    key: str
    path: str
    index: int
    family: str
    style: str
    bold: bool
    italic: bool
    source: str
    coverage: frozenset
    outline: str = 'truetype'
    variable: bool = False
    generic: str = 'sans-serif'


class FontLibrary:
    def __init__(self, directories=None):
        if directories is None:
            roaming=Path(os.environ.get('APPDATA',str(Path.home()/'AppData/Roaming')))
            local=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'AppData/Local')))
            directories=[(roaming/'kingsoft/office6/docerFonts','WPS'),
                         (Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts','system'),
                         (local/'Microsoft/Windows/Fonts','system'),
                         (Path(__file__).resolve().parents[1]/'fonts','custom')]
            directories += [(Path(p),'custom') for p in os.environ.get('OCR_FONT_DIRS','').split(os.pathsep) if p]
            if os.name!='nt':
                directories += [(Path('/usr/share/fonts'),'system'),(Path.home()/'.local/share/fonts','system')]
        self.faces={}
        seen=set()
        for entry in directories:
            folder,source=entry if isinstance(entry,tuple) else (Path(entry),'custom')
            folder=Path(folder)
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob('*')):
                if path.suffix.lower() not in {'.ttf','.otf','.ttc'} or not path.is_file():
                    continue
                # Subset and UI icon caches do not provide general text fonts.
                if 'subset' in path.parts or any(s in path.stem.lower() for s in ['icon','katex','symbol','wingding','webding']):
                    continue
                try:
                    with path.open('rb') as handle:
                        collection=handle.read(4)==b'ttcf'
                    if collection:
                        with TTCollection(path,lazy=True) as fonts:
                            count=len(fonts.fonts)
                    else:
                        count=1
                    for index in range(count):
                        font=TTFont(path,fontNumber=index,lazy=True)
                        try:
                            family=font['name'].getBestFamilyName() or path.stem
                            style=font['name'].getBestSubFamilyName() or 'Regular'
                            flags=font['OS/2'].fsSelection if 'OS/2' in font else 0
                            embedding=font['OS/2'].fsType if 'OS/2' in font else 0
                            if embedding & (0x2|0x100|0x200):
                                continue
                            # Some cached downloads have a readable name/cmap
                            # but a truncated or invalid head table.
                            head=font['head']
                            if 'hmtx' in font and font['hmtx'].metrics.get('.notdef',(0,0))[0]>head.unitsPerEm*5:
                                continue
                            coverage=frozenset((font.getBestCmap() or {}).keys())
                            if len(coverage)<70:
                                continue
                            bold=bool(flags&32) or 'bold' in style.lower() or 'black' in style.lower()
                            italic=bool(flags&1) or any(s in style.lower() for s in ['italic','oblique'])
                            identity=(_normal(family),_normal(style))
                            if identity in seen:
                                continue
                            seen.add(identity)
                            key=hashlib.sha256(f'{path.resolve()}:{index}'.encode()).hexdigest()[:20]
                            generic=_generic(family)
                            if 'post' in font and font['post'].isFixedPitch:
                                generic='monospace'
                            elif 'OS/2' in font and font['OS/2'].panose.bFamilyType==2:
                                serif=font['OS/2'].panose.bSerifStyle
                                if 2<=serif<=10:
                                    generic='serif'
                                elif 11<=serif<=15:
                                    generic='sans-serif'
                            self.faces[key]=FontFace(key,str(path),index,family,style,bold,italic,source,coverage,
                                                     'truetype' if 'glyf' in font else 'cff', 'fvar' in font,generic)
                        finally:
                            font.close()
                except (OSError,ValueError,KeyError,AssertionError,TTLibError):
                    continue

    def public_fonts(self):
        return [dict(key=f.key,family=f.family,style=f.style,bold=f.bold,italic=f.italic,source=f.source,
                     chinese=0x4e2d in f.coverage) for f in self.faces.values()]

    @lru_cache(maxsize=12)
    def font_for_key(self,key):
        face=self.faces[key]
        font=TTFont(face.path,fontNumber=face.index)
        # A shared space/NBSP glyph can otherwise be exported as NBSP by MuPDF.
        for table in font['cmap'].tables:
            if table.isUnicode() and table.cmap.get(32) and table.cmap.get(160)==table.cmap.get(32):
                table.cmap.pop(160)
        stream=io.BytesIO()
        font.save(stream)
        font.close()
        return fitz.Font(fontbuffer=stream.getvalue())

    @lru_cache(maxsize=256)
    def pil_font(self,key,size=48):
        face=self.faces[key]
        return ImageFont.truetype(face.path,size,index=face.index)

    def candidates(self,text,*,safe_only=False):
        needed={ord(c) for c in text if not c.isspace()}
        common=('arial','timesnewroman','calibri','cambria','georgia','couriernew','verdana','tahoma',
                'trebuchet','segoeui','simsun','simhei','microsoftyahei','kaiti','fangsong','dejavu','liberation')
        faces=[f for f in self.faces.values() if needed<=f.coverage and
               (not safe_only or f.outline=='truetype' and not f.variable)]
        faces.sort(key=lambda f:(not any(n in _normal(f.family) for n in common),f.source!='WPS',f.family,f.style))
        return faces[:160]

    def resolve(self,text,family='',bold=False,italic=False,*,safe_only=False):
        candidates=self.candidates(text,safe_only=safe_only)
        if not candidates:
            return None
        aliases={'sansserif':'arial','serif':'timesnewroman','monospace':'couriernew'}
        wanted=aliases.get(_normal(family),_normal(family))
        known=next((f for f in self.faces.values() if _normal(f.family)==wanted),None)
        generic=known.generic if known else _generic(family)
        return min(candidates,key=lambda f:(0 if _normal(f.family)==wanted else 1,
                    f.generic!=generic,
                    f.bold!=bold,f.italic!=italic,
                    any(word in f.style.lower() for word in ('narrow','condensed','expanded','black','light')),
                    0 if 'arial' in _normal(f.family) else 1))

    @lru_cache(maxsize=32)
    def export_font_for_key(self,key):
        """Isolated TrueType export path until CFF / variable roundtrips pass.

        Give each face a content-derived PDF name: different files with the
        same installed name must not collide in a multi-page font resource.
        Installed files are read-only and never renamed or redistributed.
        """
        face=self.faces[key]
        if face.outline!='truetype' or face.variable:
            raise ValueError('该字体尚未通过重建 PDF 导出验证，请使用相近的静态 TrueType 字体')
        font=TTFont(face.path,fontNumber=face.index,recalcTimestamp=False)
        try:
            _unique_unicode_glyphs(font)
            original=io.BytesIO()
            font.save(original)
            identity=hashlib.sha256(original.getvalue()).hexdigest()[:24]
            name='PT_'+identity
            for record in list(font['name'].names):
                # Keep family/full names for editor font matching. Only the
                # unique ID and PostScript resource identity need isolation.
                if record.nameID in {3,6}:
                    font['name'].setName(name,record.nameID,record.platformID,record.platEncID,record.langID)
            stream=io.BytesIO()
            font.save(stream)
            return fitz.Font(fontbuffer=stream.getvalue())
        finally:
            font.close()


@lru_cache(maxsize=1)
@tracked('fonts.load_library')
def get_library():
    return FontLibrary()


def _tight(mask):
    points=cv2.findNonZero(mask)
    if points is None:
        return None
    x,y,w,h=cv2.boundingRect(points)
    return mask[y:y+h,x:x+w],(x,y,w,h)


def _ink(image):
    rgba=np.asarray(image.convert('RGBA'))
    white=Image.new('RGBA',image.size,'white')
    white.alpha_composite(image.convert('RGBA'))
    composite=np.asarray(white)
    rgb=composite[:,:,:3].astype(float)
    alpha=rgba[:,:,3]>20
    if not alpha.any():
        return None
    # Appearance fragments have transparent paper. With opaque fixtures, infer
    # the paper from the image edges. Both dark and reversed text are supported.
    border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
    paper=np.median(border,axis=0)
    contrast=np.max(np.abs(rgb-paper),axis=2)
    values=contrast[alpha]
    if len(values)==0 or values.max()<15:
        return None
    threshold=max(18,float(np.percentile(values,90))*.32)
    mask=np.where(alpha & (contrast>threshold),255,0).astype(np.uint8)
    return _tight(mask)


@lru_cache(maxsize=3000)
def _template(key,text):
    lib=get_library()
    font=lib.pil_font(key)
    box=font.getbbox(text)
    w,h=box[2]-box[0],box[3]-box[1]
    if w<1 or h<1 or w>16000:
        raise ValueError('Invalid font template dimensions')
    im=Image.new('L',(w+4,h+4))
    ImageDraw.Draw(im).text((2-box[0],2-box[1]),text,font=font,fill=255)
    tight=_tight(np.asarray(im))
    return tight[0] if tight else np.zeros((1,1),np.uint8)


def _distance(target,template):
    h=min(42,target.shape[0])
    w=max(2,min(1400,round(target.shape[1]*h/target.shape[0])))
    a=cv2.resize(target,(w,h))>90
    b=cv2.resize(template,(w,h))>90
    da=cv2.distanceTransform((~a).astype(np.uint8),cv2.DIST_L2,3)
    db=cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,3)
    chamfer=(da[b].mean()+db[a].mean())/max(2,h*.15) if a.any() and b.any() else 2.
    ratio=(template.shape[1]/template.shape[0])/(target.shape[1]/target.shape[0])
    density=abs(float(a.mean())-float(b.mean()))
    return float(chamfer+density*.8+abs(math.log(max(.01,ratio)))*.16)


def _rank(mask,text,faces):
    scores={}
    for face in faces:
        try:
            scores[face.key]=_distance(mask,_template(face.key,text))
        except (OSError,ValueError):
            continue
    return scores


def _rank_line(mask,text,faces):
    """Compare aligned words so missed OCR spaces do not change the family."""
    lib=get_library()
    seed=lib.resolve(text,'Arial')
    if not seed or not re.search(r'[A-Za-z]',text):
        return _rank(mask,text,faces)
    regions=_word_regions(mask,text,seed.key)
    if len(regions)==1:
        return _rank(mask,text,faces)
    totals=Counter()
    weights=Counter()
    for start,end,x0,x1 in regions:
        value=text[start:end].strip()
        ink=_tight(mask[:,x0:x1])
        if not value or not ink or len(re.sub(r'[^A-Za-z]','',value))<2:
            continue
        weight=max(1,len(value))
        for key,score in _rank(ink[0],value,faces).items():
            face=lib.faces[key]
            familiar=any(name in _normal(face.family) for name in
                         ('arial','timesnewroman','calibri','cambria','georgia','courier','verdana','tahoma','trebuchet','segoe','comic','constantia'))
            totals[key]+=(score+(0 if familiar else .04))*weight
            weights[key]+=weight
    return {key:score/weights[key] for key,score in totals.items()} or _rank(mask,text,faces)


def _load_appearance(element,document_id,output_dir):
    src=element.get('originalAppearanceSrc') or element.get('appearanceSrc')
    if not src:
        return None
    if src.startswith('data:image/'):
        raw=base64.b64decode(src.split(',',1)[1],validate=True)
    else:
        prefix=f'/files/{document_id}/'
        if not src.startswith(prefix):
            return None
        path=(output_dir/src[len(prefix):]).resolve()
        if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
            return None
        raw=path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        return image.convert('RGBA')


def _word_regions(mask,text,key):
    """Align observed whitespace with OCR spans; do not invent recognized words."""
    height,width=mask.shape
    # An even closing kernel shifts its default anchor by one pixel. A thin
    # stroke can then appear as a group over empty ORIGINAL pixels, crashing
    # alignment or losing ink at a word boundary. Keep the kernel centered.
    kernel=np.ones((1,max(3,round(height*.25) | 1)),np.uint8)
    columns=np.any(cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)>0,axis=0)
    starts=np.flatnonzero(columns & ~np.r_[False,columns[:-1]])
    ends=np.flatnonzero(columns & ~np.r_[columns[1:],False])+1
    groups=list(zip(starts.tolist(),ends.tolist()))
    if any(not np.any(mask[:,a:b]) for a,b in groups):
        # Morphology is grouping evidence, not source content. If it proposes
        # an empty crop, match the intact line; never drop recognized text.
        return [(0,len(text),0,width)]
    words=list(re.finditer(r'\S+\s*',text))
    if len(groups)==len(words) and len(groups)>1:
        return [(m.start(),m.end(),a,b) for m,(a,b) in zip(words,groups)]
    if len(groups)<2 or len(groups)>len(text) or '\n' in text:
        return [(0,len(text),0,width)]
    if len(groups)>30 or len(text)>200:
        return [(0,len(text),0,width)]
    # Align characters to observed word images. Proportional midpoint cuts can
    # put the last letter of "Test" into the separate image of "1".
    font=get_library().pil_font(key)
    advances=np.array([font.getlength(text[:i]) for i in range(len(text)+1)])
    if advances[-1]<=0:
        return [(0,len(text),0,width)]
    total=sum(b-a for a,b in groups)
    beam=[(0.,0,[])]
    consumed=0
    for index,(x0,x1) in enumerate(groups):
        consumed+=x1-x0
        expected=int(np.argmin(abs(advances-consumed/total*advances[-1])))
        target=_tight(mask[:,x0:x1])[0]
        choices=[]
        for cost,start,regions in beam:
            ends=[len(text)] if index==len(groups)-1 else range(max(start+1,expected-4),min(len(text)-(len(groups)-index-1),expected+4)+1)
            for end in ends:
                value=text[start:end].strip()
                if not value:
                    continue
                template=_template(key,value)
                ratio=(template.shape[1]/template.shape[0])/(target.shape[1]/target.shape[0])
                score=_distance(target,template)+abs(math.log(max(.01,ratio)))*.7
                choices.append((cost+score,end,regions+[(start,end,x0,x1)]))
        beam=sorted(choices,key=lambda item:item[0])[:8]
        if not beam:
            return [(0,len(text),0,width)]
    return min(beam,key=lambda item:item[0])[2]


def _run(lib,face,text,start,end,box,element,image_size):
    x,y,w,h=box
    px=element['width']/image_size[0]
    py=element['height']/image_size[1]
    sample=lib.pil_font(face.key,64)
    glyph=sample.getbbox(text.rstrip() or text,anchor='ls')
    ink_height=max(1,glyph[3]-glyph[1])
    size=h*py/ink_height*64
    return {'text':text,'start':start,'end':end,'x':x*px,'y':y*py,'width':w*px,'height':h*py,
            'fontKey':face.key,'fontFamily':face.family,'fontWeight':'bold' if face.bold else 'normal',
            'fontStyle':'italic' if face.italic else 'normal','fontSize':min(1000,max(1,size))}


def ink_metrics(key,text,size=64):
    """Pillow's advance bbox includes space and digit side bearings."""
    font=get_library().pil_font(key,size)
    mask,offset=font.getmask2(text,anchor='ls')
    box=mask.getbbox()
    if not box:
        return font.getbbox(text,anchor='ls')
    return tuple(value+offset[index%2] for index,value in enumerate(box))


@tracked('fonts.match')
def match_document_fonts(document,output_dir):
    """Match OCR appearance, keeping a paragraph uniform unless evidence differs."""
    report(0,'加载本机字体','首次加载和比较字体可能需要较长时间')
    lib=get_library()
    output_dir=Path(output_dir)
    matched=[]
    total_pages=max(1,len(document['pages']))
    for page_index,page in enumerate(document['pages']):
        with progress_span(100*page_index/total_pages,100*(page_index+1)/total_pages):
            lines=[]
            candidates=[e for e in page['elements'] if e.get('type')=='text' and e.get('source')=='ocr' and e.get('text','').strip()]
            for line_index,element in enumerate(candidates):
                report(65*line_index/max(1,len(candidates)),'匹配文字字体',current=line_index+1,total=len(candidates))
                if element.get('type')!='text' or element.get('source')!='ocr' or not element.get('text','').strip():
                    continue
                if valid_runs(element):
                    matched.append(element['fontMatch'])
                    continue
                # appearanceSrc is removed by the editor after semantic edits.
                # Never infer the edited wording from originalAppearanceSrc alone.
                if not element.get('appearanceSrc'):
                    continue
                image=_load_appearance(element,document['id'],output_dir)
                evidence=_ink(image) if image else None
                if not evidence:
                    continue
                mask,box=evidence
                text=element['text']
                scores=_rank_line(mask,text,lib.candidates(text))
                if scores:
                    lines.append(dict(element=element,image=image,mask=mask,box=box,scores=scores))
            # Neighboring lines of the same scale form a continuity group. Clear
            # local differences can still override its shared family/style.
            groups=[]
            for line in sorted(lines,key=lambda l:(l['element']['y'],l['element']['x'])):
                e=line['element']
                if groups:
                    previous=groups[-1][-1]['element']
                    gap=max(e['x'],previous['x'])-min(e['x']+e['width'],previous['x']+previous['width'])
                    same=(gap<max(e['height'],previous['height'])*2 and
                          .78<e['height']/previous['height']<1.28 and
                          bool(re.search(r'[\u3400-\u9fff]',e['text']))==bool(re.search(r'[\u3400-\u9fff]',previous['text'])) and
                          0<=e['y']-previous['y']<previous['height']*5)
                else:
                    same=False
                if same:
                    groups[-1].append(line)
                else:
                    groups.append([line])
            for group_index,group in enumerate(groups):
                report(65+35*group_index/max(1,len(groups)),'匹配段落与样式',current=group_index+1,total=len(groups))
                keys=set.intersection(*(set(line['scores']) for line in group))
                def reliability(line):
                    return min(8.,(.20/max(.05,min(line['scores'].values())))**3)
                common=min(keys,key=lambda k:sum(l['scores'][k]*reliability(l) for l in group)) if keys else None
                for line in group:
                    e,mask=line['element'],line['mask']
                    best=min(line['scores'],key=line['scores'].get)
                    shared=lib.faces[common] if common else None
                    local=lib.faces[best]
                    same_style=shared and (shared.bold,shared.italic)==(local.bold,local.italic)
                    uncertain=line['scores'][best]>.32
                    if common and (uncertain or same_style and line['scores'][common]-line['scores'][best]<.16):
                        best=common
                    base=lib.faces[best]
                    regions=_word_regions(mask,e['text'],best)
                    choices=sorted(line['scores'],key=line['scores'].get)[:22]
                    choices+= [f.key for f in lib.candidates(e['text']) if _normal(f.family) in
                                {'arial','timesnewroman','calibri','cambria','georgia','couriernew'}]
                    faces=[lib.faces[key] for key in dict.fromkeys(choices)]
                    runs=[]
                    for start,end,x0,x1 in regions:
                        text=e['text'][start:end]
                        word=_tight(mask[:,x0:x1])
                        if not word or not text.strip():
                            continue
                        target,(lx,ly,w,h)=word
                        winner=best
                        if len(text.strip())>=3:
                            scores=_rank(target,text.strip(),faces)
                            if scores:
                                candidate=min(scores,key=scores.get)
                                variants=[key for key in scores if lib.faces[key].family==base.family]
                                variant=min(variants,key=scores.get) if variants else None
                                if variant and scores[variant]<scores[candidate]+.035:
                                    candidate=variant
                                gain=scores.get(best,2)-scores[candidate]
                                related=lib.faces[candidate].family==base.family
                                if gain>(.10 if related else .30) and scores[candidate]<(.16 if related else .22):
                                    winner=candidate
                        bx,by,_,_=line['box']
                        runs.append(_run(lib,lib.faces[winner],text,start,end,(bx+x0+lx,by+ly,w,h),e,line['image'].size))
                    if ''.join(r['text'] for r in runs)!=e['text']:
                        runs=[_run(lib,base,e['text'],0,len(e['text']),line['box'],e,line['image'].size)]
                    # Small punctuation and short glyphs such as Chinese 一 must
                    # not become tiny fonts merely because their ink is short.
                    # Reliable words determine a shared size and baseline.
                    evidence=[r for r in runs if len(re.sub(r'\W','',r['text']))>=2]
                    if evidence:
                        sizes=sorted((r['fontSize'],max(1,len(r['text'].strip()))) for r in evidence)
                        half=sum(weight for _,weight in sizes)/2
                        count=0
                        for size,weight in sizes:
                            count+=weight
                            if count>=half:
                                shared_size=size
                                break
                        for run in runs:
                            if run not in evidence or .75<run['fontSize']/shared_size<1.3:
                                run['fontSize']=shared_size
                        baselines=[]
                        for run in evidence:
                            glyph=ink_metrics(run['fontKey'],run['text'].strip())
                            baselines.append(run['y']-glyph[1]/64*run['fontSize'])
                        baseline=float(np.median(baselines))
                        for run in runs:
                            run['baseline']=baseline
                    # Join subdivisions of one word back into a text run. Keeping
                    # every narrow digit in a separate run creates false PDF spaces.
                    joined=[]
                    for run in runs:
                        previous=joined[-1] if joined else None
                        if (previous and previous['fontKey']==run['fontKey'] and
                                abs(previous['fontSize']-run['fontSize'])<.01 and
                                not previous['text'][-1:].isspace() and not run['text'][:1].isspace() and
                                run['x']-(previous['x']+previous['width'])<run['fontSize']*.14):
                            right=max(previous['x']+previous['width'],run['x']+run['width'])
                            bottom=max(previous['y']+previous['height'],run['y']+run['height'])
                            previous['y']=min(previous['y'],run['y'])
                            previous.update(text=previous['text']+run['text'],end=run['end'],
                                            width=right-previous['x'],height=bottom-previous['y'])
                        else:
                            joined.append(run)
                    runs=joined
                    # Keep all normal words in this line on the continuity font;
                    # only a substantial template improvement creates a style run.
                    e['fontRuns']=runs
                    e['fontFamily']=base.family
                    e['fontWeight']='bold' if base.bold else 'normal'
                    e['fontStyle']='italic' if base.italic else 'normal'
                    e['fontMatch']={'key':best,'family':base.family,'style':base.style,'source':base.source,
                                    'method':'local-font-templates','distance':round(line['scores'][best],4),
                                    'approximate':True,'text':e['text'],'geometry':[e['width'],e['height']],
                                    'inputStyle':[e.get(k) for k in ('fontFamily','fontWeight','fontStyle','fontSize')]}
                    matched.append(e['fontMatch'])
    report(100,'字体匹配完成')
    return {'templateCount':len(lib.faces),'matchedLines':len(matched),
            'families':dict(Counter(m['family'] for m in matched)),
            'policy':'continuous-style-with-local-differences; nearest-available-font'}


def valid_runs(element):
    info=element.get('fontMatch',{})
    runs=element.get('fontRuns',[])
    if not isinstance(info,dict) or not isinstance(runs,list) or not all(isinstance(r,dict) for r in runs):
        return False
    return (bool(runs) and len(runs)<=1000 and info.get('text')==element.get('text') and
            info.get('geometry')==[element['width'],element['height']] and
            info.get('inputStyle')==[element.get(k) for k in ('fontFamily','fontWeight','fontStyle','fontSize')] and
            all(isinstance(r.get('text'),str) for r in runs) and
            ''.join(r.get('text','') for r in runs)==element.get('text') and
            all(isinstance(r.get('fontKey'),str) and r['fontKey'] in get_library().faces and
                all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) and abs(r[k])<50000
                    for k in ('x','y','width','height','fontSize')) and
                ('baseline' not in r or isinstance(r['baseline'],(int,float)) and math.isfinite(r['baseline']) and abs(r['baseline'])<50000) and
                r['width']>0 and r['height']>0 and 0<r['fontSize']<=1000 for r in runs))
