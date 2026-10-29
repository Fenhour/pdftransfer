"""PDF-independent HTML / Markdown adapters for the reconstruction model."""
from __future__ import annotations

from .usage import tracked

import copy
import html
import json
from pathlib import Path
import re

from .exporter import read_asset
from .font_library import get_library, valid_runs
from .rebuild_flow import flow_cell_text, is_cell_flow


def esc(value):
    return html.escape(str(value),quote=True)


def md_text(value):
    return re.sub(r'([\\`*_{}\[\]<>()#!|>])',r'\\\1',str(value)).replace('\n','<br>')


def _font_style(e):
    family=re.sub(r'[^\w \-]','',e.get('resolvedFontFamily') or e.get('fontFamily','sans-serif'))
    return (f'font-family:"{family}",sans-serif;font-size:{e.get("fontSize",12):.3f}pt;'
            f'font-weight:{e.get("fontWeight","normal")};font-style:{e.get("fontStyle","normal")};'
            f'color:{e.get("color","#000000")};line-height:{e.get("lineHeight",1)};')


def _geometry(box):
    return f'left:{box[0]:.3f}pt;top:{box[1]:.3f}pt;width:{box[2]-box[0]:.3f}pt;height:{box[3]-box[1]:.3f}pt;'


def _box(e):
    return [e['x'],e['y'],e['x']+e['width'],e['y']+e['height']]


def _text_node(e,origin=(0,0),tag=None):
    box=_box(e)
    local=[box[0]-origin[0],box[1]-origin[1],box[2]-origin[0],box[3]-origin[1]]
    tag=tag or ('h2' if e.get('contentType')=='heading' else 'p')
    style=_geometry(local)+_font_style(e)+f'transform:rotate({e.get("rotation",0):.3f}deg);'
    if is_cell_flow(e):
        _,layout=flow_cell_text(e)
        params=dict(e,resolvedFontFamily=layout['family'],fontSize=layout['fontSize'])
        spans=[]
        for index,line in enumerate(layout['lines']):
            placement=f'left:0;top:{index*layout["lineStep"]:.3f}pt;line-height:{layout["lineStep"]:.3f}pt;'
            spans.append(f'<span class="run" style="{esc(_font_style(params)+placement)}">{esc(line)}</span>')
        body=''.join(spans)
    elif valid_runs(e):
        spans=[]
        library=get_library()
        for run in e['fontRuns']:
            face=library.faces[run['fontKey']]
            params=dict(e,fontFamily=face.family,resolvedFontFamily=face.family,fontWeight='bold' if face.bold else 'normal',
                        fontStyle='italic' if face.italic else 'normal',fontSize=run['fontSize'])
            rbox=[run['x'],run['y'],run['x']+run['width'],run['y']+run['height']]
            spans.append(f'<span class="run" style="{esc(_geometry(rbox)+_font_style(params))}">{esc(run["text"])}</span>')
        body=''.join(spans)
    else:
        body=esc(e.get('text',''))
    return f'<{tag} class="object text" data-id="{esc(e["id"])}" aria-label="{esc(e.get("text",""))}" data-bbox="{esc(json.dumps(box))}" style="{esc(style)}" contenteditable="true">{body}</{tag}>'


def _transcript(e,physical):
    lines=[]
    for line in e.get('retainedText',[]):
        b=line['bbox']
        relative=[(b[0]-e['x'])/e['width']*100,(b[1]-e['y'])/e['height']*100,
                  (b[2]-e['x'])/e['width']*100,(b[3]-e['y'])/e['height']*100]
        lines.append(f'<p tabindex="0" role="button" data-page="{physical}" data-bbox="{esc(json.dumps(b))}" data-relative-bbox="{esc(json.dumps(relative))}">{esc(line["text"])}</p>')
    if not lines:
        return ''
    return f'<details class="transcript" data-region-id="{esc(e["id"])}"><summary>图内文字转录与位置（不作为独立正文）</summary>'+''.join(lines)+'</details>'


def _table_node(table,by_id,*,fixed=True):
    box=table['bbox']
    origin=(table['columns'][0],table['rows'][0])
    style=table['observedStyle']
    border=(f'{style["borderWidth"]:.3f}pt solid {style["borderColor"]}'
            if style.get('borderWidth',0)>0 and table.get('borderVisible',True) else 'none')
    placement=_geometry([origin[0],origin[1],table['columns'][-1],table['rows'][-1]]) if fixed else ''
    columns=''.join(f'<col style="width:{b-a:.3f}pt">' for a,b in zip(table['columns'],table['columns'][1:]))
    rows=[]
    for row in range(len(table['rows'])-1):
        cells=[]
        for cell in sorted((c for c in table['cells'] if c['row']==row),key=lambda c:c['column']):
            text=sorted((by_id[i] for i in cell['textIds'] if i in by_id),key=lambda e:(e['y'],e['x']))
            if fixed:
                body=''.join(_text_node(e,(cell['bbox'][0],cell['bbox'][1]),tag='div') for e in text)
            else:
                body='<br>'.join(esc(e['text']) for e in text)
            span=f' rowspan="{max(1,int(cell.get("rowspan",1)))}" colspan="{max(1,int(cell.get("colspan",1)))}"'
            edges=cell.get('borders')
            cell_border=(''.join(f'border-{edge}:{border if edges.get(edge) else "none"};'
                                 for edge in ('top','right','bottom','left')) if isinstance(edges,dict)
                         else f'border:{border};')
            tag='th' if cell.get('header') else 'td'
            cells.append(f'<{tag}{span} style="position:relative;{esc(cell_border)}">{body}</{tag}>')
        height=table['rows'][row+1]-table['rows'][row]
        rows.append(f'<tr style="height:{height:.3f}pt">'+''.join(cells)+'</tr>')
    return (f'<table class="{"object " if fixed else ""}rebuilt-table" data-bbox="{esc(json.dumps(box))}" '
            f'style="{esc(placement)}border-collapse:collapse;table-layout:fixed"><colgroup>{columns}</colgroup><tbody>'+''.join(rows)+'</tbody></table>')


HTML_START='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>扫描文档结构化重建</title>
<style>*{box-sizing:border-box}body{margin:0;padding:16px;background:#e8ecef;font-family:Arial,sans-serif}.page{position:relative;isolation:isolate;margin:24px auto;overflow:hidden;box-shadow:0 2px 12px #0002}.object{position:absolute;margin:0;padding:0;transform-origin:center}.text{white-space:pre;overflow:visible}.run{position:absolute;white-space:pre}.object img{display:block;width:100%;height:100%}.page td,.page th{padding:0;vertical-align:top;text-align:left}.transcript{max-width:850px;margin:12px auto}.notice{max-width:850px;margin:16px auto} @media print{body{padding:0;background:white}.page{margin:0;box-shadow:none;break-after:page}.transcript,.notice{display:none}}</style>
<p class="notice">按内容类型重建：正文为文字，表格保留可恢复的行列与合并关系，复杂图文保留原图。边界与样式需校对；本地字体未随 ZIP 分发，其他设备可能采用相近字体。点击图内转录可定位。HTML 修改不会回写模型 JSON。<button id="save-html" type="button">保存修改后的 HTML</button></p>
'''

HTML_END='''<script>
document.getElementById('save-html').onclick=()=>{const copy=document.documentElement.cloneNode(true);copy.querySelectorAll('.location-highlight').forEach(n=>n.remove());const url=URL.createObjectURL(new Blob(['<!doctype html>'+copy.outerHTML],{type:'text/html;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='rebuilt-edited.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.querySelectorAll('.text[contenteditable]').forEach(n=>n.addEventListener('input',()=>n.setAttribute('aria-label',n.innerText)));
document.querySelectorAll('.transcript [data-relative-bbox]').forEach(line=>{const locate=()=>{const id=line.closest('[data-region-id]').dataset.regionId;const figure=[...document.querySelectorAll('figure[data-id]')].find(n=>n.dataset.id===id);if(!figure)return;document.querySelectorAll('.location-highlight').forEach(n=>n.remove());const b=JSON.parse(line.dataset.relativeBbox);const spot=document.createElement('span');spot.className='location-highlight';Object.assign(spot.style,{position:'absolute',left:b[0]+'%',top:b[1]+'%',width:(b[2]-b[0])+'%',height:(b[3]-b[1])+'%',outline:'2px solid #c45a31',background:'#ffb56b44',pointerEvents:'none'});figure.append(spot);figure.scrollIntoView({block:'center',behavior:'smooth'})};line.onclick=locate;line.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();locate()}}});
</script></html>'''


@tracked('rebuild.export_page')
def export_page(scene,scratch,tree:Path,ordinal:int,format:str):
    page=scene['pages'][0]
    physical=page.get('sourcePage',ordinal)
    by_id={e['id']:e for e in page['elements']}
    groups=[]
    portable=copy.deepcopy(page)
    portable.pop('sourceSrc',None)
    portable_by_id={e['id']:e for e in portable['elements']}
    portable_tables={t['id']:t for t in portable.get('tables',[])}
    references=[]
    for e in page['elements']:
        if e.get('tableBorder'):
            continue
        if e['type']=='text' and e.get('tableId'):
            continue
        if e['type']=='text':
            kind='heading' if e.get('contentType')=='heading' else 'text'
            fragment=_text_node(e)
            markdown=('## ' if kind=='heading' else '')+md_text(e['text'])
        elif e.get('rebuildVector') and e.get('shape')=='rect':
            kind='decoration'
            fragment=(f'<div class="object" style="{esc(_geometry(_box(e)))}border:{e["strokeWidth"]:.3f}pt solid {esc(e["stroke"])}"></div>')
            markdown=f'<!-- 简单文本框位置：{esc(json.dumps(_box(e)))}；框内文字保留为独立正文。 -->'
        elif e.get('rebuildVector') and e.get('shape')=='line':
            kind='decoration'
            style=_geometry(_box(e))+f'transform:rotate({e.get("rotation",0):.3f}deg);'
            fragment=(f'<svg class="object" style="{esc(style)}" viewBox="0 0 {e["width"]} {e["height"]}" '
                      f'aria-label="原页线条"><line x1="0" y1="{e["height"]/2}" x2="{e["width"]}" '
                      f'y2="{e["height"]/2}" stroke="{esc(e.get("stroke","#000000"))}" '
                      f'stroke-width="{e.get("strokeWidth",1)}"/></svg>')
            markdown=f'<!-- 原页线条位置：{esc(json.dumps(_box(e)))} -->'
        elif e['type']=='image':
            kind='graph'
            name=f'page-{ordinal:04d}-{e["id"]}.png'
            asset=tree/'assets'/name
            asset.parent.mkdir(parents=True,exist_ok=True)
            asset.write_bytes(read_asset(e['src'],scratch,scene['id']))
            src='../assets/'+name
            portable_by_id[e['id']]['src']=src
            fragment=(f'<figure class="object" data-id="{esc(e["id"])}" data-bbox="{esc(json.dumps(_box(e)))}" data-boundary-exact="false" '
                      f'style="{esc(_geometry(_box(e)))}"><img src="{esc(src)}" alt="完整图文区域"></figure>')
            markdown=f'![完整图文区域]({src})\n\n'+_transcript(e,physical)
        else:
            continue
        groups.append(dict(kind=kind,bbox=_box(e),html=fragment,md=markdown,
                           transcript=_transcript(e,physical) if e['type']=='image' else ''))
    for table in page.get('tables',[]):
        reference=''
        if table.get('referenceSrc'):
            name=f'page-{ordinal:04d}-{table["id"]}-reference.png'
            asset=tree/'assets'/name
            asset.parent.mkdir(parents=True,exist_ok=True)
            asset.write_bytes(read_asset(table['referenceSrc'],scratch,scene['id']))
            src='../assets/'+name
            portable_tables[table['id']]['referenceSrc']=src
            references.append(dict(tableId=table['id'],asset='assets/'+name,bbox=table['referenceBbox']))
            reference=(f'<details class="transcript table-reference"><summary>表格原图参考（结构需校对）</summary>'
                       f'<img src="{esc(src)}" alt="表格原始裁片" style="max-width:100%;height:auto"></details>')
        groups.append(dict(kind='table',bbox=table['bbox'],html=_table_node(table,by_id),
                           md=_table_node(table,by_id,fixed=False)+'\n\n'+reference,transcript=reference))
    groups.sort(key=lambda g:(g['bbox'][1],g['bbox'][0]))
    for e in portable['elements']:
        e.pop('appearanceSrc',None)
        e.pop('originalAppearanceSrc',None)
    model=tree/'model'/f'page-{ordinal:04d}.json'
    model.parent.mkdir(parents=True,exist_ok=True)
    model.write_text(json.dumps(portable,ensure_ascii=False,indent=2),encoding='utf-8')
    for kind in ['summary']+sorted({g['kind'] for g in groups}):
        selected=groups if kind=='summary' else [g for g in groups if g['kind']==kind]
        target=tree/kind/f'{kind}.{format}'
        target.parent.mkdir(parents=True,exist_ok=True)
        new=not target.exists()
        with target.open('a',encoding='utf-8') as handle:
            if format=='html':
                if new:
                    handle.write(HTML_START)
                handle.write(f'<section class="page" data-page="{physical}" data-source="{esc(page.get("sourceName",""))}" '
                             f'style="width:{page["width"]:.3f}pt;height:{page["height"]:.3f}pt;background:{page.get("paperColor","#fff")}">')
                handle.write(''.join(g['html'] for g in selected)+'</section>')
                handle.write(''.join(g['transcript'] for g in selected))
            else:
                handle.write(f'\n\n# 物理第 {physical} 页\n\n'+'\n\n'.join(g['md'] for g in selected)+'\n')
    return dict(readingOrder='top-left-heuristic',fontDelivery='local-font-families-not-bundled',
                tableReferences=references,
                markdownStyleLimitations=['font-family','font-size','absolute-page-position'] if format=='md' else [])


@tracked('rebuild.finalize_html')
def finalize_html(tree):
    for path in tree.glob('*/*.html'):
        with path.open('a',encoding='utf-8') as handle:
            handle.write(HTML_END)
