"""Bounded source evidence for containers and alignment, before target flow.

Region detections are hints, never semantic containers by themselves. Only a
closed rectangle and a measured plain interior permit background reconstruction.
Complex pictures retain their original pixels and a non-rendered transcript.
"""
from copy import deepcopy

import cv2
import numpy as np


def area(b):
    return max(0.,b[2]-b[0])*max(0.,b[3]-b[1])


def overlap(a,b):
    return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))


def _inside(b,r,minimum=.95):
    return area(b)>0 and overlap(b,r)/area(b)>=minimum


def region_for(b,regions):
    matches=[]
    for region in regions:
        r=region.get('coordinate',[])
        if len(r)==4 and area(r)>0 and _inside(b,r,.9):
            matches.append((area(r),str(region.get('id','')),r))
    return min(matches,default=None)


def infer_alignment(block, peers, page, container=None):
    """Separate text alignment from the block anchor in its parent frame."""
    b=block['source']['bbox_px']; label=block['source'].get('label','')
    regions=page.get('detections',{}).get('regions',[])
    scope='page';ref=[0.,0.,float(page['width_px']),float(page['height_px'])]
    if container is not None:
        scope,ref='container',container
    else:
        region=region_for(b,regions)
        if region and region[2][2]-region[2][0] > (b[2]-b[0])*1.4:
            scope,ref='region',region[2]
        # A narrow column is supported by two overlapping body blocks, not a
        # heading's own tight Region. Do not center against an arbitrary peer.
        bodies=[p['source']['bbox_px'] for p in peers if p['id']!=block['id'] and
                p['kind']=='text' and p['source'].get('label') in {'text','abstract','content'}
                and min(b[2],p['source']['bbox_px'][2])-max(b[0],p['source']['bbox_px'][0])>0]
        columns=[p for p in bodies if p[2]-p[0]<page['width_px']*.7]
        consistent=[p for p in columns if sum(abs(p[0]-q[0])<12 and abs(p[2]-q[2])<24 for q in columns)>=2]
        if consistent:
            p=min(consistent,key=lambda p:abs(p[1]-b[3]))
            if _inside([b[0],p[1],b[2],p[3]],p,.9):
                scope,ref='column',p
    lines=block['source'].get('ocr_lines',[])
    boxes=[r['bbox_px'] for r in lines]
    if container is not None and boxes:
        # An image-labelled badge's source rectangle includes its background.
        # Its text alignment is measured from glyph boxes, not that rectangle.
        b=[min(r[0] for r in boxes),min(r[1] for r in boxes),
           max(r[2] for r in boxes),max(r[3] for r in boxes)]
    rw=max(1.,ref[2]-ref[0]);center=(ref[0]+ref[2])/2
    left,right=b[0]-ref[0],ref[2]-b[2]
    near_center=abs((b[0]+b[2])/2-center)<=max(6.,rw*.035)
    balanced=abs(left-right)<=max(12.,rw*.07) and min(left,right)>rw*(.02 if container is not None else .07)
    centers=[(r[0]+r[2])/2 for r in boxes]
    line_centered=bool(centers) and max(abs(c-center) for c in centers)<=max(8.,rw*.05)
    role=label in {'doc_title','paragraph_title','section_title','title','reference','figure_title','header','footer'}
    # A paragraph can span the page centre while every physical line starts
    # at the same left edge. Centre proximity alone is not text alignment.
    left_aligned=len(boxes)>=2 and max(r[0] for r in boxes)-min(r[0] for r in boxes)<=8
    centered=(near_center and balanced and line_centered and
              (role or container is not None or block['kind']=='composite') and
              (role or not left_aligned))
    if centered:
        align,confidence='center',.9
    elif left_aligned:
        align,confidence='left',.85
    elif len(boxes)>=2 and max(r[2] for r in boxes)-min(r[2] for r in boxes)<=8:
        align,confidence='right',.8
    else:
        align,confidence='unknown',0.
    block['style'].update(align=align,align_reference=scope,
                          alignment_evidence=dict(method='source-frame-margins-and-line-edges',
                          reference_bbox_px=list(ref),confidence=confidence,center_error_px=abs((b[0]+b[2])/2-center)))
    block['local_layout']['anchor']=dict(horizontal='center' if centered else 'flow-left',
        reference=scope,confidence=confidence,reference_bbox_px=list(ref))
    if align=='unknown':
        block['style']['alignment_evidence']['render_fallback']='left'


def _rectangles(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    ink=(gray<140).astype(np.uint8)*255
    h,w=gray.shape
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(35,w//20)),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(18,h//80),1),np.uint8))
    rules=cv2.bitwise_or(horizontal,vertical)
    contours,_=cv2.findContours(rules,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        x,y,cw,ch=cv2.boundingRect(contour)
        if cw<60 or ch<22 or cw>w*.95 or ch>h*.65:
            continue
        sample=ink[y:y+ch,x:x+cw]>0
        band=max(2,min(5,round(min(cw,ch)*.06)))
        support=[sample[:band].any(axis=0).mean(),sample[-band:].any(axis=0).mean(),
                 sample[:,:band].any(axis=1).mean(),sample[:,-band:].any(axis=1).mean()]
        if min(support)<.88:
            continue
        result.append([x,y,x+cw,y+ch])
    return sorted(result,key=area)


def _plain_panel(rgb,rect,polygons):
    x,y,z,t=map(int,rect);crop=rgb[y:t,x:z]
    if min(crop.shape[:2])<10:return None
    # Exclude the observed border and full detector polygons, including det's
    # own expansion. At least 15% must remain as actual background evidence;
    # compact title badges devote much of their area to expanded glyph boxes.
    border=max(3,min(6,round(min(z-x,t-y)*.1)))
    mask=np.zeros(crop.shape[:2],np.uint8)
    mask[:border]=mask[-border:]=1;mask[:,:border]=mask[:,-border:]=1
    textmask=np.zeros_like(mask)
    for poly in polygons:
        p=np.asarray(poly,dtype=float)
        if p.shape!=(4,2) or not np.isfinite(p).all():return None
        cv2.fillPoly(textmask,[np.rint(p-[x,y]).astype(np.int32)],1)
    textmask=cv2.dilate(textmask,np.ones((3,3),np.uint8))
    mask|=textmask
    remaining=crop[mask==0]
    if len(remaining)<max(200,crop.shape[0]*crop.shape[1]*.15):return None
    bg=np.median(remaining,axis=0)
    uniform=(np.max(np.abs(remaining.astype(float)-bg),axis=1)<=25).mean()
    if uniform<.985:return None
    samples=crop[textmask>0].astype(float)
    contrast=np.max(np.abs(samples-bg),axis=1) if len(samples) else np.array([])
    foreground=samples[contrast>70]
    if len(foreground)<5:return None
    # Estimate ink from glyph cores; antialiasing contains many blends with
    # the background and is not evidence of multiple foreground colors.
    distances=np.max(np.abs(foreground-bg),axis=1)
    cores=foreground[distances>=np.quantile(distances,.7)]
    fg=np.median(cores,axis=0)
    # A textured inset cannot be called a single text color.
    if (np.max(np.abs(cores-fg),axis=1)<=35).mean()<.9:return None
    color=lambda v:'#'+''.join(f'{round(float(c)):02x}' for c in v)
    edges=np.concatenate((crop[:2].reshape(-1,3),crop[-2:].reshape(-1,3),crop[:,:2].reshape(-1,3),crop[:,-2:].reshape(-1,3)))
    dark=edges[edges.min(axis=1)<140]
    border_color=color(np.median(dark,axis=0)) if len(dark) else '#000000'
    return dict(fill=color(bg),text_color=color(fg),border_color=border_color,
                border_width_px=1. if float(bg.mean())>180 else 0.,
                evidence=dict(method='closed-rectangle-and-masked-uniform-interior',
                    uniform_fraction=float(uniform),background_sample_pixels=len(remaining),
                    total_pixels=int(crop.shape[0]*crop.shape[1])))


def organize_source(blocks,rgb,page,raw,directory,save_image,text_content):
    """Build plain containers, retaining child IDs and a single visible owner."""
    sx,sy=page['pixel_to_point'] if 'pixel_to_point' in page else (page['width_pt']/page['width_px'],page['height_pt']/page['height_px'])
    assignments=(raw.get('ocr_pipeline') or {}).get('line_assignments',[])
    if not assignments:
        assignments=[dict(polygon=[[r['bbox_px'][0],r['bbox_px'][1]],[r['bbox_px'][2],r['bbox_px'][1]],
                     [r['bbox_px'][2],r['bbox_px'][3]],[r['bbox_px'][0],r['bbox_px'][3]]],
                     box=r['bbox_px'],state='recognized',line_id=r['line_id'])
                     for b in blocks for r in b['source'].get('ocr_lines',[])]
    absorbed=set();parents=[]
    for rect in _rectangles(rgb):
        # Tables have their own structure; never wrap one inside a guessed
        # text panel or erase a neighboring/partly overlapping content block.
        if any(b['kind']=='table' and overlap(b['source']['bbox_px'],rect)>.05*min(area(rect),area(b['source']['bbox_px'])) for b in blocks):continue
        children=[b for b in blocks if b['id'] not in absorbed and _inside(b['source']['bbox_px'],rect,.9)]
        if not children or any(b['kind'] not in {'text','image'} for b in children):continue
        if any(b['kind']=='image' and not _inside(b['source']['bbox_px'],rect,.995) for b in children):continue
        if any(b['id'] not in absorbed and b not in children and overlap(b['source']['bbox_px'],rect)>.05*area(b['source']['bbox_px']) for b in blocks):continue
        inside=[a for a in assignments if len(a.get('box',[]))==4 and _inside(a['box'],rect,.9)]
        polys=[a['polygon'] for a in inside]
        panel=_plain_panel(rgb,rect,polys)
        if panel is None:continue
        # Every detected glyph-bearing line must be recognized and represented.
        lines=[r for b in children for r in b['source'].get('ocr_lines',[])]
        represented={r['line_id'] for r in lines}
        reliable=bool(inside) and all(a.get('state')=='recognized' and a.get('line_id') in represented for a in inside)
        reliable=reliable and all(b['kind']=='text' or b['source'].get('label')=='image' for b in children)
        identifier=f'{page["id"]}-container-{len(parents)}'
        asset,pixels=save_image(rgb,rect,directory,identifier+'-reference')
        source=dict(page_id=page['id'],layout_detection_id=None,label='container',bbox_px=rect,
                    bbox_pt=[rect[0]*sx,rect[1]*sy,rect[2]*sx,rect[3]*sy],reference_asset=asset,
                    crop_to_page=[[1,0,rect[0]],[0,1,rect[1]],[0,0,1]],
                    ocr_lines=deepcopy(lines),child_detection_ids=[b['source'].get('layout_detection_id') for b in children],
                    child_sources=[deepcopy(b['source']) for b in children],container_evidence=panel['evidence'])
        parent=dict(id=identifier,kind='composite' if reliable else 'image',source=source,
                    content={},style={},local_layout={},quality=dict(state='needs_review',issues=[]))
        if reliable:
            members=[]
            for child in sorted(children,key=lambda b:(b['source']['bbox_px'][1],b['source']['bbox_px'][0])):
                child=deepcopy(child)
                child['kind']='text'
                if 'text' not in child['content']:
                    child['content']=text_content(child['source']['ocr_lines'],child['id'])
                child['style']['color']=panel['text_color']
                child['source']['parent_id']=identifier
                infer_alignment(child,children,page,container=rect)
                members.append(child)
            parent['content']=dict(children=members,layout_mode='source-relative',background=dict(fill=panel['fill']),
                decorations=dict(border_color=panel['border_color'],border_width_pt=panel['border_width_px']*sy),
                padding_pt=max(3.,min(10.,(min(rect[2]-rect[0],rect[3]-rect[1])*.1)*sy)),
                gap_pt=5.)
            parent['style']=deepcopy(members[0]['style'])
            parent['quality']['issues']=['plain-container-reconstructed-from-pixel-evidence']
        else:
            parent['content']=dict(asset=asset,pixel_size=pixels,aspect_ratio=pixels[0]/pixels[1],
                transcript=deepcopy(lines),transcript_rendering='metadata-only',
                transcript_coordinate_space='page-pixels')
            parent['quality'].update(state='image_fallback',issues=['container-ocr-incomplete-preserved-whole'])
        absorbed.update(b['id'] for b in children);parents.append(parent)
    result=[b for b in blocks if b['id'] not in absorbed]+parents
    for b in result:
        if b['kind'] in {'text','composite'}:infer_alignment(b,result,page)
    page['source_relationships']=dict(contains=[dict(parent=b['id'],child=c['id']) for b in parents
        for c in b['content'].get('children',[])],
        absorbed_block_ids=sorted(absorbed),method='region-hints-plus-closed-panel-pixel-evidence')
    return result
