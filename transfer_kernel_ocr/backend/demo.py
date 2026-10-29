"""Generate a real two-page fixture containing native text, shapes and a scan."""
from pathlib import Path
import io
import pymupdf as fitz
from PIL import Image, ImageDraw


def create_demo(destination: Path):
    pdf = fitz.open()
    p = pdf.new_page(width=595,height=842)
    navy=(.09,.19,.24)
    teal=(.08,.48,.44)
    p.draw_rect(fitz.Rect(0,0,595,12),color=None,fill=teal)
    p.insert_text((44,56),"FIELDNOTES / DOCUMENT STUDIO",fontsize=10,color=teal,fontname="hebo")
    p.insert_text((44,100),"每一处细节，都值得保留。",fontname="china-s",fontsize=26,color=navy)
    p.insert_text((44,131),"A page is more than a sequence of words.",fontsize=12,color=(.4,.46,.48))
    p.draw_line((44,154),(551,154),color=(.8,.85,.86),width=1)
    p.insert_text((44,189),"01",fontsize=22,fontname="hebo",color=teal)
    p.insert_text((88,185),"项目内容 / PROJECT DETAILS",fontsize=12,fontname="china-s",color=navy)
    p.insert_text((44,222),"在画布中选择文字、拖动图案，或调整图层顺序。",fontname="china-s",fontsize=12,color=navy)
    p.insert_text((44,244),"原始坐标始终保留，阅读顺序仅作为附加信息。",fontname="china-s",fontsize=12,color=navy)
    p.draw_rect(fitz.Rect(44,276,551,316),color=None,fill=(.91,.96,.95))
    for x in [44,260,416,551]:
        p.draw_line((x,276),(x,416),color=(.65,.77,.77),width=.8)
    for y in [276,316,366,416]:
        p.draw_line((44,y),(551,y),color=(.65,.77,.77),width=.8)
    for pos,text in [((58,301),"元素类型"),((274,301),"保留位置"),((430,301),"独立编辑"),((58,346),"文字 / 图形"),((274,346),"原始坐标"),((430,346),"支持"),((58,396),"印章 / 签名"),((274,396),"原始外观"),((430,396),"图像区域")]:
        p.insert_text(pos,text,fontname="china-s",fontsize=12,color=navy)
    p.draw_rect(fitz.Rect(44,449,57,462),color=teal,width=1.3)
    p.draw_polyline([(46,455),(50,459),(55,451)],color=teal,width=1.4)
    p.insert_text((68,460),"保持文字与图案的相对布局",fontname="china-s",fontsize=12,color=navy)
    p.insert_text((44,508),"E = mc",fontsize=24,fontname="tiit",color=navy)
    p.insert_text((116,496),"2",fontsize=13,fontname="tiit",color=navy)
    p.insert_text((44,540),"FORMULA / individual glyphs preserved",fontsize=9,color=(.4,.46,.48))
    p.draw_bezier((46,610),(98,552),(85,639),(155,583),color=navy,width=2)
    p.draw_bezier((101,610),(140,565),(182,650),(228,587),color=navy,width=1.5)
    p.insert_text((44,651),"SIGNATURE",fontsize=9,color=(.4,.46,.48))
    p.draw_circle((462,566),55,color=(.75,.15,.19),width=2)
    p.draw_circle((462,566),48,color=(.75,.15,.19),width=.8)
    p.insert_text((431,568),"APPROVED",fontsize=10,fontname="hebo",color=(.75,.15,.19))
    p.insert_text((443,587),"2026",fontsize=10,color=(.75,.15,.19))
    icon=Image.new("RGB",(240,110),(226,239,234))
    d=ImageDraw.Draw(icon)
    d.rounded_rectangle((20,20,100,90),radius=12,fill=(20,118,107))
    d.ellipse((119,18,191,90),fill=(234,174,99))
    buf=io.BytesIO();icon.save(buf,format="PNG")
    p.insert_image(fitz.Rect(340,661,551,758),stream=buf.getvalue())
    for i in range(38):
        x=44+i*4
        p.draw_rect(fitz.Rect(x,700,x+(1 if i%3 else 2.5),742),color=None,fill=navy)
    p.insert_text((44,759),"2026 1015 0038",fontsize=9,color=navy)
    p.insert_text((44,803),"STRUCTURE, NOT JUST TEXT.",fontsize=9,color=teal)
    p.insert_text((519,803),"01 / 02",fontsize=9,color=teal)
    scan=p.get_pixmap(matrix=fitz.Matrix(1.5,1.5))
    second=pdf.new_page(width=595,height=842)
    second.insert_image(second.rect,stream=scan.tobytes("png"))
    pdf.set_metadata({"title":"页面结构示例 · 原生与扫描对照"})
    pdf.save(destination)
    pdf.close()
