from pathlib import Path
from PIL import Image
from backend.exporter import element_bitmap


def test_line_matches_canvas_horizontal_center_and_zero_width(tmp_path):
    element={"id":"line","type":"shape","shape":"line","width":80,"height":40,
             "stroke":"#ff0000","strokeWidth":2,"fill":"none"}
    image=element_bitmap(element,"doc",tmp_path,1)
    assert image.getpixel((20,20))[3] > 0
    assert image.getpixel((20,10))[3] == 0
    element["strokeWidth"]=0
    assert element_bitmap(element,"doc",tmp_path,1).getbbox() is None


def test_vector_style_changes_use_current_model(tmp_path):
    element={"id":"vector","type":"vector","width":30,"height":30,
             "svg":'<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"><rect width="30" height="30" fill="red"/></svg>',
             "fill":"#0000ff","stroke":"none","strokeWidth":0}
    image=element_bitmap(element,"doc",tmp_path,1)
    assert image.getpixel((15,15)) == (0,0,255,255)
