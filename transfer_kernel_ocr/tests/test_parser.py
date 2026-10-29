"""Behavioral checks for decomposition and editable appearance ownership."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from backend.parser import parse_document, _estimate_ink_color, _svg_for_drawing


def composite(page, root, document_id, omitted=()):
    scale=page["renderScale"]
    width=round(page["width"]*scale)
    height=round(page["height"]*scale)
    canvas=Image.new("RGBA",(width,height),(255,255,255,255))
    for element in sorted(page["elements"],key=lambda e:e["zIndex"]):
        if element["id"] in omitted:
            continue
        src=element.get("appearanceSrc",element.get("src"))
        if not src:
            continue
        image=Image.open(root/src.split(f"/files/{document_id}/",1)[1]).convert("RGBA")
        if element["rotation"]:
            image=image.rotate(-element["rotation"],expand=True)
        cx=(element["x"]+element["width"]/2)*scale
        cy=(element["y"]+element["height"]/2)*scale
        canvas.alpha_composite(image,(round(cx-image.width/2),round(cy-image.height/2)))
    return np.asarray(canvas.convert("RGB"))


class ParserTests(unittest.TestCase):
    def setUp(self):
        test_root=Path(__file__).resolve().parent/".tmp"
        test_root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=test_root)
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_pixels_and_no_text_ghost_after_deletion(self):
        path=self.root/"native.pdf"
        with fitz.open() as pdf:
            page=pdf.new_page(width=300,height=300)
            page.draw_rect((20,20,280,100),fill=(.8,.9,1),color=None)
            page.insert_text((40,60),"Editable text",fontsize=20)
            page.insert_text((180,260),"Rotate",rotate=90,fontsize=16)
            page.draw_line((10,280),(290,280),color=(1,0,0),width=2)
            pdf.save(path)
        document=parse_document([path],self.root/"out","native","Native",ocr=False)
        page=document["pages"][0]
        expected=np.asarray(Image.open(self.root/"out/pages/page-001.png").convert("RGB"))
        actual=composite(page,self.root/"out","native")
        self.assertEqual(int(np.max(np.abs(actual.astype(int)-expected.astype(int)))),0)
        text=next(e for e in page["elements"] if e.get("text")=="Editable text")
        self.assertTrue(text["glyphs"])
        deleted=composite(page,self.root/"out","native",[text["id"]])
        # Removing the text reveals reconstructed blue paper, no black scan text.
        crop=deleted[80:130,75:300]
        self.assertGreater(int(crop.min()),150)
        self.assertTrue(any(e["type"]=="vector" for e in page["elements"]))
        self.assertTrue(any(e.get("rotation")==-90 for e in page["elements"]))

    def test_rotated_cropped_pdf_coordinates_match_render(self):
        path=self.root/"rotated.pdf"
        with fitz.open() as pdf:
            page=pdf.new_page(width=500,height=500)
            page.insert_text((90,140),"Crop and rotate",fontsize=18)
            page.draw_rect((80,180,270,300),color=(1,0,0),width=3)
            page.set_cropbox((50,50,450,400))
            page.set_rotation(90)
            pdf.save(path)
        document=parse_document([path],self.root/"out","rotated","Rotated",ocr=False)
        page=document["pages"][0]
        self.assertEqual((page["width"],page["height"]),(350,400))
        self.assertEqual(page["sourceRotation"],90)
        expected=np.asarray(Image.open(self.root/"out/pages/page-001.png").convert("RGB"))
        actual=composite(page,self.root/"out","rotated")
        self.assertEqual(int(np.max(np.abs(actual.astype(int)-expected.astype(int)))),0)

    def test_multiframe_input_and_transparency(self):
        first=Image.new("RGB",(120,80),"white")
        second=Image.new("RGB",(80,120),"#f0e0d0")
        path=self.root/"multi.tiff"
        first.save(path,save_all=True,append_images=[second])
        document=parse_document([path],self.root/"out","multi","Multi",ocr=False)
        self.assertEqual(len(document["pages"]),2)
        self.assertEqual([p["index"] for p in document["pages"]],[0,1])
        self.assertEqual(document["pages"][1]["nativePixelSize"],[80,120])

    def test_encrypted_pdf_has_clear_error(self):
        path=self.root/"protected.pdf"
        with fitz.open() as pdf:
            pdf.new_page()
            pdf.save(path,encryption=fitz.PDF_ENCRYPT_AES_256,owner_pw="owner",user_pw="password")
        with self.assertRaisesRegex(ValueError,"已加密"):
            parse_document([path],self.root/"out","protected","Protected",ocr=False)

    def test_unsupported_graphics_keep_all_visible_pixels(self):
        path=self.root/"unsupported.pdf"
        with fitz.open() as pdf:
            page=pdf.new_page(width=150,height=150)
            page.insert_text((20,40),"Keep all pixels")
            page.draw_circle((70,90),20,color=(1,0,0))
            pdf.save(path)
        with patch("backend.parser._native_graphics",side_effect=RuntimeError("unsupported clip")):
            document=parse_document([path],self.root/"out","fallback","Fallback",ocr=False)
        page=document["pages"][0]
        expected=np.asarray(Image.open(self.root/"out/pages/page-001.png").convert("RGB"))
        actual=composite(page,self.root/"out","fallback")
        self.assertTrue(np.array_equal(actual,expected))
        self.assertTrue(any("无法提取" in warning for warning in page["warnings"]))

    def test_real_local_ocr_and_no_unchanged_scan_under_text(self):
        path=self.root/"scan.png"
        image=Image.new("RGB",(700,260),"white")
        draw=ImageDraw.Draw(image)
        draw.text((40,70),"LAYOUT TEST 123",font=ImageFont.load_default(size=40),fill="black")
        image.save(path)
        document=parse_document([path],self.root/"out","scan","Scan",ocr=True)
        page=document["pages"][0]
        texts=[e for e in page["elements"] if e["type"]=="text"]
        self.assertTrue(texts)
        self.assertIn("LAYOUT"," ".join(e["text"] for e in texts))
        self.assertTrue(all(e["source"]=="ocr" for e in texts))
        self.assertTrue(all(max(int(e["color"][i:i+2],16) for i in (1,3,5))<80 for e in texts))
        deleted=composite(page,self.root/"out","scan",[e["id"] for e in texts])
        # A scan background containing the original text would fail this check.
        self.assertGreater(float(deleted[60:130,30:450].mean()),250)

    def test_sparse_ink_and_light_lettering_color_remain_visible(self):
        sparse=np.full((100,200,3),255,dtype=np.uint8)
        sparse[40:50,60:80]=[20,30,40]  # 1% ink: median/20th percentile is paper.
        self.assertEqual(_estimate_ink_color(sparse),"#141e28")
        reversed_colors=np.full((100,200,3),20,dtype=np.uint8)
        reversed_colors[40:50,60:80]=[240,245,250]
        self.assertEqual(_estimate_ink_color(reversed_colors),"#f0f5fa")

    def test_svg_missing_opacity_is_valid_numeric_default(self):
        drawing={"rect":fitz.Rect(10,10,80,80),"items":[("re",fitz.Rect(10,10,80,80),1)],
                 "fill":(1,0,0),"color":(0,0,0),"fill_opacity":None,"stroke_opacity":None}
        _,svg=_svg_for_drawing(drawing,fitz.Matrix(1,1))
        self.assertNotIn('opacity="None"',svg)
        self.assertIn('fill-opacity="1.0"',svg)
        self.assertIn('stroke-opacity="1.0"',svg)


if __name__=="__main__":
    unittest.main()
