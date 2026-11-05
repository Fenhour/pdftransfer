"""Page detection preserves block identity and shares recognition across types."""
import base64
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pdf_extractor import usage
from native_worker import NativeRuntime, handle_request, load_config
from rebuild_pipeline import POLICY, _ocr_records, recognize_rebuild


class TypedModelTests(unittest.TestCase):
    def setUp(self):
        factory=patch('rebuild_ocr._ocr_result',lambda value:value)
        factory.start()
        self.addCleanup(factory.stop)

    def pipeline(self):
        self.det_inputs=[]
        self.rec_inputs=[]
        self.crop_inputs=[]
        self.table_inputs=[]
        def layout(image):
            yield {'boxes':[
                {'cls_id':2,'label':'text','score':.99,'coordinate':[0,0,80,40]},
                {'cls_id':1,'label':'image','score':.99,'coordinate':[50,0,80,40]},
                {'cls_id':8,'label':'table','score':.99,'coordinate':[0,40,80,80]}]}
        def regions(image,**kwargs):
            yield {'boxes':[{'cls_id':0,'label':'Region','score':.9,'coordinate':[0,0,80,80]}]}
        def det(image,**kwargs):
            self.det_inputs.append(image.copy())
            yield {'dt_polys':np.array([[[2,2],[30,2],[30,12],[2,12]],
                [[55,2],[75,2],[75,12],[55,12]],[[2,42],[30,42],[30,52],[2,52]]])}
        def rec(images,**kwargs):
            self.rec_inputs.append(images)
            for image in images:
                yield {'rec_text':'Visible','rec_score':.95}
        def crop(image,polys):
            self.crop_inputs.append(image.copy())
            for poly in polys:
                x0,y0=np.floor(poly.min(axis=0)).astype(int)
                x1,y1=np.ceil(poly.max(axis=0)).astype(int)
                yield image[y0:y1,x0:x1].copy()
        ocr=SimpleNamespace(text_det_model=det,text_rec_model=rec,_crop_by_polys=crop)
        def table(image,**kwargs):
            self.table_inputs.append(kwargs)
            yield {'table_res_list':[{'pred_html':'<table><tr><td>Visible</td></tr></table>',
                                     'cell_box_list':np.array([[0,0,80,40]]),
                                     'table_ocr_pred':{'rec_texts':['Visible']}}]}
        return SimpleNamespace(layout_det_model=layout,region_detection_model=regions,
                               general_ocr_pipeline=ocr,table_recognition_pipeline=table)

    def test_single_page_detection_recognizes_image_transcript_and_reuses_table_ocr(self):
        rgb=np.zeros((80,80,3),dtype=np.uint8)
        model=self.pipeline()
        answer=recognize_rebuild(rgb,SimpleNamespace(paddlex_pipeline=model))
        self.assertEqual(len(self.det_inputs),1)
        self.assertEqual(self.det_inputs[0].shape,(80,80,3))
        self.assertEqual([len(batch) for batch in self.rec_inputs],[3])
        self.assertEqual(self.crop_inputs[0][:40,50:80,:].min(),255)
        self.assertEqual(self.crop_inputs[0][41:,50:80,:].max(),0)
        self.assertEqual(rgb.max(),0,'Input pixels stay unchanged')
        self.assertEqual(answer['overall_ocr_res']['block_ids'],['layout-0','layout-1','layout-2'])
        self.assertEqual(answer['overall_ocr_res']['labels'],['text','image','table'])
        self.assertEqual(answer['overall_ocr_res']['line_ids'],['line-0','line-1','line-2'])
        self.assertEqual(answer['overall_ocr_res']['rec_boxes'][2],[2.,42.,30.,52.])
        self.assertEqual(answer['table_res_list'][0]['block_id'],'layout-2')
        self.assertEqual(answer['table_res_list'][0]['cell_box_list'][0],[0.,40.,80.,80.])
        self.assertFalse(self.table_inputs[0]['use_ocr_model'])
        self.assertIsInstance(self.table_inputs[0]['overall_ocr_res']['rec_boxes'],np.ndarray)
        self.assertEqual(self.table_inputs[0]['overall_ocr_res']['rec_boxes'].tolist(),[[2.,2.,30.,12.]])
        self.assertEqual(answer['table_res_list'][0]['table_ocr_coordinate_space'],'table-local-pixels')
        self.assertEqual(answer['region_det_res']['boxes'][0]['id'],'region-0')
        self.assertEqual(answer['layout_det_res']['boxes'][1]['label'],'image')

    def test_unsupported_regions_do_not_invoke_ocr(self):
        model=self.pipeline()
        model.layout_det_model=lambda image:iter([{'boxes':[dict(label='formula',coordinate=[0,0,40,40])]}])
        result=recognize_rebuild(np.zeros((40,40,3),dtype=np.uint8),model)
        self.assertEqual(result['overall_ocr_res']['rec_texts'],[])
        self.assertEqual(len(self.det_inputs),1)
        self.assertEqual(self.rec_inputs,[])
        self.assertEqual(result['layout_det_res']['boxes'][0]['label'],'formula')

    def test_native_table_survives_paused_content_and_overlap_gate(self):
        model=self.pipeline()
        native_html='<table><tr><td>Native text</td><td>Second cell</td></tr></table>'
        def table(image,**kwargs):
            yield {'table_res_list':[dict(pred_html=native_html,
                cell_box_list=[[0,0,60,40],[40,0,80,40]])]}
        model.table_recognition_pipeline=table
        result=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model)
        self.assertEqual(len(result['table_res_list']),1)
        published=result['table_res_list'][0]
        self.assertEqual(published['pred_html'],native_html)
        self.assertEqual(published['cell_box_list'],[[0.,40.,60.,80.],[40.,40.,80.,80.]])
        self.assertEqual(published['validation_policy'],'native-v3-specialized-validation-paused')
        self.assertNotIn('reconstruction_evidence',published)
        self.assertTrue(any(d['reason']=='table-native-result-specialized-validation-paused'
                            for d in result['diagnostics']))

    def test_region_expansion_uses_actual_detector_keyword_and_is_reported(self):
        model=self.pipeline()
        calls=[]
        original=model.region_detection_model
        def regions(image,**kwargs):
            calls.append(kwargs)
            return original(image,**kwargs)
        model.region_detection_model=regions
        answer=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model,region_unclip_ratio=1.03)
        self.assertEqual(calls[0]['layout_unclip_ratio'],1.03)
        self.assertEqual(answer['region_det_res']['unclip_ratio'],1.03)
        with self.assertRaises(ValueError):
            recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model,region_unclip_ratio=1.3)

    def test_page_mapping_preserves_crop_local_float_polygons_for_tables(self):
        polygons=np.array([[[2.,2.],[30.,2.],[30.,12.],[2.,12.]]])
        polygons.setflags(write=False)
        diagnostics=[]
        records=_ocr_records({'rec_texts':['Cell'], 'rec_polys':polygons},
                             'layout-0',10,40,np.ones((20,40),dtype=bool),diagnostics)
        self.assertEqual(records[0]['box'],[12.,42.,40.,52.])
        self.assertEqual(polygons[0].tolist(),[[2.,2.],[30.,2.],[30.,12.],[2.,12.]])
        self.assertEqual(diagnostics,[])

    def test_polygons_are_bounded_by_the_crop_before_page_mapping(self):
        diagnostics=[]
        records=_ocr_records({'rec_texts':['Cell'],
                              'rec_polys':[[[-2,-3],[30,-3],[30,12],[-2,12]]]},
                             'layout-0',10,40,np.ones((20,40),dtype=bool),diagnostics)
        self.assertEqual(records[0]['box'],[10.,40.,40.,52.])
        self.assertEqual(diagnostics[0]['reason'],'ocr-polygon-clipped-to-crop')

    def test_visual_blocks_publish_unique_pixel_ownership_without_ocr(self):
        model=self.pipeline()
        def layout(image):
            yield {'boxes':[dict(label='image',coordinate=[10,10,30,30]),
                                dict(label='image',coordinate=[0,0,40,40]),
                                dict(label='formula',coordinate=[10,10,30,30])]}
        model.layout_det_model=layout
        model.general_ocr_pipeline.text_det_model=lambda image,**kwargs:iter([{'dt_polys':[]}])
        result=recognize_rebuild(np.zeros((40,40,3),dtype=np.uint8),model)
        self.assertEqual(self.rec_inputs,[])
        ownership=result['block_ownership']
        self.assertEqual(ownership['layout-0']['owned_pixel_count'],400)
        self.assertEqual(ownership['layout-2']['owned_pixel_count'],0)
        self.assertEqual(ownership['layout-1']['owned_pixel_count'],1200)
        self.assertEqual(ownership['layout-1']['total_pixel_count'],1600)
        self.assertEqual(ownership['layout-1']['excluded_rects'],[[10,10,30,30],[10,10,30,30]])
        self.assertTrue(any(d.get('block_id')=='layout-2' and
                            d['reason']=='fully-owned-by-another-block' for d in result['diagnostics']))

    def test_contained_image_does_not_block_table_structure(self):
        model=self.pipeline()
        def layout(image):
            yield {'boxes':[dict(label='image',coordinate=[0,0,20,40]),
                            dict(label='table',coordinate=[0,0,80,80])]}
        model.layout_det_model=layout
        model.general_ocr_pipeline.text_det_model=lambda image,**kwargs:iter([
            {'dt_polys':np.array([[[30,45],[60,45],[60,55],[30,55]]])}])
        result=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model)
        self.assertEqual(len(self.table_inputs),1)
        self.assertEqual(len(result['table_res_list']),1)
        evidence=result['block_ownership']['layout-1']
        self.assertEqual(evidence['excluded_block_ids'],['layout-0'])
        self.assertEqual(evidence['excluded_rects'],[[0,0,20,40]])
        self.assertEqual(evidence['owned_pixel_count'],5600)
        diagnostic=next(d for d in result['diagnostics'] if d['reason']=='table-contained-blocks-accepted')
        self.assertEqual(diagnostic['contained_ids'],['layout-0'])
        self.assertEqual(result['source_containment']['nodes']['layout-1']['contained_ids'],['layout-0'])

    def test_crossing_image_still_blocks_table_structure(self):
        model=self.pipeline()
        model.layout_det_model=lambda image:iter([{'boxes':[
            dict(label='image',coordinate=[0,0,20,40]),
            dict(label='table',coordinate=[10,0,80,80])]}])
        model.general_ocr_pipeline.text_det_model=lambda image,**kwargs:iter([
            {'dt_polys':np.array([[[30,45],[60,45],[60,55],[30,55]]])}])
        result=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model)
        self.assertEqual(self.table_inputs,[])
        self.assertEqual(result['table_res_list'],[])
        self.assertTrue(any(d['reason']=='table-overlap-image-fallback' for d in result['diagnostics']))

    def test_contained_formula_ocr_conflict_does_not_skip_table(self):
        model=self.pipeline()
        model.layout_det_model=lambda image:iter([{'boxes':[
            dict(label='formula',coordinate=[20,10,40,20]),
            dict(label='table',coordinate=[0,0,80,80])]}])
        model.general_ocr_pipeline.text_det_model=lambda image,**kwargs:iter([
            {'dt_polys':np.array([[[2,10],[60,10],[60,20],[2,20]],
                                 [[2,45],[60,45],[60,55],[2,55]]])}])
        result=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model)
        self.assertTrue(any(d['reason']=='ocr-line-crosses-visual-block' for d in result['diagnostics']))
        self.assertEqual(len(self.table_inputs),1)
        self.assertFalse(any(d['reason']=='table-ocr-conflict-image-fallback' for d in result['diagnostics']))

    def test_new_operation_cache_is_separate_and_preserves_region_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=load_config()
            config.update(results_dir=temporary)
            runtime=NativeRuntime(config)
            runtime.pipeline=self.pipeline()
            stream=io.BytesIO()
            Image.new('RGB',(80,80),'black').save(stream,format='PNG')
            request=dict(id='new',op='recognize_rebuild',pngBase64=base64.b64encode(stream.getvalue()).decode())
            first=handle_request(runtime,request)
            second=handle_request(runtime,request)
            self.assertFalse(first['cacheHit'])
            self.assertTrue(second['cacheHit'])
            self.assertEqual(first['result'],second['result'])
            self.assertEqual(first['result']['rebuild_policy'],POLICY)
            self.assertEqual(len(self.det_inputs),1)
            self.assertEqual([len(batch) for batch in self.rec_inputs],[3])
            self.assertEqual(second['modelProfile'],[])
            old_key=__import__('hashlib').sha256(__import__('native_worker').BUILD.encode()+runtime.cache_signature+stream.getvalue()).hexdigest()
            self.assertFalse((Path(temporary)/(old_key+'.json')).exists())

    def test_table_without_ocr_does_not_publish_empty_grid_as_recovered_structure(self):
        model=self.pipeline()
        model.layout_det_model=lambda image:iter([{'boxes':[dict(label='table',coordinate=[0,0,80,80])]}])
        model.general_ocr_pipeline.text_det_model=lambda image,**kwargs:iter([{'dt_polys':[]}])
        def empty_grid(image,**kwargs):
            self.table_inputs.append(kwargs)
            yield {'table_res_list':[{'pred_html':'<table><tr><td></td></tr></table>'}]}
        model.table_recognition_pipeline=empty_grid
        result=recognize_rebuild(np.zeros((80,80,3),dtype=np.uint8),model)
        self.assertEqual(self.rec_inputs,[])
        self.assertEqual(self.table_inputs,[],'Empty OCR must stop before table structure inference')
        self.assertEqual(result['table_res_list'],[])
        diagnostic=next(d for d in result['diagnostics'] if d['reason']=='table-empty-ocr-image-fallback')
        self.assertEqual(diagnostic['block_id'],'layout-0')

    def test_containment_scale_is_forwarded_and_separates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=load_config();config.update(results_dir=temporary)
            runtime=NativeRuntime(config);runtime.pipeline=self.pipeline()
            stream=io.BytesIO();Image.new('RGB',(80,80),'black').save(stream,format='PNG')
            request=dict(id='scale',op='recognize_rebuild',pngBase64=base64.b64encode(stream.getvalue()).decode())
            first=handle_request(runtime,request)
            request['pixelToPoint']=[1.,.25]
            second=handle_request(runtime,request)
            self.assertFalse(second['cacheHit'])
            self.assertEqual(first['result']['source_containment']['tolerance_px'],[2.,2.])
            self.assertEqual(second['result']['source_containment']['tolerance_px'],[1.,4.])
            self.assertTrue(handle_request(runtime,request)['cacheHit'])
            request['pixelToPoint']=[0,1]
            with self.assertRaises(ValueError):handle_request(runtime,request)


if __name__=='__main__':
    unittest.main()
