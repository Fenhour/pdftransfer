"""Synthetic ownership truth and recognizer scheduling, without model weights."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pdf_extractor import usage
from rebuild_ocr import recognize_blocks


def polygon(x0,y0,x1,y1):
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]


def block(identifier,label,box):
    return dict(id=identifier,label=label,coordinate=box)


class Recognizer:
    def __init__(self):
        self.batch_sampler=SimpleNamespace(batch_size=1)
        self.pre_tfs={'ReisizeNorm':SimpleNamespace(rec_image_shape=[3,48,320],max_imgW=3200)}
        self.calls=[]
        self.error=None
        self.drop_last=False

    def __call__(self,images,batch_size,**kwargs):
        self.batch_sampler.batch_size=batch_size
        self.calls.append((list(images),batch_size,kwargs))
        if self.error:
            raise self.error
        for image in images[:-1] if self.drop_last else images:
            yield dict(rec_text=f'width-{image.shape[1]}',rec_score=.99)


class OCR:
    def __init__(self,polys):
        self.polys=polys
        self.det_calls=[]
        self.crop_calls=[]
        self.text_rec_model=Recognizer()

    def text_det_model(self,image,**kwargs):
        self.det_calls.append((image.copy(),kwargs))
        yield {'dt_polys':self.polys}

    def _crop_by_polys(self,image,polys):
        self.crop_calls.append((image.copy(),np.array(polys)))
        for poly in polys:
            x0,y0=np.floor(poly.min(axis=0)).astype(int)
            x1,y1=np.ceil(poly.max(axis=0)).astype(int)
            yield image[y0:y1,x0:x1].copy()


class PageOCRTests(unittest.TestCase):
    def setUp(self):
        factory=patch('rebuild_ocr._ocr_result',lambda value:value)
        factory.start()
        self.addCleanup(factory.stop)

    def run_ocr(self,polys,blocks,ownership=None,shape=(200,400,3),ocr=None):
        ocr=ocr or OCR(polys)
        self.ocr=ocr
        self.diagnostics=[]
        self.image=np.zeros(shape,dtype=np.uint8)
        return recognize_blocks(self.image,SimpleNamespace(general_ocr_pipeline=ocr),
                                blocks,ownership or {},self.diagnostics)

    def test_full_page_once_cross_class_batch_and_ids_survive_ratio_sort(self):
        # Input order is deliberately different from page and rec width order.
        polys=np.asarray([polygon(10,70,40,80),polygon(10,10,110,20),polygon(10,130,60,140)],dtype=float)
        original=polys.copy()
        polys.setflags(write=False)
        records,locals_,info=self.run_ocr(polys,[block('title','paragraph_title',[0,0,150,40]),
            block('body','text',[0,50,150,100]),block('table','table',[0,110,150,160])])
        self.assertEqual(len(self.ocr.det_calls),1)
        self.assertEqual(self.ocr.det_calls[0][0].shape,(200,400,3))
        self.assertEqual([c[1] for c in self.ocr.text_rec_model.calls],[3])
        self.assertEqual([v['block_id'] for v in records],['title','body','table'])
        self.assertEqual([v['text'] for v in records],['width-100','width-30','width-50'])
        self.assertEqual([v['label'] for v in records],['paragraph_title','text','table'])
        self.assertEqual([v['line_id'] for v in records],['line-0','line-1','line-2'])
        self.assertEqual(info['recognitionBatches'][0]['line_ids'],['line-1','line-2','line-0'])
        np.testing.assert_array_equal(polys,original)
        self.assertEqual(self.ocr.text_rec_model.batch_sampler.batch_size,1)

    def test_table_result_is_numpy_and_local_not_page_coordinates(self):
        records,locals_,info=self.run_ocr([polygon(22,43,62,53)],
            [block('t','table',[20.5,40.5,100.5,90.5])])
        local=locals_['t']
        self.assertEqual(records[0]['box'],[22.,43.,62.,53.])
        self.assertEqual(local['rec_boxes'].shape,(1,4))
        self.assertEqual(local['rec_boxes'].dtype,np.float32)
        self.assertEqual(local['rec_boxes'].tolist(),[[2.,3.,42.,13.]])
        self.assertEqual(local['rec_polys'].shape,(1,4,2))
        self.assertEqual(local['doc_preprocessor_res']['output_img'].shape,(51,81,3))
        self.assertEqual(local['line_ids'],['line-0'])
        self.assertEqual(local['block_id'],'t')
        self.assertFalse(local['model_settings']['use_doc_preprocessor'])

    def test_image_transcript_enters_shared_rec_but_cross_visual_line_does_not(self):
        records,locals_,info=self.run_ocr([polygon(5,5,30,15),polygon(55,5,75,15),polygon(40,25,60,35)],
            [block('visual','image',[50,0,90,50]),block('text','text',[0,0,100,50])])
        self.assertEqual([r['block_id'] for r in records],['text','visual'])
        self.assertEqual(records[1]['label'],'image')
        self.assertEqual(sum(len(c[0]) for c in self.ocr.text_rec_model.calls),2)
        self.assertEqual(info['skippedVisualLineCount'],1)
        # Padding extends below the picture; only its actual pixels are masked.
        self.assertEqual(self.ocr.crop_calls[0][0][:50,50:90].min(),255)
        self.assertEqual(self.ocr.crop_calls[0][0][51:,50:90].max(),0)
        cross=next(d for d in self.diagnostics if d.get('line_id')=='line-2')
        self.assertEqual(cross['reason'],'ocr-line-crosses-visual-block')
        self.assertEqual(set(cross['candidate_ids']),{'text','visual'})
        self.assertEqual(self.image.max(),0)

    def test_polygon_ownership_avoids_false_bbox_visual_conflict(self):
        diagonal=[[0,0],[10,0],[30,20],[20,20]]
        records,_,info=self.run_ocr([diagonal],
            [block('visual','image',[0,15,10,25]),block('text','text',[0,0,80,40])])
        self.assertEqual(len(records),1)
        self.assertEqual(info['skippedVisualLineCount'],0)

    def test_cross_block_and_unmatched_lines_are_auditable_and_not_recognized(self):
        records,_,info=self.run_ocr([polygon(40,10,60,20),polygon(120,10,140,20)],
            [block('a','text',[0,0,50,50]),block('b','table',[50,0,100,50])])
        self.assertEqual(records,[])
        self.assertEqual(self.ocr.text_rec_model.calls,[])
        conflict=next(d for d in self.diagnostics if d['reason']=='ocr-line-crosses-layout-blocks')
        self.assertEqual(conflict['candidate_ids'],['a','b'])
        self.assertEqual(info['unmatchedLineCount'],1)

    def test_small_layout_edge_error_keeps_full_line_without_relabeling(self):
        records,local,info=self.run_ocr([polygon(91,41,207,66)],
            [block('heading','paragraph_title',[90,42.1,206.7,63.4])])
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['label'],'paragraph_title')
        self.assertEqual(records[0]['box'],[91.,41.,207.,66.])
        self.assertEqual(info['line_assignments'][0]['assignment_method'],'bounded-layout-edge-tolerance')
        self.assertEqual(self.ocr.crop_calls[0][1].min(axis=(0,1)).tolist(),[13.,11.])
        self.assertEqual(self.diagnostics,[])

    def test_large_boundary_error_still_rejects_instead_of_reclaiming_content(self):
        records,_,info=self.run_ocr([polygon(0,0,100,30)],[block('a','text',[0,0,100,16])])
        self.assertEqual(records,[])
        self.assertEqual(info['line_assignments'][0]['state'],'partial-owner-conflict')

    def test_ownership_exclusions_cannot_be_reclaimed_by_enclosing_text(self):
        records,_,_=self.run_ocr([polygon(20,10,40,20)],
            [block('parent','text',[0,0,100,50])],
            {'parent':{'excluded_rects':[[15,0,45,30]]}})
        self.assertEqual(records,[])
        self.assertEqual(self.ocr.text_rec_model.calls,[])
        self.assertEqual(self.diagnostics[0]['reason'],'ocr-line-unmatched-block')

    def test_bounded_batches_for_narrow_and_extreme_width_lines(self):
        polys=[polygon(1,2+i*12,41,10+i*12) for i in range(9)]
        polys += [polygon(1,120+i*12,2500,128+i*12) for i in range(4)]
        records,_,info=self.run_ocr(polys,[block('body','text',[0,0,2600,200])],shape=(200,2600,3))
        self.assertEqual(len(records),13)
        self.assertEqual(sum(v['batch_size'] for v in info['recognitionBatches']),13)
        self.assertGreater(len(info['recognitionBatches']),1)
        for batch in info['recognitionBatches']:
            self.assertLessEqual(batch['batch_size'],8)
            self.assertLessEqual(batch['padded_pixels'],614400)
            self.assertLessEqual(batch['padding_ratio'],1.5)

    def test_sampler_restored_when_recognizer_raises(self):
        ocr=OCR([polygon(0,0,20,10)])
        ocr.text_rec_model.error=RuntimeError('synthetic recognizer failure')
        with self.assertRaisesRegex(RuntimeError,'synthetic recognizer failure'):
            self.run_ocr(ocr.polys,[block('a','text',[0,0,100,50])],ocr=ocr)
        self.assertEqual(ocr.text_rec_model.batch_sampler.batch_size,1)

    def test_rec_result_count_mismatch_raises_and_restores_sampler(self):
        ocr=OCR([polygon(0,0,20,10),polygon(0,20,20,30)])
        ocr.text_rec_model.drop_last=True
        with self.assertRaisesRegex(ValueError,'result count'):
            self.run_ocr(ocr.polys,[block('a','text',[0,0,100,50])],ocr=ocr)
        self.assertEqual(ocr.text_rec_model.batch_sampler.batch_size,1)

    def test_empty_low_score_and_nonfinite_results_are_rejected_with_line_ids(self):
        class RejectedRecognizer(Recognizer):
            def __call__(self,images,batch_size,**kwargs):
                self.batch_sampler.batch_size=batch_size
                yield dict(rec_text='   ',rec_score=.99)
                yield dict(rec_text='Low confidence',rec_score=.1)
                yield dict(rec_text='Invalid score',rec_score=float('nan'))
        ocr=OCR([polygon(0,y,20,y+10) for y in (0,20,40)])
        ocr.text_rec_model=RejectedRecognizer()
        ocr.text_rec_score_thresh=.5
        records,locals_,info=self.run_ocr(ocr.polys,[block('a','text',[0,0,100,70])],ocr=ocr)
        self.assertEqual(records,[])
        self.assertEqual(locals_['a']['rec_boxes'].shape,(0,4))
        self.assertEqual([d['line_id'] for d in self.diagnostics],['line-0','line-1','line-2'])
        self.assertTrue(all(d['reason']=='ocr-line-recognition-rejected' for d in self.diagnostics))
        self.assertEqual([v['state'] for v in info['line_assignments']],['recognition-rejected']*3)
        self.assertEqual(ocr.text_rec_model.batch_sampler.batch_size,1)

    def test_empty_detector_result_has_valid_empty_local_arrays(self):
        records,locals_,info=self.run_ocr([], [block('a','text',[0,0,100,50])])
        self.assertEqual(records,[])
        self.assertEqual(locals_['a']['rec_boxes'].shape,(0,4))
        self.assertEqual(locals_['a']['rec_polys'].shape,(0,4,2))
        self.assertEqual(info['recognitionBatches'],[])

    def test_invalid_detector_geometry_is_reported_without_rec(self):
        records,_,info=self.run_ocr([[[0,0],[10,10]],polygon(0,0,0,10),
                                    [[float('nan'),0],[10,0],[10,10],[0,10]]],
                                   [block('a','text',[0,0,100,50])])
        self.assertEqual(records,[])
        self.assertEqual(info['inputDetCount'],3)
        self.assertEqual([d['reason'] for d in self.diagnostics],['invalid-ocr-polygon']*3)

    def test_empty_page_does_not_fabricate_ocr_owners(self):
        records,locals_,info=self.run_ocr([polygon(0,0,20,10)],[])
        self.assertEqual(records,[])
        self.assertEqual(locals_,{})
        self.assertEqual(info['unmatchedLineCount'],1)


if __name__=='__main__':
    unittest.main()
