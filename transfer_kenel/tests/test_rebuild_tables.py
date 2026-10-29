"""Manual geometry truth for horizontally ruled tables without vertical rules."""
import copy
from html.parser import HTMLParser
import unittest
import numpy as np
from pdf_extractor import usage
from rebuild_tables import validated_table


def bands():
    image=np.full((240,300,3),180,dtype=np.uint8)
    for y in (48,96,144,192):
        image[y:y+3]=255
    cells=[[0,y,300,y+48] for y in (0,48,96,144,192)]
    cells += [[115,y,185,y+48] for y in (48,96)]
    texts=[]
    boxes=[]
    for r in range(5):
        values=[(110,190,f'shared{r}')] if r in (0,3,4) else [
            (25,80,f'left{r}'),(140,158,f'choice{r}'),(205,280,f'right{r}')]
        for x,z,text in values:
            texts.append(text)
            boxes.append([x,r*48+12,z,r*48+24])
    raw=dict(cell_box_list=cells,pred_html='<table><tr><td>incorrect repeated choice1</td></tr></table>')
    return image,raw,dict(rec_texts=texts,rec_boxes=np.array(boxes,dtype=float))


class TableGeometryTests(unittest.TestCase):
    def test_overlapping_model_cells_use_repeated_gutters_and_shared_row_spans(self):
        image,table,ocr=bands()
        original=copy.deepcopy(table)
        result=validated_table(table,ocr,image)
        self.assertEqual(table,original)
        self.assertEqual(len(result['cell_box_list']),9)
        self.assertEqual(result['pred_html'].count('colspan="3"'),3)
        evidence=result['reconstruction_evidence']
        self.assertEqual(len(evidence['column_boundaries']),4)
        self.assertEqual(sorted(i for c in evidence['line_assignments'] for i in c['ocr_indexes']),list(range(9)))
        self.assertEqual([c['column'] for c in evidence['line_assignments'] if c['row']==1],[0,1,2])
        self.assertIn('right1',result['pred_html'])
        self.assertEqual(result['raw_model_pred_html'],original['pred_html'])

    def test_without_measured_row_rules_no_topology_is_invented(self):
        image,table,ocr=bands()
        image[:]=180
        self.assertIsNone(validated_table(table,ocr,image))

    def test_one_split_row_is_insufficient_to_infer_repeated_columns(self):
        image,table,ocr=bands()
        ocr['rec_texts']=ocr['rec_texts'][:4]+['merged2']+ocr['rec_texts'][7:]
        ocr['rec_boxes']=np.vstack((ocr['rec_boxes'][:4],[[110,108,190,120]],ocr['rec_boxes'][7:]))
        self.assertIsNone(validated_table(table,ocr,image))

    def test_valid_geometry_still_rejects_html_lost_or_duplicated_text(self):
        image=np.full((30,100,3),255,np.uint8)
        table=dict(cell_box_list=[[0,0,50,30],[50,0,100,30]],pred_html='<table><tr><td>left</td><td>left</td></tr></table>')
        ocr=dict(rec_texts=['left','right'],rec_boxes=np.array([[5,5,30,15],[60,5,90,15]]))
        self.assertIsNone(validated_table(table,ocr,image))
        table['pred_html']='<table><tr><td>left</td><td>right</td></tr></table>'
        self.assertIs(validated_table(table,ocr,image),table)


if __name__=='__main__':
    unittest.main()
