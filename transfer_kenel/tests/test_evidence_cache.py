import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'services/structure_v3'))
from evidence_cache import read_evidence, LEGACY_POLICIES


class EvidenceCacheTests(unittest.TestCase):
    def test_exact_key_only_reuses_model_evidence_and_strips_old_policy_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data=b'exact image bytes'
            config=dict(model_dir='own-models',cpu_threads=6,results_dir='ignored')
            identity=json.dumps({k:v for k,v in config.items() if k!='results_dir'},
                sort_keys=True,ensure_ascii=False).encode('utf8')
            policy=LEGACY_POLICIES[0]
            key=hashlib.sha256(b'build'+identity+policy.encode()+data).hexdigest()
            box=dict(id='a',coordinate=[0,0,20,10],label='text')
            poly=[[0,0],[20,0],[20,10],[0,10]]
            old=dict(width=100,height=100,rebuild_policy=policy,layout_det_res=dict(boxes=[box]),
                region_det_res=dict(boxes=[box]),overall_ocr_res=dict(rec_texts=['keep'],rec_scores=[.99],rec_polys=[poly]),
                ocr_pipeline=dict(dt_polys=[poly]),diagnostics=[dict(reason='old-conflict')],
                table_res_list=[dict(old=True)],block_ownership=dict(old=True))
            path=Path(directory)/(key+'.json');path.write_text(json.dumps(old),encoding='utf8')
            found=read_evidence(data,config,'build',[directory],width=100,height=100)
            self.assertEqual(found['overall_ocr_res']['rec_texts'],['keep'])
            self.assertNotIn('diagnostics',found)
            self.assertNotIn('table_res_list',found)
            self.assertNotIn('block_ownership',found)
            self.assertIsNone(read_evidence(b'other image',config,'build',[directory],width=100,height=100))
            self.assertIsNone(read_evidence(data,dict(config,cpu_threads=1),'build',[directory],width=100,height=100))
            old['overall_ocr_res']['rec_scores']=[float('nan')]
            path.write_text(json.dumps(old),encoding='utf8')
            self.assertIsNone(read_evidence(data,config,'build',[directory],width=100,height=100))
