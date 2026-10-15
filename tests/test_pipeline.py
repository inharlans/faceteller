import io,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from PIL import Image
from facework import server
from scripts import finalize_labels
from scripts.import_workbook import collect_edits
ROOT=Path(__file__).resolve().parents[1]

class AppTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(server.app)
    def test_knowledge_and_exports(self):
        r=self.client.get('/api/knowledge');self.assertEqual(r.status_code,200)
        self.assertEqual(len(r.json()['interpretations']['records']),103)
        self.assertEqual(len(r.json()['measurements']['features']),17)
        self.assertEqual(self.client.get('/knowledge-ui.js').status_code,200)
        self.assertEqual(self.client.get('/api/knowledge/export/interpretations').status_code,200)
        self.assertEqual(self.client.get('/api/knowledge/export/unknown').status_code,404)
    def test_project_and_media(self):
        r=self.client.get('/api/project');self.assertEqual(r.status_code,200);self.assertEqual(r.json()['summary']['subjects'],102)
        r=self.client.get('/media/frll_001_03/eye_right');self.assertEqual(r.status_code,200);self.assertGreater(Image.open(io.BytesIO(r.content)).width,20)
        self.assertEqual(self.client.get('/media/unknown/face').status_code,404)
    def test_reject_cross_origin_write(self):
        self.assertEqual(self.client.post('/api/review',json={}).status_code,403)
        self.assertEqual(self.client.post('/api/review',json={},headers={'X-Face-Workbench':'1','Origin':'https://example.com'}).status_code,403)
    def test_review_persistence_conflicts_and_invalid_label(self):
        # Isolated copy: no test edits masquerade as human reviews in delivered data.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data/annotations').mkdir(parents=True);(root/'data/derived').mkdir()
            for name in ['annotations','crops','images','summary']:
                shutil.copyfile(ROOT/f'data/derived/{name}.json',root/f'data/derived/{name}.json')
            for name in ['visual_v1.tsv']:shutil.copyfile(ROOT/f'data/annotations/{name}',root/f'data/annotations/{name}')
            shutil.copyfile(ROOT/'data/label_schema.json',root/'data/label_schema.json')
            with patch.object(server,'ROOT',root),patch.object(server,'DATA',root/'data/derived'),patch.object(server,'OVERRIDES',root/'data/annotations/overrides.json'),patch.object(finalize_labels,'ROOT',root):
                h={'X-Face-Workbench':'1'};payload={'annotation_id':'frll_001_03:eye_right','label':'narrow','notes':'test','revision':0}
                r=self.client.post('/api/review',json=payload,headers=h);self.assertEqual(r.status_code,200)
                self.assertEqual(self.client.post('/api/review',json=payload,headers=h).status_code,409)
                payload.update(label='not_a_label',revision=1);self.assertEqual(self.client.post('/api/review',json=payload,headers=h).status_code,422)
                rows=json.loads((root/'data/derived/annotations.json').read_text(encoding='utf-8'));self.assertEqual(next(r for r in rows if r['annotation_id']=='frll_001_03:eye_right')['manual_label'],'narrow')
    def test_blank_and_invalid_images(self):
        h={'X-Face-Workbench':'1'}
        r=self.client.post('/api/analyze',files={'file':('bad.png',b'not image','image/png')},headers=h);self.assertEqual(r.status_code,422)
        buff=io.BytesIO();Image.new('RGB',(512,512),'white').save(buff,format='PNG')
        r=self.client.post('/api/analyze',files={'file':('blank.png',buff.getvalue(),'image/png')},headers=h)
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['reason'],'no_face')
    def test_real_upload_and_no_persistence(self):
        before=list((ROOT/'data').rglob('*'))
        p=ROOT/'data/raw/frll/neutral_front/neutral_front/025_03.jpg'
        r=self.client.post('/api/analyze',files={'file':('image.jpg',p.read_bytes(),'image/jpeg')},headers={'X-Face-Workbench':'1'})
        self.assertEqual(r.status_code,200,r.text);self.assertEqual(r.json()['status'],'ok');self.assertEqual(len(r.json()['predictions']),7)
        self.assertEqual(len(r.json()['measurements']),17)
        self.assertTrue(all(m['size_label'] is None for m in r.json()['measurements']))
        self.assertEqual(before,list((ROOT/'data').rglob('*')))
    def test_workbook_unedited_import(self):
        path=ROOT/'outputs/face-dataset-v1/얼굴형태_데이터셋_검수.xlsx'
        self.assertEqual(collect_edits(path),[])
if __name__=='__main__':unittest.main()
