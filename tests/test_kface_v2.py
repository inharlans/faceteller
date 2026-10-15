import io, json, math, subprocess, unittest
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]

class MultiviewTests(unittest.TestCase):
    def test_factorization_recovers_known_depth(self):
        from facework.multiview import factorize, canonical_frame, POINTS
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(POINTS), 3)) * [1, 1.3, .4]
        idx = {k: i for i, k in enumerate(POINTS)}
        X[idx[1], 2] = 1.2  # nose tip well in front
        views = []
        for yaw, pitch in [(-40, -5), (-27, -5), (-13, -5), (0, -5), (13, -5), (27, -5), (40, -5), (0, 17), (-12, 18), (12, 18)]:
            a, b = math.radians(yaw), math.radians(pitch)
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
            views.append(((X @ (Rx @ Ry).T)[:, :2] * 200 + [400, 300]))
        Y, residual = factorize(np.stack(views))
        self.assertLess(residual, 1e-6)
        A = canonical_frame(Y, idx); B = canonical_frame(X, idx)
        self.assertTrue(np.allclose(A, B, atol=1e-4))

    def test_reflection_puts_nose_in_front(self):
        from facework.multiview import canonical_frame, POINTS
        idx = {k: i for i, k in enumerate(POINTS)}
        X = np.random.default_rng(1).normal(size=(len(POINTS), 3)); X[idx[1], 2] = -2
        Y = canonical_frame(X, idx)
        self.assertGreater(Y[idx[1], 2], (Y[idx[133], 2] + Y[idx[362], 2]) / 2)

class LabelLogicTests(unittest.TestCase):
    def test_classify_and_borderline(self):
        from facework.analysis_v2 import classify
        self.assertEqual(classify(0.1, [0.2, 0.3], ['low', 'medium', 'high'], 0.01), ('low', 'confident'))
        self.assertEqual(classify(0.205, [0.2, 0.3], ['low', 'medium', 'high'], 0.01), ('medium', 'borderline'))
        self.assertEqual(classify(0.35, [0.2, 0.3], ['low', 'medium', 'high'], 0.01), ('high', 'confident'))

    def test_percentile_is_clipped(self):
        from facework.analysis_v2 import percentile
        table = [[round(q, 2), q] for q in np.arange(0.05, 0.951, 0.05)]
        self.assertEqual(percentile(-1, table), 5); self.assertEqual(percentile(2, table), 95)
        self.assertEqual(percentile(0.5, table), 50)

    def test_type_matching_requires_confident_agreement(self):
        from facework import analysis_v2
        feats = {'brow_thickness': {'label': 'thick', 'status': 'confident', 'part': 'brow'},
                 'brow_eye_gap': {'label': 'close', 'status': 'confident', 'part': 'brow'}}
        names = {t['name_ko']: t for t in analysis_v2.match_types(feats)}
        self.assertEqual(names['귀미']['match'], 'close')
        feats['brow_eye_gap']['status'] = 'borderline'
        names = {t['name_ko']: t for t in analysis_v2.match_types(feats)}
        self.assertEqual(names['귀미']['match'], 'partial')
        feats['brow_eye_gap'].update(status='confident', label='wide')
        self.assertNotIn('귀미', {t['name_ko'] for t in analysis_v2.match_types(feats)})
        feats['brow_eye_gap']['status'] = 'withheld'
        self.assertEqual({t['name_ko']: t for t in analysis_v2.match_types(feats)}['귀미']['match'], 'partial')

    def test_schema_and_rules_share_vocabulary(self):
        schema = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
        rules = json.loads((ROOT / 'data/type_rules_v3.json').read_text(encoding='utf-8'))
        known = set(schema['features']) | set(schema['withheld_features'])
        for r in rules['records']:
            for c in r['conditions']:
                self.assertIn(c['feature'], known, r['record_id'])
                if c['feature'] in schema['features']:
                    self.assertTrue(set(c['values']) <= set(schema['features'][c['feature']]['classes']), r['record_id'])

class PrivacyTests(unittest.TestCase):
    def test_kface_material_not_tracked_or_packaged(self):
        tracked = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True).stdout.split()
        self.assertFalse([p for p in tracked if p.startswith(('data/raw/', 'data/derived/', 'models/', 'tmp/kface_review'))])
        ignored = (ROOT / '.gitignore').read_text(encoding='utf-8')
        for entry in ['data/raw/', 'data/derived/', 'models/', 'tmp/kface_review/']:
            self.assertIn(entry, ignored)
        # Reports and deliverables must not contain provided K-FACE photos or crops.
        leaked = [p for p in (ROOT / 'outputs').rglob('*') if p.is_file() and ('kface' in p.as_posix().lower()) and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bin')]
        self.assertEqual(leaked, [])

class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from facework import server
        cls.client = TestClient(server.app)

    def test_status_and_types(self):
        r = self.client.get('/api/v2/types'); self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()['records']), 103)
        self.assertTrue(all(x['original'] for x in r.json()['records']))
        r = self.client.get('/api/v2/status'); self.assertEqual(r.status_code, 200)
        self.assertEqual(self.client.get('/v2').status_code, 200)

    def test_analyze_frll_photo_without_persistence(self):
        if not (ROOT / 'models/kface_v2').exists():
            self.skipTest('v2 checkpoints not trained')
        before = sorted(p.as_posix() for p in (ROOT / 'data').rglob('*'))
        path = ROOT / 'data/raw/frll/neutral_front/neutral_front/025_03.jpg'
        r = self.client.post('/api/v2/analyze', files={'front': ('f.jpg', path.read_bytes(), 'image/jpeg')}, headers={'X-Face-Workbench': '1'})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json(); self.assertEqual(body['status'], 'ok')
        self.assertTrue(body['features'])
        for t in body['types']:
            self.assertTrue(t['service_text'] and t['overall'] in ('길', '흉', '길흉혼재'))
        self.assertEqual(before, sorted(p.as_posix() for p in (ROOT / 'data').rglob('*')))

    def test_rejects_blank_and_cross_origin(self):
        buff = io.BytesIO(); Image.new('RGB', (400, 400), 'white').save(buff, format='PNG')
        r = self.client.post('/api/v2/analyze', files={'front': ('b.png', buff.getvalue(), 'image/png')}, headers={'X-Face-Workbench': '1'})
        self.assertEqual(r.json()['reason'], 'no_face')
        self.assertEqual(self.client.post('/api/v2/analyze', files={'front': ('b.png', buff.getvalue(), 'image/png')}).status_code, 403)

class ToneTests(unittest.TestCase):
    def test_displayed_texts_have_no_stigmatising_terms(self):
        svc = json.loads((ROOT / 'data/interpretation_service_v1.json').read_text(encoding='utf-8'))
        banned = [w for w in svc['tone_policy']['service_text_banned_terms'] if w != '당신은']
        for r in svc['records']:
            shown = [r['service_text'], r['one_line']] + [v['풀이'] for v in r.get('domains_service', r['domains']).values()]
            for t in shown:
                bad = [w for w in banned if w in t and not (w == '거지' and '불거지' in t)]
                self.assertFalse(bad, f"{r['record_id']}: {bad} in {t[:60]}")

class SplitTests(unittest.TestCase):
    def test_participant_split_is_disjoint_and_label_based(self):
        import csv
        refs = list(csv.DictReader((ROOT / 'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')))
        split = {r['subject_id']: r['split'] for r in refs}
        self.assertEqual(len(split), 400)
        counts = {sp: sum(v == sp for v in split.values()) for sp in ('train', 'val', 'test')}
        self.assertEqual(counts, {'train': 276, 'val': 62, 'test': 62})
        proto = json.loads((ROOT / 'data/derived/kface/split_protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(proto['selection_uses'], 'labels_only_no_model_outputs')
        self.assertEqual(proto['classes_below_min'], 0)
        caps = list(csv.DictReader((ROOT / 'data/derived/kface/dataset_v3/captures.csv').open(encoding='utf-8-sig')))
        self.assertTrue(all(split[c['subject_id']] == c['split'] for c in caps))       # no capture-level leakage
        self.assertTrue(all(c['split'] == 'train' for c in caps if c['used_for_training'] == 'True'))
        # checkpoints were trained on the current labels
        import torch
        summary = json.loads((ROOT / 'data/derived/kface/reference_summary.json').read_text(encoding='utf-8'))
        from facework.kface_ensemble import checkpoints
        folds = json.loads((ROOT / 'models/kface_v4/folds.json').read_text(encoding='utf-8'))['assignment']
        self.assertFalse(set(folds) & {s for s, v in split.items() if v == 'test'})   # test people never in any fold
        for part in ('eye', 'brow', 'nose', 'mouth', 'face'):
            paths = checkpoints(part, ROOT / 'models/kface_v4'); self.assertEqual(len(paths), 10, part)
            for pth in paths:
                meta = torch.load(pth, map_location='cpu', weights_only=False)['meta']
                self.assertEqual(meta['labels_sha256'], summary['labels_sha256'], str(pth))

class ReviewTests(unittest.TestCase):
    def test_blind_review_validation_and_isolated_save(self):
        import tempfile
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from facework import server
        client = TestClient(server.app); h = {'X-Face-Workbench': '1'}
        items = client.get('/api/v2/review/items').json(); s = items['subjects'][0]
        self.assertEqual(set(items), {'subjects', 'tasks', 'done'})  # no AI labels exposed to the reviewer
        self.assertTrue(all(isinstance(x, str) for x in items['subjects']))
        self.assertEqual(client.get(f'/api/v2/review/crop/{s}/face').status_code, 200)
        self.assertEqual(client.get('/api/v2/review/crop/..%2F..%2Fx/face').status_code, 404)
        self.assertEqual(client.get(f'/api/v2/review/crop/{s}/nose').status_code, 404)
        body = {'subject_id': s, 'reviewer': 't', 'brow_shape': 'straight', 'brow_thickness': 'thin', 'face_shape': 'oval'}
        with tempfile.TemporaryDirectory() as td, patch.object(server, 'REVIEW_ANSWERS', Path(td) / 'a.csv'):
            self.assertEqual(client.post('/api/v2/review', json=body | {'face_shape': 'heart'}, headers=h).status_code, 422)
            self.assertEqual(client.post('/api/v2/review', json=body, headers=h).status_code, 200)
            self.assertIn(s, (Path(td) / 'a.csv').read_text(encoding='utf-8-sig'))

if __name__ == '__main__':
    unittest.main()
