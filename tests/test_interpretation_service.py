import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_KEYS = {'재물', '관운·명예', '수명·건강', '가족·배우자·자녀', '성품·처세', '기타'}
OVERALL = {'길', '흉', '길흉혼재', '중립/불명'}
DOMAIN_GRADES = {'길', '흉', '혼재'}
SERVICE_FIELDS = ['record_id', 'name_zh', 'name_ko', 'part', 'hanja_meaning', 'one_line', 'overall', 'domains',
                  'translation_ko', 'easy_explanation', 'service_text', 'caution', 'evidence_quotes']
RULE_FIELDS = ['record_id', 'name_ko', 'part', 'conditions', 'unobservable_conditions', 'rule_coverage',
               'min_conditions_to_suggest', 'notes']


def load(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf8'))


class InterpretationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = {r['record_id']: r for r in load('interpretation_catalog_v1.json')['records']}
        cls.order = list(cls.catalog)
        cls.service = load('interpretation_service_v1.json')
        cls.rules = load('type_rules_v2.json')

    def check_ids(self, rows):
        self.assertEqual(len(rows), 103)
        self.assertEqual([r['record_id'] for r in rows], self.order)
        for r in rows:
            src = self.catalog[r['record_id']]
            self.assertEqual((r['name_ko'], r['part']), (src['name_ko'], src['part']), r['record_id'])

    def test_service_records_complete(self):
        self.assertEqual(self.service['version'], 'interpretation-service-v1')
        self.assertIn('tone_policy', self.service)
        rows = self.service['records']
        self.check_ids(rows)
        for r in rows:
            rid = r['record_id']
            self.assertEqual(r['name_zh'], self.catalog[rid]['name_zh'], rid)
            for f in SERVICE_FIELDS:
                self.assertIn(f, r, rid)
                self.assertTrue(r[f] or f == 'domains', f'{rid} empty {f}')
            self.assertIn(r['overall'], OVERALL, rid)

    def test_service_evidence_is_source_substring(self):
        for r in self.service['records']:
            src = self.catalog[r['record_id']]['original_section']
            self.assertTrue(r['evidence_quotes'], r['record_id'])
            for q in r['evidence_quotes']:
                self.assertTrue(q and q in src, f"{r['record_id']}: {q!r}")

    def test_domains_valid_and_consistent_with_overall(self):
        for r in self.service['records']:
            rid = r['record_id']
            self.assertLessEqual(set(r['domains']), DOMAIN_KEYS, rid)
            grades = set()
            for k, v in r['domains'].items():
                self.assertEqual(set(v), {'평가', '풀이'}, rid)
                self.assertIn(v['평가'], DOMAIN_GRADES, f'{rid} {k}')
                self.assertTrue(v['풀이'], f'{rid} {k}')
                grades.add(v['평가'])
            if r['overall'] == '길':
                self.assertEqual(grades, {'길'}, rid)
            elif r['overall'] == '흉':
                self.assertEqual(grades, {'흉'}, rid)
            elif r['overall'] == '길흉혼재':
                self.assertTrue('혼재' in grades or {'길', '흉'} <= grades, rid)

    def test_service_text_follows_tone_policy(self):
        banned = self.service['tone_policy']['service_text_banned_terms']
        for r in self.service['records']:
            text = r['service_text']
            self.assertTrue('옛 관상서' in text or '문헌' in text, r['record_id'])
            for term in banned:
                self.assertNotIn(term, text, f"{r['record_id']}: {term}")

    def test_rules_records_and_evidence(self):
        self.assertEqual(self.rules['version'], 'type-rules-v2-draft')
        self.assertIn('matching_policy', self.rules)
        rows = self.rules['records']
        self.check_ids(rows)
        for r in rows:
            src = self.catalog[r['record_id']]['original_section']
            for f in RULE_FIELDS:
                self.assertIn(f, r, r['record_id'])
            for c in r['conditions'] + r['unobservable_conditions']:
                self.assertTrue(c['evidence'] and c['evidence'] in src, f"{r['record_id']}: {c['evidence']!r}")

    def test_rule_features_in_vocabulary(self):
        vocab = {k: set(v['values']) for k, v in self.rules['feature_vocabulary'].items()}
        for r in self.rules['records']:
            rid = r['record_id']
            for c in r['conditions']:
                self.assertIn(c['feature'], vocab, rid)
                self.assertTrue(c['values'], rid)
                self.assertLessEqual(set(c['values']), vocab[c['feature']], f"{rid} {c['feature']}")
                self.assertTrue(0 < c['weight'] <= 1, rid)
            for u in r['unobservable_conditions']:
                self.assertTrue(u['description'] and u['reason'], rid)

    def test_rule_coverage_consistent(self):
        for r in self.rules['records']:
            rid, n = r['record_id'], len(r['conditions'])
            self.assertIn(r['rule_coverage'], {'full', 'partial', 'none'}, rid)
            if r['rule_coverage'] == 'none':
                self.assertEqual((n, r['min_conditions_to_suggest']), (0, 0), rid)
            else:
                self.assertGreater(n, 0, rid)
                self.assertTrue(1 <= r['min_conditions_to_suggest'] <= n, rid)
            if r['rule_coverage'] == 'full':
                self.assertFalse(r['unobservable_conditions'], rid)
            if r['rule_coverage'] == 'partial':
                self.assertTrue(r['unobservable_conditions'], rid)


if __name__ == '__main__':
    unittest.main()
