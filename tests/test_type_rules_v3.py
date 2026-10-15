import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_FIELDS = ['record_id', 'name_ko', 'part', 'conditions', 'unobservable_conditions', 'rule_coverage',
               'min_conditions_to_suggest', 'notes', 'photo_suggestion_hold']
V3_ADDED = {'brow_density', 'brow_tail_clarity', 'brow_glabella', 'brow_gap', 'eyelid', 'sclera_show', 'under_eye',
            'eye_depth', 'eye_spacing', 'nostril_show', 'alar_fullness', 'nose_dorsum', 'lip_bow',
            'philtrum_definition', 'philtrum_length', 'lip_projection', 'chin_shape', 'cheekbone', 'cheek_fullness',
            'lower_face_ratio', 'chin_projection'}


def load(name):
    return json.loads((ROOT / 'data' / name).read_text(encoding='utf8'))


class TypeRulesV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = {r['record_id']: r for r in load('interpretation_catalog_v1.json')['records']}
        cls.order = list(cls.catalog)
        cls.v2 = load('type_rules_v2.json')
        cls.v3 = load('type_rules_v3.json')
        cls.rows = cls.v3['records']
        cls.v2_rows = {r['record_id']: r for r in cls.v2['records']}

    def test_top_level(self):
        self.assertEqual(self.v3['version'], 'type-rules-v3-draft')
        for key in ('feature_vocabulary', 'matching_policy', 'weight_legend', 'changes_from_v2'):
            self.assertIn(key, self.v3)
        self.assertTrue(self.v3['matching_policy'].startswith(self.v2['matching_policy']))

    def test_records_match_catalog(self):
        self.assertEqual(len(self.rows), 103)
        self.assertEqual([r['record_id'] for r in self.rows], self.order)
        for r in self.rows:
            src = self.catalog[r['record_id']]
            self.assertEqual((r['name_ko'], r['part']), (src['name_ko'], src['part']), r['record_id'])
            for f in RULE_FIELDS:
                self.assertIn(f, r, r['record_id'])

    def test_evidence_is_source_substring(self):
        for r in self.rows:
            src = self.catalog[r['record_id']]['original_section']
            for c in r['conditions'] + r['unobservable_conditions']:
                self.assertTrue(c['evidence'] and c['evidence'] in src, f"{r['record_id']}: {c['evidence']!r}")

    def test_vocabulary_extends_v2(self):
        vocab = self.v3['feature_vocabulary']
        for k, v in self.v2['feature_vocabulary'].items():
            self.assertIn(k, vocab)
            self.assertEqual(vocab[k]['values'], v['values'], k)
        self.assertEqual(set(vocab) - set(self.v2['feature_vocabulary']), V3_ADDED)
        for k, v in vocab.items():
            self.assertTrue(v['values'] and v['definition'], k)

    def test_conditions_in_vocabulary(self):
        vocab = {k: set(v['values']) for k, v in self.v3['feature_vocabulary'].items()}
        for r in self.rows:
            rid = r['record_id']
            for c in r['conditions']:
                self.assertIn(c['feature'], vocab, rid)
                self.assertTrue(c['values'], rid)
                self.assertLessEqual(set(c['values']), vocab[c['feature']], f"{rid} {c['feature']}")
                self.assertLess(len(set(c['values'])), len(vocab[c['feature']]), f"{rid} {c['feature']} allows all values")
                self.assertIn(c['weight'], (0.5, 0.7, 1.0), rid)
            for u in r['unobservable_conditions']:
                self.assertTrue(u['description'] and u['reason'], rid)

    def test_coverage_and_min_consistent(self):
        for r in self.rows:
            rid, n = r['record_id'], len(r['conditions'])
            expected = 'none' if n == 0 else ('partial' if r['unobservable_conditions'] else 'full')
            self.assertEqual(r['rule_coverage'], expected, rid)
            self.assertEqual(r['min_conditions_to_suggest'], n, rid)

    def test_unobservable_not_increased(self):
        for r in self.rows:
            old = self.v2_rows[r['record_id']]
            self.assertLessEqual(len(r['unobservable_conditions']), len(old['unobservable_conditions']), r['record_id'])
        total = lambda rows: sum(len(r['unobservable_conditions']) for r in rows)
        self.assertLess(total(self.rows), total(self.v2['records']))

    def test_single_weak_condition_notes_shared_types(self):
        for r in self.rows:
            if len(r['conditions']) == 1 and r['conditions'][0]['weight'] < 1.0:
                self.assertIn('공유', r['notes'], r['record_id'])

    def test_hold_flag(self):
        for r in self.rows:
            self.assertIsInstance(r['photo_suggestion_hold'], bool, r['record_id'])
            if r['photo_suggestion_hold']:
                self.assertTrue(r.get('hold_reason'), r['record_id'])

    def test_changes_summary_matches_records(self):
        ch = self.v3['changes_from_v2']
        self.assertEqual(ch['unobservable_conditions_total']['v3'], sum(len(r['unobservable_conditions']) for r in self.rows))
        self.assertEqual(ch['conditions_total']['v3'], sum(len(r['conditions']) for r in self.rows))
        changed = [r['record_id'] for r in self.rows if r != {**self.v2_rows[r['record_id']],
                                                               'photo_suggestion_hold': r['photo_suggestion_hold'],
                                                               **({'hold_reason': r['hold_reason']} if 'hold_reason' in r else {})}]
        self.assertEqual(sorted(changed), sorted(ch['records_changed_ids'] + ch['notes_only_updated_ids']))
        for rid in ch['notes_only_updated_ids']:
            r, old = next(x for x in self.rows if x['record_id'] == rid), self.v2_rows[rid]
            self.assertEqual((r['conditions'], r['unobservable_conditions']),
                             (old['conditions'], old['unobservable_conditions']), rid)
        self.assertEqual(ch['records_changed'], len(ch['records_changed_ids']))

    def test_v2_file_untouched_version(self):
        self.assertEqual(self.v2['version'], 'type-rules-v2-draft')


if __name__ == '__main__':
    unittest.main()
