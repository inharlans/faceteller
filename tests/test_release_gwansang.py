from copy import deepcopy
from pathlib import Path
import unittest

from facework.pilot_contract import Decision, build_result
from facework.release_gwansang import GwansangCatalog
from facework.release_readings import load_reading_catalog

ROOT = Path(__file__).resolve().parents[1]


class Snapshot:
    def __init__(self, catalog):
        self.document = deepcopy(catalog.document)
        self.sha = catalog.identity['gwansang_catalog_sha256']

    def gwansang_catalog(self):
        return self.document, self.sha


def decision(label, confidence=.9, status='CONFIRMED'):
    feature = label.split(':')[0]
    return Decision(feature, 'brow', status, label if status == 'CONFIRMED' else None,
                    label, confidence, None, None, 'test', (), 'classification')


class GroupedReadingsTests(unittest.TestCase):
    def setUp(self):
        self.catalog = GwansangCatalog.load(ROOT)

    def test_complete_user_archive_is_preserved(self):
        self.assertEqual(self.catalog.identity['catalog_shape_count'], 58)
        self.assertEqual(self.catalog.identity['catalog_source_count'], 136)
        self.assertEqual(self.catalog.identity['active_mapping_count'], 18)
        self.assertIsInstance(load_reading_catalog(ROOT), GwansangCatalog)

    def test_one_shape_not_one_card_per_source(self):
        items = [decision('brow_shape:straight')]
        result = self.catalog.apply(build_result(items, self.catalog.repository(Snapshot(self.catalog))))
        self.assertEqual(len(result['passages']), 1)
        p = result['passages'][0]
        self.assertEqual(p['passage_id'], 'GW-S15')
        self.assertEqual(p['reading']['title'], '일자미 · 일자 눈썹')
        self.assertEqual(p['source']['source_id'], 'GS036')
        self.assertEqual(p['source']['archived_source_count'], 2)
        self.assertAlmostEqual(p['G'], .9)
        self.assertEqual(p['score_feature_count'], 1)
        self.assertEqual(result['named_types'], [])

    def test_source_priority_does_not_change_feature_score(self):
        doc = deepcopy(self.catalog.document)
        sources = next(s for s in doc['data']['생김새'] if s['생김새코드'] == 'S15')['출처']
        sources[0]['순위'], sources[1]['순위'] = 2, 1
        revised = GwansangCatalog(doc)
        output = revised.apply(build_result([decision('brow_shape:straight', .82)], revised.repository(Snapshot(revised))))
        self.assertEqual(output['passages'][0]['source']['source_id'], 'GS037')
        self.assertAlmostEqual(output['passages'][0]['G'], .82)

    def test_gender_specific_record_cannot_win_or_enable_a_shape(self):
        doc = deepcopy(self.catalog.document)
        shape = next(s for s in doc['data']['생김새'] if s['생김새코드'] == 'S15')
        shape['출처'][0]['적용성별'] = '여'
        revised = GwansangCatalog(doc)
        self.assertEqual(revised.candidates({'brow_shape:straight'})[0].source['source_id'], 'GS037')
        doc['bindings']['shapes']['S36'].update(enabled=True, required_labels=['cheekbone:prominent'])
        with self.assertRaises(ValueError):
            GwansangCatalog(doc)

    def test_all_conditions_and_held_evidence_are_enforced(self):
        self.assertEqual(self.catalog.candidates({'nose_length:long'}), [])
        self.assertEqual([p.passage_id for p in self.catalog.candidates({'nose_length:long', 'nose_width:wide'})], ['GW-S30'])
        result = build_result([decision('brow_shape:straight', status='HELD')], self.catalog.repository(Snapshot(self.catalog)))
        self.assertEqual(result['passages'], [])

    def test_literal_longer_than_eye_requires_actual_brow_measurement(self):
        labels = {'brow_length:long'}
        self.assertEqual(self.catalog.candidates(labels), [])
        for value, source in ((1.14, 'segmentation'), (1.9, 'landmark'), (float('nan'), 'segmentation')):
            self.assertEqual(self.catalog.candidates(labels, measurements={'brow_length':{
                'value':value, 'source':source, 'unit':'ratio'}}), [])
        candidates = self.catalog.candidates(labels, measurements={'brow_length':{
            'value':1.15, 'source':'segmentation', 'unit':'ratio'}})
        self.assertEqual([p.passage_id for p in candidates], ['GW-S12'])

    def test_old_overclaim_does_not_return_through_new_archive(self):
        all_labels = {label for b in self.catalog.bindings.values() for label in b['required_labels']}
        all_labels.update({'eye_tilt:up','chin_projection:prominent','under_eye:visible'})
        ids = {p.passage_id for p in self.catalog.candidates(all_labels)}
        self.assertNotIn('GW-S20', ids)
        self.assertNotIn('GW-S19', ids)
        self.assertNotIn('GW-S56', ids)
        self.assertNotIn('GW-S28', ids)

    def test_db_text_tamper_is_rejected_even_with_unchanged_hash_column(self):
        db = Snapshot(self.catalog)
        db.document['data']['생김새'][0]['출처'][0]['번역'] = 'changed'
        with self.assertRaises(ValueError):
            self.catalog.repository(db)

    def test_unknown_or_ambiguous_binding_fails(self):
        doc = deepcopy(self.catalog.document)
        del doc['bindings']['shapes']['S01']
        with self.assertRaises(ValueError):
            GwansangCatalog(doc)
        doc = deepcopy(self.catalog.document)
        doc['data']['생김새'][14]['출처'][1]['순위'] = 1
        with self.assertRaises(ValueError):
            GwansangCatalog(doc)


if __name__ == '__main__':
    unittest.main()
