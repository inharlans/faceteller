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
        self.assertEqual(self.catalog.identity['active_mapping_count'], 15)
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
        self.assertEqual(output['passages'][0]['source']['source_id'], 'GS036')
        self.assertAlmostEqual(output['passages'][0]['G'], .82)

    def test_gender_specific_record_cannot_win_or_enable_a_shape(self):
        doc = deepcopy(self.catalog.document)
        shape = next(s for s in doc['data']['생김새'] if s['생김새코드'] == 'S15')
        shape['출처'][0]['적용성별'] = '여'
        with self.assertRaises(ValueError):
            GwansangCatalog(doc)
        doc = deepcopy(self.catalog.document)
        doc['bindings']['shapes']['S36'].update(enabled=True, required_labels=['cheekbone:prominent'])
        with self.assertRaises(ValueError):
            GwansangCatalog(doc)

    def test_all_conditions_and_held_evidence_are_enforced(self):
        self.assertEqual(self.catalog.candidates({'nose_length:long'}), [])
        self.assertEqual([p.passage_id for p in self.catalog.candidates({'nose_length:long', 'nose_width:wide'})], ['GW-S30'])
        result = build_result([decision('brow_shape:straight', status='HELD')], self.catalog.repository(Snapshot(self.catalog)))
        self.assertEqual(result['passages'], [])

    def test_narrow_eyes_alone_do_not_mean_long_eyes(self):
        for labels in ({'eye_roundness:narrow'}, {'eye_size:large'}):
            self.assertNotIn('GW-S18', [p.passage_id for p in self.catalog.candidates(labels)])
        passages = self.catalog.candidates({'eye_roundness:narrow', 'eye_size:large'})
        self.assertEqual([p.passage_id for p in passages], ['GW-S18'])
        self.assertEqual(passages[0].source['source_id'], 'GS047')

    def test_true_aliases_resolve_without_merging_related_types(self):
        for name in ('일자미', '일자 눈썹', '일자로 뻗은 눈썹', ' 일자  눈썹 '):
            self.assertEqual(self.catalog.shape_for_alias(name), 'S15')
        self.assertEqual(self.catalog.shape_for_alias('봉안'), 'S19')
        self.assertEqual(self.catalog.shape_for_alias('와잠'), 'S28')
        for name in ('와잠미', '복코', '절통비', '현담비', '신월미', '만궁구', '산근'):
            self.assertIsNone(self.catalog.shape_for_alias(name), name)

    def test_all_active_sources_are_explicit_and_meaning_matches_that_source(self):
        expected = dict(zip(
            ['S01','S09','S12','S13','S14','S15','S18','S30','S31','S33','S45','S49','S52','S55','S57'],
            ['GS003','GS014','GS026','GS029','GS032','GS036','GS047','GS072','GS075','GS077','GS105','GS115','GS123','GS129','GS134']))
        self.assertEqual({k:b['presentation']['representative_source_id'] for k,b in self.catalog.bindings.items() if b['enabled']}, expected)
        for key, sid in expected.items():
            b = self.catalog.bindings[key]
            result = self.catalog.apply(build_result([decision(label) for label in b['required_labels']],
                self.catalog.repository(Snapshot(self.catalog), measurements={'brow_length':{
                    'value':1.2, 'source':'segmentation', 'unit':'ratio'}})))
            p = next(p for p in result['passages'] if p['passage_id'] == 'GW-'+key)
            self.assertEqual(p['source']['source_id'], sid)
            self.assertEqual(p['source']['reading_basis'], 'reviewed_representative')
            self.assertEqual(p['reading']['meaning_basis'], 'representative')
            self.assertEqual(p['reading']['meaning'], b['presentation']['meaning'])
            self.assertNotEqual(p['reading']['meaning'], self.catalog.shapes[key]['종합풀이'])
        self.assertNotIn('무관', self.catalog.bindings['S15']['presentation']['meaning'])
        self.assertNotIn('높은 지위', self.catalog.bindings['S45']['presentation']['meaning'])
        self.assertNotIn('장수', self.catalog.bindings['S33']['presentation']['meaning'])

    def test_unobserved_whole_body_conditions_disable_readings_only(self):
        for key in ('S02','S03','S04'):
            b = self.catalog.bindings[key]
            items = [decision(label) for label in b['required_labels']]
            result = self.catalog.apply(build_result(items, self.catalog.repository(Snapshot(self.catalog))))
            self.assertNotIn('GW-'+key, [p['passage_id'] for p in result['passages']])
            self.assertEqual(len(result['items']), len(items))
            self.assertTrue(all(i['status'] == 'CONFIRMED' for i in result['items']))

    def test_incomplete_review_or_wrong_source_cannot_be_served(self):
        for field, value in (('representative_source_id','GS075'), ('aliases',['와잠']),
                             ('aliases',['복코']), ('excluded_sources',{}), ('meaning','')):
            doc = deepcopy(self.catalog.document)
            key = 'S31' if value == ['복코'] else 'S15'
            doc['bindings']['shapes'][key]['presentation'][field] = value
            with self.assertRaises(ValueError, msg=f'{key} {field}'):
                GwansangCatalog(doc)

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
