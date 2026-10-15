"""Synthetic CPU tests for the reviewed (editorial/semantic + overlay QA) service verifier.

No service, GPU, photograph or database is used. Fixtures are generated text.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import unittest

from facework.pilot_contract import Decision, Passage, build_result
from facework.release_readings import ReadingCatalog
from scripts.release.verify_service import OVERLAY_ITEMS, summarize_result, overlay_summary

REVISION = 'f' * 64
ITEMS = [('eye', .95), ('nose', .9), ('mouth', .85), ('face', .8), ('chin', .75)]
# (passage id, required item, literature order); x_eye ties 'a' on G and wins the tie, then is excluded.
PASSAGES = [('x_eye', 'eye', 0), ('a', 'eye', 1), ('b', 'nose', 2), ('c', 'mouth', 3), ('d', 'face', 4), ('e', 'chin', 5)]


def text(pid): return f'原文{pid}，古籍。'


class Repository:
    def __init__(self, passages): self.passages = passages; self.revision = REVISION
    def corpus_revision(self): return self.revision
    @contextmanager
    def snapshot(self): yield self
    def load_candidates(self, labels):
        # Fresh objects per query, like PostgreSQL rows; AND semantics.
        return [copy.deepcopy(p) for p in self.passages if p.required_labels <= labels]


def decision(key, value):
    return Decision(key, key, 'CONFIRMED', key + ':a', 'A', value, None, None, 'synthetic', (), 'classification')


def build(excluded=('x_eye',)):
    passages = [Passage(pid, frozenset({item + ':a'}), text(pid), 'db translation', 'db commentary',
                        {'title': 'Synthetic', 'url': 'https://example.invalid/' + pid}, 1, order)
                for pid, item, order in PASSAGES]
    corpus = {'passages': [{'passage_id': p.passage_id, 'original_text': p.original_text,
                            'required_labels': sorted(p.required_labels), 'service_eligible': True} for p in passages]}
    readings = {pid: {'title': 'T' + pid, 'translation': '직역 ' + pid, 'meaning': '풀이 ' + pid, 'valence': 'neutral',
                      'valence_label': '중립', 'topics': ['other']} for pid, _, _ in PASSAGES}
    named = [{'id': 'nt-b', 'name': '유형B', 'required_labels': ['nose:a'], 'source_passage_ids': ['b'], 'original_text': text('b'),
              'meaning': '옛 글의 이름', 'valence': 'neutral', 'valence_label': '중립'},
             {'id': 'nt-x', 'name': '유형X', 'required_labels': ['eye:a'], 'source_passage_ids': ['x_eye'], 'original_text': text('x_eye'),
              'meaning': '제외된 출처', 'valence': 'neutral', 'valence_label': '중립'}]
    document = {'version': 'readings-test', 'authoring': 'ai_edited_unreviewed', 'readings': readings, 'named_types': named}
    policy = {'version': 'semantic-test', 'excluded_passages': [
        {'passage_id': pid, 'original_sha256': hashlib.sha256(text(pid).encode()).hexdigest(), 'reason': 'synthetic exclusion'} for pid in excluded]}
    catalog = ReadingCatalog(document, corpus, policy)
    catalog.identity = {'readings_sha256': '1' * 64, 'semantic_review_sha256': '2' * 64,
                        'readings_version': document['version'], 'semantic_review_version': policy['version']}
    return catalog, Repository(passages)


def overlay(missing=()):
    features = []
    for key in OVERLAY_ITEMS:
        if key in missing: continue
        source = 'segmentation' if key in ('brow_length', 'brow_tail') else 'landmark'
        features.append({'item_id': key, 'label': key, 'source': source, 'paths': [[[10.5, 20.0], [30.25, 40.0]], [[0, 0], [100, 80]]],
                         'value': 0.5, 'unit': 'ratio'})
    return {'width': 100, 'height': 80, 'features': features}


def event(catalog, repo, *, filtered=True, overlay_value=None):
    decisions = [decision(k, v) for k, v in ITEMS]
    data = build_result(decisions, catalog.repository(repo) if filtered else repo)
    data = catalog.apply(data)
    data['runtime'] = {'corpus_sha256': repo.revision, **catalog.identity,
                       'eligible_passages': len(PASSAGES), 'active_mapping_count': len(PASSAGES) - len(catalog.excluded)}
    data['face_overlay'] = overlay() if overlay_value is None else overlay_value
    return {'type': 'result', 'data': data}


class SemanticFilterBeforeTopFive(unittest.TestCase):
    def test_exclusion_is_replaced_by_next_candidate(self):
        catalog, repo = build(); ev = event(catalog, repo)
        self.assertEqual([p['passage_id'] for p in ev['data']['passages']], ['a', 'b', 'c', 'd', 'e'])
        summary = summarize_result(ev, repo, catalog)
        self.assertTrue(summary['selection_verified']); self.assertTrue(summary['editorial_verified'])
        self.assertEqual(summary['semantic_excluded_candidates'], 1); self.assertEqual(summary['passage_count'], 5)
        self.assertEqual(summary['named_type_ids'], ['nt-b'])  # excluded source suppresses nt-x
        self.assertTrue(summary['overlay_verified'])

    def test_display_only_filter_after_cap_is_rejected(self):
        catalog, repo = build()
        decisions = [decision(k, v) for k, v in ITEMS]
        data = build_result(decisions, repo)  # unfiltered ranking selects x_eye and drops e
        self.assertEqual([p['passage_id'] for p in data['passages']], ['x_eye', 'a', 'b', 'c', 'd'])
        data['passages'] = [p for p in data['passages'] if p['passage_id'] != 'x_eye']
        for rank, p in enumerate(data['passages'], 1): p['rank'] = rank
        data = catalog.apply(data)
        data['runtime'] = {'corpus_sha256': REVISION, **catalog.identity}; data['face_overlay'] = overlay()
        with self.assertRaisesRegex(RuntimeError, 'selected IDs'): summarize_result({'type': 'result', 'data': data}, repo, catalog)

    def test_excluded_passage_returned_is_rejected(self):
        catalog, repo = build(); ev = event(catalog, repo, filtered=False)
        with self.assertRaisesRegex(RuntimeError, 'Selection verification|Editorial verification'): summarize_result(ev, repo, catalog)

    def test_wrong_active_mapping_count_rejected(self):
        catalog, repo = build(); ev = event(catalog, repo); ev['data']['runtime']['active_mapping_count'] = len(PASSAGES)
        with self.assertRaisesRegex(RuntimeError, 'active mapping'): summarize_result(ev, repo, catalog)


class EditorialIdentityAndText(unittest.TestCase):
    def setUp(self):
        self.catalog, self.repo = build(); self.event = event(self.catalog, self.repo)
        summarize_result(self.event, self.repo, self.catalog)

    def reject(self, mutate, pattern='Editorial verification'):
        bad = copy.deepcopy(self.event); mutate(bad['data'])
        with self.assertRaisesRegex(RuntimeError, pattern): summarize_result(bad, self.repo, self.catalog)

    def test_identity_mismatch(self):
        for key in ('readings_sha256', 'semantic_review_sha256', 'readings_version', 'semantic_review_version'):
            with self.subTest(key=key): self.reject(lambda d, k=key: d['runtime'].__setitem__(k, 'other'))
        self.reject(lambda d: d['runtime'].pop('readings_sha256'))
        self.reject(lambda d: d['ranking_policy'].__setitem__('semantic_review_version', 'other'))

    def test_text_and_reading_mismatch(self):
        p = lambda d: d['passages'][1]
        self.reject(lambda d: p(d).__setitem__('translation', 'db translation'))
        self.reject(lambda d: p(d).__setitem__('commentary', 'changed'))
        self.reject(lambda d: p(d)['reading'].__setitem__('valence', 'favorable'))
        self.reject(lambda d: p(d).__setitem__('original_text', 'changed'))
        self.reject(lambda d: p(d)['source'].__setitem__('url', 'https://elsewhere.invalid'))
        self.reject(lambda d: p(d)['source'].__setitem__('editorial_version', 'other'))
        self.reject(lambda d: p(d)['source'].pop('editorial_status'))

    def test_named_type_mismatch(self):
        self.reject(lambda d: d['named_types'][0].__setitem__('meaning', 'changed'))
        self.reject(lambda d: d.__setitem__('named_types', []))
        self.reject(lambda d: d['named_types'].append(copy.deepcopy(d['named_types'][0])))

    def test_database_text_must_match_reading_bound_archive(self):
        catalog, repo = build()
        repo.passages = [p if p.passage_id != 'b' else Passage(p.passage_id, p.required_labels, 'changed in DB', p.translation,
                                                               p.commentary, p.source, p.literature_priority, p.passage_order)
                         for p in repo.passages]
        ev = event(catalog, repo)
        with self.assertRaisesRegex(RuntimeError, 'archived corpus'): summarize_result(ev, repo, catalog)


class OverlayQA(unittest.TestCase):
    def test_scalar_summary_and_missing_mask(self):
        summary = overlay_summary(overlay())
        self.assertEqual(summary['overlay_feature_count'], 19); self.assertEqual(summary['overlay_sources'], {'landmark': 17, 'segmentation': 2})
        self.assertEqual(summary['overlay_point_count'], 19 * 4)
        self.assertNotIn('accuracy', summary['overlay_meaning'].replace('not anatomical or human-verified accuracy', ''))
        missing = overlay_summary(overlay(missing=('brow_length', 'brow_tail')))
        self.assertTrue(missing['overlay_verified']); self.assertEqual(missing['overlay_feature_count'], 17)
        self.assertEqual(missing['overlay_missing_items'], ['brow_length', 'brow_tail'])
        # scalar-only: no coordinate arrays anywhere in the summary
        self.assertFalse(any(isinstance(v, list) and v and isinstance(v[0], (list, int, float)) for v in summary.values()))
        empty = overlay_summary({'width': 100, 'height': 80, 'features': []})
        self.assertFalse(empty['overlay_verified']); self.assertEqual(empty['overlay_reason'], 'OVERLAY_EMPTY')

    def test_bad_overlays_rejected(self):
        def bad(mutate):
            value = overlay(); mutate(value); return value
        cases = {
            'nan': bad(lambda o: o['features'][0]['paths'][0][0].__setitem__(0, float('nan'))),
            'inf': bad(lambda o: o['features'][0]['paths'][0][0].__setitem__(1, math.inf)),
            'outside_x': bad(lambda o: o['features'][1]['paths'][0][1].__setitem__(0, 100.5)),
            'negative_y': bad(lambda o: o['features'][1]['paths'][0][1].__setitem__(1, -0.1)),
            'string_coordinate': bad(lambda o: o['features'][2]['paths'][0][0].__setitem__(0, '10')),
            'three_values': bad(lambda o: o['features'][2]['paths'][0].__setitem__(0, [1, 2, 3])),
            'single_point': bad(lambda o: o['features'][2]['paths'].__setitem__(0, [[1, 2]])),
            'segmentation_non_brow': bad(lambda o: o['features'][0].__setitem__('source', 'segmentation')),
            'unknown_source': bad(lambda o: o['features'][0].__setitem__('source', 'cnn_attention')),
            'non_brow_missing': bad(lambda o: o['features'].pop(0)),
            'duplicate': bad(lambda o: o['features'].append(copy.deepcopy(o['features'][0]))),
            'bad_size': bad(lambda o: o.__setitem__('width', 0)),
            'nan_value': bad(lambda o: o['features'][0].__setitem__('value', float('nan'))),
        }
        for name, value in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(RuntimeError, 'Overlay verification'): overlay_summary(value)

    def test_bad_overlay_fails_full_result(self):
        catalog, repo = build(); bad = overlay(); bad['features'][3]['paths'][0][0] = [500, 5]
        with self.assertRaisesRegex(RuntimeError, 'Overlay verification'): summarize_result(event(catalog, repo, overlay_value=bad), repo, catalog)

    def test_current_result_cannot_silently_omit_overlay(self):
        catalog, repo=build();ev=event(catalog,repo)
        del ev['data']['face_overlay']
        with self.assertRaisesRegex(RuntimeError,'omitted face_overlay'):
            summarize_result(ev,repo,catalog)


class LegacyBehaviour(unittest.TestCase):
    def legacy(self):
        _, repo = build(excluded=())
        decisions = [decision(k, v) for k, v in ITEMS]
        data = build_result(decisions, repo); data['runtime'] = {'corpus_sha256': repo.revision}
        return {'type': 'result', 'data': data}, repo

    def test_no_readings_legacy_direct_db_comparison(self):
        ev, repo = self.legacy(); summary = summarize_result(ev, repo)
        self.assertTrue(summary['db_source_verified']); self.assertNotIn('editorial_verified', summary); self.assertNotIn('overlay_verified', summary)
        bad = copy.deepcopy(ev); bad['data']['passages'][0]['translation'] = 'changed'
        with self.assertRaisesRegex(RuntimeError, 'differs from PostgreSQL'): summarize_result(bad, repo)

    def test_editorial_response_without_catalog_is_refused(self):
        catalog, repo = build(); ev = event(catalog, repo)
        with self.assertRaisesRegex(RuntimeError, 'no pinned reading catalog'): summarize_result(ev, repo)

    def test_error_event_shape(self):
        catalog, repo = build()
        out = summarize_result({'type': 'error', 'code': 'NO_FACE'}, repo, catalog)
        self.assertEqual(out['status'], 'error'); self.assertFalse(out['editorial_verified'])
        self.assertNotIn('editorial_verified', summarize_result({'type': 'error', 'code': 'NO_FACE'}, repo))

    def test_summary_is_json_scalar_safe(self):
        catalog, repo = build(); summary = summarize_result(event(catalog, repo), repo, catalog)
        encoded = json.dumps(summary, allow_nan=False)
        self.assertNotIn(text('a'), encoded); self.assertNotIn('10.5', encoded)  # no original text or coordinates saved


if __name__ == '__main__':
    unittest.main()
