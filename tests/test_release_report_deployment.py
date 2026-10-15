"""CPU checks for the report's deployment schema: single ResNet-18 inference over historical training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from facework.release_thresholds import POLICY
from scripts.release import build_report as report
from scripts.release.build_report import EvidenceError, SPLITS

PARTS = ['eye', 'brow', 'nose', 'mouth', 'face', 'forehead', 'glabella', 'under_eye', 'cheekbone', 'ear', 'philtrum', 'mole']
CANDIDATES = ['resnet18', 'efficientnet_b0', 'convnext_tiny']
EAR_VERSION = 'visible-ear-eye-scale-v2'
EAR_SHA = 'e' * 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    return digest(path)


def metrics(total, support, threshold=.6):
    valid = sum(support)
    return {'total_count': total, 'valid_count': valid, 'missing_count': total - valid, 'class_count': len(support),
            'accuracy': .5, 'macro_f1': .4, 'balanced_accuracy': .5, 'nll': .7, 'ece': .1, 'brier': .3,
            'support': support, 'predicted_support': support,
            'macro_f1_classes': [i for i, n in enumerate(support) if n],
            'threshold': {'value': threshold, 'coverage': .5, 'accuracy': .6, 'accepted_count': valid // 2},
            'ece_bins': [{'lower': 0, 'upper': 1, 'count': valid, 'accuracy': .5, 'mean_confidence': .6}]}


def select_threshold(cal, policy, classes):
    """Deterministic synthetic candidate counts, chosen with the policy's own tie-break."""
    n = cal['n_val']
    base = {'policy_version': policy['version'], 'source_split': 'calibration', 'test_used': False, 'n_valid': n}
    if not cal['calibrated']:
        cal.update(threshold=policy['initial_threshold'],
                   threshold_selection={**base, 'threshold': policy['initial_threshold'], 'status': 'disabled_uncalibrated', 'candidates': []})
    else:
        floor = policy['binary_floor'] if classes == 2 else policy['multiclass_floor']
        candidates = []
        for t in policy['candidates']:
            if t < floor:
                continue
            accepted = max(0, n - round((t - .5) * 40)); wrong = accepted // 5; held = n - accepted
            candidates.append({'threshold': t, 'accepted': accepted, 'withheld': held, 'wrong_confirmed': wrong,
                               'coverage': accepted / n, 'accepted_error_rate': wrong / accepted if accepted else None,
                               'loss': (wrong * policy['wrong_confirmation_cost'] + held * policy['withheld_cost']) / n})
        best = min(candidates, key=lambda c: (c['loss'], -c['accepted'], abs(c['threshold'] - .6), c['threshold']))
        cal.update(threshold=best['threshold'], threshold_selection={**base, 'threshold': best['threshold'],
                   'status': 'selected_on_calibration', 'selected': best, 'candidates': candidates})
    for m in (cal['raw'], cal.get('calibrated_metrics')):
        if m:
            m['threshold']['value'] = cal['threshold']


def weight_hash(arch, part):
    return hashlib.sha256(f'{arch}/{part}'.encode()).hexdigest()


def build(root, legacy=False, hooks=None, heads_per_part=1):
    """Write a consistent synthetic bundle; hooks mutate one document before downstream hashing."""
    hooks = hooks or {}
    hook = lambda name, doc: hooks.get(name, lambda value: None)(doc)
    subjects, n = {}, 0
    for split, size in SPLITS.items():
        for _ in range(size):
            subjects[f'S{n:04d}'] = {'split': split}
            n += 1
    heads = {part: [f'{part}_shape'] + [f'{part}_shape{i}' for i in range(1, heads_per_part)] for part in PARTS}
    features = {key: {'ko': f'{part} 형태', 'part': part, 'kind': 'classification', 'classes': ['small', 'large'],
                      'classes_ko': ['작음', '큼'], 'annotation_provenance': 'automatic morphology'}
                for part in PARTS for key in heads[part]}
    support = {split: {key: {'counts': [size // 2, size - size // 2], 'unknown': 0} for key in features} for split, size in SPLITS.items()}
    registry = {'synthetic': True, 'split_sizes': SPLITS, 'subjects': subjects, 'features': features,
                'parts': {part: {'features': heads[part], 'input_size': [64, 64], 'crops': [part]} for part in PARTS},
                'class_support': support, 'preprocess_version': 'eye-centre-isotropic-v2', 'human_review_count': 0,
                'depth_features': {'nose_depth': {'ko': '코 깊이', 'part': 'nose', 'classes': ['low', 'medium', 'high'],
                                                  'classes_ko': ['낮음', '보통', '높음']}}}
    hook('registry', registry)
    registry_sha = dump(root / 'data/release/training_registry.json', registry)
    config = {'version': 'release-classification-v1', 'registry_sha256': registry_sha, 'architectures': CANDIDATES,
              'preprocess_version': 'eye-centre-isotropic-v2', 'seed': 20261014, 'epochs_max': 4, 'human_review_count': 0,
              'calibration': 'calibration34 one temperature per head after averaging deployed models', 'split_sizes': SPLITS}
    training_sha = dump(root / 'models/release/training_frozen.json', config)
    active = list(CANDIDATES) if legacy else ['resnet18']
    selection_fields = {}
    if not legacy:
        selection = {'version': 'release-inference-choice-v1', 'architectures': ['resnet18'], 'registry_sha256': registry_sha,
                     'training_config_sha256': training_sha, 'selection_basis': 'synthetic user choice: ResNet18 only',
                     'decided_utc': '2026-10-15T00:00:00+00:00'}
        hook('selection', selection)
        selection_fields = {'inference_architectures': ['resnet18'],
                            'inference_selection_sha256': dump(root / 'models/release/inference_frozen.json', selection)}
    for arch in CANDIDATES:
        for part in PARTS:
            meta = {'arch': arch, 'part': part, 'registry_sha256': registry_sha, 'classification_features': heads[part],
                    'preprocess_version': 'eye-centre-isotropic-v2', 'input_size': [64, 64], 'crops': [part],
                    'classes': {key: ['small', 'large'] for key in heads[part]}}
            if part == 'ear':
                meta.update(ear_transform_version=EAR_VERSION, ear_preprocessing_sha256=EAR_SHA)
            model = {'config': config, 'meta': meta, 'selected_epoch': 2, 'elapsed_seconds': 100.0,
                     'history': [{'epoch': e, 'loss': 1.0 / e, 'train_seconds': 30.0, 'validation_seconds': 5.0,
                                  'selection_macro_f1': .4} for e in (1, 2, 3)],
                     'selection': {key: {'metrics': metrics(34, [17, 17])} for key in heads[part]}, 'train_people_with_crops': 270,
                     'train_people_by_feature': {key: 270 for key in heads[part]}, 'train_crops': 540}
            hook(f'model/{arch}/{part}', model)
            dump(root / f'models/release/{arch}/{part}.json', model)
    deployment = {'synthetic': True, 'config': config, 'test_used': False, 'ear_transform_version': EAR_VERSION,
                  'ear_preprocessing_sha256': EAR_SHA, **selection_fields,
                  'weights': {f'models/release/{arch}/{part}.pt': weight_hash(arch, part) for arch in active for part in PARTS},
                  'features': {key: {'calibrated': True, 'status': 'fitted', 'temperature': 1.2, 'test_used_for_fit': False,
                                     'train_support': [135, 135], 'raw': metrics(34, [17, 17]), 'support': [17, 17],
                                     'n_val': 34, 'calibrated_metrics': metrics(34, [17, 17])} for key in features},
                  'calibration_timings': {'total_seconds': 60.0, 'parts': {p: 4.0 for p in PARTS}, 'scope': 'synthetic'}}
    hook('deployment', deployment)
    if not legacy:
        policy = dict(POLICY)
        hook('policy', policy)
        deployment['threshold_policy_sha256'] = dump(root / 'models/release/threshold_policy_frozen.json', policy)
        for key, cal in deployment['features'].items():
            select_threshold(cal, policy, len(features[key]['classes']))
    hook('deployment_final', deployment)
    deployment_sha = dump(root / 'models/release/deployment.json', deployment)
    test = {'synthetic': True, 'deployment_sha256': deployment_sha, 'human_review_count': 0,
            'timings': {'total_seconds': 90.0, 'parts': {p: 6.0 for p in PARTS}, 'scope': 'synthetic'},
            'features': {key: {'participants': 62, 'photo_available': 62, 'annotation_provenance': 'automatic morphology',
                               'raw': metrics(62, [31, 31], deployment['features'][key].get('threshold', .6)),
                               'calibrated': metrics(62, [31, 31], deployment['features'][key].get('threshold', .6))
                               if deployment['features'][key]['calibrated'] else None}
                         for key in features}}
    hook('test', test)
    dump(root / 'outputs/release/test_metrics.json', test)
    depth = {'synthetic': True, 'cohort_n': 400, 'boundary_scope': 'all400', 'records_sha256': 'f' * 64,
             'features': {'nose_depth': {'n': 400, 'quantiles': [.25, .75], 'lower': -1.0, 'upper': 1.0,
                                         'sigma_repeat_train': .2, 'repeat_train_people': 10}},
             'quality': {'min_confidence': .1, 'max_cross_view_error': .2, 'fit_scope': 'synthetic'}}
    depth_sha = dump(root / 'data/release/depth_config.json', depth)
    dump(root / 'data/release/landmark_labels.json', {'features': {}})
    corpus = {'synthetic': True, 'schema_version': 'synthetic', 'literature_coverage': [],
              'sources': [{'literature_id': 'L1', 'title': 'Synthetic', 'title_ko': '합성 문헌', 'source_urls': []}],
              'passages': [{'passage_id': 'P1', 'literature_id': 'L1', 'service_eligible': True, 'exclusion_reasons': [],
                            'annotation_status': 'ai_annotated', 'original_text': '原', 'translation': '번역', 'source': 'L1',
                            'required_labels': ['eye_shape:large']}]}
    dump(root / 'data/release/literature/reference_corpus_v1.json', corpus)
    corpus_sha = report.canonical_sha(corpus)
    items = [{'item_id': key, 'kind': 'classification', 'region': f['part'], 'status': 'CONFIRMED', 'hold_reasons': [],
              'confidence': .8, 'measurement_score': None} for key, f in features.items()]
    items.append({'item_id': 'nose_depth', 'kind': 'depth', 'region': 'nose', 'status': 'HELD',
                  'hold_reasons': ['DEPTH_QUALITY_FAILED'], 'confidence': None, 'measurement_score': None})
    timings = {stage: .1 for stage in report.STAGES} | {'received_to_result_seconds': 1.0}
    rows = [{'subject_id': sid, 'split': s['split'], 'seconds': 2.0, 'http_status': 200, 'status': 'PARTIAL',
             'progress_event_seconds': [.5, 1.0], 'score_formula_verified': True, 'score_formula_verified_count': 0,
             'score_components_verified_count': 0, 'selection_verified': True, 'deployment_sha256': deployment_sha,
             'depth_config_sha256': depth_sha, 'corpus_sha256': corpus_sha, 'items': items, 'features': len(items),
             'confirmed': len(items) - 1, 'held': 1, 'timings': timings, 'passages': [], 'passage_count': 0,
             'db_source_verified': True, 'db_verified_passages': 0, 'db_candidates': 0} for sid, s in subjects.items()]
    identity = {'ready': True, 'deployment_sha256': deployment_sha, 'depth_config_sha256': depth_sha, 'registry_sha256': registry_sha,
                'score_mode': 'include_measurement_scores', 'repository': 'postgresql_readonly', 'eligible_passages': 1,
                'corpus_sha256': corpus_sha, 'calibrated_head_count': sum(c['calibrated'] for c in deployment['features'].values()),
                'uncalibrated_heads': {k: {'calibrated': False} for k, c in deployment['features'].items() if not c['calibrated']}}
    if not legacy:
        identity.update(classifier_architectures=['resnet18'], classifier_model_count=len(PARTS))
    service = {'synthetic': True, 'schema_version': 'release-service-verification-v1', 'complete': True, 'all400': True,
               'participants': rows, 'service': identity, 'warmup': [], 'errors_tested': [], 'error_checks_passed': False,
               'score_formula_verified': True, 'score_formula_verified_count': 0, 'score_components_verified_count': 0,
               'selection_verified_participants': 400, 'corpus_sha256': corpus_sha}
    hook('service', service)
    dump(root / 'outputs/release/service-verification.json', service)
    non_ear = {f'models/release/{arch}/{part}.pt': weight_hash(arch, part) for arch in CANDIDATES for part in PARTS if part != 'ear'}
    ear = {'pass': True, 'unchanged_non_ear_models': 33, 'retrained_ear_models': 3, 'ear_transform_version': EAR_VERSION,
           'ear_preprocessing_sha256': EAR_SHA, 'deployment_sha256': 'a' * 64, 'before_non_ear_weights': non_ear,
           'after_non_ear_weights': dict(non_ear), 'baseline_ear_training_seconds': 480.0, 'baseline_calibration_seconds': 44.0,
           'baseline_test_seconds': 347.0, 'correction_wall_seconds': 1347.0}
    hook('ear', ear)
    dump(root / 'outputs/release/ear-input-correction/verification.json', ear)


def write_supplemental(root, physical=False):
    """Identity-bound TLS/camera and DB metadata evidence for an already built fixture."""
    from scripts.release.sync_label_metadata import load_inputs
    deployment = json.loads((root / 'models/release/deployment.json').read_text(encoding='utf-8'))
    corpus = json.loads((root / 'data/release/literature/reference_corpus_v1.json').read_text(encoding='utf-8'))
    shas = {k: digest(root / p) for k, p in (('deployment', 'models/release/deployment.json'),
            ('registry', 'data/release/training_registry.json'), ('depth', 'data/release/depth_config.json'))}
    camera = {'pass': True, 'physical_camera': physical, 'recorded_source': not physical, 'detector_mocked': False,
              'actual_models': True, 'photos_saved': False, 'participants': 1, 'split': 'train', 'capture_posts': 1,
              'total_posts': 2, 'tracks_stopped': True, 'equivalent_item_count': 13, 'max_score_difference': 0,
              'external_requests': 0, 'page_errors': [], 'deployment_sha256': shas['deployment'], 'quality_version': 'q-v3'}
    tls = {'synthetic': True, 'pass': True, 'actual_models': True, 'url': 'https://127.0.0.1:8768/',
           'certificate_verified_by_scoped_ca': True, 'browser_certificate_spki_pinned': True, 'global_trust_modified': False,
           'self_signed_development': True, 'physical_camera_tested': physical, 'camera': camera,
           'health': {'ready': True, 'deployment_sha256': shas['deployment'], 'registry_sha256': shas['registry'],
                      'depth_config_sha256': shas['depth'], 'corpus_sha256': report.canonical_sha(corpus),
                      'classifier_architectures': deployment.get('inference_architectures', CANDIDATES),
                      'classifier_model_count': len(deployment['weights']), 'quality_version': 'q-v3'}}
    dump(root / 'outputs/release/tls-camera-verification.json', tls)
    dump(root / 'outputs/release/frontend-camera-verification.json', camera)
    rows, revision, inputs = load_inputs({'training_registry': root / 'data/release/training_registry.json',
                                          'depth_config': root / 'data/release/depth_config.json',
                                          'landmark_labels': root / 'data/release/landmark_labels.json'})
    eligible = sum(p['service_eligible'] for p in corpus['passages'])
    dump(root / 'outputs/release/database-label-metadata-check.json', {'synthetic': True, 'metadata_revision': revision,
         'matches': True, 'differing_labels': [], 'eligible_without_metadata': []})
    dump(root / 'outputs/release/database-label-metadata-sync.json', {'synthetic': True, 'metadata_revision': revision,
         'registry_labels': len(rows), 'inserted': len(rows), 'updated': 0, 'unchanged': 0, 'stale_cleared': 0,
         'unmapped_db_labels': [], 'inputs_sha256': inputs})
    dump(root / 'outputs/release/database-final-status.json', {'synthetic': True, 'corpus_sha256': report.canonical_sha(corpus),
         'passages': len(corpus['passages']), 'eligible': eligible, 'labels': len(rows), 'labels_with_metadata': len(rows)})


class DeploymentSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def load(self, **kwargs):
        build(self.root, **kwargs)
        return report.load_bundle(self.root, synthetic=True)

    def rejects(self, pattern, **kwargs):
        with self.assertRaisesRegex(EvidenceError, pattern):
            self.load(**kwargs)

    def test_single_resnet18_twelve_models_over_historical_three_candidates(self):
        b = self.load()
        self.assertEqual(b['architectures'], ['resnet18'])
        self.assertEqual(b['candidates'], CANDIDATES)
        self.assertEqual(len(b['deployment']['weights']), 12)
        self.assertEqual({m['meta']['arch'] for m in b['training']}, {'resnet18'})
        self.assertEqual(len(b['training']), 12)
        self.assertEqual(len(b['historical_training']), 24)
        self.assertEqual(b['ear_correction_scope'], {'history_non_ear': 33, 'history_ear': 3, 'active_non_ear': 11, 'active_ear': 1})
        self.assertIn('inference_selection', b['hashes'])
        self.assertIn('history/convnext_tiny/eye', b['hashes'])
        tables = report.make_tables(b)
        report.privacy_check(b, tables)
        self.assertEqual(len(tables['모델학습시간'][1]), 12)
        self.assertEqual(len(tables['역사적후보학습시간'][1]), 24)
        self.assertEqual(len(tables['부위요약'][1]), 12)
        selection = dict(tables['추론선택'][1])
        self.assertEqual(selection['현재 배포 가중치 수'], 12)
        self.assertEqual(selection['원 학습 후보 모델 수(역사적)'], 36)
        self.assertIn('앙상블 아님', report.brief(b)[1])
        self.assertIn('현재 배포는 그중 비귀 11모델', report.ear_correction_note(b))

    def test_legacy_thirty_six_model_history_still_supported(self):
        b = self.load(legacy=True)
        self.assertEqual(b['architectures'], CANDIDATES)
        self.assertIsNone(b['inference_selection'])
        self.assertEqual((len(b['training']), len(b['historical_training'])), (36, 0))
        self.assertEqual(b['ear_correction_scope']['active_non_ear'], 33)
        tables = report.make_tables(b)
        self.assertNotIn('추론선택', tables)
        self.assertNotIn('역사적후보학습시간', tables)
        self.assertNotIn('현재 배포는 그중', report.ear_correction_note(b))

    def test_selection_identity_is_strict(self):
        cases = [
            ('ResNet-18 단일', {'selection': lambda s: s.update(architectures=['resnet18', 'efficientnet_b0'])}),
            ('schema', {'selection': lambda s: s.update(version='release-inference-choice-v0')}),
            ('registry', {'selection': lambda s: s.update(registry_sha256='0' * 64)}),
            ('학습 설정 파일', {'selection': lambda s: s.update(training_config_sha256='0' * 64)}),
            ('선택 근거', {'selection': lambda s: s.update(selection_basis='')}),
            ('추론 선택 파일 SHA256', {'deployment': lambda d: d.update(inference_selection_sha256='0' * 64)}),
            ('ResNet-18 단일', {'deployment': lambda d: d.update(inference_architectures=['efficientnet_b0'])}),
            ('일부 누락', {'deployment': lambda d: d.pop('inference_selection_sha256')}),
            ('weights', {'deployment': lambda d: d['weights'].update({'models/release/efficientnet_b0/eye.pt': 'b' * 64})}),
            ('SHA256 형식', {'deployment': lambda d: d['weights'].update({'models/release/resnet18/eye.pt': 'X' * 64})}),
            ('분류 모델 계열', {'service': lambda s: s['service'].update(classifier_model_count=36)}),
            ('분류 모델 계열', {'service': lambda s: s['service'].pop('classifier_architectures')}),
        ]
        for pattern, hooks in cases:
            with self.subTest(pattern=pattern, hooks=list(hooks)):
                self.rejects(pattern, hooks=hooks)

    def test_training_config_must_stay_historical(self):
        build(self.root)
        path = self.root / 'models/release/training_frozen.json'
        changed = json.loads(path.read_text(encoding='utf-8'))
        changed['architectures'] = ['resnet18']
        dump(path, changed)
        with self.assertRaisesRegex(EvidenceError, '학습 설정 파일'):
            report.load_bundle(self.root, synthetic=True)

    def test_selection_file_without_manifest_fields_is_stale(self):
        build(self.root, legacy=True)
        dump(self.root / 'models/release/inference_frozen.json', {'version': 'release-inference-choice-v1'})
        with self.assertRaisesRegex(EvidenceError, '재보정 필요'):
            report.load_bundle(self.root, synthetic=True)

    def test_historical_candidate_records_are_required(self):
        build(self.root)
        (self.root / 'models/release/convnext_tiny/eye.json').unlink()
        with self.assertRaisesRegex(EvidenceError, '필수 자료 없음'):
            report.load_bundle(self.root, synthetic=True)

    def test_ear_history_subset_must_match(self):
        cases = [
            ('비귀 모델 해시', {'deployment': lambda d: d['weights'].update({'models/release/resnet18/eye.pt': 'c' * 64})}),
            ('33모델', {'ear': lambda e: e['after_non_ear_weights'].pop('models/release/convnext_tiny/eye.pt')}),
            ('33모델', {'ear': lambda e: e['before_non_ear_weights'].update({'models/release/convnext_tiny/eye.pt': 'c' * 64})}),
            ('33\\+3', {'ear': lambda e: e.update(unchanged_non_ear_models=11)}),
            ('전처리 SHA256', {'ear': lambda e: e.update(ear_preprocessing_sha256='0' * 64)}),
            ('귀 입력 전처리', {'model/resnet18/ear': lambda m: m['meta'].update(ear_transform_version='old')}),
        ]
        for pattern, hooks in cases:
            with self.subTest(pattern=pattern):
                self.rejects(pattern, hooks=hooks)

    def test_single_class_calibration_guard_and_no_human_review(self):
        def single(d, calibrated):
            cal = d['features']['eye_shape']
            cal.update(raw=metrics(34, [34, 0]), support=[34, 0], n_val=34)
            if calibrated:
                cal.update(calibrated_metrics=metrics(34, [34, 0]))
            else:
                cal.update(calibrated=False, status='single_class_calibration_support', temperature=None, calibrated_metrics=None)
        b = self.load(hooks={'deployment': lambda d: single(d, False)})
        self.assertEqual(b['service']['service']['calibrated_head_count'], 11)
        self.assertEqual(b['test']['features']['eye_shape']['calibrated'], None)
        self.assertEqual([row[2] for row in report.make_tables(b)['온도보정'][1] if row[0] == 'eye_shape'], [False])
        self.rejects('관측 클래스 1개 이하', hooks={'deployment': lambda d: single(d, True)})
        self.rejects('단일 클래스 보정 보류', hooks={'deployment': lambda d: (single(d, False), d['features']['eye_shape'].update(temperature=1.0))})
        self.rejects('보정/미보정 헤드', hooks={'deployment': lambda d: single(d, False),
                                           'service': lambda s: s['service'].update(calibrated_head_count=12, uncalibrated_heads={})})
        self.rejects('human_review_count', hooks={'test': lambda t: t.update(human_review_count=1)})

    def test_per_head_thresholds_bound_to_frozen_calibration_policy(self):
        b = self.load()
        self.assertEqual(b['threshold_policy'], POLICY)
        self.assertEqual({c['threshold'] for c in b['deployment']['features'].values()}, {.65})
        tables = report.make_tables(b)
        self.assertEqual(len(tables['헤드기준값'][1]), 12)
        self.assertEqual({row[5] for row in tables['헤드기준값'][1]}, {.65})
        self.assertIn('threshold_policy_frozen.json SHA256', dict(tables['기준값정책'][1]))
        def head(update):
            return {'deployment_final': lambda d: update(d['features']['eye_shape'])}
        def other_candidate(cal):
            pick = cal['threshold_selection']['candidates'][0]
            cal['threshold'] = cal['threshold_selection']['threshold'] = pick['threshold']
            cal['threshold_selection']['selected'] = pick
            for m in (cal['raw'], cal['calibrated_metrics']):
                m['threshold']['value'] = pick['threshold']
        cases = [
            ('정책 파일 SHA256', {'deployment_final': lambda d: d.update(threshold_policy_sha256='0' * 64)}),
            ('설계 하한', {'policy': lambda p: p.update(binary_floor=.55)}),
            ('보정군 전용', {'policy': lambda p: p.update(test_used=True)}),
            ('재계산한 기준값', head(other_candidate)),
            ('loss/coverage', head(lambda c: c['threshold_selection']['candidates'][1].update(loss=0.0))),
            ('선택 기록 불일치', head(lambda c: c.update(threshold=.9))),
            ('출처', head(lambda c: c['threshold_selection'].update(test_used=True))),
            ('정책 격자', head(lambda c: c['threshold_selection']['candidates'].pop(0))),
            ('선택 기준값을 쓰지 않았습니다', {'test': lambda t: t['features']['eye_shape']['raw']['threshold'].update(value=.6)}),
            ('미선택 기준값', {'deployment': lambda d: d['features']['eye_shape'].update(
                calibrated=False, status='fitted_but_disabled', temperature=None, calibrated_metrics=None),
                'deployment_final': lambda d: (d['features']['eye_shape']['threshold_selection'].update(threshold=.7),
                                               d['features']['eye_shape'].update(threshold=.7),
                                               d['features']['eye_shape']['raw']['threshold'].update(value=.7))}),
        ]
        for pattern, hooks in cases:
            with self.subTest(pattern=pattern):
                self.rejects(pattern, hooks=hooks)

    def test_threshold_policy_file_without_manifest_hash_is_stale(self):
        build(self.root, legacy=True)
        dump(self.root / 'models/release/threshold_policy_frozen.json', POLICY)
        with self.assertRaisesRegex(EvidenceError, '기준값 정책 해시'):
            report.load_bundle(self.root, synthetic=True)

    def test_supplemental_tls_camera_and_db_metadata_are_identity_bound(self):
        build(self.root)
        write_supplemental(self.root)
        b = report.load_bundle(self.root, synthetic=True)
        tables = report.make_tables(b)
        self.assertIn('녹화 영상 입력(물리 웹캠 아님)', [row[2] for row in tables['검증근거'][1] if row[0] == 'HTTPS·카메라 경로'][0])
        self.assertEqual(dict(tables['TLS카메라검증'][1])['물리 웹캠 시험'], False)
        self.assertEqual(dict(tables['DB설계메타데이터'][1])['레이블(메타데이터 완비)'], 27)
        self.assertIn('tls_camera', b['hashes'])
        def tamper(name, update):
            path = self.root / 'outputs/release' / name
            doc = json.loads(path.read_text(encoding='utf-8')); update(doc); dump(path, doc)
        cases = [
            ('신원', 'tls-camera-verification.json', lambda d: d['health'].update(deployment_sha256='0' * 64)),
            ('신원', 'tls-camera-verification.json', lambda d: d['health'].update(classifier_model_count=36)),
            ('전역 신뢰', 'tls-camera-verification.json', lambda d: d.update(global_trust_modified=True)),
            ('입력 출처', 'tls-camera-verification.json', lambda d: d.update(physical_camera_tested=True)),
            ('카메라 경로', 'tls-camera-verification.json', lambda d: d['camera'].update(detector_mocked=True)),
            ('TLS 카메라 기록이 다릅니다', 'frontend-camera-verification.json', lambda d: d.update(max_score_difference=.1)),
            ('동결 정의', 'database-label-metadata-check.json', lambda d: d.update(metadata_revision='0' * 64)),
            ('동결 정의', 'database-label-metadata-check.json', lambda d: d.update(matches=False)),
            ('입력/범위', 'database-label-metadata-sync.json', lambda d: d['inputs_sha256'].update(depth_config='0' * 64)),
            ('DB 최종 상태', 'database-final-status.json', lambda d: d.update(eligible=0)),
        ]
        for pattern, name, update in cases:
            with self.subTest(pattern=pattern, name=name):
                write_supplemental(self.root)
                tamper(name, update)
                with self.assertRaisesRegex(EvidenceError, pattern):
                    report.load_bundle(self.root, synthetic=True)

    def test_resumed_http_run_is_disclosed_and_recovery_must_preserve_prefix(self):
        resumed = {'service': lambda s: s.update(resume_events=[{'completed_before_resume': 397, 'reason': 'continue preserved'}],
                                                 errors_tested=[{'case': 'bad', 'passed': True}, {'case': 'bad', 'passed': True}],
                                                 error_checks_passed=True)}
        recovery = {'preserved_committed': 396, 'recovered_actual_response_count': 397, 'prefix_identical': True, 'reason': 'replace lock'}
        build(self.root, hooks=resumed)
        dump(self.root / 'outputs/release/windows-report-write-recovery/recovery.json', recovery)
        b = report.load_bundle(self.root, synthetic=True)
        tables = report.make_tables(b)
        self.assertEqual([row[2] for row in tables['응답기록재개'][1]], [397, 396])
        self.assertIn('2회 / 1종 / 통과 2회', [row[2] for row in tables['검증근거'][1] if row[0] == '오류 경로'][0])
        dump(self.root / 'outputs/release/windows-report-write-recovery/recovery.json', {**recovery, 'prefix_identical': False})
        with self.assertRaisesRegex(EvidenceError, '기존 응답 보존'):
            report.load_bundle(self.root, synthetic=True)
        build(self.root)  # recovery without a resume record
        dump(self.root / 'outputs/release/windows-report-write-recovery/recovery.json', recovery)
        with self.assertRaisesRegex(EvidenceError, '재개 기록이 없습니다'):
            report.load_bundle(self.root, synthetic=True)
        (self.root / 'outputs/release/windows-report-write-recovery/recovery.json').unlink()
        self.rejects('재개 기록', hooks={'service': lambda s: s.update(resume_events=[{'completed_before_resume': 400, 'reason': 'x'}])})

    def test_direct_cli_works_from_another_working_directory(self):
        import subprocess, sys
        build(self.root)
        write_supplemental(self.root)  # exercises the lazy metadata-helper import
        script = Path(report.__file__).resolve()
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(script), '--root', str(self.root), '--synthetic',
                                 '--no-copy', '--validate-only', '--output-dir', str(self.root / 'out')],
                                cwd=self.root, capture_output=True, text=True, encoding='utf-8', timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_synthetic_report_renders_when_fonts_available(self):
        fonts = Path('C:/Windows/Fonts') if Path('C:/Windows/Fonts/malgun.ttf').is_file() else Path('/mnt/c/Windows/Fonts')
        if not (fonts / 'malgun.ttf').is_file():
            self.skipTest('Malgun font unavailable')
        try:
            import openpyxl, reportlab  # noqa: F401
        except ImportError:
            self.skipTest('report dependencies unavailable')
        build(self.root, heads_per_part=4, hooks={'service': lambda s: s.update(  # 48 heads + resume: worst page load
            resume_events=[{'completed_before_resume': 397, 'reason': 'continue preserved'}])})
        write_supplemental(self.root)
        output = self.root / 'out'
        code = report.main(['--root', str(self.root), '--output-dir', str(output), '--synthetic', '--no-copy', '--font-dir', str(fonts)])
        self.assertEqual(code, 0)
        text = (output / f'{report.TITLE}.txt').read_text(encoding='utf-8')
        self.assertIn('[추론선택]', text)
        self.assertIn('[역사적후보학습시간]', text)
        self.assertIn('[헤드기준값]', text)
        self.assertIn('[TLS카메라검증]', text)
        self.assertIn('[DB설계메타데이터]', text)
        self.assertIn('합성 TEST ONLY', text)


if __name__ == '__main__':
    unittest.main()
