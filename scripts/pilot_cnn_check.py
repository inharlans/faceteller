"""CPU-only, local, aggregate-only audit of the existing FaceTeller CNNs.

Run with .venv/Scripts/python.exe scripts/pilot_cnn_check.py.
Does not train, change service code, persist images, or tune on test subjects.
"""
import os
import sys

# Set these before importing any numeric / model library. Never query a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import csv
import hashlib
import io
import json
import math
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def deny_network(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.sendto', 'socket.sendmsg',
                 'subprocess.Popen', 'os.system'):
        raise RuntimeError('Pilot prohibits network connections and child processes')


sys.addaudithook(deny_network)
import numpy as np
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.set_default_device('cpu')


def deny_cuda(*args, **kwargs):
    raise RuntimeError('GPU use prohibited for this pilot')


torch.cuda._lazy_init = deny_cuda
import cv2
import mediapipe as mp
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
from facework import analysis_v2 as a
from facework.kface import Store, member
from facework.kface_ensemble import checkpoints

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)
OUT = ROOT / 'outputs/pilot-20261014'
REF = ROOT / 'data/derived/kface/reference_labels.csv'
MODEL = ROOT / 'models/kface_v4'
SCHEMA = a.SCHEMA['features']


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def rows(path, delimiter=','):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def apply_temperature(probabilities, temperature):
    """Exact normalized p**(1/T), accepting one vector or a batch. CPU numpy only."""
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid probabilities or temperature')
    if p.ndim not in (1, 2) or (p.sum(axis=-1) <= 0).any():
        raise ValueError('Expected nonempty probability vector or batch')
    logits = np.full_like(p, -np.inf)
    np.log(p, out=logits, where=p > 0)
    logits /= temperature
    logits -= logits.max(axis=-1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=-1, keepdims=True)


def ids_hash(ids):
    return hashlib.sha256(','.join(sorted(ids)).encode()).hexdigest()


def ratio(n, d):
    return n / d if d else None


def score(records, total):
    # One record per person / feature, never two independent records for two eyes.
    n = len(records)
    accepted = [r for r in records if r['accepted']]
    return {'n_scored': n, 'n_correct': sum(r['correct'] for r in records),
            'accuracy': ratio(sum(r['correct'] for r in records), n),
            'n_accepted': len(accepted), 'coverage_on_scored': ratio(len(accepted), n),
            'accepted_coverage_on_all_test_images': ratio(len(accepted), total),
            'accepted_accuracy': ratio(sum(r['correct'] for r in accepted), len(accepted))}


def probability_diagnostics(records, temperature=None):
    if not records:
        return None
    p = np.array([r['prob'] for r in records], dtype=float)
    if temperature is not None:
        p = apply_temperature(p, temperature)
    y = np.array([r['truth_index'] for r in records])
    conf = p.max(1)
    correct = p.argmax(1) == y
    bins = []
    for i in range(10):
        keep = (conf >= i / 10) & ((conf < (i + 1) / 10) if i < 9 else (conf <= 1))
        if keep.any():
            bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'n': int(keep.sum()),
                         'confidence': float(conf[keep].mean()), 'accuracy': float(correct[keep].mean())})
    return {'n': len(y), 'temperature': temperature, 'calibrated': temperature is not None,
            'nll': float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
            'brier_multiclass_sum': float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
            'ece_10_equal_width_bins': sum(b['n'] * abs(b['accuracy'] - b['confidence']) for b in bins) / len(y),
            'bins': bins,
            'fixed_thresholds': {str(t): {'n_accepted': int((conf >= t).sum()),
                                          'coverage': float((conf >= t).mean()),
                                          'accuracy': float(correct[conf >= t].mean()) if (conf >= t).any() else None}
                                 for t in (0.6, 0.7)},
            'interpretation': 'Fixed-threshold diagnostics; no test fitting. Calibration flag denotes a fitted transformation, not independently certified calibration quality.'}


def audit_splits(refs):
    groups = {sp: {s for s, r in refs.items() if r['split'] == sp} for sp in ('train', 'val', 'test')}
    frozen = read('data/derived/kface/split_frozen.json')['split']
    assignment = read('models/kface_v4/folds.json')['assignment']
    assert all(frozen[s] == r['split'] for s, r in refs.items())
    assert set(assignment) == groups['train'] | groups['val']
    assert len(groups['test']) == 62
    folds = []
    for f in sorted(set(assignment.values())):
        expected = {'test': groups['test'], 'val': {s for s, fold in assignment.items() if fold == f},
                    'train': {s for s, fold in assignment.items() if fold != f}}
        actual = rows(f'models/kface_v4/refs/fold{f}/reference_labels.csv')
        assert len(actual) == len(refs)
        for sp in expected:
            assert {r['subject_id'] for r in actual if r['split'] == sp} == expected[sp]
        assert not (expected['train'] & expected['val'] or expected['train'] & expected['test'] or expected['val'] & expected['test'])
        # Fold labels must differ from the global labels only in their split column.
        assert all(all(v == refs[r['subject_id']][k] for k, v in r.items() if k != 'split') for r in actual)
        folds.append({'fold': f, **{sp + '_people': len(ids) for sp, ids in expected.items()},
                      **{sp + '_ids_sha256': ids_hash(ids) for sp, ids in expected.items()},
                      'overlap_train_val': 0, 'overlap_train_test': 0, 'overlap_val_test': 0})
    return {'original_split_people': {sp: len(ids) for sp, ids in groups.items()}, 'development_people': len(assignment),
            'folds': folds, 'frozen_split_matches': True, 'fold_manifests_disjoint': True,
            'test_ids_sha256': ids_hash(groups['test']),
            'unit': 'participant; all captures and left/right crops inherit the participant split',
            'limitations': ['Fold validation was also used for epoch selection.',
                           'Shared cut points and normalization came from the original 276 training participants, not each fold alone.',
                           'Existing within-subject repeatability statistics use participants beyond training.',
                           'Existing test results informed service reliability/type-use policy; this is a repeat audit, not an untouched new final test.']}


def calibration_audit():
    candidates = []
    for base in ('models', 'outputs', 'tmp', 'data/annotations', 'data/derived/kface'):
        for parent, dirs, files in os.walk(ROOT / base):
            dirs[:] = [d for d in dirs if d not in ('subjects', 'subjects_backup_v3', 'dataset_v3', 'vggt',
                                                    'kface_review', 'torch_cache', '__pycache__')]
            for name in files:
                low = name.lower()
                if any(t in low for t in ('oof', 'logit', 'prediction', 'raw_pred', 'calibration', 'temperature')):
                    if Path(name).suffix.lower() in ('.json', '.npz', '.npy', '.csv', '.pt', '.pkl'):
                        path = Path(parent) / name
                        entry = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size}
                        if path.suffix == '.json':
                            obj = json.loads(path.read_text(encoding='utf-8'))
                            entry['top_level_keys'] = list(obj)[:20] if isinstance(obj, dict) else 'list'
                            entry['kind'] = 'aggregate_metrics' if isinstance(obj, dict) and 'parts' in obj else 'other_or_legacy_predictions'
                        candidates.append(entry)
    # Current evaluator retains R/C only in RAM; its persisted OOF JSON has aggregate metrics.
    oof = read('models/kface_v4/evaluation_oof.json')
    assert 'parts' in oof and not any(k in oof for k in ('logits', 'probabilities', 'predictions'))
    return {'status': 'UNVERIFIED_NO_CURRENT_OOF_RAW_PREDICTIONS', 'fit_performed': False,
            'test_used_for_fit': False, 'temperature': None, 'candidates_inspected': candidates,
            'reason': 'Current v4 OOF artifacts contain aggregate scores, not participant-linked probabilities/logits and labels. Legacy models/pilot test_predictions are not v4 OOF validation data.',
            'scope': 'Existing saved raw-prediction audit only; no new development-set inference or training.',
            'next_step': 'Generate participant-held-out validation/OOF raw probabilities with fold provenance, fit per-head temperature using only development labels, freeze before final test.',
            'caveats': ['Original val 62 people are part of the final 338-person fold-training pool; using the full 50-model ensemble on them is not held-out calibration.',
                        'OOF uses two backbones from one held-out fold; test uses ten members per part. Calibration transfer needs validation.',
                        '23 regression heads do not provide class probabilities; temperature scaling applies only to the 15 visual classification heads.']}


def detect_cpu(image, detector):
    arr = np.ascontiguousarray(np.asarray(image))
    h, w = arr.shape[:2]
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if len(result.face_landmarks) != 1:
        return None
    p = np.array([[v.x * w, v.y * h, v.z * w] for v in result.face_landmarks[0]], np.float32)
    pitch, yaw, roll = Rotation.from_matrix(np.asarray(result.facial_transformation_matrixes[0])[:3, :3]).as_euler('xyz', degrees=True)
    bs = {b.category_name: b.score for b in result.face_blendshapes[0]}
    return {'landmarks': p, 'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll),
            'smile': max(bs['mouthSmileLeft'], bs['mouthSmileRight']), 'blink': max(bs['eyeBlinkLeft'], bs['eyeBlinkRight']),
            'brow_down': max(bs['browDownLeft'], bs['browDownRight'])}


def design_regions():
    entries = [('눈', 'trained_legacy_region', 'eye', []), ('눈썹', 'trained_legacy_region', 'brow', []),
               ('코', 'trained_legacy_region', 'nose', []), ('입', 'trained_legacy_region', 'mouth', []),
               ('얼굴형', 'trained_legacy_region', 'face', []), ('이마', 'UNTRAINED_NO_HEAD', None, []),
               ('미간', 'partial_legacy_feature_no_dedicated_region_model', 'face', ['brow_gap']),
               ('애교살', 'UNTRAINED_EXPLICITLY_WITHHELD', None, []),
               ('광대', 'partial_legacy_feature_no_dedicated_region_model', 'face', ['cheekbone']),
               ('귀', 'UNTRAINED_NO_HEAD', None, []),
               ('인중', 'partial_legacy_feature_no_dedicated_region_model', 'mouth', ['philtrum_length']),
               ('점', 'UNTRAINED_NO_HEAD', None, [])]
    return [{'region': name, 'status': status, 'legacy_model_part': part,
             'existing_features': features or ([k for k, f in SCHEMA.items() if f['part'] == part] if part else []),
             'dedicated_model': status == 'trained_legacy_region'} for name, status, part, features in entries]


def fold0_models(res):
    return {part: ([model for path, model in zip(checkpoints(part, MODEL), members)
                    if path.parent.name == 'fold0'], meta, tf)
            for part, (members, meta, tf) in res['models'].items()}


def predict_fold0_visual(crops, selected):
    result = {}
    with torch.inference_mode():
        for part, (members, meta, tf) in selected.items():
            assert len(members) == 2
            batch = torch.stack([tf(crops[c]) for c in meta['crops']])
            outputs = [model(batch)[1] for model in members]
            for j, k in enumerate(meta['classification_features']):
                result[k] = torch.stack([v[j].softmax(1) for v in outputs]).mean(0).mean(0).tolist()
    return result


def fit_fold0(res, refs, report, detector, cache):
    selected = fold0_models(res)
    assignment = read('models/kface_v4/folds.json')['assignment']
    val_ids = sorted(s for s, fold in assignment.items() if fold == 0)
    assert len(val_ids) == report['split_audit']['folds'][0]['val_people']
    visual = [k for k, f in SCHEMA.items() if f['source'] == 'visual']
    records = {k: [] for k in visual}
    counts = Counter(); store = Store(); started = time.perf_counter()
    try:
        for index, sid in enumerate(val_ids):
            key = ids_hash([sid])
            saved = cache['fold0_val'].get(key)
            if saved is not None:
                probs = saved
                counts['reused_images'] += 1
            else:
                with Image.open(io.BytesIO(store.read(sid, member('S001', 'L1', 'E01', 'C7')))) as original:
                    im = original.convert('RGB')
                counts['images_read'] += 1
                d = detect_cpu(im, detector)
                if d is None:
                    counts['detection_failure'] += 1
                    continue
                crops = a.normalized_crops(im, d['landmarks'][:, :2])
                probs = predict_fold0_visual(crops, selected)
                cache['fold0_val'][key] = probs
                save_raw_cache(cache)
                del im, d, crops
            counts['inferred_images'] += 1
            for k in visual:
                classes = SCHEMA[k]['classes']; truth = refs[sid].get(k)
                if truth in classes:
                    records[k].append({'prob': probs[k], 'truth_index': classes.index(truth)})
            if (index + 1) % 10 == 0 or index + 1 == len(val_ids):
                print('FOLD0_VALIDATION', index + 1, '/', len(val_ids), flush=True)
    finally:
        store.close()
    config = {'schema_version': 'faceteller.temperature.v1', 'model_selection': 'fold0_two_arch',
              'features': {}, 'checkpoint_count': 10,
              'checkpoints': [m for m in report['models']['checkpoints'] if m['fold'] == 0],
              'aggregation': 'per-member softmax -> arithmetic mean of two fold0 backbones -> arithmetic mean of left/right crops -> softmax(log(prob)/temperature)',
              'validation_capture': 'S001/L1/E01/C7.jpg', 'validation_people': len(val_ids),
              'validation_ids_sha256': ids_hash(val_ids), 'validation_counts': dict(counts),
              'reference_labels_sha256': sha(REF), 'model_train_people': report['split_audit']['folds'][0]['train_people'],
              'test_used_for_fit': False, 'independent_human_gt_count': 0,
              'optimizer': {'method': 'scipy.optimize.minimize_scalar bounded', 'parameter': 'logT',
                            'bounds': [-3, 3], 'objective': 'mean multiclass negative log likelihood', 'xatol': 1e-6},
              'limitations': ['Fold0 validation was reused for checkpoint epoch selection; this is not an independently reserved calibration set.',
                             'Current checkpoints lack participant identity hashes; disjoint fold manifests and recorded training counts were checked.',
                             'Labels are AI consensus, with no independent human ground truth.',
                             'Only this 10-model fold0 deployment is calibrated; never apply these temperatures to the full 50-model ensemble.',
                             'Regression heads have no classification confidence; runtime must hold with CONFIDENCE_NOT_CALIBRATED.'],
              'fit_seconds': None}
    for k, rec in records.items():
        classes = SCHEMA[k]['classes']
        item = {'temperature': None, 'classes': classes, 'n_val': len(rec), 'calibrated': False}
        if len(rec) < 10:
            item['reason'] = 'INSUFFICIENT_VALID_REFERENCE_SAMPLES'
            config['features'][k] = item
            continue
        p = np.array([v['prob'] for v in rec], float); y = np.array([v['truth_index'] for v in rec])
        def nll(log_t):
            probs = apply_temperature(p, math.exp(float(log_t)))
            return float(-np.log(probs[np.arange(len(y)), y].clip(1e-300)).mean())
        optimum = minimize_scalar(nll, bounds=(-3, 3), method='bounded', options={'xatol': 1e-6})
        baseline = nll(0)
        # T=1 is a feasible fallback if bounded optimization fails to improve NLL.
        best_log_t = float(optimum.x) if optimum.success and optimum.fun <= baseline else 0.0
        success = bool(optimum.success and np.isfinite(optimum.fun))
        item.update(temperature=math.exp(best_log_t) if success else None, calibrated=success,
                    optimizer_success=bool(optimum.success), optimizer_logT=float(optimum.x),
                    optimization_at_boundary=abs(float(optimum.x)) > 2.99,
                    val_nll_before=baseline, val_nll_after=nll(best_log_t) if success else None,
                    val_class_counts=np.bincount(y, minlength=len(classes)).tolist(),
                    missing_validation_classes=[c for i, c in enumerate(classes) if not np.any(y == i)],
                    calibration_method='temperature_scaling_of_ensemble_probabilities')
        config['features'][k] = item
    config['fit_seconds'] = time.perf_counter() - started
    # Freeze and publish before evaluating held-out test predictions.
    (OUT / 'cnn_fold0_calibration.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report['calibration']['fold0_config'] = 'outputs/pilot-20261014/cnn_fold0_calibration.json'
    report['calibration']['status'] = 'FOLD0_TEMPERATURE_FITTED_FULL50_UNCALIBRATED'
    report['calibration']['fit_performed'] = True
    report['calibration']['reason'] = 'No usable existing v4 raw OOF predictions; freshly inferred fold0-held-out validation participants using only the same ten fold0 deployment models.'
    report['calibration']['scope'] = 'Fresh fold0 validation inference and bounded per-visual-head temperature fit; full50 audit is separate and remains uncalibrated.'
    report['calibration']['calibrated_model_selection'] = 'fold0_two_arch'
    report['calibration']['fold0_fit'] = config
    print('CALIBRATION_CONFIG_READY', sum(v['calibrated'] for v in config['features'].values()), 'heads', flush=True)
    return config


def save_raw_cache(cache):
    # Numeric outputs only: probabilities / metrics, opaque participant hash, never an image or landmark.
    (OUT / 'cnn_raw_predictions.json').write_text(json.dumps(cache, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    started = time.perf_counter()
    protected = [ROOT / p for p in ('facework/analysis_v2.py', 'facework/kface_training.py',
                 'facework/kface_ensemble.py', 'facework/features.py', 'facework/morphometrics.py',
                 'data/derived/kface/reference_labels.csv', 'data/derived/kface/reference_summary.json',
                 'data/annotations/kface_visual_v4.tsv', 'data/label_schema_kface_v3.json',
                 'models/kface_v4/evaluation.json', 'models/kface_v4/evaluation_oof.json',
                 'models/kface_v4/folds.json', 'tmp/pilot-20261014/design.txt')]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    ref_rows = rows(REF)
    refs = {r['subject_id']: r for r in ref_rows}
    assert len(refs) == len(ref_rows) == 400
    splits = audit_splits(refs)
    consensus = {r['subject_id']: r for r in rows('data/annotations/kface_visual_v4.tsv', '\t')}
    summary = read('data/derived/kface/reference_summary.json')
    assert summary['labels_sha256'] == sha(REF)
    visual = [k for k, f in SCHEMA.items() if f['source'] == 'visual']
    mismatches = sum(refs[s].get(k) != (consensus[s][k] if consensus[s][k] in SCHEMA[k]['classes'] or consensus[s][k] == 'unknown' else '')
                     for s in refs for k in visual)
    assert mismatches == 0
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'scope': 'Fresh local inference on actual K-FACE held-out front images, one per participant. No training or service integration.',
              'runtime': {'python': sys.executable, 'torch': torch.__version__, 'device': 'cpu',
                          'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                          'mediapipe_delegate': 'CPU', 'opencv_threads': cv2.getNumThreads(),
                          'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'network_and_subprocess_audit_guard': True,
                          'photos_saved': 0, 'photos_transmitted': 0},
              'split_audit': splits, 'design_5_3_regions': design_regions(),
              'reference_provenance': {'labels': 'data/derived/kface/reference_labels.csv',
                  'visual_consensus': 'data/annotations/kface_visual_v4.tsv', 'visual_reference_mismatches': mismatches,
                  'visual': 'Three AI raters A/B/C consensus; not three human raters. Correlated AI errors remain possible.',
                  'landmark': 'Participant median from >=3 valid S001/E01/C7 captures under eight bright lights; classes use original train P25/P75 or feature-specific cuts.',
                  'multiview': 'Existing multi-view factorization-derived relative depth reference, not VGGT or measured physical 3D truth.',
                  'independent_human_gt_count': 0, 'L1_in_reference_lights': 'L1' in summary['reference_lights'],
                  'warning': 'L1 contributes to reference construction. Landmark/hybrid agreement on this capture is not independent measurement validation. AI rater A also saw L1.',
                  'excluded_labels': ['borderline', 'unknown', 'uncertain', 'missing']},
              'calibration': calibration_audit(), 'source_sha256_before': before,
              'metric_definition': {'unit': 'one subject/photo/feature; left/right visual probabilities and regression outputs averaged',
                  'cnn_only': 'Raw CNN heads, regression mean mapped through reference cuts; eye_asymmetry uses its trained raw regression head here.',
                  'current_hybrid': 'Existing feature_table plus quality/occlusion gating; OOF-selected landmarks and derived eye_asymmetry retained.',
                  'coverage': 'accepted/scored references, with accepted/all 62 images also reported; reference coverage is separate.',
                  'cnn_acceptance': 'Visual raw max probability >=0.70 (existing service); regression distance from cuts >= existing mean validation margin. No quality gating.',
                  'hybrid_acceptance': 'status starts with confident after quality/occlusion gating; literature eligibility is a separate used_for_types flag.',
                  'not_claimed': 'No calibrated confidence for regression; no 12-region implementation, VGGT validation, or end-to-end API test.'}}
    print('AUDIT split, reference consensus and OOF raw-prediction search complete', flush=True)
    t = time.perf_counter()
    res = a.resources()
    report['runtime']['model_load_seconds'] = time.perf_counter() - t
    model_audit = []
    for part, (members, meta, tf) in res['models'].items():
        paths = checkpoints(part, MODEL)
        assert len(members) == len(paths) == 10
        for path, model in zip(paths, members):
            assert all(v.device.type == 'cpu' for v in model.parameters())
            # Read metadata with mmap to avoid an extra full model copy.
            ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            m = ck['meta']; f = int(path.parent.name[4:]); fd = splits['folds'][f]
            assert m['train_participants'] == fd['train_people']
            assert m['labels_sha256'] == summary['labels_sha256']
            assert m['classes'] == meta['classes'] and m['norm'] == meta['norm']
            assert m['cuts'] == meta['cuts']
            row = {'checkpoint': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                   'mtime_ns': path.stat().st_mtime_ns,
                   'arch': m['arch'], 'part': part, 'fold': f, 'train_people': m['train_participants'],
                   'val_people_from_manifest': fd['val_people'], 'train_crop_pool': m['train_captures'],
                   'classification_heads': m['classification_features'], 'regression_heads': m['regression_features'],
                   'labels_sha256_match': True}
            if f == 0:
                row['sha256'] = sha(path)
            for sp in ('train', 'val'):
                value = m.get(sp + '_participants_sha256')
                row[sp + '_checkpoint_ids_hash_match'] = value == fd[sp + '_ids_sha256'] if value else None
                assert value is None or row[sp + '_checkpoint_ids_hash_match']
            model_audit.append(row)
            del ck
    report['models'] = {'shape_checkpoint_count': len(model_audit), 'occlusion_checkpoint_count': int(res['occlusion'] is not None),
                        'per_part': {p: len(v[0]) for p, v in res['models'].items()}, 'checkpoints': model_audit,
                        'missing_train_identity_hash_count': sum(m['train_checkpoint_ids_hash_match'] is None for m in model_audit),
                        'identity_evidence_limit': 'Manifest disjointness and recorded participant counts verified; absent checkpoint ID hashes cannot prove historical training membership.',
                        'landmark_selected_by_existing_oof': sorted(res['landmark_preferred']),
                        'source_counts': dict(Counter(f['source'] for f in SCHEMA.values()))}
    print('LOADED', len(model_audit), 'shape CNNs on CPU', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({'sources': before, 'models': model_audit,
                                'script': sha(Path(__file__))}, sort_keys=True).encode()).hexdigest()
    cache_path = OUT / 'cnn_raw_predictions.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    cache_valid = cache.get('fingerprint') == fingerprint
    if not cache_valid:
        cache = {'fingerprint': fingerprint, 'schema': 'aggregate_and_numeric_predictions_v2', 'fold0_val': {}, 'test': {}}
    report['raw_prediction_cache'] = {'path': 'outputs/pilot-20261014/cnn_raw_predictions.json',
        'existing_cache_validated': cache_valid, 'validation': 'exact script/source/reference hashes and 50 checkpoint metadata/size/mtime; fold0 checkpoint SHA256',
        'photos_or_landmarks_stored': False, 'existing_legacy_raw_predictions_reused': False}
    # Capture fold0 outputs during the full-50 forward pass, avoiding a second test inference.
    captured = {part: [] for part in res['models']}
    hooks = []
    for part, (members, meta, tf) in fold0_models(res).items():
        for model in members:
            def capture(module, inputs, output, part=part):
                captured[part].append([v.softmax(1) for v in output[1]])
            hooks.append(model.register_forward_hook(capture))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT / 'models/face_landmarker.task'),
                                         delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    cnn = {k: [] for k in SCHEMA}; hybrid = {k: [] for k in SCHEMA}
    fold0_test = {k: [] for k in visual}
    times = []; failures = Counter(); method_counts = {k: Counter() for k in SCHEMA}; gate_counts = Counter()
    accepted_all = Counter(); images_read = 0; detected = 0; inferred = 0
    reused_test = 0
    test_ids = sorted(s for s, r in refs.items() if r['split'] == 'test')
    store = Store()
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            config = fit_fold0(res, refs, report, detector, cache)
            # Calibration inference hooks are not needed after each validation sample.
            for values in captured.values():
                values.clear()
            for index, sid in enumerate(test_ids):
                opaque_id = ids_hash([sid])
                saved = cache['test'].get(opaque_id)
                if saved is not None:
                    images_read += 1; detected += 1; inferred += 1; reused_test += 1
                    gate_counts.update(saved['blocked'])
                    for k in SCHEMA:
                        accepted_all[k] += saved['accepted_all'][k]
                        method_counts[k][saved['methods'][k]] += 1
                        if k in saved['cnn']:
                            cnn[k].append(saved['cnn'][k]); hybrid[k].append(saved['hybrid'][k])
                    for k, rec in saved['fold0'].items():
                        fold0_test[k].append(rec)
                    times.append(saved['seconds'])
                    continue
                t = time.perf_counter()
                sample_cnn = {}; sample_hybrid = {}; sample_fold0 = {}; sample_methods = {}; sample_accept = {}
                try:
                    blob = store.read(sid, member('S001', 'L1', 'E01', 'C7'))
                    with Image.open(io.BytesIO(blob)) as original:
                        im = original.convert('RGB')
                    images_read += 1
                    d = detect_cpu(im, detector)
                    if d is None:
                        failures['face_detection'] += 1
                        continue
                    detected += 1
                    p = d['landmarks']; metrics = a.measure(im, p[:, :2]); crops = a.normalized_crops(im, p[:, :2])
                    flags = a.quality_flags(d, metrics, im)
                    occ = a.occlusion_check(crops)
                    blocked = {scope for _, scope in flags}
                    if occ:
                        if occ['eyewear']['sunglasses'] > 0.5 or occ['eyewear']['glasses'] > 0.5:
                            blocked.update(('eye', 'brow', 'nose_depth'))
                        if occ['hat']['yes'] > 0.5:
                            blocked.add('brow')
                    gate_counts.update(blocked)
                    for values in captured.values():
                        values.clear()
                    pred = a.run_models(im, p, crops)
                    fold0_probs = {}
                    for part, outputs in captured.items():
                        assert len(outputs) == 2
                        for j, k in enumerate(res['models'][part][1]['classification_features']):
                            fold0_probs[k] = torch.stack([v[j] for v in outputs]).mean(0).mean(0).tolist()
                    table = a.feature_table(pred, metrics, depth=None)
                    inferred += 1
                    for k, f in SCHEMA.items():
                        ft = table[k]
                        if 'all' in blocked or f['part'] in blocked or ('nose_depth' in blocked and f['source'] == 'multiview'):
                            ft['status'] = 'withheld'
                        h_accept = ft['status'].startswith('confident')
                        accepted_all[k] += int(h_accept)
                        method = ft.get('method', 'cnn_visual')
                        method_counts[k][method] += 1
                        sample_methods[k] = method; sample_accept[k] = int(h_accept)
                        truth = refs[sid].get(k)
                        if truth not in f['classes']:
                            continue
                        meta = pred['_meta'][f['part']]
                        if f['source'] == 'visual':
                            prob = pred[k]['prob']
                            assert np.isfinite(prob).all() and abs(sum(prob) - 1) < 1e-5
                            label = pred[k]['classes'][int(np.argmax(prob))]
                            entry = {'correct': label == truth, 'accepted': max(prob) >= 0.7,
                                     'prob': prob, 'truth_index': pred[k]['classes'].index(truth)}
                            fold0_entry = {'prob': fold0_probs[k], 'truth_index': pred[k]['classes'].index(truth)}
                            fold0_test[k].append(fold0_entry)
                            sample_fold0[k] = fold0_entry
                        else:
                            value = float(np.mean(pred[k]['values']))
                            label = f['classes'][int(np.searchsorted(summary['features'][k]['cuts'], value))]
                            margin = meta['uncertainty_margin'].get(k) or 0
                            entry = {'correct': label == truth, 'accepted': min(abs(value - c) for c in meta['cuts'][k]) >= margin}
                        cnn[k].append(entry)
                        hybrid_entry = {'correct': ft['label'] == truth, 'accepted': h_accept,
                                        'used_for_types': h_accept and ft.get('used_for_types', True)}
                        hybrid[k].append(hybrid_entry)
                        sample_cnn[k] = entry; sample_hybrid[k] = hybrid_entry
                    cache['test'][opaque_id] = {'cnn': sample_cnn, 'hybrid': sample_hybrid, 'fold0': sample_fold0,
                        'accepted_all': sample_accept, 'methods': sample_methods, 'blocked': sorted(blocked),
                        'seconds': time.perf_counter() - t}
                    save_raw_cache(cache)
                    del blob, im, crops, pred, table, d, p
                except (KeyError, FileNotFoundError, OSError) as exc:
                    failures[type(exc).__name__] += 1
                finally:
                    times.append(time.perf_counter() - t)
                    if (index + 1) % 5 == 0 or index + 1 == len(test_ids):
                        print('INFER', index + 1, '/', len(test_ids), 'last_s', round(times[-1], 3), flush=True)
    finally:
        store.close()
        for hook in hooks:
            hook.remove()
    report['raw_prediction_cache']['test_images_reused'] = reused_test
    report['raw_prediction_cache']['test_images_newly_inferred'] = inferred - reused_test
    calibration_test = {}
    for k, rec in fold0_test.items():
        item = config['features'][k]
        baseline = probability_diagnostics(rec)
        adjusted = probability_diagnostics(rec, item['temperature']) if item['calibrated'] else None
        calibration_test[k] = {'n_test': len(rec), 'baseline': baseline, 'temperature_scaled': adjusted,
                              'temperature': item['temperature'], 'calibrated': item['calibrated']}
    report['calibration']['fold0_test62'] = calibration_test
    report['calibration']['test_summary'] = {
        'n_images': inferred, 'head_count': len(calibration_test),
        'mean_head_nll_before': float(np.mean([v['baseline']['nll'] for v in calibration_test.values()])),
        'mean_head_nll_after': float(np.mean([v['temperature_scaled']['nll'] for v in calibration_test.values() if v['calibrated']])),
        'mean_head_ece_before': float(np.mean([v['baseline']['ece_10_equal_width_bins'] for v in calibration_test.values()])),
        'mean_head_ece_after': float(np.mean([v['temperature_scaled']['ece_10_equal_width_bins'] for v in calibration_test.values() if v['calibrated']])),
        'note': 'Macro average across visual heads; missing/borderline/unknown reference labels excluded per head. Full50 remains uncalibrated.'}
    report['fresh_measurement'] = {'capture': 'S001/L1/E01/C7.jpg', 'planned_people': 62, 'planned_images': 62,
        'actual_images_read': images_read, 'detected_people': detected, 'actual_inferred_images': inferred,
        'crop_inputs_per_successful_image': 7, 'shape_crop_inputs_total': 7 * inferred,
        'failures': dict(failures), 'quality_occlusion_blocked_image_counts_by_scope': dict(gate_counts),
        'timing': {'scope': 'ZIP read + decode + CPU detection/crops + 50 shape CNNs + occlusion CNN + feature policy/scoring; excludes model load, API, depth and literature matching',
                   'n': len(times), 'total_s': sum(times), 'mean_s': float(np.mean(times)),
                   'median_s': float(np.median(times)), 'p95_s': float(np.percentile(times, 95)),
                   'first_s': times[0], 'warm_mean_s': float(np.mean(times[1:]))}, 'features': {}}
    agreement = read('data/annotations/kface_rater_agreement.json')
    for k, f in SCHEMA.items():
        eligible = sum(refs[s].get(k) in f['classes'] for s in test_ids)
        report['fresh_measurement']['features'][k] = {'name_ko': f['ko'], 'legacy_part': f['part'], 'reference_source': f['source'],
            'classes': f['classes'], 'reference_eligible_people': eligible, 'reference_coverage': eligible / 62,
            'excluded_reference_counts': dict(Counter(refs[s].get(k) or 'missing' for s in test_ids if refs[s].get(k) not in f['classes'])),
            'cnn_only': score(cnn[k], 62), 'current_hybrid': score(hybrid[k], 62),
            'hybrid_accepted_images_including_unscorable_reference': accepted_all[k],
            'hybrid_method_counts': dict(method_counts[k]),
            'hybrid_literature_eligible_scored_count': sum(r['used_for_types'] for r in hybrid[k]),
            'uncalibrated_probability_diagnostics': probability_diagnostics(cnn[k]) if f['source'] == 'visual' else None,
            'ai_rater_fleiss_kappa_existing': agreement.get(k, {}).get('fleiss_kappa')}
    report['fresh_measurement']['by_legacy_part'] = {part: {
        'cnn_micro': score([r for k, f in SCHEMA.items() if f['part'] == part for r in cnn[k]],
                           62 * sum(f['part'] == part for f in SCHEMA.values())),
        'hybrid_micro': score([r for k, f in SCHEMA.items() if f['part'] == part for r in hybrid[k]],
                              62 * sum(f['part'] == part for f in SCHEMA.values()))} for part in res['models']}
    report['fresh_measurement']['by_reference_source'] = {src: score([r for k, f in SCHEMA.items() if f['source'] == src for r in cnn[k]],
                                        62 * sum(f['source'] == src for f in SCHEMA.values())) for src in ('visual', 'landmark', 'multiview')}
    existing = read('models/kface_v4/evaluation.json')
    report['previous_results_cited_not_remeasured'] = {
        'source': 'models/kface_v4/evaluation.json', 'group': a.SERVICE,
        'warning': 'Existing multi-light, mostly per-crop metrics; not directly comparable to this one-photo, side-averaged participant score.',
        'features': {k: {'part': part, **v.get(a.SERVICE, {})} for part, pr in existing['parts'].items() for k, v in pr['features'].items()},
        'visual_threshold_source': 'models/kface_v4/visual_confidence.json',
        'visual_threshold_metrics': read('models/kface_v4/visual_confidence.json')}
    report['design_gaps'] = [
        '12-region design not implemented: four regions untrained, three only partially represented by legacy heads.',
        'Existing CNN has 15 visual classification and 23 regression heads; regression classes have no calibrated probabilities.',
        'Existing full50 threshold is raw probability 0.70. Only the separate fold0 10-model candidate has fitted visual-head temperatures; runtime integration is performed by main.',
        'Existing alignment makes the two outer eye corners horizontal, not the design eye centers. Crop scale is FACE_PX=256 / distance(234,454), not eye-distance normalization; fixed Resize slightly changes aspect ratios for rounded-to-8 sizes.',
        'Runtime must preserve the trained face-width crop convention. Replacing it with eye-distance normalization without retraining/revalidation is an input distribution change; this pilot is not fully compliant with design 5.2.',
        'Current feature_table selects landmarks using OOF, overrides eye_asymmetry with roundness-derived asymmetry, and uses prior test-derived reliability/type gating.',
        'Four legacy single-frontal depth regression heads are not the design three-view VGGT depth measurements.',
        'Some old service/evaluation strings still incorrectly say single AI rater; current labels match three-AI consensus.',
        'No independent human GT: reference agreement cannot establish independently reviewed morphology accuracy.']
    report['runtime']['cuda_initialized'] = torch.cuda.is_initialized()
    assert not report['runtime']['cuda_initialized']
    report['runtime']['total_seconds'] = time.perf_counter() - started
    report['source_integrity'] = {p.relative_to(ROOT).as_posix(): sha(p) == before[p.relative_to(ROOT).as_posix()] for p in protected}
    assert all(report['source_integrity'].values())
    report['status'] = 'complete' if inferred == 62 else 'partial_with_failures'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cnn_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_review(report)
    print('DONE', report['status'], 'images', inferred, 'cuda_initialized', report['runtime']['cuda_initialized'], flush=True)


def write_review(r):
    fresh = r['fresh_measurement']
    pct = lambda v: '-' if v is None else f'{100 * v:.2f}%'
    lines = ['FaceTeller CNN 파일럿 검증 — 2026-10-14', '',
        '범위: 기존 코드 수정·학습·main 연결 없이 별도 CPU 검증. 결과는 집계 수치만 저장.',
        f"실측: heldout 62명 S001/L1/E01/C7 각 1장. 읽기 {fresh['actual_images_read']}장, 검출 {fresh['detected_people']}명, 추론 {fresh['actual_inferred_images']}장.",
        f"CPU: torch threads=2, interop=2, MediaPipe CPU, CUDA initialized={r['runtime']['cuda_initialized']}. 이미지 저장 0 / 전송 0.",
        '원래 사람 분할: train 276 / val 62 / test 62. 최종 개발집단 338명, 5-fold.',
        *[f"  fold {f['fold']}: train {f['train_people']} / val {f['val_people']} / test {f['test_people']}명; manifest 교집합 0." for f in r['split_audit']['folds']],
        f"형태 모델 50개(5부위 × 2백본 × 5-fold), 가림 모델 1개. checkpoint 학습 ID hash 미기록 {r['models']['missing_train_identity_hash_count']}개: manifest와 인원 일치까지 검증, 과거 학습 명단 완전 증명 불가.",
        '', '정답 출처: reference_labels.csv와 kface_visual_v4.tsv(visual consensus) 대조 불일치 0.',
        '시각 15항목은 AI 판정자 A/B/C 합의. 독립 human GT 0건. 3명 인간 전문가 합의로 표현하면 안 됨.',
        '랜드마크 19항목은 반복 사진 중앙값, 깊이 4항목은 기존 다중시점 상대 깊이 기준.',
        'L1은 기준값 생성에 포함되고 AI A가 본 사진이다. 본 점수는 reference 일치율이며 독립 측정 정확도가 아님.',
        'OOF epoch 선택 재사용·초기 train 경계값 공유·반복오차 전체 사람 통계·기존 test 기반 신뢰도 정책의 한계 유지.',
        '', '확률보정: 전체50 모델은 미보정. 별도 fold0 두 백본 × 5부위 = 10모델 후보만 실제 temperature fitting.',
        '기존 v4 OOF 파일에는 집계 지표만 있어 raw 재사용 불가. fold0 heldout 68명을 실제 추론하여 각 시각 head logT∈[-3,3] NLL 최소화.',
        '동일 fold0 10모델의 test62 raw 확률은 전체50 추론 중 hook으로 함께 수집: test 재추론 없이 보정 전후 NLL/ECE 평가.',
        '시험 62명으로 fitting/threshold tuning 하지 않음. calibration config는 test 평가 전에 확정·저장.',
        'fold0 검증 참여자는 epoch early stopping에도 사용됨: 독립 calibration 전용 표본이 아님. 독립 human GT 없음.',
        'JSON에 모델군별 시각 항목 NLL/Brier/ECE와 고정 theta 0.60·0.70 정확도/coverage 기록. 보정 후 시험 지표 악화도 그대로 공개.',
        '원래 val 62명도 최종 앙상블의 학습집단에 포함됨. 전체 앙상블로 재예측하여 보정하면 누수.',
        '', '설계 5.3 대상 12부위:',
        *[f"  {x['region']}: {x['status']} / 기존 특징 {','.join(x['existing_features']) or '없음'}" for x in r['design_5_3_regions']],
        '미간 너비·광대 돌출·인중 길이의 기존 head는 전용 부위 crop 모델 구현을 뜻하지 않음.',
        '애교살은 스키마에 학습불가로 보류; 인중 골 뚜렷함도 낮은 AI 일치도로 제거. 이마·귀·점 head 없음.',
        '', '이번 실측: 기존 5부위별 micro reference 일치율 (서로 다른 특징의 분모를 합친 설명용 수치)',
        '부위 | CNN 정확도 | CNN coverage | 확신 CNN 정확도 | 현재 hybrid 정확도 | hybrid coverage']
    for part, v in fresh['by_legacy_part'].items():
        c, h = v['cnn_micro'], v['hybrid_micro']
        lines.append(f"{part} | {pct(c['accuracy'])} | {pct(c['coverage_on_scored'])} | {pct(c['accepted_accuracy'])} | {pct(h['accuracy'])} | {pct(h['coverage_on_scored'])}")
    lines += ['', '특징별 실측 (62명 중 reference 유효인원 N; 좌우 평균 사진 1장 단위)',
              '특징 | N/62 | CNN acc | CNN cov | CNN accepted acc | hybrid acc | hybrid cov | hybrid accepted acc']
    for k, v in fresh['features'].items():
        c, h = v['cnn_only'], v['current_hybrid']
        lines.append(f"{k} | {v['reference_eligible_people']}/62 | {pct(c['accuracy'])} | {pct(c['coverage_on_scored'])} | {pct(c['accepted_accuracy'])} | {pct(h['accuracy'])} | {pct(h['coverage_on_scored'])} | {pct(h['accepted_accuracy'])}")
    tm = fresh['timing']
    lines += ['', 'coverage 정의: 유효 reference 중 보류하지 않은 비율. 정답 누락·경계·unknown 제외.',
              'JSON에는 reference coverage와 전체 62장 대비 accepted coverage도 별도로 기록.',
              'CNN은 raw head, hybrid는 기존 OOF landmark 선택·가림/품질·눈 비대칭 파생식을 적용. 동일한 평가로 혼동 금지.',
              '회귀 결과에는 softmax 확률이 없어 확률 기준 대신 기존 validation 오차 margin으로 보류.',
              f"실측 시간: 평균 {tm['mean_s']:.3f}초/장, 중앙값 {tm['median_s']:.3f}, P95 {tm['p95_s']:.3f}, 62장 합계 {tm['total_s']:.1f}초. 모델 load {r['runtime']['model_load_seconds']:.1f}초 별도.",
              '시간에 ZIP 읽기·전처리·50 CNN·가림·feature policy 포함. 깊이/VGGT·문헌선정·HTTP 제외: 전체 서비스 5초 충족 주장 불가.',
              '', '기존 평가 재인용과 구분:',
              'JSON previous_results_cited_not_remeasured = 기존 evaluation.json service(C7,bright)와 visual_confidence.json의 재인용.',
              '기존 여러 조명/좌우 crop 분모와 이번 L1 사진 62장 평균 분모가 다르므로 직접 개선율 계산 금지.',
              'fresh_measurement만 이번 실제 추론 결과. 과거 학습 결과를 이번 실측으로 표시하지 않음.',
              '', 'fold0 보정 전후 test62 (head별):',
              'feature | n_val | T | n_test | NLL before -> after | ECE before -> after']
    for k, v in r['calibration']['fold0_test62'].items():
        fit = r['calibration']['fold0_fit']['features'][k]
        b, c = v['baseline'], v['temperature_scaled']
        if c:
            lines.append(f"{k} | {fit['n_val']} | {fit['temperature']:.6f} | {v['n_test']} | {b['nll']:.6f} -> {c['nll']:.6f} | {b['ece_10_equal_width_bins']:.6f} -> {c['ece_10_equal_width_bins']:.6f}")
        else:
            lines.append(f"{k} | {fit['n_val']} | 미보정 | {v['n_test']}")
    lines += ['', 'main runtime 연결 계약:',
              'outputs/pilot-20261014/cnn_fold0_calibration.json: model_selection="fold0_two_arch", features[k]={temperature,classes,n_val,calibrated}.',
              '부위당 fold0 백본 2개만 선택 -> 각 softmax 확률의 산술평균 -> 좌우 crop 산술평균 -> softmax(log(prob)/T). 클래스 순서 그대로 유지.',
              '실제 fit 성공한 visual head만 calibrated=true. 회귀 23항목은 config에 없으며 CONFIDENCE_NOT_CALIBRATED로 보류.',
              '전체50모델에 이 config 적용 금지. config checkpoint SHA256으로 fold0 10개 가중치 확인 가능.',
              '기존 trained crop 유지: 외안각 수평정렬, 얼굴폭234-454 기준 FACE_PX=256. 설계 눈중심/눈사이거리 정규화와 다름.',
              '새 눈기준 crop으로 무작정 교체하면 학습 입력 분포가 달라짐. 재학습/재검증 없이 설계 완전 준수 주장 금지.',
              '', 'main 연결 전 차이 및 남은 검증:', *['- ' + v for v in r['design_gaps']],
              '', '완료: scripts/pilot_cnn_check.py / outputs/pilot-20261014/cnn_results.json / cnn_review.txt',
              '사용한 기존 source/reference 파일 SHA256 전후 일치 확인. 기존 파일 수정 없음.']
    (OUT / 'cnn_review.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
