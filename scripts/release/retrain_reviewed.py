"""Reviewed relabel training candidate v3 (isolated; never writes the active bundle).

Implements the 2026-10-15 Claude audit, main second review and coordinator follow-ups:
  * ResNet18 only; fixed participant split train270/selection34/calibration34/test62.
  * The 41 classification features keep their definitions. Face-width/height
    denominators are NOT replaced; context parts (facework.release_context,
    part-face-context-v1: eye, brow, nose, mouth, glabella, philtrum) get an
    H x 2W input (local part left, whole face letterboxed right). Raw crop
    contract (registry input_size/crops) is unchanged; model_input_size recorded.
    Face, ear, forehead, under_eye, cheekbone, mole inputs are unchanged.
  * Measured targets are soft class probabilities: Phi interval membership of
    the subject (or side) value with sigma = train270 median SINGLE-CAPTURE repeat
    SD (not SD/sqrt(n)); no near-boundary exclusion. Physical 0/1 references and
    original train270 cuts are kept. Brow length/tail use visible-brow-mask-v3
    (facework.release_brows) with train-only refit cuts (new measurement).
    Forehead/ear use their existing metric values and tertile cuts with sigma
    from train repeat ranges (range/d2). pigmented_marks stays a hard label.
  * Per-side targets for bilateral crops (eye, brow, ear); subject targets for
    selection/calibration/test evaluation.
  * Visual AI soft votes retained; ties excluded from hard evaluation only.
  * Calibration on calibration34 only: temperature only if >=2 classes have >=5
    hard-labelled participants; allowed_confirmation_classes = those classes with
    actual training exposure. Threshold: existing empirical calibration-cost
    policy (facework.release_thresholds, wrong 1 / withheld 0.25), restricted to
    allowed classes; always a finite float.
  * Unsupported heads trained for interface completeness, never confirmable:
    eye_asymmetry, under_eye, philtrum_definition, pigmented_marks.
All targets are automated measurements or same-family AI votes; human_review_count=0.

CLI (Windows): .venv/Scripts/python.exe -B -X utf8 scripts/release/retrain_reviewed.py {prepare,train,calibrate,test,all}
  prepare   CPU: identity checks, re-measurement parity, targets, protocol freeze, caches.
  train     GPU: 12 ResNet18 part models (fixed 16max/6min/patience4 protocol).
  calibrate GPU: calibration34 temperatures, allowed classes, thresholds.
  test      GPU: test62 once, after deployment_candidate.json is frozen.
"""
import argparse
import gc
import hashlib
import io
import json
import math
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from facework.release_context import CONTEXT_PARTS, CONTEXT_REASON, CONTEXT_VERSION, compose_context, model_input_size

VERSION = 'reviewed-relabel-candidate-v3'
CANDIDATE = ROOT / 'experiments/relabel-20261015/candidate-v3'
# Frozen earlier candidate kept untouched; superseded by main's independent brow-mask review.
SUPERSEDES = {'candidate': 'experiments/relabel-20261015/candidate-v2', 'version': 'reviewed-relabel-candidate-v2',
              'protocol_sha256': '35a7d48067571f86f93617fb4d72712728573dea085909d7e88193fe872ddb9d',
              'reason': 'brow-mask-independent-review-20261015 BROW-V2-01 (thin-tail extent) and BROW-V2-02 (unstable side hidden by bilateral mean)'}
UNSUPPORTED = {
    'eye_asymmetry': 'single-capture ICC 0.413; top-quantile cut within repeat noise',
    'under_eye': 'single-pass AI label, 8 positives in 400, no calibration positives',
    'philtrum_definition': 'AI Fleiss kappa 0.173 recorded as withhold',
    'pigmented_marks': 'absent: train 4 / calibration 0; uncalibratable'}
MASK_FEATURES = {'brow_length': 'length_eye_ratio', 'brow_tail': 'tail_angle_deg'}
EXTENDED_SOFT = ('forehead_height', 'forehead_width', 'ear_length', 'ear_width')
PHYSICAL_CENTRES = {'eye_tilt': 0.0, 'mouth_corner': 0.0, 'lip_balance': 1.0, 'brow_tail': 0.0}
D2 = {2: 1.128, 3: 1.693}   # expected range / SD for n normal draws
MIN_LANDMARK_CAPTURES = 3
MIN_MASK_FRAMES = 2
MIN_CALIBRATION_CLASS = 5
COVERAGE_POLICY = 'min5_per_class_two_classes_v1'
TRAIN_SUPPORT_POLICY = 'actual_unique_people_with_training_crop_target_argmax_v1'
SIDE_OF_CROP = {'eye_right': 'right', 'eye_left': 'left', 'brow_right': 'right', 'brow_left': 'left',
                'ear_left_oblique': 'C4', 'ear_right_oblique': 'C10'}
SEED = 20261014

AUDITS = {
    'outputs/release/claude-full-label-audit-20261015.json': 'a5fb5268164e845b25b24f4a8918d203ce05de72b55ab2606d413f26b21da12c',
    'outputs/release/main-all-label-audit-20261015.json': '1c0d3b06de1cc9b7d8fab4f2d63b9637fabc74ea2c0bf323420828fb292b9283',
    'outputs/release/main-second-review-20261015.json': '67de25b7c369c3adee2db74dabf338a9919230381f65c143cd75d344727890cb',
}
INPUTS = ['data/release/training_registry.json', 'data/release/landmark_labels.json',
          'data/release/extended_labels.json', 'models/selection-20261014/protocol.json',
          'data/label_schema_kface_v3.json', 'data/derived/release/crop_manifest.json',
          'data/derived/release/ears-v2/manifest.json', 'models/release/ear_preprocessing_frozen.json',
          'data/relabel-20261015/measurements.json', 'data/relabel-20261015/brow_mask_audit.json',
          'outputs/release/brow-mask-independent-review-20261015.json',
          'facework/release_preprocess.py', 'facework/release_brows.py', 'facework/release_context.py',
          'facework/release_thresholds.py', 'scripts/release/prepare_landmark_labels.py',
          'scripts/release/retrain_reviewed.py']
FROZEN = ['models/release/training_frozen.json', 'models/release/inference_frozen.json',
          'models/release/deployment.json', 'models/release/threshold_policy_frozen.json',
          'outputs/release/test_metrics.json']


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    for active in ('models/release', 'data/release'):
        if path.resolve().is_relative_to((ROOT / active).resolve()):
            raise RuntimeError('Candidate must not write the active bundle')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8'); tmp.replace(path)


# ---------------------------------------------------------------- identities
def check_audits(root=ROOT, audits=AUDITS):
    """Both audits and the second review must exist with the reviewed identity."""
    for rel, digest in audits.items():
        path = Path(root) / rel
        if not path.is_file(): raise RuntimeError(f'Required review artifact missing: {rel}')
        if sha(path) != digest: raise RuntimeError(f'Review artifact identity changed: {rel}')
    review = load(Path(root) / 'outputs/release/main-second-review-20261015.json')
    if (review.get('claude_audit_sha256') != audits['outputs/release/claude-full-label-audit-20261015.json'] or
            review.get('independent_audit_sha256') != audits['outputs/release/main-all-label-audit-20261015.json']):
        raise RuntimeError('Second review does not reference the pinned audits')
    return review


def identities(root=ROOT):
    return {'audits': dict(AUDITS), 'inputs': {p: sha(Path(root) / p) for p in INPUTS},
            'frozen_active_bundle': {p: sha(Path(root) / p) for p in FROZEN if (Path(root) / p).exists()}}


def verify_protocol(candidate=CANDIDATE, root=ROOT):
    check_audits(root)
    path = Path(candidate) / 'protocol.json'
    if not path.exists(): raise RuntimeError('Run prepare first (protocol.json missing)')
    protocol = load(path); now = identities(root)
    for group in ('audits', 'inputs'):
        if protocol['identities'][group] != now[group]:
            changed = [k for k in now[group] if protocol['identities'][group].get(k) != now[group][k]]
            raise RuntimeError(f'Identity changed since protocol freeze ({group}): {changed}')
    # This candidate never writes the active bundle; a later change there is main's
    # (runtime/promotion owner) and is reported loudly instead of blocking training.
    moved = [k for k in now['frozen_active_bundle'] if protocol['identities']['frozen_active_bundle'].get(k) != now['frozen_active_bundle'][k]]
    if moved: print('WARNING_ACTIVE_BUNDLE_CHANGED_SINCE_FREEZE', moved, flush=True)
    if sha(Path(candidate) / 'targets.json') != protocol['targets_sha256']:
        raise RuntimeError('targets.json changed after protocol freeze')
    return protocol


# ---------------------------------------------------------------- targets
def phi(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_soft(value, cuts, sigma):
    """Interval-membership probabilities of N(value, sigma^2) for ordered cuts."""
    if not (math.isfinite(value) and math.isfinite(sigma) and sigma > 0): raise ValueError('invalid soft target input')
    cdf = [phi((c - value) / sigma) for c in cuts]
    probs = [cdf[0]] + [cdf[i] - cdf[i - 1] for i in range(1, len(cdf))] + [1 - cdf[-1]]
    probs = [max(p, 0.0) for p in probs]; total = sum(probs)
    return [p / total for p in probs]


def hard_class(value, cuts): return int(np.searchsorted(cuts, value, side='right'))


def median_repeat_sd(groups, minimum=3):
    sds = [statistics.stdev(v) for v in groups if len(v) >= minimum]
    if not sds: raise ValueError('no repeat variability')
    return float(statistics.median(sds))


def soft_entry(value, n, cuts, sigma, **extra):
    """Soft target from one measured value; sigma is the single-capture repeat SD."""
    return {'value': float(value), 'n': int(n), 'sigma': float(sigma), 'soft': phi_soft(value, cuts, sigma),
            'hard': hard_class(value, cuts), 'tie': False, **extra}


def landmark_frames(sid, root=ROOT):
    from scripts.kface_labels import REF_LIGHTS, quality_ok, mouth_ok
    from scripts.release.prepare_landmark_labels import measurements_from_points
    frames = []
    with np.load(Path(root) / f'data/derived/kface/subjects/{sid}.npz') as archive:
        lookup = {str(k): i for i, k in enumerate(archive['keys'])}
        for light in REF_LIGHTS:
            i = lookup.get(f'S001|{light}|E01|C7|core')
            if i is None or archive['status'][i] != 1: continue
            m = measurements_from_points(archive['landmarks'][i])
            frames.append({'light': light, 'quality_ok': bool(quality_ok(archive['pose'][i], archive['blend'][i], m)),
                           'mouth_ok': bool(mouth_ok(m)), 'metrics': m})
    return frames


def brow_frames(sid, root=ROOT):
    """visible-brow-mask-v3 per parsed frontal frame (L1/L2/L3), with frame quality gates."""
    from facework.release_brows import measure_brows
    folder = Path(root) / 'data/derived/release/ears'
    meta = load(folder / f'{sid}.json'); frames = []
    with np.load(Path(root) / f'data/derived/kface/subjects/{sid}.npz') as archive, (folder / f'{sid}.bin').open('rb') as blob:
        lookup = {str(k): i for i, k in enumerate(archive['keys'])}
        for frame in meta['parsing_frames']:
            if frame['camera'] != 'C7' or frame['status'] != 'parsed': continue
            i = lookup.get(frame['capture'] + '|core')
            if i is None:
                frames.append({'light': frame['light'], 'sides': {}, 'rejected': {}, 'frame_rejected': 'NO_LANDMARKS'}); continue
            blob.seek(frame['mask']['offset'])
            with Image.open(io.BytesIO(blob.read(frame['mask']['length']))) as image: mask = np.asarray(image).copy()
            r = measure_brows(mask, archive['landmarks'][i], pose=archive['pose'][i], blend=archive['blend'][i], include_path=False)
            frames.append({'light': frame['light'], 'sides': r['sides'], 'rejected': r['rejected'], 'frame_rejected': r['frame_rejected']})
    return frames


def _check_parity(name, ours, theirs, tol=1e-9):
    if abs(ours - theirs) > tol: raise RuntimeError(f'Parity failure against audit artifact: {name} {ours} vs {theirs}')


def _range_sd(entry):
    """Single-capture SD estimate from a repeat relative range (range/d2), or None."""
    n = entry.get('n'); rel = entry.get('relative_range'); value = entry.get('value')
    if value is None or rel is None or n not in D2: return None
    return float(rel * abs(value) / D2[n])


def _count(items):
    out = {}
    for x in items: out[x] = out.get(x, 0) + 1
    return out


def build_targets(root=ROOT, *, subjects=None, parity=True, log=None):
    """Candidate target table. Pure CPU; reads frozen artifacts only."""
    from facework import release_brows
    root = Path(root)
    registry = load(root / 'data/release/training_registry.json')
    landmark = load(root / 'data/release/landmark_labels.json')
    extended = load(root / 'data/release/extended_labels.json')
    schema_v3 = load(root / 'data/label_schema_kface_v3.json')['features']
    groups = load(root / 'models/selection-20261014/protocol.json')['groups']
    split_of = {s: g for g, ids in groups.items() for s in ids}
    if sorted(len(v) for v in groups.values()) != [34, 34, 62, 270] or len(split_of) != 400:
        raise RuntimeError('Participant split must be fixed train270/selection34/calibration34/test62')
    remeasure = load(root / 'data/relabel-20261015/measurements.json')['subjects'] if parity else {}
    F = registry['features']; ids = sorted(split_of) if subjects is None else sorted(subjects)
    train = [s for s in ids if split_of[s] == 'train']
    landmark_keys = [k for k in landmark['features'] if k not in MASK_FEATURES]
    raw_lm, raw_brow = {}, {}
    for n, sid in enumerate(ids, 1):
        frames = landmark_frames(sid, root)
        if parity:
            theirs = remeasure[sid]['frames']
            if [f['light'] for f in frames] != [f['light'] for f in theirs]: raise RuntimeError(f'Parity frames differ {sid}')
            for a, b in zip(frames, theirs):
                if a['quality_ok'] != b['quality_ok'] or a['mouth_ok'] != b['mouth_ok']: raise RuntimeError(f'Parity gate {sid}')
                for key, value in b['metrics'].items(): _check_parity(f'{sid}/{key}', a['metrics'][key], value)
        values = {}
        for key in landmark_keys:
            metric = landmark['features'][key]['metric']; mouth_gate = schema_v3[key]['part'] == 'mouth'
            usable = [f for f in frames if f['quality_ok'] and (f['mouth_ok'] or not mouth_gate)]
            values[key] = {'subject': [f['metrics'][metric] for f in usable]}
            if F[key].get('per_side'):
                values[key]['sides'] = {s: [f['metrics'][f'{metric}_{s}'] for f in usable] for s in ('right', 'left')}
        raw_lm[sid] = values
        raw_brow[sid] = brow_frames(sid, root)
        if log and n % 50 == 0: log(n, len(ids))

    definitions = {}
    for key in landmark_keys:
        original = landmark['features'][key]
        definitions[key] = {
            'source': 'landmark_soft', 'metric': original['metric'], 'classes': F[key]['classes'],
            'cuts': list(original['cuts']), 'cut_provenance': original['method'] + ' (original train270 release cuts retained)',
            'physical_centre': PHYSICAL_CENTRES.get(key), 'sigma_subject': float(original['median_within_subject_sd']),
            'sigma_definition': 'train270 median single-capture repeat SD (original release value)',
            'min_captures': MIN_LANDMARK_CAPTURES, 'per_side': bool(F[key].get('per_side'))}
        if definitions[key]['per_side']:
            definitions[key]['sigma_side'] = median_repeat_sd([raw_lm[s][key]['sides'][side] for s in train for side in ('right', 'left')])
        if subjects is None:
            recomputed = median_repeat_sd([raw_lm[s][key]['subject'] for s in train])
            if abs(recomputed - definitions[key]['sigma_subject']) > 1e-9: raise RuntimeError(f'Repeat SD differs for {key}')

    def frame_sides(sid): return [f['sides'] for f in raw_brow[sid] if not f['frame_rejected']]
    brow_sigma = {'subject': {}, 'side': {}}
    for metric in MASK_FEATURES.values():
        brow_sigma['subject'][metric] = median_repeat_sd(
            [[(fs['left'][metric] + fs['right'][metric]) / 2 for fs in frame_sides(s) if 'left' in fs and 'right' in fs] for s in train])
        brow_sigma['side'][metric] = median_repeat_sd([[fs[side][metric] for fs in frame_sides(s) if side in fs] for s in train for side in ('right', 'left')])
    brow_summary = {sid: release_brows.aggregate(frame_sides(sid), single_capture_sd=brow_sigma) for sid in ids}
    for key, metric in MASK_FEATURES.items():
        medians = [brow_summary[s]['subject'][metric]['median'] for s in train
                   if brow_summary[s]['subject'][metric]['median'] is not None and brow_summary[s]['subject'][metric]['n'] >= MIN_MASK_FRAMES]
        if key in PHYSICAL_CENTRES:
            tol = 1.96 * brow_sigma['subject'][metric]
            cuts = [PHYSICAL_CENTRES[key] - tol, PHYSICAL_CENTRES[key] + tol]
            provenance = 'physical 0 deg reference +/- 1.96 x train270 median single-capture SD of the v3 visible-hair tail angle'
        else:
            cuts = np.quantile(medians, [.25, .75]).tolist()
            provenance = 'train270-only 0.25/0.75 quantiles of v3 visible-hair length (new measurement; mesh and v2 cuts do not apply)'
        definitions[key] = {
            'source': 'brow_mask_soft', 'metric': metric, 'mask_definition': release_brows.VERSION, 'mask_gates': release_brows.GATES,
            'classes': F[key]['classes'], 'cuts': [float(c) for c in cuts], 'cut_provenance': provenance,
            'physical_centre': PHYSICAL_CENTRES.get(key), 'train_measured_subjects': len(medians),
            'sigma_subject': brow_sigma['subject'][metric], 'sigma_side': brow_sigma['side'][metric],
            'sigma_definition': 'train270 median single-capture repeat SD of the v3 mask metric',
            'min_captures': MIN_MASK_FRAMES, 'per_side': True,
            'note': 'bottom quartile is population-relative; visible_span_shorter_than_eye recorded separately, never as this class'}

    extended_keys = set(next(iter(extended['subjects'].values())))
    for key, f in F.items():
        if key in definitions: continue
        if key not in extended_keys:
            definitions[key] = {'source': 'visual_vote', 'classes': f['classes'], 'per_side': False, 'provenance': f.get('annotation_provenance')}
        elif key in EXTENDED_SOFT:
            t = extended['thresholds'][key]; side_based = key.startswith('ear_')
            subject_sds = [_range_sd(extended['metrics'][s][key]) for s in train if s in extended['metrics']]
            side_sds = [_range_sd((extended['metrics'][s][key].get('side_measurements') or {}).get(c) or {})
                        for s in train if s in extended['metrics'] for c in ('C4', 'C10')] if side_based else []
            sigma_side = float(statistics.median([x for x in side_sds if x])) if side_based else None
            sigma_subject = sigma_side if side_based else float(statistics.median([x for x in subject_sds if x]))
            definitions[key] = {'source': 'extended_soft', 'classes': f['classes'], 'per_side': side_based,
                                'cuts': [t['lower'], t['upper']], 'cut_provenance': 'original train270 tertiles (extended_labels.json)',
                                'sigma_subject': sigma_subject, 'sigma_side': sigma_side,
                                'sigma_definition': 'train270 median single-capture SD estimated from repeat range / d2 (n=2:1.128, n=3:1.693)'
                                                    + ('; ear sides pooled, bilateral subject value uses the same sigma' if side_based else ''),
                                'provenance': f.get('annotation_provenance')}
        else:
            definitions[key] = {'source': 'extended_hard', 'classes': f['classes'], 'per_side': False, 'provenance': f.get('annotation_provenance'),
                                'note': 'binary weak automated label kept hard; no repeat-scale soft target'}
    for key, reason in UNSUPPORTED.items(): definitions[key].update(supported=False, unsupported_reason=reason)
    for d in definitions.values(): d.setdefault('supported', True)

    table = {}
    for sid in ids:
        labels = registry['subjects'][sid]['labels']; votes = registry['subjects'][sid]['soft_targets']
        entry = {'split': split_of[sid], 'targets': {}}
        for key, d in definitions.items():
            target = {'subject': None, 'sides': {}}
            if d['source'] == 'landmark_soft':
                vals = raw_lm[sid][key]
                if len(vals['subject']) >= d['min_captures']:
                    target['subject'] = soft_entry(statistics.median(vals['subject']), len(vals['subject']), d['cuts'], d['sigma_subject'])
                if d['per_side']:
                    target['sides'] = {s: soft_entry(statistics.median(v), len(v), d['cuts'], d['sigma_side']) if len(v) >= d['min_captures'] else None
                                       for s, v in vals['sides'].items()}
            elif d['source'] == 'brow_mask_soft':
                summary = brow_summary[sid]; metric = d['metric']
                def pick(s, sigma):
                    if s['median'] is None or s['n'] < d['min_captures']: return None
                    return soft_entry(s['median'], s['n'], d['cuts'], sigma)
                target['subject'] = pick(summary['subject'][metric], d['sigma_subject'])
                target['sides'] = {s: pick(summary['sides'][s][metric], d['sigma_side']) for s in ('right', 'left')}
                target['withheld'] = {'subject_unstable': summary['subject'][metric]['unstable'],
                                      'subject_contributing_side_failed': summary['subject'][metric]['contributing_side_failed'],
                                      'sides_unstable': {s: summary['sides'][s][metric]['unstable'] for s in ('right', 'left')}}
                if key == 'brow_length': target['visible_span_shorter_than_eye'] = summary['visible_span_shorter_than_eye_frames']
            elif d['source'] == 'extended_soft':
                m = extended['metrics'].get(sid, {}).get(key, {})
                if m.get('value') is not None and labels.get(key) in d['classes']:
                    target['subject'] = soft_entry(m['value'], m.get('n') or 0, d['cuts'], d['sigma_subject'])
                    if d['classes'][target['subject']['hard']] != labels[key]: raise RuntimeError(f'Extended label/value mismatch {sid}/{key}')
                if d['per_side']:
                    for camera in ('C4', 'C10'):
                        side = (m.get('side_measurements') or {}).get(camera) or {}
                        value = side.get('value')
                        target['sides'][camera] = (soft_entry(value, side.get('n') or 0, d['cuts'], d['sigma_side'])
                                                   if value is not None and math.isfinite(value) else None)
            else:
                label = labels.get(key)
                if label in d['classes']:
                    index = d['classes'].index(label)
                    vote = votes.get(key) if d['source'] == 'visual_vote' else None
                    soft = [float(x) for x in vote] if vote else [1.0 if i == index else 0.0 for i in range(len(d['classes']))]
                    top = max(soft); tie = sum(abs(x - top) < 1e-9 for x in soft) > 1
                    target['subject'] = {'soft': soft, 'hard': index, 'tie': bool(tie), 'source_label': label}
            entry['targets'][key] = target
        table[sid] = entry

    support = {}
    for split in groups:
        support[split] = {}
        for key, d in definitions.items():
            k = len(d['classes']); hard = [0] * k; mass = [0.0] * k; ties = unmeasured = 0
            for sid in groups[split]:
                if sid not in table: continue
                t = table[sid]['targets'][key]['subject']
                if t is None: unmeasured += 1; continue
                mass = [a + b for a, b in zip(mass, t['soft'])]
                if t['tie']: ties += 1
                else: hard[t['hard']] += 1
            support[split][key] = {'hard': hard, 'soft_mass': [round(x, 3) for x in mass], 'ties': ties, 'unmeasured': unmeasured}
    for key, d in definitions.items():
        d['train_zero_support_classes'] = [c for c, n in zip(d['classes'], support['train'][key]['hard']) if n == 0]
    brow_qa = {'frames': sum(len(raw_brow[s]) for s in ids),
               'frame_rejected': _count(f['frame_rejected'] for s in ids for f in raw_brow[s] if f['frame_rejected']),
               'side_rejected': _count(r for s in ids for f in raw_brow[s] for r in f['rejected'].values()),
               'side_accepted': sum(len(f['sides']) for s in ids for f in raw_brow[s]),
               'unstable_subjects': {m: sum(brow_summary[s]['subject'][m]['unstable'] for s in ids) for m in MASK_FEATURES.values()},
               'contributing_side_failed_subjects': {m: sum(bool(brow_summary[s]['subject'][m]['contributing_side_failed']) and brow_summary[s]['subject'][m]['n'] >= MIN_MASK_FRAMES
                                                            and not brow_summary[s]['subject'][m]['unstable'] for s in ids) for m in MASK_FEATURES.values()},
               'published_subjects': {split: {m: sum(brow_summary[s]['subject'][m]['median'] is not None for s in ids if split_of[s] == split)
                                              for m in MASK_FEATURES.values()} for split in groups},
               'visible_span_shorter_than_eye_subjects': sum(1 for s in ids if brow_summary[s]['visible_span_shorter_than_eye_frames']['true'])}
    return {'version': VERSION, 'human_review_count': 0, 'participants': len(table),
            'meaning': 'automated landmark/mask measurements and same-family AI votes; not human ground truth',
            'definitions': definitions, 'subjects': table, 'support': support, 'brow_mask_qa': brow_qa}


# ---------------------------------------------------------------- inputs
def _png(blob, crop):
    blob.seek(crop['offset'])
    with Image.open(io.BytesIO(blob.read(crop['length']))) as image: return np.asarray(image.convert('RGB')).copy()


def crop_targets(targets, sid, heads, crop_name):
    """Per-crop soft vectors: side target for per-side features, subject target otherwise."""
    side = SIDE_OF_CROP.get(crop_name); vectors = []
    for key in heads:
        t = targets['subjects'][sid]['targets'][key]; d = targets['definitions'][key]
        entry = t['sides'].get(side) if (d['per_side'] and side is not None) else t['subject']
        vectors.append(None if entry is None else entry['soft'])
    return vectors


def subject_truth(targets, sid, heads):
    """Subject hard truth for evaluation; ties and unmeasured are -1."""
    out = []
    for key in heads:
        t = targets['subjects'][sid]['targets'][key]['subject']
        out.append(-1 if t is None or t['tie'] else int(t['hard']))
    return out


def exposure_support(owners, votes, heads, classes):
    """Actual training exposure: unique people per class among crops whose target argmax is that class."""
    owners = np.asarray(owners); votes = np.asarray(votes); people = {}; crops = {}
    for j, key in enumerate(heads):
        k = len(classes[key]); v = votes[:, j, :k]; has = v.sum(1) > 0; arg = v.argmax(1)
        people[key] = [int(len(set(owners[has & (arg == c)].tolist()))) for c in range(k)]
        crops[key] = [int((has & (arg == c)).sum()) for c in range(k)]
    return people, crops


def build_cache(part, split, registry, targets, protocol, candidate=CANDIDATE, root=ROOT):
    import torch
    path = Path(candidate) / 'cache' / f'{part}_{split}.pt'
    if path.exists():
        result = torch.load(path, map_location='cpu', weights_only=True)
        if result['targets_sha256'] != protocol['targets_sha256']: raise RuntimeError('Stale candidate cache')
        return result
    spec = registry['parts'][part]; heads = spec['features']; F = registry['features']
    width = max(len(F[k]['classes']) for k in heads)
    needs_eye_quality = part in ('eye', 'under_eye') or any(F[k].get('region', part) == 'eye' for k in heads)
    ids = sorted(s for s, d in targets['subjects'].items() if d['split'] == split)
    folder = Path(root) / 'data/derived/release' / ('ears-v2' if part == 'ear' else 'subjects')
    images, owners, votes, rows = [], [], [], []; skipped = {'face_context_invalid': 0, 'no_target': 0}
    for owner, sid in enumerate(ids):
        meta_path = folder / f'{sid}.json'
        if not meta_path.exists(): continue
        meta = load(meta_path)
        with (folder / f'{sid}.bin').open('rb') as blob:
            for frame in meta['frames']:
                if frame['status'] != 'ready' or (split != 'train' and frame['light'] != 'L1'): continue
                blend = frame.get('blend', [0] * 5)
                if needs_eye_quality and (max(blend[1:3]) > .5 or blend[0] > .3): continue
                if part in ('mouth', 'philtrum') and blend[0] > .3: continue
                face = None
                if part in CONTEXT_PARTS:
                    fc = frame['crops'].get('face')
                    if fc is None or fc.get('valid_fraction', 1) < .9: skipped['face_context_invalid'] += 1; continue
                    face = _png(blob, fc)
                for name in spec['crops']:
                    crop = frame['crops'].get(name)
                    if crop is None or crop.get('valid_fraction', 1) < .9: continue
                    array = _png(blob, crop)
                    if list(array.shape[:2]) != spec['input_size']: raise RuntimeError('Crop shape mismatch')
                    if face is not None: array = compose_context(array, face, part, mirror_face=name.endswith('_left'))
                    v = np.zeros((len(heads), width), np.float32)
                    for j, vec in enumerate(crop_targets(targets, sid, heads, name)):
                        if vec is not None: v[j, :len(vec)] = vec
                    if split == 'train' and not v.any(): skipped['no_target'] += 1; continue
                    images.append(torch.from_numpy(array).permute(2, 0, 1)); owners.append(owner); votes.append(v)
                    rows.append({'subject_id': sid, 'capture': frame.get('capture', frame['light']), 'crop': name})
    if not images: raise RuntimeError(f'No usable crops: {part}/{split}')
    result = {'images': torch.stack(images), 'owners': torch.tensor(owners), 'votes': torch.from_numpy(np.stack(votes)),
              'subject_truth': torch.tensor([subject_truth(targets, s, heads) for s in ids]),
              'ids': ids, 'rows': rows, 'heads': heads, 'skipped': skipped,
              'context_version': CONTEXT_VERSION if part in CONTEXT_PARTS else None,
              'model_input_size': model_input_size(part), 'targets_sha256': protocol['targets_sha256']}
    if list(result['images'].shape[2:]) != model_input_size(part): raise RuntimeError('Model input size mismatch')
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(result, path)
    return result


# ---------------------------------------------------------------- training
def augment_input(x, g, part, part_width=None):
    """Existing fixed augmentation applied per tile so content never crosses tiles.

    The two equal-size tiles receive identical random draws (same rotation,
    shift, photometric and blur parameters): the generator state is saved before
    the local tile and restored before the whole-face tile, so their relative
    geometry is never perturbed (independent +/-3 deg rotations could differ by
    6 deg, more than the tilt cut-offs). One shared flip decision follows
    (nose/mouth only, as before).
    """
    import torch
    from scripts.model_selection_train import augment
    if part_width is None or x.shape[-1] == part_width: return augment(x, g, part)
    if x.shape[-1] != 2 * part_width: raise ValueError('context input must be two equal-width tiles')
    state = g.get_state()
    a = augment(x[..., :part_width], g, 'context')
    g.set_state(state)
    b = augment(x[..., part_width:], g, 'context')
    if part in ('nose', 'mouth'):
        flip = torch.rand((len(x), 1, 1, 1), device=x.device, generator=g) < .5
        a = torch.where(flip, a.flip(-1), a); b = torch.where(flip, b.flip(-1), b)
    return torch.cat([a, b], -1)


def soft_loss(logits_list, votes, weights):
    """Class-weighted soft cross-entropy over heads with target mass (inherited weighting)."""
    import torch
    per_head = []
    for j, logits in enumerate(logits_list):
        target = votes[:, j, :logits.shape[1]]; valid = target.sum(1) > 0
        if not valid.any(): continue
        weight = (target[valid] * weights[j]).sum(1)
        per_head.append((-(target[valid] * logits[valid].float().log_softmax(1)).sum(1) * weight).sum() / weight.sum().clamp_min(1e-8))
    return torch.stack(per_head).mean() if per_head else None


def class_weights(exposure, heads):
    """Inherited inverse-support weights (cap 5), from actual training exposure in people."""
    return [np.minimum(sum(exposure[k]) / (len(exposure[k]) * np.maximum(exposure[k], 1)), 5).astype(np.float32) for k in heads]


def predict(models, data, heads, classes, device):
    import torch
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    outputs = {k: [] for k in heads}
    with torch.inference_mode():
        for start in range(0, len(data['images']), 16):
            x = (data['images'][start:start + 16].to(device).float() / 255 - mean) / std
            if device == 'cuda': x = x.contiguous(memory_format=torch.channels_last)  # baseline GPU layout
            predictions = [m(x)[1] for m in models]
            for j, k in enumerate(heads): outputs[k].append(torch.stack([p[j].softmax(1) for p in predictions]).mean(0).cpu().numpy())
    owners = data['owners'].numpy(); truth = data['subject_truth'].numpy(); result = {}
    for j, k in enumerate(heads):
        raw = np.concatenate(outputs[k]); prob = []; present = []
        for i in range(len(data['ids'])):
            mask = owners == i; present.append(bool(mask.any()))
            prob.append(raw[mask].mean(0) if mask.any() else np.full(len(classes[k]), 1 / len(classes[k])))
        y = np.where(np.asarray(present), truth[:, j], -1)  # no usable photo -> not evaluated
        result[k] = {'prob': np.asarray(prob).tolist(), 'truth': y.tolist(), 'available': present}
    return result


def checkpoint_meta(part, registry, protocol, exposure, root=ROOT):
    spec = registry['parts'][part]
    meta = {'arch': 'resnet18', 'part': part, 'classification_features': spec['features'],
            'classes': {k: registry['features'][k]['classes'] for k in spec['features']},
            'crops': spec['crops'], 'input_size': spec['input_size'], 'model_input_size': model_input_size(part),
            'context_version': CONTEXT_VERSION if part in CONTEXT_PARTS else None,
            'registry_sha256': sha(Path(root) / 'data/release/training_registry.json'),
            'preprocess_version': registry['preprocess_version'], 'train_support': exposure,
            'train_support_policy': TRAIN_SUPPORT_POLICY, 'targets_sha256': protocol['targets_sha256'],
            'candidate_version': VERSION, 'human_review_count': 0}
    if part == 'ear':
        from facework.release_parsing import EAR_TRANSFORM_VERSION
        meta.update(ear_transform_version=EAR_TRANSFORM_VERSION,
                    ear_preprocessing_sha256=sha(Path(root) / 'models/release/ear_preprocessing_frozen.json'))
    return meta


def train_part(part, registry, targets, protocol, candidate=CANDIDATE, device='cuda', pretrained=True, epochs=None):
    import torch
    from facework.kface_training import PartNet
    from scripts.model_selection_metrics import metrics
    out = Path(candidate) / 'models/resnet18'; path = out / f'{part}.json'
    if path.exists(): return load(path)
    cfg = protocol['config']; heads = registry['parts'][part]['features']; F = registry['features']
    classes = {k: F[k]['classes'] for k in heads}
    train = build_cache(part, 'train', registry, targets, protocol, candidate)
    selection = build_cache(part, 'selection', registry, targets, protocol, candidate)
    exposure, crop_exposure = exposure_support(train['owners'].numpy(), train['votes'].numpy(), heads, classes)
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    model = PartNet('resnet18', 0, [len(classes[k]) for k in heads], pretrained=pretrained).to(device)
    if device == 'cuda': model = model.to(memory_format=torch.channels_last)  # baseline GPU layout
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    batch = cfg['batch_size']; steps = math.ceil(len(train['images']) / batch); max_epochs = epochs or cfg['epochs_max']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs_max'] * steps, eta_min=3e-6)
    use_amp = device == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    rng = torch.Generator(device=device).manual_seed(SEED + 1)
    data = train['images'].to(device); votes = train['votes'].to(device)
    weights = [torch.tensor(w, device=device) for w in class_weights(exposure, heads)]
    part_width = registry['parts'][part]['input_size'][1] if part in CONTEXT_PARTS else None
    supported = [k for k in heads if targets['definitions'][k]['supported']]
    best, best_epoch, best_state, best_prediction, history = -1.0, 0, None, None, []
    for epoch in range(1, max_epochs + 1):
        model.train(); order = torch.randperm(len(data), device=device, generator=rng); losses = []; started = time.perf_counter()
        for offset in range(0, len(data), batch):
            idx = order[offset:offset + batch]
            x = augment_input(data[idx].float() / 255, rng, part, part_width)
            if use_amp: x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                _, predictions = model(x)
            loss = soft_loss(predictions, votes[idx], weights)
            if loss is None: continue
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            old = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if not use_amp or scaler.get_scale() >= old: scheduler.step()
            losses.append(float(loss.detach()))
        model.eval(); prediction = predict([model], selection, heads, classes, device); scores = []
        for k in heads:
            y = np.asarray(prediction[k]['truth']); m = metrics(np.asarray(prediction[k]['prob']), y); prediction[k]['metrics'] = m
            if k in supported and m['macro_f1'] is not None and len(set(y[y >= 0].tolist())) >= 2: scores.append(m['macro_f1'])
        score = float(np.mean(scores)) if scores else 0.0
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None, 'selection_macro_f1_supported': score,
                        'scored_heads': len(scores), 'train_seconds': time.perf_counter() - started})
        if score > best:
            best, best_epoch, best_prediction = score, epoch, prediction
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print('REVIEWED_EPOCH', part, epoch, round(score, 4), flush=True)
        if epoch >= cfg['epochs_min'] and epoch - best_epoch >= cfg['patience']: break
    meta = checkpoint_meta(part, registry, protocol, exposure)
    meta['protocol_sha256'] = sha(Path(candidate) / 'protocol.json')
    meta['pretrained'] = 'torchvision ResNet18_Weights.DEFAULT' if pretrained else None
    out.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'meta': meta}, out / f'{part}.pt')
    record = {'meta': meta, 'config': cfg, 'history': history, 'selected_epoch': best_epoch, 'selection': best_prediction,
              'train_crops': len(data), 'train_people': len(set(train['owners'].tolist())), 'skipped': train['skipped'],
              'train_crop_support': crop_exposure, 'subject_train_support': {k: targets['support']['train'][k]['hard'] for k in heads}}
    dump(path, record)
    del model, optimizer, scaler, data, votes, best_state, train, selection
    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------- calibration
def calibration_rule(support, train_support, minimum=MIN_CALIBRATION_CLASS):
    """Temperature only with >=2 adequately supported classes; allowed = adequate and actually trained."""
    adequate = [i for i, n in enumerate(support) if n >= minimum]
    allowed = [i for i in adequate if train_support[i] > 0]
    return {'fit_temperature': len(adequate) >= 2, 'adequate_classes': adequate, 'allowed_classes': allowed,
            'minimum_per_class': minimum, 'coverage_policy': COVERAGE_POLICY}


def evaluate_policy(prob, truth, threshold, allowed):
    from facework.release_thresholds import POLICY
    p = np.asarray(prob); y = np.asarray(truth); valid = y >= 0; p, y = p[valid], y[valid]
    if not len(y): return {'n': 0}
    pred = p.argmax(1); conf = p.max(1); accepted = (conf >= threshold) & np.isin(pred, allowed)
    wrong = int(((pred != y) & accepted).sum()); held = int((~accepted).sum())
    return {'n': int(len(y)), 'accepted': int(accepted.sum()), 'withheld': held, 'wrong_confirmed': wrong,
            'declared_cost_mean': (wrong * POLICY['wrong_confirmation_cost'] + held * POLICY['withheld_cost']) / len(y),
            'accepted_error_rate': wrong / int(accepted.sum()) if accepted.any() else None}


def calibrate_head(prob, truth, definition, train_support, *, validation_ids, test_ids):
    """One head on calibration34: coverage rule, optional temperature, allowed classes, empirical threshold."""
    from scripts.model_selection_metrics import fit_temperature, apply_temperature, metrics
    from facework.release_thresholds import select_threshold
    p = np.asarray(prob, dtype=float); y = np.asarray(truth, dtype=int); d = definition; k = p.shape[1]
    if set(validation_ids) & set(test_ids): raise ValueError('Calibration cannot use test participants')
    support = np.bincount(y[y >= 0], minlength=k).tolist()
    rule = calibration_rule(support, train_support)
    entry = {'classes': d['classes'], 'class_count': k, 'n_val': int((y >= 0).sum()), 'support': support,
             'raw_calibration_support': support, 'train_support': [int(x) for x in train_support],
             'train_support_policy': TRAIN_SUPPORT_POLICY, 'coverage_policy': COVERAGE_POLICY, 'rule': rule,
             'supported': d['supported'], 'unsupported_reason': d.get('unsupported_reason'),
             'split': 'validation', 'test_used_for_fit': False, 'participant_separation_checked': True}
    calibrated = False; adjusted = p
    if d['supported'] and rule['fit_temperature']:
        masked = np.where(np.isin(y, rule['adequate_classes']), y, -1)  # inadequate classes do not shape T
        fit = fit_temperature(p, masked, validation_ids=validation_ids, test_ids=test_ids)
        ok = fit.get('calibrated') is True and fit.get('optimizer_success') is True and fit.get('temperature') is not None
        entry['fit'] = fit
        if ok:
            calibrated = True; adjusted = apply_temperature(p, fit['temperature'])
            entry.update(temperature=fit['temperature'], log_temperature=fit['log_temperature'], optimizer_success=True,
                         nll_before=fit['nll_before'], nll_after=fit['nll_after'], at_bound=fit['at_bound'], status=fit['status'])
        else:
            entry.update(status='temperature_fit_failed')
    else:
        entry.update(status='unsupported_head' if not d['supported'] else 'insufficient_calibration_support')
    if not calibrated:
        entry.update(temperature=None, log_temperature=None, optimizer_success=False)
    allowed = rule['allowed_classes'] if calibrated else []
    # Existing empirical policy; classes outside `allowed` count as untrained so they are never accepted.
    masked_support = [int(train_support[i]) if i in allowed else 0 for i in range(k)]
    selection = select_threshold(adjusted, y, calibrated=calibrated, train_support=masked_support,
                                 validation_ids=validation_ids, test_ids=test_ids)
    entry.update(calibrated=calibrated, threshold=float(selection['threshold']), threshold_selection=selection,
                 allowed_confirmation_classes=[d['classes'][i] for i in allowed], raw=metrics(p, y, threshold=float(selection['threshold'])))
    if calibrated:
        entry['calibrated_metrics'] = metrics(adjusted, y, threshold=entry['threshold'])
        entry['policy_on_calibration34'] = evaluate_policy(adjusted, y, entry['threshold'], allowed)
    return entry


def _load_model(part, candidate, device):
    import torch
    from facework.kface_training import PartNet
    ck = torch.load(Path(candidate) / f'models/resnet18/{part}.pt', map_location='cpu', weights_only=True); meta = ck['meta']
    model = PartNet('resnet18', 0, [len(meta['classes'][k]) for k in meta['classification_features']], pretrained=False)
    model.load_state_dict(ck['state_dict']); model.eval(); model = model.to(device)
    if device == 'cuda': model = model.to(memory_format=torch.channels_last)
    return [model], meta


def calibrate(registry, targets, protocol, candidate=CANDIDATE, device='cuda'):
    import torch
    from facework.release_thresholds import POLICY
    path = Path(candidate) / 'deployment_candidate.json'
    if path.exists(): return load(path)
    test_ids = [s for s, d in targets['subjects'].items() if d['split'] == 'test']
    features, weights, files = {}, {}, {}
    for part, spec in registry['parts'].items():
        models, meta = _load_model(part, candidate, device)
        data = build_cache(part, 'calibration', registry, targets, protocol, candidate)
        prediction = predict(models, data, spec['features'], meta['classes'], device)
        for k, r in prediction.items():
            features[k] = calibrate_head(r['prob'], r['truth'], targets['definitions'][k], meta['train_support'][k],
                                         validation_ids=data['ids'], test_ids=test_ids)
            features[k]['subject_train_support'] = targets['support']['train'][k]['hard']
        local = Path(candidate) / f'models/resnet18/{part}.pt'
        weights[f'models/release/resnet18/{part}.pt'] = sha(local)   # promotion target key
        files[f'models/release/resnet18/{part}.pt'] = local.relative_to(ROOT).as_posix() if local.is_relative_to(ROOT) else str(local)
        del models; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    manifest = {'version': VERSION, 'status': 'candidate_not_promoted', 'protocol_sha256': sha(Path(candidate) / 'protocol.json'),
                'targets_sha256': protocol['targets_sha256'], 'registry_sha256': sha(ROOT / 'data/release/training_registry.json'),
                'weights': weights, 'candidate_weight_files': files, 'inference_architectures': ['resnet18'],
                'parts': protocol['inputs'], 'context_version': CONTEXT_VERSION,
                'threshold_policy': {**POLICY, 'restricted_to_allowed_confirmation_classes': True},
                'coverage_policy': {'version': COVERAGE_POLICY, 'minimum_per_class': MIN_CALIBRATION_CLASS, 'adequate_classes_for_temperature': 2},
                'train_support_policy': TRAIN_SUPPORT_POLICY,
                'ear_transform_version': protocol['ear_transform_version'], 'ear_preprocessing_sha256': protocol['ear_preprocessing_sha256'],
                'features': features, 'test_used': False, 'human_review_count': 0}
    dump(path, manifest)
    return manifest


def test(registry, targets, protocol, candidate=CANDIDATE, device='cuda'):
    import torch
    from scripts.model_selection_metrics import apply_temperature, metrics
    manifest = calibrate(registry, targets, protocol, candidate, device)
    for key, digest in manifest['weights'].items():
        local = Path(candidate) / 'models/resnet18' / Path(key).name
        if sha(local) != digest: raise RuntimeError('Candidate weight changed after calibration')
    report = {}
    for part, spec in registry['parts'].items():
        models, meta = _load_model(part, candidate, device)
        data = build_cache(part, 'test', registry, targets, protocol, candidate)
        prediction = predict(models, data, spec['features'], meta['classes'], device)
        for k, r in prediction.items():
            c = manifest['features'][k]; p = np.asarray(r['prob']); y = np.asarray(r['truth'], dtype=int)
            q = apply_temperature(p, c['temperature']) if c['calibrated'] else p
            boundary = [bool(t is not None and 'sigma' in t and max(t['soft']) < .9)
                        for t in (targets['subjects'][sid]['targets'][k]['subject'] for sid in data['ids'])]
            yb = np.where(np.asarray(boundary), y, -1)
            allowed = [meta['classes'][k].index(x) for x in c['allowed_confirmation_classes']]
            report[k] = {'raw': metrics(p, y, threshold=c['threshold']),
                         'calibrated': metrics(q, y, threshold=c['threshold']) if c['calibrated'] else None,
                         'near_boundary_subset': metrics(q, yb, threshold=c['threshold']) if (yb >= 0).any() else None,
                         'policy': evaluate_policy(q, y, c['threshold'], allowed) if c['calibrated'] else None,
                         'photo_available': int(sum(r['available'])), 'participants': len(data['ids']),
                         'supported': c['supported'], 'calibrated_head': c['calibrated']}
        del models; gc.collect()
        if device == 'cuda': torch.cuda.empty_cache()
    result = {'version': VERSION, 'deployment_candidate_sha256': sha(Path(candidate) / 'deployment_candidate.json'),
              'features': report, 'human_review_count': 0,
              'evaluation_meaning': 'agreement with automated measurement/AI-vote targets on held-out test62; ties and unmeasured excluded from hard metrics'}
    dump(Path(candidate) / 'test_metrics.json', result)
    return result


# ---------------------------------------------------------------- stages
def protocol_inputs(registry):
    out = {}
    for part, spec in registry['parts'].items():
        entry = {'crops': spec['crops'], 'input_size': spec['input_size'], 'model_input_size': model_input_size(part)}
        if part in CONTEXT_PARTS:
            entry.update(context_version=CONTEXT_VERSION, context={
                'layout': 'local part left, whole frontal face right', 'helper': 'facework.release_context.compose_context',
                'face_source': 'face crop network input 256x224 letterboxed into the part input size, gray 127',
                'mirror_face_for_left_crops': True, 'reason': CONTEXT_REASON[part]})
        out[part] = entry
    return out


def prepare(candidate=CANDIDATE, root=ROOT, build_caches=True):
    from facework.release_preprocess import VERSION as PREPROCESS
    from facework.release_parsing import EAR_TRANSFORM_VERSION
    check_audits(root)
    candidate = Path(candidate); protocol_path = candidate / 'protocol.json'
    registry = load(Path(root) / 'data/release/training_registry.json')
    if protocol_path.exists():
        protocol = verify_protocol(candidate, root)
    else:
        if len(registry['features']) != 41 or sum(len(p['features']) for p in registry['parts'].values()) != 41:
            raise RuntimeError('Registry must define exactly 41 classification features')
        targets = build_targets(root, log=lambda n, total: print(f'REVIEWED_TARGETS {n}/{total}', flush=True))
        dump(candidate / 'targets.json', targets)
        archive = candidate / 'archive'; archive.mkdir(parents=True, exist_ok=True)
        for rel in AUDITS: shutil.copy2(Path(root) / rel, archive / Path(rel).name)
        protocol = {'version': VERSION, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'architecture': 'resnet18', 'preprocess_version': PREPROCESS,
                    'ear_transform_version': EAR_TRANSFORM_VERSION,
                    'ear_preprocessing_sha256': sha(Path(root) / 'models/release/ear_preprocessing_frozen.json'),
                    'split': {s: len(v) for s, v in load(Path(root) / 'models/selection-20261014/protocol.json')['groups'].items()},
                    'config': {'epochs_max': 16, 'epochs_min': 6, 'patience': 4, 'batch_size': 16,
                               'optimizer': 'AdamW lr3e-4 wd0.05 cosine (eta_min 3e-6)', 'seed': SEED,
                               'checkpoint': 'selection34 mean subject macroF1 over supported heads with >=2 classes',
                               'loss': 'class-weighted soft cross-entropy (inherited weights from actual exposure, cap 5)',
                               'augmentation': 'scripts.model_selection_train.augment per tile with identical random draws for both tiles (generator state saved/restored: same rotation, shift, photometric, blur); one shared flip (nose/mouth)',
                               'train_lights': 'all ready frames', 'eval_lights': 'L1',
                               'torch_threads': {'intra': 4, 'interop': 2}, 'gpu_memory_format': 'channels_last (CUDA only)',
                               'amp': 'float16 autocast + GradScaler (CUDA only)'},
                    'inputs': protocol_inputs(registry), 'context_version': CONTEXT_VERSION, 'context_parts': list(CONTEXT_PARTS),
                    'unsupported': UNSUPPORTED,
                    'calibration': {'split': 'calibration34', 'coverage_policy': COVERAGE_POLICY, 'minimum_per_class': MIN_CALIBRATION_CLASS,
                                    'oof_cross_fitting': False},
                    'threshold': 'facework.release_thresholds.POLICY (empirical calibration cost, wrong 1 / withheld 0.25), allowed classes only',
                    'train_support_policy': TRAIN_SUPPORT_POLICY,
                    'identities': identities(root), 'targets_sha256': sha(candidate / 'targets.json'),
                    'supersedes': SUPERSEDES, 'human_review_count': 0, 'gpu_started': False}
        dump(protocol_path, protocol)
        protocol = verify_protocol(candidate, root)
    if build_caches:
        targets = load(candidate / 'targets.json')
        for part in registry['parts']:
            for split in ('train', 'selection', 'calibration', 'test'):
                data = build_cache(part, split, registry, targets, protocol, candidate, root)
                print('REVIEWED_CACHE', part, split, tuple(data['images'].shape), data['skipped'], flush=True)
                del data; gc.collect()
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('stage', choices=['prepare', 'train', 'calibrate', 'test', 'all'])
    parser.add_argument('--targets-only', action='store_true', help='prepare without building image caches')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(2)
    if args.stage == 'prepare':
        protocol = prepare(build_caches=not args.targets_only)
        print(json.dumps({'protocol_sha256': sha(CANDIDATE / 'protocol.json'), 'targets_sha256': protocol['targets_sha256']}))
        return
    if not torch.cuda.is_available(): raise RuntimeError('GPU stages require CUDA; main starts them explicitly')
    torch.hub.set_dir(str(ROOT / 'models/torch_cache/hub')); torch.backends.cudnn.benchmark = True
    protocol = verify_protocol()
    registry = load(ROOT / 'data/release/training_registry.json'); targets = load(CANDIDATE / 'targets.json')
    if args.stage in ('train', 'all'):
        for part in registry['parts']: train_part(part, registry, targets, protocol)
    if args.stage in ('calibrate', 'all'): calibrate(registry, targets, protocol)
    if args.stage in ('test', 'all'): test(registry, targets, protocol)
    verify_protocol()


if __name__ == '__main__': main()
