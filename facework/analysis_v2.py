"""Photo -> morphology features -> nearest literature types -> traditional reading (v2).

Pipeline for one frontal photo (optionally with left/right oblique photos for depth):
MediaPipe landmarks -> quality checks -> scale-normalised crops -> part CNNs (models/kface_v2)
-> feature classes against the K-FACE cohort P25/P75 cut points, with an uncertainty band
-> literature type rules (data/type_rules_v3.json) -> reading texts (interpretation_service_v1).
Nothing is written to disk.
"""
import json, math
from functools import lru_cache
import numpy as np
from .features import ROOT, detector
from .morphometrics import measure, normalized_crops
from .multiview import POINTS, depth_features

SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
MODEL_DIR = ROOT / 'models/kface_v2'      # single checkpoints, occlusion model, depth calibration
ENSEMBLE_DIR = ROOT / 'models/kface_v4'   # cross-validated fold ensembles (used when present)
EVAL_DIR = ENSEMBLE_DIR if (ENSEMBLE_DIR / 'evaluation.json').exists() else MODEL_DIR
C7_PITCH = -5.0
VISUAL_CONFIDENCE = 0.7  # test accuracy 0.68-0.73 at 64-71% coverage (models/kface_v2/visual_confidence.json)
QUALITY = {'max_abs_yaw_deg': 12, 'max_abs_pitch_offset_deg': 16, 'min_face_width_px': 150,
           'max_smile': 0.3, 'max_blink': 0.5, 'max_mouth_open': 0.06,
           'max_brow_down': 0.2}  # K-FACE browDown: neutral P97 ~0.2, frown (E03) median 0.20

@lru_cache(maxsize=1)
def resources():
    import torch
    from .kface_training import PartNet, eval_transform, PART_CROPS
    summary = json.loads((ROOT / 'data/derived/kface/reference_summary.json').read_text(encoding='utf-8')) \
        if (ROOT / 'data/derived/kface/reference_summary.json').exists() else None
    from .kface_ensemble import checkpoints, load as load_ck, combined_meta
    models = {}
    for part in PART_CROPS:
        paths = checkpoints(part, ENSEMBLE_DIR) if (ENSEMBLE_DIR / 'evaluation.json').exists() else checkpoints(part, MODEL_DIR)
        if paths:
            members = [load_ck(pth) for pth in paths]
            models[part] = ([m for m, _ in members], combined_meta([mt for _, mt in members]), eval_transform(part))
    rules = json.loads((ROOT / 'data/type_rules_v3.json').read_text(encoding='utf-8'))
    service = json.loads((ROOT / 'data/interpretation_service_v1.json').read_text(encoding='utf-8'))
    catalog = json.loads((ROOT / 'data/interpretation_catalog_v1.json').read_text(encoding='utf-8'))
    load = lambda name: json.loads((EVAL_DIR / name).read_text(encoding='utf-8')) if (EVAL_DIR / name).exists() else None
    # method selection comes from held-out non-test data: out-of-fold for ensembles, the val split otherwise
    evaluation, evaluation_val = load('evaluation.json'), load('evaluation_oof.json') or load('evaluation_val.json')
    occlusion = None
    if (MODEL_DIR / 'occlusion.pt').exists():
        ck = torch.load(MODEL_DIR / 'occlusion.pt', map_location='cpu', weights_only=False)
        m = PartNet(ck['meta']['arch'], 0, [len(v) for v in ck['meta']['classes'].values()], pretrained=False)
        m.load_state_dict(ck['state_dict']); m.eval(); occlusion = (m, ck['meta'], eval_transform('face'))
    return {'models': models, 'summary': summary, 'rules': rules, 'service': service,
            'catalog': {r['record_id']: r for r in catalog['records']}, 'evaluation': evaluation, 'occlusion': occlusion,
            'landmark_preferred': landmark_preferred(evaluation_val)}

STD = 'standard(front,ref_bright)'
SERVICE = 'service(C7,bright)'  # frontal captures that pass quality_flags (scripts/kface_evaluate.py)

def landmark_preferred(ev):
    """Features where the photo's own landmark ratio beat the CNN on the validation split (not test)."""
    out = set()
    for pr in (ev or {}).get('parts', {}).values():
        for k, g in pr['features'].items():
            v = g.get(SERVICE) or g.get(STD, {})
            if v.get('landmark_accuracy') is not None and v['landmark_accuracy'] > v['cnn_accuracy'] + 0.02:
                out.add(k)
    return out

@lru_cache(maxsize=1)
def landmarker():
    return detector()

def detect(image):
    """Landmarks (N x 3 px), pose (pitch, yaw, roll), blendshape summary; None when not exactly one face."""
    import mediapipe as mp
    from scipy.spatial.transform import Rotation
    arr = np.ascontiguousarray(np.asarray(image.convert('RGB')))
    h, w = arr.shape[:2]
    r = landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if len(r.face_landmarks) != 1:
        return None, ('no_face' if not r.face_landmarks else 'multiple_faces')
    p = np.array([[t.x * w, t.y * h, t.z * w] for t in r.face_landmarks[0]], np.float32)
    pitch, yaw, roll = Rotation.from_matrix(np.asarray(r.facial_transformation_matrixes[0])[:3, :3]).as_euler('xyz', degrees=True)
    bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
    return {'landmarks': p, 'pitch': float(pitch), 'yaw': float(yaw), 'roll': float(roll),
            'smile': max(bs['mouthSmileLeft'], bs['mouthSmileRight']), 'blink': max(bs['eyeBlinkLeft'], bs['eyeBlinkRight']),
            'brow_down': max(bs['browDownLeft'], bs['browDownRight'])}, None

def percentile(value, table):
    """Linear interpolation inside the cohort quantile table; clipped to 5..95."""
    if not table or value is None:
        return None
    qs = [q for q, _ in table]; vs = [v for _, v in table]
    if value <= vs[0]:
        return 5
    if value >= vs[-1]:
        return 95
    return int(round(100 * float(np.interp(value, vs, qs))))

def quality_flags(d, metrics, image):
    """(message, scope) pairs. Scope 'all' withholds every result; a part scope withholds that part only."""
    flags = []
    if abs(d['yaw']) > QUALITY['max_abs_yaw_deg']:
        flags.append(('얼굴이 옆으로 돌아가 있습니다. 카메라를 정면으로 바라봐 주세요.', 'all'))
    if abs(d['pitch'] - C7_PITCH) > QUALITY['max_abs_pitch_offset_deg']:
        flags.append(('고개가 위나 아래로 기울어 있습니다. 카메라를 눈높이에 맞춰 주세요.', 'all'))
    if metrics['face_width_px'] < QUALITY['min_face_width_px']:
        flags.append(('얼굴이 작게 찍혔습니다. 더 가까이(얼굴 폭 150픽셀 이상) 촬영해 주세요.', 'all'))
    gray = np.asarray(image.convert('L'), dtype=np.float32)
    if gray.mean() < 45:
        flags.append(('사진이 어둡습니다. 밝고 고른 조명에서 촬영해 주세요.', 'all'))
    if d['smile'] > QUALITY['max_smile']:
        flags.append(('웃는 표정은 입과 눈 모양을 바꿉니다. 입·눈 결과는 보류했습니다.', 'mouth'))
        flags.append(('', 'eye'))
    elif metrics['mouth_open'] > QUALITY['max_mouth_open']:
        flags.append(('입술이 벌어져 있어 입 결과는 보류했습니다. 입을 편하게 다문 사진을 사용해 주세요.', 'mouth'))
    if d['brow_down'] > QUALITY['max_brow_down']:
        flags.append(('눈썹을 찌푸린 표정이라 눈썹 결과는 보류했습니다. 이마와 눈썹에 힘을 빼고 촬영해 주세요.', 'brow'))
    if d['blink'] > QUALITY['max_blink']:
        flags.append(('눈을 감았거나 가늘게 뜬 사진이라 눈 결과는 보류했습니다.', 'eye'))
    return flags

def classify(value, cuts, classes, margin):
    idx = int(np.searchsorted(cuts, value))
    near = min(abs(value - c) for c in cuts) < (margin or 0)
    return classes[idx], ('borderline' if near else 'confident')

def occlusion_check(crops):
    """Eyewear/hat probabilities from the K-FACE accessory model; None when the model is absent."""
    import torch
    occ = resources()['occlusion']
    if occ is None:
        return None
    model, meta, tf = occ
    with torch.inference_mode():
        _, cs = model(tf(crops['face']).unsqueeze(0))
    return {name: dict(zip(classes, [round(float(v), 3) for v in cs[j].softmax(1)[0]])) for j, (name, classes) in enumerate(meta['classes'].items())}

def run_models(image, p, crops=None):
    import torch
    res = resources(); crops = crops or normalized_crops(image, p[:, :2]); out = {}
    for part, (members, meta, tf) in res['models'].items():
        batch = torch.stack([tf(crops[c]) for c in meta['crops']])
        with torch.inference_mode():
            outs = [m(batch) for m in members]
        r = torch.stack([o[0] for o in outs]).mean(0)
        cs = [torch.log(torch.stack([o[1][j].softmax(1) for o in outs]).mean(0)) for j in range(len(outs[0][1]))]  # mean probabilities
        for j, k in enumerate(meta['regression_features']):
            mu, sd = meta['norm'][k]
            vals = (r[:, j] * sd + mu).tolist()
            out[k] = {'values': vals, 'sides': [c.split('_')[1] if '_' in c else None for c in meta['crops']]}
        for j, k in enumerate(meta['classification_features']):
            prob = cs[j].softmax(1).mean(0)  # left/right crops averaged
            out[k] = {'prob': prob.tolist(), 'classes': meta['classes'][k]}
        out.setdefault('_meta', {})[part] = meta
    return out

def feature_table(pred, metrics, depth=None):
    res = resources(); summary = res['summary']; rows = {}
    metas = pred.get('_meta', {})
    for k, f in SCHEMA['features'].items():
        meta = metas.get(f['part'])
        if meta is None:
            continue
        entry = {'feature': k, 'part': f['part'], 'ko': f['ko'], 'source': f['source']}
        if f['source'] == 'visual':
            if k not in pred:
                continue
            prob = pred[k]['prob']; i = int(np.argmax(prob)); cls = pred[k]['classes'][i]
            entry.update(label=cls, label_ko=f['classes_ko'][f['classes'].index(cls)], probability=round(prob[i], 3),
                         status='confident' if prob[i] >= VISUAL_CONFIDENCE else 'borderline')
        else:
            if k == 'eye_asymmetry':
                vals = pred.get('eye_roundness', {}).get('values')
                value = abs(math.log(max(vals[0], 1e-3) / max(vals[1], 1e-3))) if vals else None
                margin = None
                if value is None:
                    continue
            elif k in pred:
                value = float(np.mean(pred[k]['values'])); margin = meta['uncertainty_margin'].get(k)
            else:
                continue
            method = 'cnn_roundness_derived' if k == 'eye_asymmetry' else 'cnn_single_photo'
            if k in resources()['landmark_preferred'] and metrics.get(f.get('metric')) is not None:
                value = metrics[f['metric']]; method = 'landmark_single_photo'
                margin = 0.5 * summary['features'][k]['median_within_subject_sd']
            if depth and k in depth:
                value = depth[k]; method = 'multiview_photos'; margin = depth['margins'][k]
            cuts = summary['features'][k]['cuts']; cls, status = classify(value, cuts, f['classes'], margin if margin is not None else 0.25 * summary['features'][k]['median_within_subject_sd'])
            if k in SCHEMA.get('reliability_notes', {}) and status == 'confident':
                status = 'confident_low_reliability_feature'
            entry.update(value=round(value, 4), label=cls, label_ko=f['classes_ko'][f['classes'].index(cls)], status=status, method=method,
                         cohort_percentile=percentile(value, summary['features'][k].get('quantile_table')),
                         landmark_value=round(metrics[f['metric']], 4) if f.get('metric') in metrics and metrics[f['metric']] is not None else None)
            if f['source'] == 'multiview' and method == 'cnn_single_photo':
                entry['note'] = '정면 사진 한 장에서 추정한 깊이입니다. 좌·우 사선 사진을 함께 올리면 다중 시점으로 측정합니다.'
        entry.update(reliability(k, entry.get('method')))
        if entry.get('method') == 'multiview_photos':
            cal = json.loads((MODEL_DIR / 'fewview_depth.json').read_text(encoding='utf-8'))[depth['calibration_set']][k]
            acc = cal['class_agreement_calibrated']
            entry.update(test_accuracy=acc, reliability='A' if acc >= 0.85 else 'B' if acc >= 0.7 else 'C', used_for_types=True)
        rows[k] = entry
    return rows

def reliability(k, method=None):
    """Grade from held-out test accuracy of the method actually used.

    A >= 0.85, B >= 0.70, C otherwise; D when a visual head's overall test accuracy does not beat
    always answering the most common class by 5 points. Features graded D, or below 0.60, are shown but not used for
    literature-type matching.
    """
    res = resources(); ev = res['evaluation']; f = SCHEMA['features'][k]; acc = None; base = None; overall = None
    if method == 'cnn_roundness_derived':
        # the evaluated eye_asymmetry head is not what the service shows; this estimator was never tested
        return {'test_accuracy': None, 'reliability': None, 'used_for_types': False}
    g = (ev or {}).get('parts', {}).get(f['part'], {}).get('features', {}).get(k, {}).get(SERVICE) or ((ev or {}).get('parts', {}).get(f['part'], {}).get('features', {}).get(k, {}).get(STD, {}))
    if method == 'landmark_single_photo':
        acc = g.get('landmark_accuracy')
    elif f['source'] == 'visual':
        base = g.get('majority_baseline_accuracy'); acc = g.get('accuracy'); overall = acc
        vc = EVAL_DIR / 'visual_confidence.json'
        if vc.exists():
            row = next((r for r in json.loads(vc.read_text(encoding='utf-8')).get(k, []) if r['threshold'] == VISUAL_CONFIDENCE), None)
            acc = row['accuracy'] if row else acc
    else:
        acc = g.get('cnn_confident_accuracy') or g.get('cnn_accuracy')
    if acc is None:
        return {'test_accuracy': None, 'reliability': None, 'used_for_types': True}
    # D: the whole-coverage accuracy does not beat answering the most common class; filtering to
    # confident predictions alone can hide a head that only ever outputs the majority class
    if base is not None and (overall or 0) < base + 0.05:
        grade = 'D'
    elif k in SCHEMA.get('reliability_notes', {}) and f['source'] != 'visual':
        grade = 'C'
    else:
        grade = 'A' if acc >= 0.85 else 'B' if acc >= 0.7 else 'C'
    return {'test_accuracy': round(acc, 3), 'reliability': grade, 'used_for_types': grade != 'D' and acc >= 0.6}

def match_types(features):
    """Literature types whose observable conditions agree with confident feature classes."""
    res = resources(); service = {r['record_id']: r for r in res['service']['records']}; out = []
    for rule in res['rules']['records']:
        if rule['rule_coverage'] == 'none' or not rule['conditions'] or rule.get('photo_suggestion_hold'):
            continue  # hold: stigmatising readings are not attached to photos (type_rules_v3)
        met = []; missed = []; unknown = []
        for c in rule['conditions']:
            f = features.get(c['feature'])
            if f is None:
                unknown.append(c); continue
            if f['status'] in ('borderline', 'withheld') or not f.get('used_for_types', True):
                unknown.append(c)
            elif f['label'] in c['values']:
                met.append(c)
            else:
                missed.append(c)
        if missed or not met:
            continue
        weight = sum(c['weight'] for c in met)
        status = 'close' if not unknown and len(met) >= rule['min_conditions_to_suggest'] else 'partial'
        s = service[rule['record_id']]
        out.append({'record_id': rule['record_id'], 'name_ko': rule['name_ko'], 'name_zh': s['name_zh'], 'part': rule['part'],
                    'match': status, 'weight': round(weight, 2), 'met': [{'feature': c['feature'], 'values': c['values'], 'evidence': c['evidence']} for c in met],
                    'unconfirmed': [{'feature': c['feature'], 'evidence': c['evidence']} for c in unknown],
                    'unobservable': rule['unobservable_conditions'], 'rule_coverage': rule['rule_coverage'],
                    'overall': s['overall'], 'one_line': s['one_line'], 'hanja_meaning': s['hanja_meaning'],
                    'service_text': s['service_text'], 'easy_explanation': s['easy_explanation'], 'domains': s.get('domains_service', s['domains']), 'domains_full': s['domains'],
                    'translation_ko': s['translation_ko'], 'caution': s['caution'],
                    'original': res['catalog'].get(rule['record_id'], {}).get('original_section'),
                    'source_url': res['catalog'].get(rule['record_id'], {}).get('source_url')})
    rank = {'close': 0, 'partial': 1}
    out.sort(key=lambda t: (rank[t['match']], -len(t['met']), -t['weight']))
    return out

def group_types(types):
    """Group types of one part that were matched on exactly the same photo evidence.

    Many literature types share a single observable condition (e.g. seven eye types need only
    'large eyes'); the photo cannot tell them apart, so they are shown together, not asserted.
    Groups with more matched conditions come first.
    """
    groups = {}
    for t in types:
        key = (t['part'], t['match'], tuple(sorted((c['feature'], tuple(c['values'])) for c in t['met'])))
        groups.setdefault(key, []).append(t)
    rank = {'close': 0, 'partial': 1}
    out = [{'part': part, 'match': match, 'evidence': [{'feature': f, 'values': list(v)} for f, v in sig],
            'conditions_met': len(sig), 'record_ids': [t['record_id'] for t in ts],
            'overall_mix': sorted({t['overall'] for t in ts})}
           for (part, match, sig), ts in groups.items()]
    out.sort(key=lambda g: (g['part'], rank[g['match']], -g['conditions_met'], len(g['record_ids'])))
    return out

def oblique_depth(front, sides):
    """Depth from a frontal photo plus oblique photos, mapped to the 12-view scale.

    Calibration lines were fitted on training participants (scripts/kface_fewview_eval.py);
    the set is chosen from the mean absolute yaw of the oblique photos.
    """
    views = [front] + sides
    xy = np.stack([v['landmarks'][POINTS, :2] for v in views]).astype(float)
    feats, _ = depth_features(xy)
    path = MODEL_DIR / 'fewview_depth.json'
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding='utf-8'))
    yaw = float(np.mean([abs(s['yaw']) for s in sides]))
    key = 'front+-40deg (C4,C7,C10)' if yaw >= 33 else 'front+-27deg (C5,C7,C9)'
    out = {'residual': feats['residual'], 'calibration_set': key}
    out['margins'] = {}
    for k in ('nose_projection', 'nose_bridge', 'lip_projection', 'chin_projection'):
        if k not in cal[key]:
            continue
        a, b = cal[key][k]['calibration']; out[k] = a * feats[k] + b
        out['margins'][k] = 0.6 * cal[key][k]['mae_calibrated']  # held-out few-view error, not rig repeatability
    return out

def analyze(image, side_images=()):
    d, reason = detect(image)
    if d is None:
        return {'status': 'reject', 'reason': reason}
    p = d['landmarks']; metrics = measure(image, p[:, :2])
    flags = quality_flags(d, metrics, image)
    crops = normalized_crops(image, p[:, :2])
    occ = occlusion_check(crops)
    if occ:
        # Eyewear also rests on the nasal bridge: K-FACE test accuracy of nose depth fell from 0.75 to 0.53-0.64.
        if occ['eyewear']['sunglasses'] > 0.5:
            flags += [('선글라스가 감지되어 눈·눈썹·코 높이 결과는 보류했습니다. 벗고 촬영해 주세요.', 'eye'), ('', 'brow'), ('', 'nose_depth')]
        elif occ['eyewear']['glasses'] > 0.5:
            flags += [('안경이 감지되어 눈·눈썹·코 높이 결과는 보류했습니다. 안경을 벗고 촬영해 주세요.', 'eye'), ('', 'brow'), ('', 'nose_depth')]
        if occ['hat']['yes'] > 0.5:
            flags.append(('모자가 감지되었습니다. 눈썹이 가려질 수 있어 눈썹 결과는 보류했습니다.', 'brow'))
    blocked = {scope for _, scope in flags}
    depth = None; depth_note = None
    if side_images:
        sides = []
        for im in side_images:
            s, why = detect(im)
            if s is None:
                depth_note = f'사선 사진에서 얼굴을 찾지 못했습니다({why}).'; break
            if abs(s['yaw']) < 20 or abs(s['yaw']) > 55:
                depth_note = '사선 사진은 좌우로 약 40도(20~50도) 돌린 얼굴이어야 합니다.'; break
            sides.append(s)
        else:
            depth = oblique_depth(d, sides)
            if depth is not None and depth['residual'] > 0.04:  # K-FACE held-out p95 was 0.027
                depth_note = '사진 간 표정·자세 차이가 커서 다중 시점 깊이를 사용하지 않았습니다.'; depth = None
    pred = run_models(image, p, crops)
    features = feature_table(pred, metrics, depth)
    for f in features.values():
        if 'all' in blocked or f['part'] in blocked or ('nose_depth' in blocked and f['source'] == 'multiview'):
            f['status'] = 'withheld'
    types = [] if 'all' in blocked else [t for t in match_types(features) if t['part'] not in blocked]
    groups = group_types(types)
    res = resources()
    return {'status': 'ok', 'quality_flags': [m for m, _ in flags if m], 'withheld_parts': sorted(b for b in blocked if b != 'nose_depth'), 'pose': {k: round(d[k], 1) for k in ('yaw', 'pitch', 'roll')},
            'features': list(features.values()), 'types': types, 'type_groups': groups,
            'depth': {'used': depth is not None, 'note': depth_note}, 'occlusion': occ,
            'types_withheld_reason': '촬영 조건을 먼저 맞춰 주세요. 자세·조명·크기 조건이 맞지 않는 사진으로는 관상 유형을 제시하지 않습니다.' if 'all' in blocked else None,
            'disclaimer': res['service']['tone_policy']['display_disclaimer'],
            'model_status': 'K-FACE 400명 기반 실험 모델. 측정 레이블은 다회 촬영 랜드마크 중앙값, 시각 레이블은 AI 단일 검수자. 인간 전문가 검수 전.'}
