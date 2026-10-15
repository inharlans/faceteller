"""Test-split evaluation of morphology-kface-v2 CNNs, broken down by capture condition.

For every measured feature the CNN is compared with the single-photo landmark measurement of
the same capture (the 'measurement baseline'), both against the participant reference label.
Borderline references are excluded from class metrics. Writes models/kface_v2/evaluation.json.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED
from facework.kface_training import (SCHEMA, OUT, PART_CROPS, PartNet, Crops, capture_rows, eval_transform,
                                     load_references, part_features, predict)

LIGHT_GROUPS = {'ref_bright': ['L1', 'L2', 'L3', 'L8', 'L9', 'L12', 'L13', 'L22'],
                'medium_front': ['L4', 'L14', 'L23'], 'side_light': ['L16', 'L17', 'L19', 'L20', 'L25', 'L26', 'L28', 'L29'],
                'dark': ['L5', 'L6', 'L7', 'L10', 'L11', 'L15', 'L18', 'L21', 'L24', 'L27', 'L30']}
CAMERA_GROUPS = {'front_C7': ['C7'], 'yaw13_C6_C8': ['C6', 'C8'], 'from_above_C15_C17': ['C15', 'C16', 'C17']}

def group_of(meta):
    lg = next(g for g, ls in LIGHT_GROUPS.items() if meta['light'] in ls)
    cg = next((g for g, cs in CAMERA_GROUPS.items() if meta['camera'] in cs), 'other')
    if meta['accessory'] != 'S001':
        return [f"accessory_{meta['accessory']}"]
    if meta['expression'] != 'E01':
        return [f"expression_{meta['expression']}"]
    out = [f'light:{lg}', f'camera:{cg}', 'all_core']
    if lg == 'ref_bright' and cg != 'other':
        out.append('standard(front,ref_bright)')  # 5 of 6 cameras here are C6/C8 (yaw 13) or C15-17 (pitch +18)
    if cg == 'front_C7' and lg in ('ref_bright', 'medium_front'):
        out.append('service(C7,bright)')  # captures that pass the service quality gate (|yaw|<=12, |pitch+5|<=16)
    return out

def is_reference_capture(meta):
    """Captures whose landmark medians define the reference (scripts/kface_labels.py REF_LIGHTS)."""
    return (meta['accessory'] == 'S001' and meta['expression'] == 'E01' and meta['camera'] == 'C7'
            and meta['purpose'] == 'core' and meta['light'] in LIGHT_GROUPS['ref_bright'])

def classify(values, cuts, classes):
    return [classes[int(np.searchsorted(cuts, v))] for v in values]

def main(split='test', model_root=None):
    """split: test | val | oof (out-of-fold over train+val, needs a cross-validated model_root)."""
    from facework.kface_ensemble import checkpoints, predict_rows, combined_meta
    root = Path(model_root) if model_root else OUT
    refs, summary = load_references(); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'schema': SCHEMA['version'], 'labels_sha256': summary['labels_sha256'], 'parts': {}, 'models': str(root)}
    folds = json.loads((root / 'folds.json').read_text(encoding='utf-8'))['assignment'] if (root / 'folds.json').exists() else None
    for part in PART_CROPS:
        if not checkpoints(part, root):
            continue
        filt = lambda m: m['camera'] in ('C6', 'C7', 'C8', 'C15', 'C16', 'C17')
        if split == 'oof':
            rows = capture_rows(part, refs, ['train', 'val'], purposes=('core', 'expression', 'accessory'), filt=filt)
            R = None; C = None; metas = []
            for f in sorted(set(folds.values())):
                idx = [n for n, r in enumerate(rows) if folds[r[0]] == f]
                Rf, Cf, mf = predict_rows(part, [rows[n] for n in idx], refs, checkpoints(part, root, f), device); metas.append(mf)
                if R is None:
                    R = np.zeros((len(rows), Rf.shape[1])); C = [np.zeros((len(rows), c.shape[1])) for c in Cf]
                R[idx] = Rf
                for j in range(len(Cf)):
                    C[j][idx] = Cf[j]
            meta = combined_meta(metas)  # margins averaged over every fold, not fold 0 only
        else:
            rows = capture_rows(part, refs, [split], purposes=('core', 'expression', 'accessory'), filt=filt)
            R, C, meta = predict_rows(part, rows, refs, checkpoints(part, root), device)
        reg, cls = meta['regression_features'], meta['classification_features']
        # Landmark measurement of the same capture (baseline).
        metrics_cache = {}
        def measured(s, i, key):
            if s not in metrics_cache:
                metrics_cache[s] = json.loads((DERIVED / 'subjects' / f'{s}.json').read_text())
            m = metrics_cache[s][i]
            return None if m is None else m.get(key)
        groups = defaultdict(list)
        for n, r in enumerate(rows):
            for g in group_of(r[4]):
                groups[g].append(n)
        part_report = {'test_participants': len({r[0] for r in rows}), 'captures': len(rows), 'features': {}}
        for j, k in enumerate(reg):
            f = SCHEMA['features'][k]; cuts = meta['cuts'][k]; classes = meta['classes'][k]; mu, sd = meta['norm'][k]
            feat = {}
            for g, idx in sorted(groups.items()):
                ref_vals, cnn_vals, lm_vals, ref_cls = [], [], [], []
                for n in idx:
                    s, i, crop, side, cmeta = rows[n]
                    key = f'{k}_{side}_value' if f.get('per_side') and side else f'{k}_value'
                    if refs[s].get(key) in ('', None):
                        continue
                    ref_vals.append(float(refs[s][key])); cnn_vals.append(R[n, j] * sd + mu)
                    mk = f"{f['metric']}_{side}" if f.get('per_side') and side else f.get('metric')
                    # the baseline must not be scored on the very captures whose median is the reference
                    lm_vals.append(measured(s, i, mk) if f['source'] != 'multiview' and not is_reference_capture(cmeta) else None)
                    ref_cls.append(refs[s][k])
                if len(ref_vals) < 10:
                    continue
                ref_vals = np.array(ref_vals); cnn_vals = np.array(cnn_vals)
                keep = [q for q, c in enumerate(ref_cls) if c != 'borderline']
                truth = [classify([ref_vals[q]], cuts, classes)[0] for q in keep]
                pred = classify(cnn_vals[keep], cuts, classes)
                margin = meta['uncertainty_margin'].get(k) or 0
                confident = [q for q in range(len(keep)) if min(abs(cnn_vals[keep][q] - c) for c in cuts) >= margin]
                entry = {'n': len(ref_vals), 'n_class_eval': len(keep),
                         'cnn_pearson_r': round(float(np.corrcoef(cnn_vals, ref_vals)[0, 1]), 3),
                         'cnn_mae_sd_units': round(float(np.mean(np.abs(cnn_vals - ref_vals)) / sd), 3),
                         'cnn_accuracy': round(accuracy_score(truth, pred), 3),
                         'cnn_macro_f1': round(f1_score(truth, pred, labels=classes, average='macro', zero_division=0), 3),
                         'cnn_confident_rate': round(len(confident) / max(len(keep), 1), 3),
                         'cnn_confident_accuracy': round(accuracy_score([truth[q] for q in confident], [pred[q] for q in confident]), 3) if confident else None}
                lm_ok = [q for q in keep if lm_vals[q] is not None]
                if lm_ok and f['source'] != 'multiview':
                    lm = np.array([lm_vals[q] for q in lm_ok]); lm_truth = [classify([ref_vals[q]], cuts, classes)[0] for q in lm_ok]
                    entry['landmark_pearson_r'] = round(float(np.corrcoef(lm, ref_vals[lm_ok])[0, 1]), 3)
                    entry['landmark_accuracy'] = round(accuracy_score(lm_truth, classify(lm, cuts, classes)), 3)
                    entry['landmark_macro_f1'] = round(f1_score(lm_truth, classify(lm, cuts, classes), labels=classes, average='macro', zero_division=0), 3)
                if g in ('standard(front,ref_bright)', 'service(C7,bright)'):
                    entry['confusion_matrix'] = {'labels': classes, 'matrix': confusion_matrix(truth, pred, labels=classes).tolist()}
                feat[g] = entry
            part_report['features'][k] = feat
        for j, k in enumerate(cls):
            classes = meta['classes'][k]; feat = {}
            for g, idx in sorted(groups.items()):
                keep = [n for n in idx if refs[rows[n][0]].get(k) in classes]
                if len(keep) < 10:
                    continue
                truth = [refs[rows[n][0]][k] for n in keep]; pred = [classes[int(C[j][n].argmax())] for n in keep]
                train_labels = [refs[r][k] for r in refs if refs[r]['split'] == 'train' and refs[r][k] in classes]
                majority = max(classes, key=train_labels.count)
                feat[g] = {'n': len(keep), 'accuracy': round(accuracy_score(truth, pred), 3),
                           'macro_f1': round(f1_score(truth, pred, labels=classes, average='macro', zero_division=0), 3),
                           'majority_baseline_accuracy': round(accuracy_score(truth, [majority] * len(truth)), 3)}
                if g == 'standard(front,ref_bright)':
                    feat[g]['confusion_matrix'] = {'labels': classes, 'matrix': confusion_matrix(truth, pred, labels=classes).tolist()}
            part_report['features'][k] = feat
        report['parts'][part] = part_report
        print(part, json.dumps({k: v.get('standard(front,ref_bright)') for k, v in part_report['features'].items()}, ensure_ascii=False)[:1500], flush=True)
    report['caveats'] = ['참조 레이블은 같은 MediaPipe 측정의 다회 촬영 중앙값이다. 인간 전문가 정답이 아니다.',
                         '시각 레이블은 AI 단일 검수자 기준이다.',
                         '시험 참여자는 학습·검증과 겹치지 않는다. 같은 참여자의 여러 촬영은 서로 독립이 아니다.']
    report['split'] = split
    root.mkdir(parents=True, exist_ok=True)
    (root / ('evaluation.json' if split == 'test' else f'evaluation_{split}.json')).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'test', sys.argv[2] if len(sys.argv) > 2 else None)
