"""Can three photos (front + left/right oblique) replace the 12-view rig for depth features?

For val/test participants and each lighting session, depth from {C7 + two obliques} is
compared with the participant's 12-view reference (median over the other sessions, so the
same captures are never on both sides). Writes models/kface_v2/fewview_depth.json.
"""
import csv, json, statistics, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED
from facework.features import ROOT
from facework.multiview import POINTS, depth_features

SETS = {'front+-27deg (C5,C7,C9)': ['C5', 'C7', 'C9'], 'front+-40deg (C4,C7,C10)': ['C4', 'C7', 'C10'],
        'front+-13deg (C6,C7,C8)': ['C6', 'C7', 'C8'], 'five views (C4,C5,C7,C9,C10)': ['C4', 'C5', 'C7', 'C9', 'C10']}
SESSIONS = ['L1', 'L2', 'L3', 'L8', 'L12']
KEYS = ['nose_projection', 'nose_bridge', 'lip_projection', 'chin_projection']

def collect(refs, depth, cams, splits):
    rows = []
    for s, r in refs.items():
        if r['split'] not in splits:
            continue
        d = np.load(DERIVED / 'subjects' / f'{s}.npz'); pos = {k: i for i, k in enumerate(d['keys'])}
        for light in SESSIONS:
            idx = [pos.get(f'S001|{light}|E01|{c}|' + ('core' if c in ('C6', 'C7', 'C8') else 'multiview')) for c in cams]
            if any(i is None or d['status'][i] != 1 for i in idx):
                continue
            feats, _ = depth_features(d['landmarks'][idx][:, POINTS, :2].astype(float))
            others = [v for l, v in depth[s]['sessions'].items() if l != light]
            if len(others) >= 3:
                rows.append((feats, {k: statistics.median(o[k] for o in others) for k in KEYS}))
    return rows

def main():
    refs = {r['subject_id']: r for r in csv.DictReader((DERIVED / 'reference_labels.csv').open(encoding='utf-8-sig'))}
    depth = json.loads((DERIVED / 'depth.json').read_text(encoding='utf-8'))['subjects']
    summary = json.loads((DERIVED / 'reference_summary.json').read_text(encoding='utf-8'))
    out = {}
    for name, cams in SETS.items():
        train = collect(refs, depth, cams, ['train']); held = collect(refs, depth, cams, ['val', 'test'])
        entry = {'cameras': cams, 'captures_train': len(train), 'captures_heldout': len(held),
                 'residual_p50_p95': [round(float(np.quantile([f['residual'] for f, _ in held], q)), 4) for q in (.5, .95)]}
        for k in KEYS:
            a, b = np.polyfit([f[k] for f, _ in train], [r[k] for _, r in train], 1)  # few-view -> 12-view scale
            pred = np.array([a * f[k] + b for f, _ in held]); ref = np.array([r[k] for _, r in held])
            cuts = summary['features'][k]['cuts']
            entry[k] = {'calibration': [float(a), float(b)], 'pearson_r': round(float(np.corrcoef(pred, ref)[0, 1]), 3),
                        'mae_calibrated': round(float(np.mean(np.abs(pred - ref))), 4),
                        'class_agreement_calibrated': round(float(np.mean(np.searchsorted(cuts, pred) == np.searchsorted(cuts, ref))), 3),
                        'between_subject_sd': summary['features'][k]['between_subject_sd']}
        out[name] = entry
        print(name, json.dumps(entry))
    (ROOT / 'models/kface_v2').mkdir(parents=True, exist_ok=True)
    (ROOT / 'models/kface_v2/fewview_depth.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main()
