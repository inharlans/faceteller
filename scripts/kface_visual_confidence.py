"""Accuracy vs softmax threshold for every visual head (test participants, standard conditions).

Writes models/kface_v2/visual_confidence.json; the service uses the 0.7 row for its reliability grade.
"""
import sys, json, torch, numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch.utils.data import DataLoader
from facework.kface_training import OUT, PartNet, Crops, capture_rows, eval_transform, load_references, predict, SCHEMA

def main(root=None):
    from facework.kface_ensemble import checkpoints, predict_rows
    root = Path(root) if root else OUT
    refs, _ = load_references(); out = {}; cache = {}
    for part, k in [(f['part'], k) for k, f in SCHEMA['features'].items() if f['source'] == 'visual']:
        if part not in cache:
            rows = capture_rows(part, refs, ['test'], filt=lambda x: x['camera'] == 'C7' and x['light'] in ('L1', 'L2', 'L3', 'L8', 'L9', 'L12', 'L13', 'L22', 'L4', 'L14', 'L23'))  # service condition: frontal only
            _, C, meta = predict_rows(part, rows, refs, checkpoints(part, root), 'cuda'); cache[part] = (meta, rows, C)
        meta, rows, C = cache[part]
        j = meta['classification_features'].index(k); P = C[j]; classes = meta['classes'][k]
        g = defaultdict(list); t = {}
        for n, r in enumerate(rows):
            g[(r[0], r[1])].append(P[n]); t[(r[0], r[1])] = refs[r[0]][k]
        keys = [x for x in g if t[x] in classes]; prob = np.array([np.mean(g[x], 0) for x in keys]); truth = np.array([classes.index(t[x]) for x in keys])
        out[k] = []
        for th in (0, 0.5, 0.6, 0.7, 0.8):
            sel = prob.max(1) >= th
            out[k].append({'threshold': th, 'coverage': round(float(sel.mean()), 3), 'accuracy': round(float((prob.argmax(1) == truth)[sel].mean()), 3) if sel.any() else None,
                           'pred_dist': np.bincount(prob.argmax(1)[sel], minlength=len(classes)).tolist(), 'truth_dist': np.bincount(truth[sel], minlength=len(classes)).tolist()})
            print(k, out[k][-1], flush=True)
    (root / 'visual_confidence.json').write_text(json.dumps(out, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
