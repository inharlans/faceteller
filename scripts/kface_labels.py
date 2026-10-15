"""Subject-level reference labels for K-FACE (measurement-derived, schema morphology-kface-v2).

Reference value = median over S001/E01/C7 captures under 8 balanced bright lights, after
quality gates. Cut points = training-split tertiles (one value per participant). A label is
'borderline' when the 95% band of the participant's repeat-capture mean crosses a cut point.
Depth features come from data/derived/kface/depth.json (median over lighting sessions).
Visual labels (16 features) are merged from data/annotations/kface_visual_v3.tsv.
"""
import csv, hashlib, json, math, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED, subjects
from facework.features import ROOT

REF_LIGHTS = ['L1', 'L2', 'L3', 'L8', 'L9', 'L12', 'L13', 'L22']
SCHEMA = ROOT / 'data/label_schema_kface_v3.json'
VISUAL = ROOT / 'data/annotations/kface_visual_v4.tsv'  # three-rater consensus
FROZEN = DERIVED / 'split_frozen.json'  # keeps the test participants fixed across label revisions
SEED = 20260926

C7_PITCH = -5.0  # median head pitch reported for the frontal camera across participants (capture_probe.json)

def quality_ok(pose, blend, m):
    pitch, yaw, roll = pose
    return abs(yaw) < 8 and abs(pitch - C7_PITCH) < 16 and blend[1] < 0.5 and blend[2] < 0.5 and blend[0] < 0.3

def mouth_ok(m):
    return m['mouth_open'] < 0.06  # lips apart changes every mouth ratio; other features stay usable

def split_subjects(ids):
    """Participant-level 70/15/15 split stratified by the ID's last two digits (meaning not verified)."""
    rng = random.Random(SEED); groups = defaultdict(list)
    for s in ids:
        groups[s[-2:]].append(s)
    split = {}
    for g in sorted(groups):
        members = sorted(groups[g]); rng.shuffle(members); n = len(members)
        n_test = round(n * .15); n_val = round(n * .15)
        for i, s in enumerate(members):
            split[s] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
    return split

def multilabel_split(ids, values, visual, schema, candidates=30000):
    """Participant split 276/62/62 chosen by labels only (never model outputs).

    Stratification classes: every measured feature binned at all-participant P25/P75 plus every
    visual label. Random candidates (seeded) are scored by (1) classes with fewer than their
    attainable minimum (3, or fewer for rare classes) in val or test, then (2) the summed largest
    class-share gap between each evaluation split and train. The best candidate is kept.
    """
    cols = []
    for k, f in schema['features'].items():
        if f['source'] == 'visual':
            labs = [visual.get(s, {}).get(k, '') for s in ids]
            classes = f['classes']
        else:
            vals = np.array([values[s].get(k, np.nan) for s in ids], float); ok = ~np.isnan(vals)
            qs = f.get('quantiles', [0.25, 0.75]); cuts = [np.nanquantile(vals, q) for q in qs]
            labs = [f['classes'][int(np.searchsorted(cuts, v))] if o else '' for v, o in zip(vals, ok)]
            classes = f['classes']
        for c in classes:
            col = np.array([l == c for l in labs], float)
            if col.sum() > 0:
                cols.append((k, c, col))
    X = np.stack([c[2] for c in cols], 1)                      # participants x classes
    feat_of = [c[0] for c in cols]; total = X.sum(0)
    need = np.minimum(3, np.floor(total * 0.15 * 0.5))          # attainable minimum per evaluation split
    n = len(ids); rng = np.random.default_rng(SEED); best = None
    G = np.array([[a == b for b in feat_of] for a in feat_of], float)  # same-feature mask for class shares
    groups = defaultdict(list)
    for i, s in enumerate(ids):
        groups[s[-2:]].append(i)                               # keep the ID-suffix balance as well
    groups = [np.array(v) for _, v in sorted(groups.items())]
    for _ in range(candidates):
        assign = np.zeros(n, int)
        for m in groups:
            m = rng.permutation(m); k = round(len(m) * .15)
            assign[m[:k]] = 1; assign[m[k:2 * k]] = 2
        cnt = np.stack([X[assign == j].sum(0) for j in range(3)])
        short = int(((cnt[1] < need) | (cnt[2] < need)).sum())
        share = cnt / np.maximum(1, cnt @ G)
        gap = float(np.abs(share[1:] - share[0]).sum())
        if best is None or (short, gap) < best[0]:
            best = ((short, gap), assign.copy())
    assign = best[1]
    split = {s: ('train', 'val', 'test')[assign[i]] for i, s in enumerate(ids)}
    return split, {'version': 'multilabel-v3', 'seed': SEED, 'candidates': candidates, 'selection_uses': 'labels_only_no_model_outputs',
                   'stratification_classes': len(cols), 'classes_below_min': best[0][0], 'summed_share_gap': round(best[0][1], 3),
                   'sizes': dict(Counter(split.values()))}

def icc_1(mat):
    a = np.asarray(mat, float); n, k = a.shape
    ms_b = k * ((a.mean(1) - a.mean()) ** 2).sum() / (n - 1)
    ms_w = ((a - a.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))

def main():
    schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
    measured = {k: v for k, v in schema['features'].items() if v['source'] in ('landmark', 'image', 'multiview')}
    ids = subjects()
    depth = json.loads((DERIVED / 'depth.json').read_text(encoding='utf-8'))['subjects']
    values = {}; shots = {}; within = defaultdict(list); icc_rows = defaultdict(list); per_all = {}
    for s in ids:
        d = np.load(DERIVED / 'subjects' / f'{s}.npz'); metrics = json.loads((DERIVED / 'subjects' / f'{s}.json').read_text())
        pos = {k: i for i, k in enumerate(d['keys'])}
        per = defaultdict(list); used = 0
        for light in REF_LIGHTS:
            i = pos[f'S001|{light}|E01|C7|core']
            m = metrics[i]
            if d['status'][i] != 1 or not quality_ok(d['pose'][i], d['blend'][i], m):
                continue
            used += 1
            for k, f in measured.items():
                if f['part'] == 'mouth' and not mouth_ok(m):
                    continue
                if f['source'] in ('landmark', 'image') and m.get(f['metric']) is not None:
                    per[k].append(m[f['metric']])
                if f.get('per_side'):
                    for side in ('right', 'left'):
                        if m.get(f"{f['metric']}_{side}") is not None:
                            per[f'{k}_{side}'].append(m[f"{f['metric']}_{side}"])
        for k, f in measured.items():
            if f['source'] == 'multiview':
                per[k] = [v[f['metric']] for v in depth.get(s, {}).get('sessions', {}).values()]
        values[s] = {k: statistics.median(v) for k, v in per.items() if len(v) >= 3}
        shots[s] = {'reference_captures': used, **{f'n_{k}': len(v) for k, v in per.items()}}
        for k, v in per.items():
            if len(v) >= 3:
                within[k].append(statistics.stdev(v))
        per_all[s] = per
    visual = {}
    if VISUAL.exists():
        for r in csv.DictReader(VISUAL.open(encoding='utf-8'), delimiter='\t'):
            visual[r['subject_id']] = r
    if FROZEN.exists():
        split = json.loads(FROZEN.read_text(encoding='utf-8'))['split']; protocol = json.loads((DERIVED / 'split_protocol.json').read_text(encoding='utf-8'))
    else:
        split, protocol = multilabel_split(ids, values, visual, schema)
    (DERIVED / 'split_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=1), encoding='utf-8')
    for s in ids:
        for k, v in per_all[s].items():
            if len(v) >= 5 and split[s] == 'train':
                icc_rows[k].append(v[:5])
    features = {}
    for k, f in measured.items():
        train = [values[s][k] for s in ids if split[s] == 'train' and k in values[s]]
        qs = f.get('quantiles', [0.25, 0.75])  # measurement_definitions reference_policy: P25/P75
        cuts = [float(np.quantile(train, q)) for q in qs]
        sd = statistics.median(within[k])
        features[k] = {'cuts': cuts, 'quantiles': qs, 'classes': f['classes'], 'train_subjects': len(train),
                       'median_within_subject_sd': sd, 'between_subject_sd': statistics.pstdev(train),
                       'icc_1_1_train': round(icc_1(icc_rows[k]), 3) if len(icc_rows[k]) > 10 else None,
                       'train_median': statistics.median(train), 'train_p05_p95': [float(np.quantile(train, .05)), float(np.quantile(train, .95))],
                       # Aggregate quantile table (P5..P95) for percentile display; no per-person values leave this script.
                       'quantile_table': [[round(q, 2), float(np.quantile(train, q))] for q in np.arange(0.05, 0.951, 0.05)]}
    rows = []
    for s in ids:
        row = {'subject_id': s, 'split': split[s], 'id_suffix': s[-2:], **shots[s]}
        for k, f in features.items():
            v = values[s].get(k)
            if v is None:
                row[k] = ''; row[f'{k}_value'] = ''; continue
            n = shots[s].get(f'n_{k}', 1)
            band = 1.96 * f['median_within_subject_sd'] / math.sqrt(max(n, 1))
            cls = f['classes'][int(np.searchsorted(f['cuts'], v))]
            row[k] = 'borderline' if any(abs(v - c) < band for c in f['cuts']) else cls
            row[f'{k}_value'] = round(v, 6)
            if measured[k].get('per_side'):
                for side in ('right', 'left'):
                    sv = values[s].get(f'{k}_{side}')
                    row[f'{k}_{side}_value'] = round(sv, 6) if sv is not None else ''
        for k, f in schema['features'].items():
            if f['source'] == 'visual':
                label = visual.get(s, {}).get(k, '')
                row[k] = label if label in f['classes'] or label == 'unknown' else ''
        rows.append(row)
    out = DERIVED / 'reference_labels.csv'
    with out.open('w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {'version': schema['version'], 'participants': len(ids), 'split': dict(Counter(split.values())),
               'reference_lights': REF_LIGHTS, 'features': features,
               'label_counts': {k: {sp: dict(Counter(r[k] for r in rows if r['split'] == sp)) for sp in ['train', 'val', 'test']} for k in schema['features']},
               'labels_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    (DERIVED / 'reference_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, f in features.items():
        c = summary['label_counts'][k]['train']
        print(f"{k:20s} icc={f['icc_1_1_train']} within={f['median_within_subject_sd']:.4f} between={f['between_subject_sd']:.4f} cuts={[round(x, 3) for x in f['cuts']]} train={c}")

if __name__ == '__main__':
    main()
