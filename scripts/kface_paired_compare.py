"""Paired, participant-level comparison of two model sets on the same test participants.

Both sets are scored against the *current* reference labels (three-rater consensus for visual features),
on standard captures (S001/E01, frontal and upper cameras, bright frontal lights). For every feature each
test participant gets an accuracy over his/her captures (borderline references excluded); the paired
difference is bootstrapped over participants (10,000 resamples). A pooled estimate averages each
participant's differences over all features before resampling, which respects that features of one
person are correlated. Writes models/kface_v4/paired_comparison.json with per-participant accuracies.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.kface_training import SCHEMA, PART_CROPS, capture_rows, load_references
from facework.kface_ensemble import checkpoints, predict_rows

STD_LIGHTS = ('L1', 'L2', 'L3', 'L8', 'L9', 'L12', 'L13', 'L22')   # reference lights (medians come from C7 here)
NONREF_FRONT = ('L4', 'L14', 'L23')                                  # frontal lights never used for the reference

def per_person(part, rows, R, C, meta, refs):
    out = {}
    for j, k in enumerate(meta['regression_features']):
        f = SCHEMA['features'][k]; cuts = meta['cuts'][k]; mu, sd = meta['norm'][k]; acc = defaultdict(list)
        for n, (s, i, crop, side, _) in enumerate(rows):
            if refs[s].get(k) in ('borderline', '', None, 'unknown'):
                continue
            key = f'{k}_{side}_value' if f.get('per_side') and side else f'{k}_value'
            if refs[s].get(key) in ('', None):
                continue
            truth = int(np.searchsorted(cuts, float(refs[s][key]))); pred = int(np.searchsorted(cuts, R[n, j] * sd + mu))
            acc[s].append(truth == pred)
        out[k] = {s: float(np.mean(v)) for s, v in acc.items()}
    for j, k in enumerate(meta['classification_features']):
        classes = meta['classes'][k]; acc = defaultdict(list)
        for n, (s, *_rest) in enumerate(rows):
            if refs[s].get(k) in classes:
                acc[s].append(classes[int(C[j][n].argmax())] == refs[s][k])
        out[k] = {s: float(np.mean(v)) for s, v in acc.items()}
    return out

_metrics = {}
def metrics_of(s):
    if s not in _metrics:
        _metrics[s] = json.loads((ROOT / 'data/derived/kface/subjects' / f'{s}.json').read_text())
    return _metrics[s]

def boot(d, rng, n=10000):
    d = np.asarray(d); idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975)), float((m <= 0).mean())

def main(old='models/kface_v2', new='models/kface_v4'):
    refs, _ = load_references(); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scores = {'old': {}, 'new': {}}
    for part in PART_CROPS:
        # service condition: frontal camera only (C6/C8/C15-17 fail the service pose gate)
        rows = capture_rows(part, refs, ['test'], filt=lambda m: m['camera'] == 'C7' and m['light'] in STD_LIGHTS + NONREF_FRONT)
        for tag, root in (('old', ROOT / old), ('new', ROOT / new)):
            R, C, meta = predict_rows(part, rows, refs, checkpoints(part, root), device)
            scores[tag].update(per_person(part, rows, R, C, meta, refs))
        # third arm: the photo's own landmark ratio, scored only on non-reference lights (no circularity)
        lm = {}
        for k in [x for x in scores['new'] if SCHEMA['features'].get(x, {}).get('source') == 'landmark' and SCHEMA['features'][x]['part'] == part]:
            f = SCHEMA['features'][k]; cuts = json.loads((ROOT / 'data/derived/kface/reference_summary.json').read_text(encoding='utf-8'))['features'][k]['cuts']; acc = defaultdict(list)
            for s, i, crop, side, meta in rows:
                if meta['light'] not in NONREF_FRONT or refs[s].get(k) in ('borderline', '', None, 'unknown'):
                    continue
                key = f'{k}_{side}_value' if f.get('per_side') and side else f'{k}_value'
                mk = f"{f['metric']}_{side}" if f.get('per_side') and side else f['metric']
                mv = metrics_of(s)[i]
                if refs[s].get(key) in ('', None) or mv is None or mv.get(mk) is None:
                    continue
                acc[s].append(int(np.searchsorted(cuts, float(refs[s][key]))) == int(np.searchsorted(cuts, mv[mk])))
            lm[k] = {s: float(np.mean(v)) for s, v in acc.items()}
        scores.setdefault('landmark', {}).update(lm)
    rng = np.random.default_rng(20260929); feats = {}; pooled = defaultdict(list)
    for k in scores['new']:
        if k not in scores['old']:
            continue
        people = sorted(set(scores['old'][k]) & set(scores['new'][k]))
        diff = [scores['new'][k][s] - scores['old'][k][s] for s in people]
        mean, lo, hi, p_le0 = boot(diff, rng)
        feats[k] = {'source': SCHEMA['features'][k]['source'], 'participants': len(people),
                    'old': round(float(np.mean([scores['old'][k][s] for s in people])), 3), 'new': round(float(np.mean([scores['new'][k][s] for s in people])), 3),
                    'diff': round(mean, 3), 'ci95': [round(lo, 3), round(hi, 3)], 'boot_p_diff_le_0': round(p_le0, 4)}
        for s, d in zip(people, diff):
            pooled[(SCHEMA['features'][k]['source'] == 'visual', s)].append(d)
        if k in scores.get('landmark', {}):
            pl = sorted(set(scores['landmark'][k]) & set(scores['new'][k]))
            dl = [scores['new'][k][s] - scores['landmark'][k][s] for s in pl]
            m2, l2, h2, p2 = boot(dl, rng)
            feats[k]['landmark'] = round(float(np.mean([scores['landmark'][k][s] for s in pl])), 3)
            feats[k]['new_minus_landmark'] = round(m2, 3); feats[k]['ci95_vs_landmark'] = [round(l2, 3), round(h2, 3)]
    summary = {}
    for vis, name in ((False, 'measured_and_depth'), (True, 'visual_vs_consensus')):
        per = [np.mean(v) for (v_, s), v in pooled.items() if v_ == vis]
        mean, lo, hi, p = boot(per, rng)
        summary[name] = {'participants': len(per), 'mean_diff': round(mean, 3), 'ci95': [round(lo, 3), round(hi, 3)], 'boot_p_diff_le_0': round(p, 4)}
    report = {'method': __doc__.strip().splitlines()[0], 'old': old, 'new': new, 'truth': 'current reference labels (consensus for visual)',
              'pooled': summary, 'features': feats, 'per_participant': scores}
    (ROOT / new / 'paired_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    for k, v in feats.items():
        extra = f"  | landmark {v['landmark']:.3f} new-lm={v['new_minus_landmark']:+.3f} CI[{v['ci95_vs_landmark'][0]:+.3f},{v['ci95_vs_landmark'][1]:+.3f}]" if 'landmark' in v else ''
        print(f"{k:20s} {v['old']:.3f}->{v['new']:.3f} d={v['diff']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] p={v['boot_p_diff_le_0']}{extra}")

if __name__ == '__main__':
    main(*sys.argv[1:])
