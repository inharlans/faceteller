"""Three-rater consensus for the 16 visual features (A = first pass, B and C = blind independent passes).

Outputs
  data/annotations/kface_visual_v4.tsv        consensus hard labels (same columns as v3)
  data/annotations/kface_visual_v4_soft.json  per participant/feature vote shares over classes
  data/annotations/kface_rater_agreement.json pairwise Cohen kappa, Fleiss kappa, agreement, decisions
Rules: unknown when two or more raters say unknown; otherwise majority of the remaining votes.
Three different votes -> median for ordinal features, 'uncertain' for nominal ones (kept out of
hard-label training/evaluation, still used as a soft target).
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.kface import subjects

SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
VIS = {k: f for k, f in SCHEMA['features'].items() if f['source'] == 'visual'}
NOMINAL = {'brow_shape', 'face_shape', 'chin_shape', 'sclera_show'}  # everything else has an order
FILES = {'brows': ['brow_shape', 'brow_thickness', 'brow_density', 'brow_tail_clarity'], 'eyes': ['eyelid', 'sclera_show', 'eye_depth'],
         'nose': ['nose_tip', 'nostril_show', 'alar_fullness'], 'mouth': ['lip_bow', 'philtrum_definition'],
         'face': ['face_shape', 'chin_shape', 'cheekbone', 'cheek_fullness']}
RATERS = ROOT / 'tmp/kface_review/raters'

def load_rater(r):
    key = json.loads((RATERS / '_keys' / f'{r}.json').read_text(encoding='utf-8')); out = {}
    for name, feats in FILES.items():
        for row in csv.DictReader((RATERS / r / f'{name}.tsv').open(encoding='utf-8-sig'), delimiter='\t'):
            s = key[str(int(row['number']))]
            for k in feats:
                v = (row.get(k) or '').strip()
                out.setdefault(s, {})[k] = v if v in VIS[k]['classes'] else 'unknown'
    return out

def kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n; ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def fleiss(table):
    """table: items x categories counts, every row sums to the same number of raters."""
    t = np.asarray(table, float); n = t.sum(1)[0]
    p_j = t.sum(0) / t.sum(); P_i = ((t * t).sum(1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return float((P_i.mean() - pe) / (1 - pe)) if pe < 1 else 1.0

def main():
    ids = subjects()
    A = {r['subject_id']: r for r in csv.DictReader((ROOT / 'data/annotations/kface_visual_v3.tsv').open(encoding='utf-8'), delimiter='\t')}
    B, C = load_rater('B'), load_rater('C')
    raters = {'A': {s: {k: A[s][k] for k in VIS} for s in ids}, 'B': B, 'C': C}
    report = {}; consensus = {s: {} for s in ids}; soft = {s: {} for s in ids}
    for k, f in VIS.items():
        cls = f['classes']; votes = {s: [raters[r][s][k] for r in 'ABC'] for s in ids}
        both = lambda x, y: [s for s in ids if raters[x][s][k] in cls and raters[y][s][k] in cls]
        pair = {f'{x}-{y}': round(kappa([raters[x][s][k] for s in both(x, y)], [raters[y][s][k] for s in both(x, y)], cls) or 0, 3) for x, y in (('A', 'B'), ('A', 'C'), ('B', 'C'))}
        full = [s for s in ids if all(v in cls for v in votes[s])]
        fk = fleiss([[votes[s].count(c) for c in cls] for s in full]) if full else None
        stats = Counter()
        for s in ids:
            v = votes[s]; known = [x for x in v if x in cls]
            soft[s][k] = {c: round(known.count(c) / len(known), 3) for c in cls} if known else None
            if len(known) <= 1:
                consensus[s][k] = 'unknown'; stats['unknown'] += 1; continue
            top, n = Counter(known).most_common(1)[0]
            if n >= 2:
                consensus[s][k] = top; stats['unanimous' if n == len(known) == 3 else 'majority'] += 1
            elif k in NOMINAL:
                consensus[s][k] = 'uncertain'; stats['uncertain'] += 1
            else:
                consensus[s][k] = sorted(known, key=cls.index)[len(known) // 2]; stats['ordinal_median'] += 1
        decision = 'keep' if fk is not None and fk >= 0.4 else 'keep_low_agreement' if fk is not None and fk >= 0.2 else 'withhold'
        report[k] = {'ko': f['ko'], 'pairwise_cohen_kappa': pair, 'fleiss_kappa': round(fk, 3) if fk is not None else None,
                     'items_all_known': len(full), 'consensus_types': dict(stats), 'decision': decision,
                     'consensus_distribution': dict(Counter(consensus[s][k] for s in ids))}
    rows = [{'subject_id': s, **{k: consensus[s][k] for k in VIS}, 'rater': 'consensus_A_B_C', 'notes': A[s].get('notes', '')} for s in ids]
    with (ROOT / 'data/annotations/kface_visual_v4.tsv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    (ROOT / 'data/annotations/kface_visual_v4_soft.json').write_text(json.dumps(soft), encoding='utf-8')
    (ROOT / 'data/annotations/kface_rater_agreement.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for k, v in report.items():
        print(f"{k:20s} fleiss={v['fleiss_kappa']} pair={v['pairwise_cohen_kappa']} {v['decision']} {v['consensus_types']}")

if __name__ == '__main__':
    main()
