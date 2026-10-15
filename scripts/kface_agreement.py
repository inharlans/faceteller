"""Blind human review of the AI visual labels (16 v3 visual features).

  python scripts/kface_agreement.py form --n 80        # blank form, stratified sample, no AI labels shown
  python scripts/kface_agreement.py score FILLED.csv   # Cohen's kappa + confusion vs AI labels

The form lists participant IDs only; the reviewer opens the crops in the local workbench
(K-FACE photos must stay with approved users). Filled forms may come from several reviewers
(column 'reviewer'); each reviewer is scored separately and pairwise between reviewers.
"""
import argparse, csv, itertools, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT

_SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
TASKS = {k: f['classes'] + ['unknown'] for k, f in _SCHEMA['features'].items() if f['source'] == 'visual'}
AI = ROOT / 'data/annotations/kface_visual_v4.tsv'  # three-rater AI consensus
OUT = ROOT / 'outputs/kface-v2'

def kappa(a, b, labels):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po

def form(n):
    ai = list(csv.DictReader(AI.open(encoding='utf-8'), delimiter='\t'))
    rng = random.Random(20260926); strata = defaultdict(list)
    for r in ai:
        strata[(r['brow_shape'], r['face_shape'])].append(r['subject_id'])
    pick = []
    while len(pick) < n and any(strata.values()):  # round-robin over label strata so rare classes appear
        for key in sorted(strata):
            if strata[key] and len(pick) < n:
                pick.append(strata[key].pop(rng.randrange(len(strata[key]))))
    rng.shuffle(pick); OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / '독립검수_시각레이블_양식.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['subject_id', 'reviewer', *TASKS, 'notes'])
        for s in pick:
            w.writerow([s, '', '', '', '', ''])
    (OUT / '독립검수_안내.txt').write_text(
        '작업대 /v2 는 사진을 보여주지 않습니다. 검수자는 승인된 로컬 환경에서 참여자 ID의 기준 사진(S001/L1/E01/C7)을 보고\n'
        f'허용값만 입력합니다: ' + json.dumps(TASKS, ensure_ascii=False) + '\n'
        'AI 레이블을 먼저 보지 마세요. 입력 후: python scripts/kface_agreement.py score <파일>\n', encoding='utf-8')
    print(path, len(pick))

def score(path):
    ai = {r['subject_id']: r for r in csv.DictReader(AI.open(encoding='utf-8'), delimiter='\t')}
    if str(path).lower().endswith('.json'):  # export of the standalone labelling page (kface-human-review-v1)
        j = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        who = j.get('reviewer') or 'reviewer'
        rows = [{**{k: (v or '') for k, v in r.items()}, 'reviewer': who} for r in j['labels']]
    else:
        rows = [r for r in csv.DictReader(Path(path).open(encoding='utf-8-sig')) if r.get('reviewer')]
    by = defaultdict(dict)
    for r in rows:
        for t, labels in TASKS.items():
            if r.get(t) and r[t] not in labels:
                raise SystemExit(f"{r['subject_id']} {t}: 허용되지 않은 값 {r[t]}")
        by[r['reviewer']][r['subject_id']] = r
    report = {}
    for reviewer, items in by.items():
        report[reviewer] = {}
        for t, labels in TASKS.items():
            ids = [s for s, r in items.items() if r.get(t) and s in ai]
            if ids:
                k, po = kappa([items[s][t] for s in ids], [ai[s][t] for s in ids], labels)
                report[reviewer][t] = {'n': len(ids), 'agreement': round(po, 3), 'cohen_kappa_vs_ai': round(k, 3),
                                       'confusion_human_to_ai': {h: dict(Counter(ai[s][t] for s in ids if items[s][t] == h)) for h in labels}}
    for r1, r2 in itertools.combinations(by, 2):
        for t, labels in TASKS.items():
            ids = [s for s in by[r1] if s in by[r2] and by[r1][s].get(t) and by[r2][s].get(t)]
            if ids:
                k, po = kappa([by[r1][s][t] for s in ids], [by[r2][s][t] for s in ids], labels)
                report.setdefault('between_reviewers', {})[f'{r1}|{r2}|{t}'] = {'n': len(ids), 'agreement': round(po, 3), 'cohen_kappa': round(k, 3)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / '독립검수_일치도.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('form'); f.add_argument('--n', type=int, default=80)
    s = sub.add_parser('score'); s.add_argument('path')
    a = ap.parse_args()
    form(a.n) if a.cmd == 'form' else score(a.path)
