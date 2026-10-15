"""Export the v3 dataset as browsable folders plus manifests (local only; contains K-FACE crops).

data/derived/kface/dataset_v3/
  participants.csv            one row per participant: split + 39 labels (+ measured values)
  captures.csv                every usable capture (training pool) with split and capture conditions
  split_summary.json / .md    class counts per feature and split, with balance check
  <part>/<feature>/<split>/<class>/<participant>.jpg   reference crop (S001/E01/C7, brightest usable light)
Borderline and unknown participants go to <class> = _borderline / _unknown so nothing is silently dropped.
"""
import csv, io, json, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED, ACCESSORY_KO, EXPRESSION_KO
from facework.features import ROOT

SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
OUT = DERIVED / 'dataset_v3'
CROP = {'eye': 'eye_right', 'brow': 'brow_right', 'nose': 'nose', 'mouth': 'mouth', 'face': 'face'}
SPLITS = ['train', 'val', 'test']

def main():
    refs = list(csv.DictReader((DERIVED / 'reference_labels.csv').open(encoding='utf-8-sig')))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    feats = list(SCHEMA['features'])
    cols = ['subject_id', 'split'] + feats + [f'{k}_value' for k in feats if SCHEMA['features'][k]['source'] != 'visual']
    with (OUT / 'participants.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(refs)
    cap_rows = []; written = 0
    for r in refs:
        s = r['subject_id']; d = np.load(DERIVED / 'subjects' / f'{s}.npz'); blob = (DERIVED / 'subjects' / f'{s}.bin').read_bytes()
        names = list(d['parts']); keys = list(d['keys'])
        for i, key in enumerate(keys):
            if d['status'][i] != 1:
                continue
            acc, light, expr, cam, purpose = key.split('|')
            cap_rows.append({'subject_id': s, 'split': r['split'], 'capture': key, 'accessory': ACCESSORY_KO[acc], 'light': light,
                             'expression': EXPRESSION_KO[expr], 'camera': cam, 'purpose': purpose,
                             'yaw': round(float(d['pose'][i, 1]), 1), 'pitch': round(float(d['pose'][i, 0]), 1),
                             'used_for_training': bool(purpose == 'core' and r['split'] == 'train' and abs(d['pose'][i, 1]) < 20 and abs(d['pose'][i, 0]) < 25)})
        ref = next((keys.index(f'S001|{l}|E01|C7|core') for l in ('L1', 'L2', 'L8', 'L12') if d['status'][keys.index(f'S001|{l}|E01|C7|core')] == 1), None)
        if ref is None:
            continue
        for k, f in SCHEMA['features'].items():
            label = r.get(k) or 'unknown'
            label = {'borderline': '_borderline', 'unknown': '_unknown'}.get(label, label)
            o, n = d['crop_offsets'][ref, names.index(CROP[f['part']])]
            target = OUT / f['part'] / k / r['split'] / label / f'{s}.jpg'
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(blob[o:o + n]); written += 1
    with (OUT / 'captures.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(cap_rows[0])); w.writeheader(); w.writerows(cap_rows)
    # Class counts per split and a balance check: largest gap between a split's class share and train's share.
    summary = {'participants': dict(Counter(r['split'] for r in refs)),
               'captures': {sp: sum(c['split'] == sp for c in cap_rows) for sp in SPLITS},
               'training_captures': int(sum(bool(c['used_for_training']) for c in cap_rows)), 'features': {}}
    for k, f in SCHEMA['features'].items():
        counts = {sp: Counter(r[k] or 'unknown' for r in refs if r['split'] == sp) for sp in SPLITS}
        share = lambda sp, c: counts[sp][c] / max(1, sum(v for kk, v in counts[sp].items() if kk in f['classes']))
        gap = max(abs(share(sp, c) - share('train', c)) for sp in ('val', 'test') for c in f['classes'])
        thin = [f'{sp}:{c}' for sp in ('val', 'test') for c in f['classes'] if counts[sp][c] < 3]
        summary['features'][k] = {'ko': f['ko'], 'source': f['source'], 'counts': {sp: dict(counts[sp]) for sp in SPLITS},
                                  'max_share_gap_vs_train': round(gap, 3), 'classes_under_3_in_eval_split': thin}
    (OUT / 'split_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['# K-FACE v3 데이터셋 분할', '', f"참여자: 학습 {summary['participants']['train']} · 검증 {summary['participants']['val']} · 시험 {summary['participants']['test']} (사람 단위, 겹침 없음)",
             f"사용 가능 촬영: 학습 {summary['captures']['train']:,} · 검증 {summary['captures']['val']:,} · 시험 {summary['captures']['test']:,} / 실제 학습 투입 {summary['training_captures']:,}", '',
             '| 특징 | 출처 | 학습 | 검증 | 시험 | 비율 차이 최대 | 검증·시험 3명 미만 클래스 |', '|---|---|---|---|---|---|---|']
    for k, v in summary['features'].items():
        fmt = lambda sp: ' / '.join(f"{c} {v['counts'][sp].get(c, 0)}" for c in SCHEMA['features'][k]['classes'])
        lines.append(f"| {v['ko']} | {v['source']} | {fmt('train')} | {fmt('val')} | {fmt('test')} | {v['max_share_gap_vs_train']:.2f} | {', '.join(v['classes_under_3_in_eval_split']) or '-'} |")
    (OUT / 'split_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'README.txt').write_text(
        'K-FACE v3 형태 데이터셋 (로컬 전용, 제공 사진 포함: 공유·업로드 금지)\n'
        '폴더: <부위>/<특징>/<train|val|test>/<클래스>/<참여자ID>.jpg  (기준 촬영 1장; 학습에는 captures.csv의 여러 조명·각도 촬영을 사용)\n'
        '_borderline: 반복 촬영 오차가 경계값에 걸쳐 분류 평가에서 제외 / _unknown: 가림 등으로 판정 불가\n'
        '분할: 사람 단위 70/15/15, ID 끝 두 자리로 층화, seed 20260926. 검증=epoch·방법 선택, 시험=최종 보고만\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('participants', 'captures', 'training_captures')}), written, 'crops')

if __name__ == '__main__':
    main()
