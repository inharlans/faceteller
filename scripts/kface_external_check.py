"""Out-of-domain consistency on FRLL neutral photos (London set, different camera and cohort).

There are no v2 reference labels for FRLL, so this compares, per photo, the K-FACE CNN output
with the landmark measurement of the same photo, and reports the same statistic on K-FACE
test captures for contrast. A large gap signals domain shift, not which of the two is right.
Writes models/kface_v2/external_consistency.json.
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.analysis_v2 import detect, run_models, SCHEMA
from facework.morphometrics import measure
from facework.kface import DERIVED

def frll_pairs():
    pairs = {}
    for path in sorted((ROOT / 'data/raw/frll/neutral_front').rglob('*.jpg')):
        im = Image.open(path).convert('RGB'); d, _ = detect(im)
        if d is None or abs(d['yaw']) > 12:
            continue
        m = measure(im, d['landmarks'][:, :2]); pred = run_models(im, d['landmarks'])
        for k, f in SCHEMA['features'].items():
            if f['source'] in ('landmark',) and k in pred and m.get(f['metric']) is not None:
                pairs.setdefault(k, []).append((float(np.mean(pred[k]['values'])), m[f['metric']]))
    return pairs

def kface_pairs():
    """Same statistic on K-FACE test participants, frontal C7 under reference lights."""
    from facework.kface import Store
    refs = {r['subject_id']: r for r in csv.DictReader((DERIVED / 'reference_labels.csv').open(encoding='utf-8-sig'))}
    store = Store(); pairs = {}
    for s, r in refs.items():
        if r['split'] != 'test':
            continue
        d = np.load(DERIVED / 'subjects' / f'{s}.npz'); keys = list(d['keys'])
        i = keys.index('S001|L1|E01|C7|core')
        if d['status'][i] != 1:
            continue
        im = store.image(s, 'S001', 'L1', 'E01', 'C7'); p = d['landmarks'][i]
        m = measure(im, p[:, :2]); pred = run_models(im, p)
        for k, f in SCHEMA['features'].items():
            if f['source'] == 'landmark' and k in pred and m.get(f['metric']) is not None:
                pairs.setdefault(k, []).append((float(np.mean(pred[k]['values'])), m[f['metric']]))
    return pairs

def stats(pairs):
    out = {}
    for k, v in pairs.items():
        a, b = np.array(v).T
        out[k] = {'n': len(v), 'pearson_r': round(float(np.corrcoef(a, b)[0, 1]), 3), 'mean_cnn_minus_landmark': round(float(np.mean(a - b)), 4)}
    return out

def main():
    report = {'frll_neutral': stats(frll_pairs()), 'kface_test_C7_L1': stats(kface_pairs()),
              'note': 'CNN 예측과 같은 사진의 랜드마크 측정 간 일치도. FRLL에는 v2 기준 레이블이 없어 정확도가 아니다.'}
    (ROOT / 'models/kface_v2/external_consistency.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for k in report['frll_neutral']:
        print(f"{k:16s} FRLL r={report['frll_neutral'][k]['pearson_r']:+.3f} bias={report['frll_neutral'][k]['mean_cnn_minus_landmark']:+.4f} | K-FACE r={report['kface_test_C7_L1'].get(k, {}).get('pearson_r')}")

if __name__ == '__main__':
    main()
