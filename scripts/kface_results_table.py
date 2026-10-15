"""Markdown results table for all features from the saved evaluations (test split, standard conditions)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.analysis_v2 import landmark_preferred

S = 'service(C7,bright)'; PK = {'eye': '눈', 'brow': '눈썹', 'nose': '코', 'mouth': '입', 'face': '얼굴'}
M = ROOT / ('models/kface_v4' if (ROOT / 'models/kface_v4/evaluation.json').exists() else 'models/kface_v2')

def table():
    ev = json.loads((M / 'evaluation.json').read_text(encoding='utf-8')); vc = json.loads((M / 'visual_confidence.json').read_text(encoding='utf-8'))
    schema = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
    summ = json.loads((ROOT / 'data/derived/kface/reference_summary.json').read_text(encoding='utf-8'))
    sel = M / ('evaluation_oof.json' if (M / 'evaluation_oof.json').exists() else 'evaluation_val.json')
    pref = landmark_preferred(json.loads(sel.read_text(encoding='utf-8')))
    f2 = lambda x: '—' if x is None else f'{x:.2f}'
    rows = ['| 부위 | 특징 | 출처 | 사용 방법 | 시험 정확도 | 확신 예측 정확도(비율) | 기준선 | 등급 |', '|---|---|---|---|---|---|---|---|']
    for part, pr in ev['parts'].items():
        for k, g in pr['features'].items():
            s = g[S]; f = schema['features'][k]; ko = f['ko'].split('(')[0]
            if f['source'] == 'visual':
                r7 = next(x for x in vc[k] if x['threshold'] == 0.7); base = s['majority_baseline_accuracy']; acc = r7['accuracy']
                grade = 'D' if s['accuracy'] < base + 0.05 else 'A' if acc >= 0.85 else 'B' if acc >= 0.7 else 'C'
                rows.append(f"| {PK[part]} | {ko} | AI 육안 | CNN | {f2(s['accuracy'])} | {f2(acc)} ({r7['coverage']:.0%}) | 최빈값 {f2(base)} | {grade} |")
            else:
                if k in pref:
                    acc = s['landmark_accuracy']; method = '랜드마크 직접'; conf = '—'
                else:
                    acc = s['cnn_confident_accuracy']; method = 'CNN'; conf = f"{f2(s['cnn_confident_accuracy'])} ({s['cnn_confident_rate']:.0%})"
                low = k in schema.get('reliability_notes', {})
                grade = 'C' if low else 'A' if acc >= 0.85 else 'B' if acc >= 0.7 else 'C'
                src = {'landmark': '측정', 'multiview': '다중 시점'}[f['source']]
                rows.append(f"| {PK[part]} | {ko} | {src} (ICC {summ['features'][k]['icc_1_1_train']}) | {method} | {f2(s['cnn_accuracy'] if k not in pref else s['landmark_accuracy'])} | {conf} | 랜드마크 {f2(s.get('landmark_accuracy'))} | {grade} |")
    return '\n'.join(rows)

if __name__ == '__main__':
    print(table())
