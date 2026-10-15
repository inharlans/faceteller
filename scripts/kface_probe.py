"""Characterise K-FACE capture conditions on a participant sample with local MediaPipe.

Cameras: head yaw/pitch and landmark success per C1..C20 (S001/L1/E01).
Lights: face luminance and landmark success per L1..L30 (S001/C7/E01).
Accessories/expressions: landmark success and blendshape summaries (L1/C7).
Nothing leaves this machine; only aggregate numbers are written.
"""
import json, statistics, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import detector, analyze
from facework.kface import Store, subjects, CAMERAS, LIGHTS, ACCESSORIES, EXPRESSIONS, DERIVED

def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {'n': len(values), 'median': round(statistics.median(values), 3),
            'min': round(min(values), 3), 'max': round(max(values), 3)}

def main():
    sample = subjects()[::40]  # 10 participants spread over capture dates
    store = Store(); out = {'participants': sample, 'cameras': {}, 'lights': {}, 'accessories': {}, 'expressions': {}}
    with detector() as lm:
        def run(subject, s, l, e, c):
            im = store.image(subject, s, l, e, c)
            r = analyze(im, lm)
            gray = np.asarray(im.convert('L'), dtype=np.float32)
            return r, gray
        for c in CAMERAS:
            rows = []
            for subject in sample:
                r, _ = run(subject, 'S001', 'L1', 'E01', c)
                ann = store.annotation(subject, 'S001', 'L1', 'E01', c)
                rows.append((r, ann))
            ok = [r['metrics'] for r, _ in rows if r['status'] == 'ok']
            # Provider points: nose tip x relative to ear midpoint indicates yaw independently of MediaPipe.
            asym = []
            for _, ann in rows:
                if ann:
                    nose, er, el = ann['points'][0], ann['points'][5], ann['points'][6]
                    span = abs(el[0] - er[0]) or 1
                    asym.append((nose[0] - (er[0] + el[0]) / 2) / span)
            out['cameras'][c] = {'detected': f'{len(ok)}/{len(rows)}',
                                 'yaw_deg': summary([m.get('mesh_yaw_deg') for m in ok]),
                                 'pitch_deg': summary([m.get('mesh_pitch_deg') for m in ok]),
                                 'roll_deg': summary([m.get('roll_deg') for m in ok]),
                                 'face_width_px': summary([m.get('face_width_px') for m in ok]),
                                 'provider_nose_offset_ratio': summary(asym)}
            print(c, out['cameras'][c]['detected'], out['cameras'][c]['yaw_deg'], out['cameras'][c]['pitch_deg'], flush=True)
        for l in LIGHTS:
            rows = [run(subject, 'S001', l, 'E01', 'C7') for subject in sample]
            ok = [r for r, _ in rows if r['status'] == 'ok']
            lum = []
            for r, gray in rows:
                if r['status'] == 'ok':
                    x0, y0, x1, y1 = r['boxes_aligned']['face']
                    lum.append(float(gray[y0:y1, x0:x1].mean()))
                else:
                    lum.append(float(gray.mean()))
            # Left/right luminance balance of the face box, a coarse light-direction indicator.
            balance = []
            for r, gray in rows:
                if r['status'] == 'ok':
                    x0, y0, x1, y1 = r['boxes_aligned']['face']; mid = (x0 + x1) // 2
                    a, b = gray[y0:y1, x0:mid].mean(), gray[y0:y1, mid:x1].mean()
                    balance.append(float((a - b) / max(a + b, 1)))
            out['lights'][l] = {'detected': f'{len(ok)}/{len(rows)}', 'face_luminance': summary(lum),
                                'left_right_balance': summary(balance),
                                'sharpness': summary([r['metrics']['face_laplacian_variance'] for r in ok])}
            print(l, out['lights'][l]['detected'], out['lights'][l]['face_luminance'], flush=True)
        for s in ACCESSORIES:
            rows = [run(subject, s, 'L1', 'E01', 'C7')[0] for subject in sample]
            ok = [r['metrics'] for r in rows if r['status'] == 'ok']
            out['accessories'][s] = {'detected': f'{len(ok)}/{len(rows)}',
                                     'eye_right_aperture_ratio': summary([m['eye_right_aperture_ratio'] for m in ok]),
                                     'brow_right_length_eye': summary([m['brow_right_length_eye'] for m in ok])}
        for e in EXPRESSIONS:
            rows = [run(subject, 'S001', 'L1', e, 'C7')[0] for subject in sample]
            ok = [r['metrics'] for r in rows if r['status'] == 'ok']
            out['expressions'][e] = {'detected': f'{len(ok)}/{len(rows)}',
                                     'smile_score': summary([m['smile_score'] for m in ok]),
                                     'mouth_open_ratio': summary([m['mouth_open_ratio'] for m in ok]),
                                     'eye_right_aperture_ratio': summary([m['eye_right_aperture_ratio'] for m in ok])}
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / 'capture_probe.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({'accessories': out['accessories'], 'expressions': out['expressions']}, indent=1))

if __name__ == '__main__':
    main()
