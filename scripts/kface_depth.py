"""Per-participant multi-view depth features, with test-retest across lighting sessions.

Method A: MediaPipe 2D landmarks in 12 views (C4-C10, C15-C17, C19-C20) per session.
Method B (independent check): provider-annotated 7 points (nose tip, eye centres, mouth
corners, ears) in all 20 views, S001/E01, sessions L1/L3/L6.
"""
import json, sys, statistics
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import Store, subjects, DERIVED, CAMERAS
from facework.multiview import POINTS, VIEWS, depth_features, factorize

SESSIONS = ['L1', 'L2', 'L3', 'L8', 'L12']

def provider_depth(store, subject, light):
    pts = []
    for c in CAMERAS:
        a = store.annotation(subject, 'S001', light, 'E01', c)
        if a is None:
            return None
        pts.append(a['points'])
    xy = np.array(pts, float)  # 20 x 7 x 2
    X, residual = factorize(xy)
    nose, er, el, mr, ml, ear_r, ear_l = X
    x = el - er; iod = np.linalg.norm(x); x /= iod
    mid_mouth = (mr + ml) / 2; down = mid_mouth - (er + el) / 2; down -= x * (down @ x); down /= np.linalg.norm(down)
    z = np.cross(x, down)
    rel = lambda p: float((p - (er + el) / 2) @ z / iod)
    tip = rel(nose); ears = (rel(ear_r) + rel(ear_l)) / 2
    if tip < ears:  # nose is in front of the ears for every head
        tip, ears = -tip, -ears
    return {'provider_nose_tip_from_eyes': tip, 'provider_ear_depth': ears, 'residual': residual}

def icc_1(values):
    """ICC(1,1) for a subjects x sessions matrix without missing cells."""
    a = np.asarray(values, float); n, k = a.shape
    grand = a.mean(); ms_b = k * ((a.mean(1) - grand) ** 2).sum() / (n - 1)
    ms_w = ((a - a.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((ms_b - ms_w) / (ms_b + (k - 1) * ms_w))

def main():
    store = Store(capacity=2); out = {}
    for n, subject in enumerate(subjects()):
        d = np.load(DERIVED / 'subjects' / f'{subject}.npz')
        keys = [k.split('|') for k in d['keys']]; pos = {tuple(k[:4]): i for i, k in enumerate(keys)}
        sessions = {}
        for light in SESSIONS:
            rows = [pos.get(('S001', light, 'E01', c)) for c in VIEWS]
            if any(r is None or d['status'][r] != 1 for r in rows):
                continue
            xy = d['landmarks'][rows][:, POINTS, :2].astype(float)
            feats, _ = depth_features(xy)
            # MediaPipe's own single-image depth prior on the frontal view, for comparison only.
            p = d['landmarks'][pos[('S001', light, 'E01', 'C7')]].astype(float)
            iod = np.linalg.norm(p[263, :2] - p[33, :2])
            feats['mediapipe_single_view_tip_from_eyes'] = float(-(p[1, 2] - (p[133, 2] + p[362, 2]) / 2) / iod)
            sessions[light] = feats
        provider = {l: provider_depth(store, subject, l) for l in ['L1', 'L3', 'L6']}
        out[subject] = {'sessions': sessions, 'provider': {k: v for k, v in provider.items() if v}}
        if n % 50 == 0:
            print(n, subject, len(sessions), flush=True)
    keys = ['nose_projection', 'nose_bridge', 'nose_dorsum', 'nose_tip_from_eyes', 'lip_projection', 'chin_projection', 'mid_face_depth']
    complete = [s for s, v in out.items() if all(l in v['sessions'] for l in SESSIONS)]
    reliability = {}
    for k in keys:
        mat = [[out[s]['sessions'][l][k] for l in SESSIONS] for s in complete]
        sd = [statistics.pstdev(r) for r in mat]
        reliability[k] = {'icc_1_1': round(icc_1(mat), 3), 'subjects': len(mat),
                          'median_within_subject_sd': round(statistics.median(sd), 4),
                          'between_subject_sd': round(statistics.pstdev([statistics.median(r) for r in mat]), 4)}
    # Cross-method agreement with provider points (tip depth relative to eye centres).
    a = []; b = []; c = []
    for s, v in out.items():
        if v['sessions'] and v['provider']:
            a.append(statistics.median(x['nose_tip_from_eyes'] for x in v['sessions'].values()))
            b.append(statistics.median(x['provider_nose_tip_from_eyes'] for x in v['provider'].values()))
            c.append(statistics.median(x['mediapipe_single_view_tip_from_eyes'] for x in v['sessions'].values()))
    agreement = {'subjects': len(a), 'pearson_multiview_vs_provider': round(float(np.corrcoef(a, b)[0, 1]), 3),
                 'pearson_singleview_prior_vs_provider': round(float(np.corrcoef(c, b)[0, 1]), 3),
                 'pearson_multiview_vs_singleview_prior': round(float(np.corrcoef(a, c)[0, 1]), 3)}
    prov = [[v['provider'][l]['provider_nose_tip_from_eyes'] for l in ['L1', 'L3', 'L6']] for v in out.values() if len(v['provider']) == 3]
    agreement['provider_icc_1_1'] = round(icc_1(prov), 3)
    summary = {'method': 'scaled-orthographic factorisation, 12 views, per lighting session', 'sessions': SESSIONS,
               'views': VIEWS, 'reliability': reliability, 'agreement': agreement}
    (DERIVED / 'depth.json').write_text(json.dumps({'summary': summary, 'subjects': out}), encoding='utf-8')
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
