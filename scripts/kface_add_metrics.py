"""Add landmark-only metrics to stored per-capture metrics (no MediaPipe rerun, no image decode)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED, subjects
from facework.morphometrics import align, landmark_metrics

NEW = ['brow_gap', 'eye_spacing', 'philtrum_length', 'lower_face_ratio']

def main():
    for s in subjects():
        d = np.load(DERIVED / 'subjects' / f'{s}.npz'); path = DERIVED / 'subjects' / f'{s}.json'
        metrics = json.loads(path.read_text())
        for i, m in enumerate(metrics):
            if m is None:
                continue
            q, _, _ = align(d['landmarks'][i][:, :2].astype(float), 864, 576)  # roll alignment only; image size unused
            lm = landmark_metrics(q)
            m.update({k: round(lm[k], 6) for k in NEW})
        path.write_text(json.dumps(metrics), encoding='utf-8')
    print('ok', len(subjects()))

if __name__ == '__main__':
    main()
