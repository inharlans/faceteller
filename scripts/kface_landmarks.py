"""Batch MediaPipe landmarks, v2 metrics and normalised part crops for K-FACE (local only).

Per participant it writes data/derived/kface/subjects/<ID>.npz (landmarks, pose, blendshapes),
<ID>.json (metrics per image) and <ID>.bin (JPEG crops; offsets in the npz). Re-running skips
participants whose outputs exist unless --force.
"""
import argparse, io, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import Store, subjects, DERIVED, LIGHTS

PARTS = ['eye_right', 'eye_left', 'brow_right', 'brow_left', 'nose', 'mouth', 'face']
OUT = DERIVED / 'subjects'

def capture_plan():
    """(accessory, light, expression, camera, purpose). Purposes drive later selection."""
    plan = []
    for l in LIGHTS:
        for c in ['C6', 'C7', 'C8', 'C15', 'C16', 'C17']:
            plan.append(('S001', l, 'E01', c, 'core'))
    for l in ['L1', 'L2', 'L3', 'L8', 'L12']:
        for c in ['C3', 'C4', 'C5', 'C9', 'C10', 'C11', 'C14', 'C18', 'C19', 'C20']:
            plan.append(('S001', l, 'E01', c, 'multiview'))
    for e in ['E02', 'E03']:
        for l in ['L1', 'L2']:
            for c in ['C6', 'C7', 'C8', 'C16']:
                plan.append(('S001', l, e, c, 'expression'))
    for s in ['S002', 'S003', 'S004', 'S005', 'S006']:
        for l in ['L1', 'L2']:
            for c in ['C7', 'C16']:
                plan.append((s, l, 'E01', c, 'accessory'))
    # v4 data expansion: wider yaw and lower camera under more lights; accessories under more lights/angles
    seen = {x[:4] for x in plan}
    bright = ['L1', 'L2', 'L3', 'L4', 'L8', 'L9', 'L12', 'L13', 'L14', 'L16', 'L19', 'L22', 'L23', 'L25', 'L28']
    for l in LIGHTS:
        for c in ['C5', 'C9'] + (['C19', 'C20'] if l in bright else []):
            if ('S001', l, 'E01', c) not in seen:
                plan.append(('S001', l, 'E01', c, 'extra_angle'))
    for s in ['S002', 'S003', 'S004', 'S005', 'S006']:
        for l in ['L1', 'L2', 'L8', 'L12']:
            for c in ['C6', 'C7', 'C8']:
                if (s, l, 'E01', c) not in seen:
                    plan.append((s, l, 'E01', c, 'extra_accessory'))
    return plan

_det = None; _store = None

def _init():
    global _det, _store
    import mediapipe as mp
    from facework.features import MODEL
    _det = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    _store = Store(capacity=2)

def process(subject, force=False):
    import mediapipe as mp
    from scipy.spatial.transform import Rotation
    from facework.morphometrics import measure, normalized_crops
    if (OUT / f'{subject}.npz').exists() and not force:
        return subject, 'kept', 0
    plan = capture_plan(); n = len(plan)
    lms = np.full((n, 478, 3), np.nan, np.float32); pose = np.full((n, 3), np.nan, np.float32)
    blend = np.full((n, 5), np.nan, np.float32); status = np.zeros(n, np.int8)
    offsets = np.zeros((n, len(PARTS), 2), np.int64); metrics = []
    buf = io.BytesIO()
    for i, (s, l, e, c, purpose) in enumerate(plan):
        im = _store.image(subject, s, l, e, c); w, h = im.size
        arr = np.ascontiguousarray(np.asarray(im))
        r = _det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        if len(r.face_landmarks) != 1:
            status[i] = 0 if not r.face_landmarks else 2; metrics.append(None); continue
        p = np.array([[t.x * w, t.y * h, t.z * w] for t in r.face_landmarks[0]], np.float32)
        lms[i] = p; status[i] = 1
        rot = np.asarray(r.facial_transformation_matrixes[0])[:3, :3]
        pose[i] = Rotation.from_matrix(rot).as_euler('xyz', degrees=True)  # pitch, yaw, roll
        bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
        blend[i] = [max(bs['mouthSmileLeft'], bs['mouthSmileRight']), bs['eyeBlinkLeft'], bs['eyeBlinkRight'], bs['jawOpen'], max(bs['browDownLeft'], bs['browDownRight'])]
        m = measure(im, p[:, :2]); metrics.append({k: (round(v, 6) if v is not None else None) for k, v in m.items()})
        for j, (part, crop) in enumerate(normalized_crops(im, p[:, :2], PARTS).items()):
            start = buf.tell(); crop.save(buf, format='JPEG', quality=92)
            offsets[i, j] = (start, buf.tell() - start)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{subject}.bin').write_bytes(buf.getvalue())
    (OUT / f'{subject}.json').write_text(json.dumps(metrics), encoding='utf-8')
    keys = np.array(['|'.join(x) for x in plan])
    np.savez_compressed(OUT / f'{subject}.npz', keys=keys, status=status, landmarks=lms, pose=pose, blend=blend, crop_offsets=offsets, parts=np.array(PARTS))
    return subject, 'done', int(status.sum())

def extend(subject):
    """Process plan rows missing from an existing <ID>.npz and append them (landmarks, metrics, crops)."""
    import mediapipe as mp
    from scipy.spatial.transform import Rotation
    from facework.morphometrics import measure, normalized_crops
    path = OUT / f'{subject}.npz'; d = dict(np.load(path)); have = set(str(k) for k in d['keys'])
    new = [x for x in capture_plan() if '|'.join(x) not in have]
    if not new:
        return subject, 'complete', 0
    metrics = json.loads((OUT / f'{subject}.json').read_text()); blob = bytearray((OUT / f'{subject}.bin').read_bytes())
    n = len(new); lms = np.full((n, 478, 3), np.nan, np.float32); pose = np.full((n, 3), np.nan, np.float32)
    blend = np.full((n, 5), np.nan, np.float32); status = np.zeros(n, np.int8); offsets = np.zeros((n, len(PARTS), 2), np.int64)
    for i, (s, l, e, c, purpose) in enumerate(new):
        im = _store.image(subject, s, l, e, c); w, h = im.size
        r = _det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(im))))
        if len(r.face_landmarks) != 1:
            status[i] = 0 if not r.face_landmarks else 2; metrics.append(None); continue
        p = np.array([[t.x * w, t.y * h, t.z * w] for t in r.face_landmarks[0]], np.float32); lms[i] = p; status[i] = 1
        pose[i] = Rotation.from_matrix(np.asarray(r.facial_transformation_matrixes[0])[:3, :3]).as_euler('xyz', degrees=True)
        bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
        blend[i] = [max(bs['mouthSmileLeft'], bs['mouthSmileRight']), bs['eyeBlinkLeft'], bs['eyeBlinkRight'], bs['jawOpen'], max(bs['browDownLeft'], bs['browDownRight'])]
        m = measure(im, p[:, :2]); metrics.append({k: (round(v, 6) if v is not None else None) for k, v in m.items()})
        for j, (part, crop) in enumerate(normalized_crops(im, p[:, :2], PARTS).items()):
            buf = io.BytesIO(); crop.save(buf, format='JPEG', quality=92); start = len(blob); blob += buf.getvalue()
            offsets[i, j] = (start, len(blob) - start)
    (OUT / f'{subject}.bin').write_bytes(bytes(blob)); (OUT / f'{subject}.json').write_text(json.dumps(metrics), encoding='utf-8')
    d.update(keys=np.concatenate([d['keys'], np.array(['|'.join(x) for x in new])]), status=np.concatenate([d['status'], status]),
             landmarks=np.concatenate([d['landmarks'], lms]), pose=np.concatenate([d['pose'], pose]), blend=np.concatenate([d['blend'], blend]),
             crop_offsets=np.concatenate([d['crop_offsets'], offsets]))
    np.savez_compressed(path, **d)
    return subject, 'extended', int(status.sum())

def recrop(subject):
    """Rewrite <ID>.bin and crop offsets from stored landmarks (after a crop-window change)."""
    from facework.morphometrics import normalized_crops
    path = OUT / f'{subject}.npz'; d = dict(np.load(path)); buf = io.BytesIO()
    offsets = np.zeros_like(d['crop_offsets'])
    for i, key in enumerate(d['keys']):
        if d['status'][i] != 1:
            continue
        s, l, e, c, _ = str(key).split('|')
        im = _store.image(subject, s, l, e, c)
        for j, (part, crop) in enumerate(normalized_crops(im, d['landmarks'][i][:, :2], PARTS).items()):
            start = buf.tell(); crop.save(buf, format='JPEG', quality=92); offsets[i, j] = (start, buf.tell() - start)
    (OUT / f'{subject}.bin').write_bytes(buf.getvalue()); d['crop_offsets'] = offsets
    np.savez_compressed(path, **d)
    return subject, 'recropped', int(d['status'].sum())

def _job(args):
    if args[2] == 'extend':
        return extend(args[0])
    if args[2]:
        return recrop(args[0])
    return process(*args[:2])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--limit', type=int)
    ap.add_argument('--force', action='store_true'); ap.add_argument('--crops-only', action='store_true'); ap.add_argument('--extend', action='store_true'); args = ap.parse_args()
    ids = subjects()[:args.limit] if args.limit else subjects()
    start = time.time(); done = 0
    with Pool(args.workers, initializer=_init) as pool:
        for subject, state, ok in pool.imap_unordered(_job, [(s, args.force, 'extend' if args.extend else args.crops_only) for s in ids]):
            done += 1
            print(f'{done}/{len(ids)} {subject} {state} ok={ok}/{len(capture_plan())} {time.time()-start:.0f}s', flush=True)

if __name__ == '__main__':
    main()
