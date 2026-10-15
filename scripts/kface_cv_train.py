"""5-fold cross-training over the 338 non-test participants; the 62 test participants stay untouched.

Each fold relabels one fifth of train+val as 'val' (epoch selection) and trains on the rest. Folds are
stratified the same way as the main split (labels only). Cut points and normalisation stay those of the
main split, so every fold predicts the same label definitions.
  python scripts/kface_cv_train.py --arch convnext_tiny --folds 5 --out models/kface_v4
"""
import argparse, csv, json, random, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.kface import DERIVED

def folds_for(rows, k, seed=20260927):
    """Round-robin over participants sorted by a label signature, so each fold gets a similar mix."""
    pool = [r for r in rows if r['split'] != 'test']
    keys = ['face_shape', 'eyelid', 'brow_thickness', 'nose_tip', 'lip_thickness', 'eye_size']
    rng = random.Random(seed); rng.shuffle(pool)
    pool.sort(key=lambda r: tuple(r.get(x, '') for x in keys))
    return {r['subject_id']: i % k for i, r in enumerate(pool)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arch', default='convnext_tiny'); ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', default='models/kface_v4'); ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--per-subject', type=int, default=36)
    ap.add_argument('--data', default='base'); ap.add_argument('--soft', default='data/annotations/kface_visual_v4_soft.json'); ap.add_argument('--only-fold', type=int)
    args = ap.parse_args()
    rows = list(csv.DictReader((DERIVED / 'reference_labels.csv').open(encoding='utf-8-sig')))
    fold = folds_for(rows, args.folds)
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    prev = out / 'folds.json'
    if prev.exists() and json.loads(prev.read_text(encoding='utf-8'))['assignment'] != fold:
        # other backbones' checkpoints in this root were trained on the old folds -> OOF would leak
        raise SystemExit('folds.json differs from the new assignment; use a new --out or retrain every arch')
    (out / 'folds.json').write_text(json.dumps({'folds': args.folds, 'assignment': fold, 'arch': args.arch}, indent=0), encoding='utf-8')
    for f in range(args.folds):
        if args.only_fold is not None and f != args.only_fold:
            continue
        d = out / 'refs' / f'fold{f}'; d.mkdir(parents=True, exist_ok=True)
        with (d / 'reference_labels.csv').open('w', newline='', encoding='utf-8-sig') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
            for r in rows:
                w.writerow(r | {'split': 'test' if r['split'] == 'test' else ('val' if fold[r['subject_id']] == f else 'train')})
        shutil.copy2(DERIVED / 'reference_summary.json', d / 'reference_summary.json')
        cmd = [sys.executable, '-m', 'facework.kface_training', '--part', 'all', '--arch', args.arch, '--data', args.data,
               '--epochs', str(args.epochs), '--per-subject', str(args.per_subject), '--workers', '6', '--seed', str(20260927 + f),
               '--refs', str(d / 'reference_labels.csv'), '--soft', args.soft, '--out', str(out / args.arch / f'fold{f}')]
        print('fold', f, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

if __name__ == '__main__':
    main()
