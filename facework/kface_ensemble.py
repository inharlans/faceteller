"""Load and run part-model ensembles (cross-validated folds x backbones, or a single checkpoint).

Layout of an ensemble root: <root>/<arch>/fold<k>/<part>.pt plus <root>/folds.json. A root without
folds is treated as holding single checkpoints <root>/<part>.pt. Regression outputs are averaged in
the shared z-space (all members use the same normalisation), class probabilities are averaged.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

def checkpoints(part, root, fold=None):
    root = Path(root)
    pattern = f'*/fold{fold}/{part}.pt' if fold is not None else f'*/fold*/{part}.pt'
    found = sorted(root.glob(pattern))
    return found or ([root / f'{part}.pt'] if (root / f'{part}.pt').exists() else [])

def load(path, device='cpu'):
    from .kface_training import PartNet
    ck = torch.load(path, map_location='cpu', weights_only=False); meta = ck['meta']
    m = PartNet(meta['arch'], len(meta['regression_features']), [len(meta['classes'][k]) for k in meta['classification_features']], pretrained=False)
    m.load_state_dict(ck['state_dict']); m.eval().to(device)
    return m, meta

def combined_meta(metas):
    meta = dict(metas[0])
    margins = {k: [m['uncertainty_margin'].get(k) for m in metas if m['uncertainty_margin'].get(k) is not None] for k in meta['uncertainty_margin']}
    # averaging several models shrinks the error; keep the mean single-model margin as a conservative band
    meta['uncertainty_margin'] = {k: (float(np.mean(v)) if v else None) for k, v in margins.items()}
    meta['ensemble'] = [{'arch': m['arch'], 'seed': m.get('seed'), 'val_score': m['val_score']} for m in metas]
    return meta

def predict_rows(part, rows, refs, paths, device):
    """Average predictions of every checkpoint in `paths` over dataset rows."""
    from .kface_training import Crops, eval_transform, predict
    Rs, Cs, metas = [], [], []
    for p in paths:
        model, meta = load(p, device); metas.append(meta)
        ds = Crops(part, rows, refs, meta['norm'], eval_transform(part))
        R, C = predict(model, DataLoader(ds, batch_size=128, num_workers=4), device)
        Rs.append(R); Cs.append(C)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    R = np.mean(Rs, 0)
    C = [np.mean([c[j] for c in Cs], 0) for j in range(len(Cs[0]))]
    return R, C, combined_meta(metas)
