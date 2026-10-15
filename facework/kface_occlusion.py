"""Occlusion checker trained on K-FACE accessory conditions (labels come from the capture design).

Heads: eyewear {none, glasses, sunglasses} (S002/S003 -> glasses, S004 -> sunglasses, S006 -> glasses)
and hat {no, yes} (S005/S006). Input: the scale-normalised face window. Participant-level split
identical to the morphology models. Usage: python -m facework.kface_occlusion
"""
import json, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from .kface_training import Crops, PartNet, capture_rows, load_references, eval_transform, train_transform, predict, SEED, OUT

EYEWEAR = {'S001': 0, 'S002': 1, 'S003': 1, 'S004': 2, 'S005': 0, 'S006': 1}
HAT = {'S001': 0, 'S002': 0, 'S003': 0, 'S004': 0, 'S005': 1, 'S006': 1}
CLASSES = {'eyewear': ['none', 'glasses', 'sunglasses'], 'hat': ['no', 'yes']}

class OcclusionCrops(Crops):
    def __init__(self, rows, refs, transform):
        super().__init__('face', rows, refs, {}, transform); self.reg = []; self.cls = []
    def __getitem__(self, idx):
        s, i, crop, side, meta = self.rows[idx]
        y = torch.tensor([EYEWEAR[meta['accessory']], HAT[meta['accessory']]])
        return self.transform(self._crop(s, i, crop)), y

def rows_for(refs, splits):
    acc = capture_rows('face', refs, splits, purposes=('accessory',))
    # Matching no-accessory captures: same lights and cameras as the accessory set.
    none = capture_rows('face', refs, splits, purposes=('core',), filt=lambda m: m['light'] in ('L1', 'L2') and m['camera'] in ('C7', 'C16'))
    return acc + none

def main(epochs=6, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(SEED); random.seed(SEED)
    refs, _ = load_references()
    train, val, test = rows_for(refs, ['train']), rows_for(refs, ['val']), rows_for(refs, ['test'])
    model = PartNet('mobilenet_v3_large', 0, [3, 2]).to(device)
    loaders = {k: DataLoader(OcclusionCrops(v, refs, train_transform('face') if k == 'train' else eval_transform('face')), batch_size=64,
                             shuffle=k == 'train', num_workers=4, persistent_workers=True) for k, v in (('train', train), ('val', val), ('test', test))}
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.02)
    w_eye = torch.tensor([1.0, 1.0, 2.0], device=device)  # sunglasses are 1/6 of accessory captures
    best = -1; state = None; history = []
    def run_eval(loader):
        model.eval(); ys = []; P = [[], []]
        with torch.inference_mode():
            for x, y in loader:
                _, cs = model(x.to(device)); ys.append(y)
                for j in range(2):
                    P[j].append(cs[j].softmax(1).cpu())
        return torch.cat(ys).numpy(), [torch.cat(p).numpy() for p in P]
    for epoch in range(epochs):
        model.train(); t0 = time.time()
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device); _, cs = model(x)
            loss = nn.functional.cross_entropy(cs[0], y[:, 0], weight=w_eye) + nn.functional.cross_entropy(cs[1], y[:, 1])
            opt.zero_grad(); loss.backward(); opt.step()
        y, P = run_eval(loaders['val'])
        score = np.mean([f1_score(y[:, j], P[j].argmax(1), average='macro') for j in range(2)])
        history.append({'epoch': epoch + 1, 'val_macro_f1': round(float(score), 4), 'seconds': round(time.time() - t0, 1)}); print(history[-1], flush=True)
        if score > best:
            best = score; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); y, P = run_eval(loaders['test'])
    result = {}
    for j, name in enumerate(CLASSES):
        pred = P[j].argmax(1)
        result[name] = {'classes': CLASSES[name], 'macro_f1': round(float(f1_score(y[:, j], pred, average='macro')), 3),
                        'accuracy': round(float((pred == y[:, j]).mean()), 3), 'confusion_matrix': confusion_matrix(y[:, j], pred).tolist()}
    meta = {'task': 'occlusion', 'arch': 'mobilenet_v3_large', 'classes': CLASSES, 'input': 'face window', 'history': history,
            'test': result, 'test_participants': len({r[0] for r in test}), 'test_captures': len(test)}
    torch.save({'state_dict': state, 'meta': meta}, OUT / 'occlusion.pt')
    (OUT / 'occlusion_evaluation.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
