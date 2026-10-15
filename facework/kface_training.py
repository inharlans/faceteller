"""Part CNNs for morphology-kface-v2: multi-head regression (measured features) + classification (visual).

Labels are participant-level references (scripts/kface_labels.py) propagated to every capture of
that participant, so the network must reproduce them across lighting and camera angle.
Usage: python -m facework.kface_training --part eye|brow|nose|mouth|face|all [--arch efficientnet_b0]
"""
import argparse, csv, hashlib, io, json, math, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from .features import ROOT
from .kface import DERIVED
from .morphometrics import WINDOWS

SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
SOFT = {}  # participant -> feature -> vote shares, loaded with --soft
OUT = ROOT / 'models/kface_v2'
PART_CROPS = {'eye': ['eye_right', 'eye_left'], 'brow': ['brow_right', 'brow_left'], 'nose': ['nose'], 'mouth': ['mouth'], 'face': ['face']}
INPUT_SCALE = {'eye': 2.0, 'brow': 2.0, 'nose': 1.75, 'mouth': 2.0, 'face': 1.25}
SEED = 20260926

def input_size(part):
    h, w = WINDOWS[part]; s = INPUT_SCALE[part]
    return int(round(h * s / 8) * 8), int(round(w * s / 8) * 8)

def part_features(part):
    reg = [k for k, f in SCHEMA['features'].items() if f['part'] == part and f['source'] != 'visual']
    cls = [k for k, f in SCHEMA['features'].items() if f['part'] == part and f['source'] == 'visual']
    return reg, cls

def eval_transform(part):
    return transforms.Compose([transforms.Resize(input_size(part), antialias=True), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

def train_transform(part):
    h, w = input_size(part)
    ops = [transforms.Resize((h, w), antialias=True),
           transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), fill=127),  # no scale: relative size is a label
           transforms.ColorJitter(0.35, 0.35, 0.3, 0.03), transforms.RandomGrayscale(0.1),
           transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.2))], p=0.2)]
    if part in ('nose', 'mouth', 'face'):
        ops.append(transforms.RandomHorizontalFlip())  # labels of these parts are left-right symmetric
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

REFS = {'labels': DERIVED / 'reference_labels.csv', 'summary': DERIVED / 'reference_summary.json', 'soft': None}

def load_references():
    rows = list(csv.DictReader(REFS['labels'].open(encoding='utf-8-sig')))
    summary = json.loads(REFS['summary'].read_text(encoding='utf-8'))
    return {r['subject_id']: r for r in rows}, summary

def capture_rows(part, refs, splits, purposes=('core',), filt=None):
    """(subject, row index, crop name, side, meta) for successful detections."""
    out = []
    for s, r in refs.items():
        if r['split'] not in splits:
            continue
        d = np.load(DERIVED / 'subjects' / f'{s}.npz')
        for i, key in enumerate(d['keys']):
            acc, light, expr, cam, purpose = key.split('|')
            if purpose not in purposes or d['status'][i] != 1:
                continue
            meta = {'subject': s, 'accessory': acc, 'light': light, 'expression': expr, 'camera': cam, 'purpose': purpose,
                    'yaw': float(d['pose'][i, 1]), 'pitch': float(d['pose'][i, 0])}
            if filt and not filt(meta):
                continue
            for crop in PART_CROPS[part]:
                side = crop.split('_')[1] if '_' in crop else None
                out.append((s, i, crop, side, meta))
    return out

class Crops(Dataset):
    def __init__(self, part, rows, refs, norm, transform):
        self.part = part; self.rows = rows; self.transform = transform; self.norm = norm
        self.reg, self.cls = part_features(part); self.refs = refs; self._blobs = {}; self._offsets = {}
        self.soft = {s: SOFT[s] for s in refs if s in SOFT}  # copied so spawned DataLoader workers see it
    def __len__(self):
        return len(self.rows)
    def _crop(self, s, i, crop):
        if s not in self._blobs:
            d = np.load(DERIVED / 'subjects' / f'{s}.npz')
            self._offsets[s] = (list(d['parts']), d['crop_offsets'])
            self._blobs[s] = np.memmap(DERIVED / 'subjects' / f'{s}.bin', dtype=np.uint8, mode='r')
        names, offs = self._offsets[s]; o, n = offs[i, names.index(crop)]
        return Image.open(io.BytesIO(self._blobs[s][o:o + n].tobytes())).convert('RGB')
    def targets(self, s, side):
        r = self.refs[s]; y = []; mask = []
        for k in self.reg:
            f = SCHEMA['features'][k]
            key = f'{k}_{side}_value' if f.get('per_side') and side else f'{k}_value'
            v = r.get(key, '')
            ok = v not in ('', None)
            y.append((float(v) - self.norm[k][0]) / self.norm[k][1] if ok else 0.0); mask.append(1.0 if ok else 0.0)
        c = [SCHEMA['features'][k]['classes'].index(r[k]) if r.get(k) in SCHEMA['features'][k]['classes'] else -1 for k in self.cls]
        return torch.tensor(y, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), torch.tensor(c, dtype=torch.long)
    def soft_targets(self, s):
        """Rater vote shares per visual head (heads x max classes); all-zero rows mean 'use the hard label'."""
        width = max([len(SCHEMA['features'][k]['classes']) for k in self.cls] or [1])
        out = torch.zeros(len(self.cls), width)
        soft = self.soft.get(s)
        for j, k in enumerate(self.cls):
            dist = soft.get(k) if soft else None
            if dist:
                out[j, :len(dist)] = torch.tensor([dist[c] for c in SCHEMA['features'][k]['classes']])
        return out
    def __getitem__(self, idx):
        s, i, crop, side, _ = self.rows[idx]
        y, mask, c = self.targets(s, side)
        return self.transform(self._crop(s, i, crop)), y, mask, c, self.soft_targets(s)

class PartNet(nn.Module):
    def __init__(self, arch, n_reg, cls_sizes, pretrained=True):
        super().__init__()
        if arch == 'efficientnet_b0':
            m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None); dim = m.classifier[1].in_features; m.classifier = nn.Identity()
        elif arch == 'efficientnet_b2':
            m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None); dim = m.classifier[1].in_features; m.classifier = nn.Identity()
        elif arch == 'convnext_tiny':
            m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None); dim = m.classifier[2].in_features; m.classifier[2] = nn.Identity()
        elif arch == 'resnet18':
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None); dim = m.fc.in_features; m.fc = nn.Identity()
        elif arch == 'resnet34':
            m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None); dim = m.fc.in_features; m.fc = nn.Identity()
        elif arch == 'mobilenet_v3_large':
            m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None); dim = m.classifier[0].in_features; m.classifier = nn.Identity()
        else:
            raise ValueError(arch)
        self.backbone = m; self.drop = nn.Dropout(0.3)
        self.reg = nn.Linear(dim, n_reg) if n_reg else None
        self.cls = nn.ModuleList(nn.Linear(dim, k) for k in cls_sizes)
    def forward(self, x):
        h = self.drop(self.backbone(x))
        return (self.reg(h) if self.reg is not None else None), [c(h) for c in self.cls]

def predict(model, loader, device):
    model.eval(); R = []; C = []
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
        for x, *_ in loader:
            r, cs = model(x.to(device, non_blocking=True))
            R.append(r.float().cpu() if r is not None else torch.zeros(len(x), 0)); C.append([c.float().softmax(1).cpu() for c in cs])
    return torch.cat(R).numpy(), [torch.cat([c[j] for c in C]).numpy() for j in range(len(C[0]))] if C and C[0] else []

class SubjectBalancedSampler(torch.utils.data.Sampler):
    """Each epoch draws up to `per_subject` captures per participant, so every person weighs the same."""
    def __init__(self, rows, per_subject, seed):
        self.by = defaultdict(list); self.per = per_subject; self.rng = random.Random(seed)
        for k, r in enumerate(rows):
            self.by[r[0]].append(k)
    def __len__(self):
        return sum(min(self.per, len(v)) for v in self.by.values())
    def __iter__(self):
        idx = []
        for ks in self.by.values():
            idx += self.rng.sample(ks, min(self.per, len(ks)))
        self.rng.shuffle(idx)
        return iter(idx)

def train_part(part, arch, epochs, per_subject, batch, device, workers, cls_weight=1.0, data='base', out=None, train_splits=('train',), val_split='val', seed=SEED):
    out = out or OUT; rng = random.Random(seed); torch.manual_seed(seed); np.random.seed(seed)
    refs, summary = load_references(); reg, cls = part_features(part)
    norm = {k: (summary['features'][k]['train_median'], summary['features'][k]['between_subject_sd']) for k in reg}
    if data == 'more':
        # wider head poses; accessory captures only for the mouth, which eyewear and hats do not cover
        purposes = ('core', 'multiview', 'extra_angle') + (('accessory', 'extra_accessory') if part == 'mouth' else ())
        train_rows = capture_rows(part, refs, train_splits, purposes=purposes, filt=lambda m: abs(m['yaw']) < 32 and abs(m['pitch']) < 32)
    else:
        train_rows = capture_rows(part, refs, train_splits, filt=lambda m: abs(m['yaw']) < 20 and abs(m['pitch']) < 25)
    val_rows = capture_rows(part, refs, [val_split], filt=lambda m: m['camera'] in ('C6', 'C7', 'C8', 'C16') and m['light'] in ('L1', 'L2', 'L3', 'L8', 'L12', 'L19', 'L25', 'L4'))
    ds_train = Crops(part, train_rows, refs, norm, train_transform(part))
    ds_val = Crops(part, val_rows, refs, norm, eval_transform(part))
    cls_sizes = [len(SCHEMA['features'][k]['classes']) for k in cls]
    model = PartNet(arch, len(reg), cls_sizes).to(device).to(memory_format=torch.channels_last)
    # Class weights for visual heads from training participants.
    cls_weights = []
    for j, k in enumerate(cls):
        counts = np.bincount([SCHEMA['features'][k]['classes'].index(refs[s][k]) for s in refs if refs[s]['split'] == 'train' and refs[s].get(k) in SCHEMA['features'][k]['classes']], minlength=cls_sizes[j]).astype(float)
        cls_weights.append(torch.tensor(np.minimum(counts.sum() / (len(counts) * np.maximum(counts, 1)), 5.0), dtype=torch.float32, device=device))  # cap: rare classes (n<10) would dominate
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = epochs * math.ceil(len(set(r[0] for r in train_rows)) * per_subject * len(PART_CROPS[part]) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')
    val_loader = DataLoader(ds_val, batch_size=batch * 2, num_workers=workers, persistent_workers=workers > 0, pin_memory=True)
    best = -1e9; history = []; state = None
    loader = DataLoader(ds_train, batch_size=batch, sampler=SubjectBalancedSampler(train_rows, per_subject * len(PART_CROPS[part]), seed),
                        num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    for epoch in range(epochs):
        model.train()
        t0 = time.time(); tot = 0.0; n = 0
        for x, y, mask, c, soft in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(device); mask = mask.to(device); c = c.to(device); soft = soft.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                r, cs = model(x)
                loss = 0.0
                if r is not None:
                    loss = loss + (nn.functional.smooth_l1_loss(r.float(), y, reduction='none', beta=0.5) * mask).sum() / mask.sum().clamp(min=1)
                for j, logit in enumerate(cs):
                    k_cls = logit.shape[1]; sj = soft[:, j, :k_cls]; has_soft = sj.sum(1) > 0
                    hard = (~has_soft) & (c[:, j] >= 0)  # samples without vote shares still use their hard label
                    if has_soft.any():   # consensus vote shares from three raters, weighted like the hard classes
                        logp = nn.functional.log_softmax(logit.float()[has_soft], 1)
                        w = (sj[has_soft] * cls_weights[j]).sum(1)
                        loss = loss + cls_weight * (-(sj[has_soft] * logp).sum(1) * w).sum() / w.sum().clamp(min=1e-6)
                    if hard.any():
                        loss = loss + cls_weight * nn.functional.cross_entropy(logit.float()[hard], c[hard, j], weight=cls_weights[j], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()) * len(x); n += len(x)
        score, detail = validate(model, ds_val, val_loader, part, device, norm, summary)
        history.append({'epoch': epoch + 1, 'train_loss': round(tot / max(n, 1), 4), 'val_score': round(score, 4), **detail, 'seconds': round(time.time() - t0, 1)})
        print(json.dumps({'part': part, **history[-1]}), flush=True)
        if np.isfinite(score) and score > best:
            best = score; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1
    if state is None:
        raise RuntimeError(f'{part}: no finite validation score in {epochs} epochs')
    model.load_state_dict(state)
    # Uncertainty margin per measured feature: half the validation residual SD (original units).
    R, _ = predict(model, val_loader, device); margins = {}
    for j, k in enumerate(reg):
        y = np.array([ds_val.targets(r[0], r[3])[0][j].item() for r in val_rows]); msk = np.array([ds_val.targets(r[0], r[3])[1][j].item() for r in val_rows]) > 0
        margins[k] = float(0.5 * np.std(R[msk, j] - y[msk]) * norm[k][1]) if msk.any() else None
    meta = {'part': part, 'arch': arch, 'input_size': input_size(part), 'crops': PART_CROPS[part], 'window': WINDOWS[part],
            'regression_features': reg, 'classification_features': cls, 'norm': norm,
            'cuts': {k: summary['features'][k]['cuts'] for k in reg}, 'classes': {k: SCHEMA['features'][k]['classes'] for k in reg + cls},
            'uncertainty_margin': margins, 'selected_epoch': best_epoch, 'val_score': best, 'history': history,
            'schema_version': SCHEMA['version'], 'labels_sha256': summary['labels_sha256'],
            'train_participants': len({r[0] for r in train_rows}), 'train_captures': len(train_rows), 'cls_loss_weight': cls_weight}
    meta.update(data=data, train_splits=list(train_splits), val_split=val_split, seed=seed, soft_labels=bool(SOFT))
    ids_hash = lambda ids: hashlib.sha256(','.join(sorted(ids)).encode()).hexdigest()
    meta.update(train_participants_sha256=ids_hash({r[0] for r in train_rows}), val_participants_sha256=ids_hash({r[0] for r in val_rows}))
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': state, 'meta': meta}, Path(out) / f'{part}.pt')
    return model, meta

def validate(model, ds, loader, part, device, norm, summary):
    R, C = predict(model, loader, device); reg, cls = part_features(part); detail = {}; scores = []
    t = [ds.targets(r[0], r[3]) for r in ds.rows]
    for j, k in enumerate(reg):
        y = np.array([a[0][j].item() for a in t]); m = np.array([a[1][j].item() for a in t]) > 0
        rr = float(np.corrcoef(R[m, j], y[m])[0, 1]); detail[f'r_{k}'] = round(rr, 3); scores.append(rr)
    for j, k in enumerate(cls):
        tc = np.array([a[2][j].item() for a in t]); m = tc >= 0
        if m.any():
            from sklearn.metrics import f1_score
            f1 = f1_score(tc[m], C[j][m].argmax(1), average='macro'); detail[f'f1_{k}'] = round(float(f1), 3); scores.append(f1)
    return float(np.mean(scores)), detail

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', help='all or comma-separated parts'); ap.add_argument('--arch', default='efficientnet_b0')
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument('--per-subject', type=int, default=36)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--cls-weight', type=float, default=0.3, help='weight of visual classification losses relative to regression')
    ap.add_argument('--data', default='base', choices=['base', 'more']); ap.add_argument('--out'); ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--refs', help='reference_labels.csv to use (summary json next to it)')
    ap.add_argument('--soft', help='rater vote-share JSON (data/annotations/kface_visual_v4_soft.json)')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    if args.soft:
        SOFT.update(json.loads(Path(args.soft).read_text(encoding='utf-8')))
    if args.refs:
        REFS['labels'] = Path(args.refs); REFS['summary'] = Path(args.refs).with_name(Path(args.refs).stem.replace('labels', 'summary') + '.json')
    for part in (PART_CROPS if args.part == 'all' else args.part.split(',')):
        train_part(part, args.arch, args.epochs, args.per_subject, args.batch, device, args.workers, args.cls_weight, args.data, args.out, seed=args.seed)

if __name__ == '__main__':
    main()
