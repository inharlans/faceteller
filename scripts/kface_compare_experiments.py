"""Compare training experiments on the validation split (models/exp/<name>/<part>.pt histories).

Prints, per head, the validation Pearson r (measured features) or macro F1 (visual features) at each
experiment's selected epoch, plus the mean per part. Test data is not touched.
"""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT

PARTS = ['eye', 'brow', 'nose', 'mouth', 'face']

def main(root=ROOT / 'models/exp'):
    exps = sorted(p for p in Path(root).iterdir() if p.is_dir())
    table = {}
    for e in exps:
        for part in PARTS:
            f = e / f'{part}.pt'
            if not f.exists():
                continue
            meta = torch.load(f, map_location='cpu', weights_only=False)['meta']
            h = meta['history'][meta['selected_epoch'] - 1]
            table.setdefault(e.name, {})[part] = {k: v for k, v in h.items() if k.startswith(('r_', 'f1_'))} | {'_score': h['val_score'], '_sec': sum(x['seconds'] for x in meta['history'])}
    heads = sorted({k for t in table.values() for p in t.values() for k in p if not k.startswith('_')}, key=lambda k: (k[:2], k))
    names = list(table)
    print('head'.ljust(26) + ''.join(n[:18].rjust(20) for n in names))
    for part in PARTS:
        for k in heads:
            vals = [table[n].get(part, {}).get(k) for n in names]
            if any(v is not None for v in vals):
                best = max(v for v in vals if v is not None)
                print(f'{part}:{k}'.ljust(26) + ''.join((f"{v:.3f}{'*' if v == best else ' '}" if v is not None else '-').rjust(20) for v in vals))
        print(f'{part}:MEAN'.ljust(26) + ''.join((f"{table[n][part]['_score']:.3f}" if part in table[n] else '-').rjust(20) for n in names))
    print('minutes'.ljust(26) + ''.join(f"{sum(p['_sec'] for p in table[n].values()) / 60:.0f}".rjust(20) for n in names))
    (ROOT / 'models/exp/comparison.json').write_text(json.dumps(table, indent=1), encoding='utf-8')

if __name__ == '__main__':
    main()
