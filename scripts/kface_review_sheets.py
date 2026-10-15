"""Numbered contact sheets of reference crops for visual labelling (local review only).

Sheets go to tmp/kface_review/ which is git-ignored and must not be copied into reports:
the K-FACE pledge forbids showing provided photos outside approved users.
"""
import io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED, subjects
from facework.features import ROOT

OUT = ROOT / 'tmp/kface_review'
LAYOUT = {'brow': dict(parts=['brow_right', 'brow_left'], scale=1.0, cols=4, rows=10),
          'nose': dict(parts=['nose'], scale=1.2, cols=10, rows=5),
          'face': dict(parts=['face'], scale=1.0, cols=8, rows=5)}

def reference_crops(subject, parts):
    d = np.load(DERIVED / 'subjects' / f'{subject}.npz'); blob = (DERIVED / 'subjects' / f'{subject}.bin').read_bytes()
    names = list(d['parts']); keys = list(d['keys'])
    for light in ['L1', 'L2', 'L8', 'L12']:
        i = keys.index(f'S001|{light}|E01|C7|core')
        if d['status'][i] == 1:
            out = []
            for part in parts:
                o, n = d['crop_offsets'][i, names.index(part)]
                out.append(Image.open(io.BytesIO(blob[o:o + n])).convert('RGB'))
            return out
    return None

def main(tasks=None, ids=None):
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    ids = ids or subjects(); index = {}
    for task, cfg in LAYOUT.items():
        if tasks and task not in tasks:
            continue
        per = cfg['cols'] * cfg['rows']
        for sheet_no, start in enumerate(range(0, len(ids), per), 1):
            chunk = ids[start:start + per]; tiles = []
            for s in chunk:
                crops = reference_crops(s, cfg['parts'])
                if crops is None:
                    continue
                crops = [c.resize((int(c.width * cfg['scale']), int(c.height * cfg['scale'])), Image.Resampling.LANCZOS) for c in crops]
                tile = Image.new('RGB', (sum(c.width for c in crops) + 4 * (len(crops) - 1), crops[0].height + 20), 'white')
                x = 0
                for c in crops:
                    tile.paste(c, (x, 20)); x += c.width + 4
                tiles.append((s, tile))
            tw = max(t.width for _, t in tiles) + 8; th = max(t.height for _, t in tiles) + 8
            sheet = Image.new('RGB', (tw * cfg['cols'], th * cfg['rows']), '#dde1e6'); draw = ImageDraw.Draw(sheet)
            for k, (s, tile) in enumerate(tiles):
                x, y = (k % cfg['cols']) * tw + 4, (k // cfg['cols']) * th + 4
                sheet.paste(tile, (x, y)); draw.text((x + 3, y + 1), f'{start + k + 1}', fill='black', font=font)
                index[f'{task}:{start + k + 1}'] = s
            sheet.save(OUT / f'{task}_{sheet_no:02}.jpg', quality=90)
    (OUT / 'index.json').write_text(json.dumps(index, indent=0), encoding='utf-8')
    print(len(index), 'tiles')

if __name__ == '__main__':
    main(sys.argv[1:] or None)
