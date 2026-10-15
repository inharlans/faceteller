"""High-resolution numbered contact sheets for the v3 visual labels (local review only, git-ignored).

Regions per participant (reference capture S001/E01/C7, obliques C4/C10 at the same light):
  brows  both brows + glabella + upper lids          -> density, tail clarity, glabella
  eyes   both eyes frontal + one oblique eye          -> eyelid, sclera show, under-eye, depth
  nose   frontal + two 40-degree obliques             -> tip, nostril show, alar fullness, dorsum
  mouth  subnasale to chin top                        -> lip bow, philtrum definition
  face   whole face                                   -> chin shape, cheekbone, cheek fullness
Numbers are the 1-based index in facework.kface.subjects(), same as the v2 sheets.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import DERIVED, Store, subjects
from facework.morphometrics import OVAL, NOSE, MOUTH, EYE, BROW
from scripts.kface_review_html import region

OUT = Path(__file__).resolve().parents[1] / 'tmp/kface_review/v3'
BROWS = BROW['right']['upper'] + BROW['right']['lower'] + BROW['left']['upper'] + BROW['left']['lower']
EYES = EYE['right']['ring'] + EYE['left']['ring']
LAYOUT = {'brows': 4, 'eyes': 2, 'nose': 3, 'mouth': 5, 'face': 5}

def tiles_for(store, s, lights=('L1', 'L2', 'L8', 'L12')):
    d = np.load(DERIVED / 'subjects' / f'{s}.npz'); pos = {k: i for i, k in enumerate(d['keys'])}
    light = next((l for l in lights if d['status'][pos[f'S001|{l}|E01|C7|core']] == 1 and pos.get(f'S001|{l}|E01|C4|multiview') is not None), None)
    if light is None:
        return {}
    ref = f'S001|{light}|E01|C7|core'
    ob = lambda c: f'S001|{light}|E01|{c}|multiview'
    r = lambda key, idx, pad, h: region(store, s, key, d, pos, idx, pad, h)
    def row(*ims, gap=4):
        ims = [i for i in ims if i is not None]
        if not ims:
            return None
        h = max(i.height for i in ims); out = Image.new('RGB', (sum(i.width for i in ims) + gap * (len(ims) - 1), h), 'white'); x = 0
        for i in ims:
            out.paste(i, (x, 0)); x += i.width + gap
        return out
    def near_eye(key):
        i = pos.get(key)
        if i is None or d['status'][i] != 1:
            return None
        p = d['landmarks'][i][:, :2]
        side = max(('right', 'left'), key=lambda sd: np.ptp(p[EYE[sd]['ring'], 0]))  # wider ring = nearer to camera
        return r(key, EYE[side]['ring'], (0.35, 1.6, 1.3), 150)
    return {'brows': r(ref, BROWS, (0.06, 1.2, 2.2), 120),
            'eyes': row(r(ref, EYES, (0.08, 1.6, 1.4), 150), near_eye(ob('C4'))),
            'nose': row(r(ob('C4'), NOSE, (0.55, 0.2, 0.35), 150), r(ref, NOSE, (0.35, 0.1, 0.3), 150), r(ob('C10'), NOSE, (0.55, 0.2, 0.35), 150)),
            'mouth': r(ref, MOUTH + [2, 164], (0.25, 0.25, 0.9), 150),
            'face': r(ref, OVAL, (0.1, 0.06, 0.05), 250)}

ONLY = sys.argv[1:]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    store = Store(capacity=2); ids = subjects(); tiles = {k: [] for k in LAYOUT}
    for n, s in enumerate(ids, 1):
        for k, im in tiles_for(store, s).items():
            if ONLY and k not in ONLY:
                continue
            if im is not None:
                tiles[k].append((n, im))
    for k, cols in LAYOUT.items():
        if ONLY and k not in ONLY:
            continue
        items = tiles[k]; tw = max(i.width for _, i in items) + 10; th = max(i.height for _, i in items) + 26
        rows = {'brows': 10, 'eyes': 9, 'nose': 10, 'mouth': 8, 'face': 6}[k]
        per = cols * rows
        for sheet, start in enumerate(range(0, len(items), per), 1):
            chunk = items[start:start + per]
            img = Image.new('RGB', (tw * cols, th * ((len(chunk) + cols - 1) // cols)), '#d9dee3'); dr = ImageDraw.Draw(img)
            for j, (n, im) in enumerate(chunk):
                x, y = (j % cols) * tw + 5, (j // cols) * th + 3
                dr.text((x, y), str(n), fill='black', font=font); img.paste(im, (x, y + 20))
            img.save(OUT / f'{k}_{sheet:02}.jpg', quality=88)
        print(k, len(items), 'tiles', f'{(len(items) + per - 1) // per} sheets', img.size)

if __name__ == '__main__':
    main()
