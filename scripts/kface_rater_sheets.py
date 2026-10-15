"""Blind contact sheets for independent visual raters (different lighting, shuffled order, fresh numbers).

python scripts/kface_rater_sheets.py B L8   -> tmp/kface_review/raters/B/{brows,eyes,nose,mouth,face}_NN.jpg
The number -> participant key is written to tmp/kface_review/raters/_keys/B.json (raters must not read it).
"""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import Store, subjects
from scripts.kface_review_sheets_v3 import tiles_for, LAYOUT

ROOT = Path(__file__).resolve().parents[1]
ROWS = {'brows': 10, 'eyes': 9, 'nose': 10, 'mouth': 8, 'face': 6}

def main(rater, light):
    out = ROOT / 'tmp/kface_review/raters' / rater; out.mkdir(parents=True, exist_ok=True)
    keys = ROOT / 'tmp/kface_review/raters/_keys'; keys.mkdir(parents=True, exist_ok=True)
    ids = subjects(); order = ids[:]; random.Random(f'rater-{rater}').shuffle(order)
    number = {s: i + 1 for i, s in enumerate(order)}
    (keys / f'{rater}.json').write_text(json.dumps({str(v): k for k, v in number.items()}), encoding='utf-8')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16); store = Store(capacity=2)
    lights = (light, 'L12', 'L2', 'L1') if light != 'L12' else ('L12', 'L8', 'L2', 'L1')
    tiles = {k: [] for k in LAYOUT}
    for s in order:
        for k, im in tiles_for(store, s, lights).items():
            if im is not None:
                tiles[k].append((number[s], im))
    for k, cols in LAYOUT.items():
        items = tiles[k]; tw = max(i.width for _, i in items) + 10; th = max(i.height for _, i in items) + 26; per = cols * ROWS[k]
        for sheet, start in enumerate(range(0, len(items), per), 1):
            chunk = items[start:start + per]
            img = Image.new('RGB', (tw * cols, th * ((len(chunk) + cols - 1) // cols)), '#d9dee3'); dr = ImageDraw.Draw(img)
            for j, (n, im) in enumerate(chunk):
                x, y = (j % cols) * tw + 5, (j // cols) * th + 3
                dr.text((x, y), str(n), fill='black', font=font); img.paste(im, (x, y + 20))
            img.save(out / f'{k}_{sheet:02}.jpg', quality=88)
    print(rater, {k: len(v) for k, v in tiles.items()})

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
