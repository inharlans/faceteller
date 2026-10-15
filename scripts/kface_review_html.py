"""Standalone labelling page (single HTML, images embedded) for human review of visual labels.

  python scripts/kface_review_html.py            # 80-person stratified form (outputs/kface-v2/독립검수_시각레이블_양식.csv)
  python scripts/kface_review_html.py --all      # all 400 participants, shuffled order

The file contains K-FACE photos, so it is written to tmp/kface_review/ (git-ignored) and must
stay with approved users: never upload, publish or e-mail it. The page never shows AI labels.
It exports JSON with participant IDs and labels only; score it with
  python scripts/kface_agreement.py score <file.json>
"""
import argparse, base64, csv, io, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from facework.kface import DERIVED, Store, subjects
from facework.morphometrics import OVAL, NOSE, align

OUT = ROOT / 'tmp/kface_review'
FORM = ROOT / 'outputs/kface-v2/독립검수_시각레이블_양식.csv'
SCHEMA = json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))
GROUP_KO = {'brow': '눈썹', 'eye': '눈', 'nose': '코', 'mouth': '입', 'face': '얼굴'}
HINTS = {  # what to look at; kept short so the page stays readable
    'brow_shape': '눈썹 띠 전체의 중심선. 털 밀도·색은 보지 않습니다.',
    'brow_thickness': '털 영역의 세로 두께(눈 크기 대비).',
    'brow_density': '털이 피부를 얼마나 촘촘히 덮는지. 굵기와 따로 봅니다.',
    'brow_tail_clarity': '바깥쪽 꼬리 끝까지 선이 또렷한지.',
    'eyelid': '윗눈꺼풀의 주름선. 사선 사진이 도움이 됩니다.',
    'sclera_show': '정면을 볼 때 검은자 위·아래로 흰자가 드러나는지.',
    'eye_depth': '사선 사진에서 눈두덩 대비 안구 위치.',
    'nose_tip': '코끝(콧방울 사이 끝부분)의 윤곽.',
    'nostril_show': '정면 사진에서 콧구멍이 보이는 정도.',
    'alar_fullness': '콧방울(양쪽 날개)의 크기와 볼록함.',
    'lip_bow': '윗입술 가운데 M자 윤곽(큐피드 활)의 뚜렷함.',
    'philtrum_definition': '코밑~윗입술 사이 골의 뚜렷함. 수염이면 판정 불가.',
    'chin_shape': '턱끝 윤곽. 수염이면 판정 불가.',
    'cheekbone': '눈 아래 옆쪽 광대가 도드라지는지.',
    'cheek_fullness': '볼의 살집. 턱선보다 볼 가운데를 봅니다.',
    'face_shape': '헤어라인은 무시하고 보이는 외곽의 주된 형태.'}
ORDER = ['brow', 'eye', 'nose', 'mouth', 'face']
TASKS = [{'key': k, 'group': GROUP_KO[f['part']], 'title': f['ko'], 'hint': HINTS.get(k, ''), 'required': True,
          'options': [[c, ko, ''] for c, ko in zip(f['classes'], f['classes_ko'])]}
         for part in ORDER for k, f in SCHEMA['features'].items() if f['source'] == 'visual' and f['part'] == part]
PANELS = [{'key': 'face', 'caption': '정면 · 무표정', 'main': True},
          {'key': 'brows', 'caption': '눈썹 · 미간'}, {'key': 'eyes', 'caption': '눈 · 정면(왼쪽) · 40° 사선(오른쪽)'},
          {'key': 'nose', 'caption': '코 · 40° · 정면 · 반대쪽 40°'}, {'key': 'mouth', 'caption': '입 · 인중 · 턱끝'}]

def jpeg(im, quality=86):
    buf = io.BytesIO(); im.convert('RGB').save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def region(store, subject, key, d, pos, idx, pad, height):
    """Crop around landmark indices from the original capture, roll-aligned, resized to `height`."""
    i = pos.get(key)
    if i is None or d['status'][i] != 1:
        return None
    s, l, e, c, _ = key.split('|')
    im = store.image(subject, s, l, e, c); arr = np.asarray(im)
    q, M, _ = align(d['landmarks'][i][:, :2].astype(float), arr.shape[1], arr.shape[0])
    import cv2
    rot = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    pts = q[idx]; lo, hi = pts.min(0), pts.max(0); ext = hi - lo
    box = [int(max(0, lo[0] - ext[0] * pad[0])), int(max(0, lo[1] - ext[1] * pad[1])),
           int(min(arr.shape[1], hi[0] + ext[0] * pad[0])), int(min(arr.shape[0], hi[1] + ext[1] * pad[2]))]
    crop = Image.fromarray(rot).crop(box)
    return crop.resize((max(1, int(crop.width * height / crop.height)), height), Image.Resampling.LANCZOS)

def stored_crop(d, blob, i, part, scale):
    names = list(d['parts']); o, n = d['crop_offsets'][i, names.index(part)]
    im = Image.open(io.BytesIO(blob[o:o + n]))
    return im.resize((int(im.width * scale), int(im.height * scale)), Image.Resampling.LANCZOS)

def item(store, subject):
    from scripts.kface_review_sheets_v3 import tiles_for
    tiles = tiles_for(store, subject)
    if not tiles:
        return None
    return {'id': subject, 'images': {k: jpeg(v) for k, v in tiles.items() if v is not None}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--all', action='store_true'); args = ap.parse_args()
    if args.all:
        ids = subjects(); random.Random(20260926).shuffle(ids); tag = f'전체{len(ids)}명'
    else:
        ids = [r['subject_id'] for r in csv.DictReader(FORM.open(encoding='utf-8-sig'))]; tag = f'표본{len(ids)}명'
    store = Store(capacity=2); items = []
    for n, s in enumerate(ids):
        it = item(store, s)
        if it:
            items.append(it)
        if n % 20 == 0:
            print(n, len(items), flush=True)
    page = (ROOT / 'web/labeling_template.html').read_text(encoding='utf-8')
    page = page.replace('/*__DATA__*/null', json.dumps({'set': tag, 'tasks': TASKS, 'panels': PANELS, 'items': items}, ensure_ascii=False))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'kface_labeling_v3_{"all" if args.all else len(items)}.html'
    path.write_text(page, encoding='utf-8')
    print(path, f'{path.stat().st_size / 1e6:.1f} MB', len(items))

if __name__ == '__main__':
    main()
