import json, hashlib, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/sources'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
 'wikisource_633': 'https://zh.wikisource.org/wiki/欽定古今圖書集成/博物彙編/藝術典/第633卷',
 'wikisource_634': 'https://zh.wikisource.org/wiki/欽定古今圖書集成/博物彙編/藝術典/第634卷',
 'ctext': 'https://ctext.org/wiki.pl?if=gb&chapter=905153&remap=gb',
 'aihub83': 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=83',
 'celeba': 'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html',
 'cfd': 'https://www.chicagofaces.org/download/',
 'mediapipe': 'https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker',
}

def fetch(item):
    from urllib.parse import quote
    from bs4 import BeautifulSoup
    name,url = item
    record = dict(id=name, url=url, retrieved_at='2026-09-21')
    try:
        req = urllib.request.Request(quote(url, safe=':/?=&%'), headers={'User-Agent':'FaceMorphologyResearch/0.1 (local academic prototype)'})
        with urllib.request.urlopen(req, timeout=45) as r:
            body = r.read()
            record['http_status'] = r.status
        (OUT/f'{name}.html').write_bytes(body)
        soup = BeautifulSoup(body, 'html.parser')
        main = soup.select_one('.mw-parser-output') if 'wikisource' in name else soup
        text = main.get_text('\n', strip=True)
        (OUT/f'{name}.txt').write_text(text, encoding='utf-8')
        record.update(sha256=hashlib.sha256(body).hexdigest(), chars=len(text))
    except Exception as e:
        record['error'] = str(e)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    (OUT/'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
