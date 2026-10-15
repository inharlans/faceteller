"""Download publisher files with checksum verification; never invent or mirror URLs."""
import hashlib, json, urllib.request, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/frll'
DEST.mkdir(parents=True, exist_ok=True)
url = 'https://api.figshare.com/v2/articles/5047666'
with urllib.request.urlopen(url, timeout=60) as response:
    metadata = json.load(response)
(DEST / 'publisher_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

def download(f):
    target = DEST / f['name']
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != f['computed_md5']:
        temp = target.with_suffix(target.suffix + '.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f['download_url'], timeout=90) as source, temp.open('wb') as out:
                    while chunk := source.read(1024 * 1024):
                        out.write(chunk)
                assert temp.stat().st_size == f['size'], 'size mismatch'
                assert hashlib.md5(temp.read_bytes()).hexdigest() == f['computed_md5'], 'MD5 mismatch'
                temp.replace(target)
                break
            except Exception:
                if attempt == 2: raise
    if target.suffix == '.zip':
        folder = DEST / target.stem
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as z:
            for member in z.infolist():
                resolved = (folder / member.filename).resolve()
                if not resolved.is_relative_to(folder.resolve()):
                    raise ValueError('Unsafe archive path')
                if not member.filename.startswith('__MACOSX/'):
                    z.extract(member, folder)
    result = dict(name=f['name'], bytes=target.stat().st_size, md5=f['computed_md5'], sha256=hashlib.sha256(target.read_bytes()).hexdigest(), url=f['download_url'])
    print(json.dumps(result), flush=True)
    return result

# Neutral front for morphology and smiling front for condition robustness.
selected = [f for f in metadata['files'] if f['name'] in ['neutral_front.zip', 'smiling_front.zip', 'london_faces_info.csv']]
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(download, selected))
(DEST / 'download_manifest.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
