"""Unpack the K-FACE High_Resolution delivery into one ZIP per participant.

The outer archive holds 400 participant ZIPs (about 10,800 JPEGs each). Expanding all
4.3M JPEGs as loose files is slow on NTFS and adds no information, so the participant
ZIPs are stored as delivered and read in place by facework.kface. CRCs are verified
while copying. Use --loose SUBJECT to expand one participant for manual browsing.
"""
import argparse, hashlib, json, sys, time, zipfile, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facework.kface import RAW, ZIPS, DEFAULT_SOURCE

def copy_member(source, name):
    target = ZIPS / name
    with zipfile.ZipFile(source) as z:
        info = z.getinfo(name)
        if target.exists() and target.stat().st_size == info.file_size:
            return name, 'kept'
        temp = target.with_suffix('.part')
        with z.open(info) as src, temp.open('wb') as dst:  # ZipExtFile raises on CRC mismatch
            shutil.copyfileobj(src, dst, 8 << 20)
        temp.replace(target)
    return name, 'extracted'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=str(DEFAULT_SOURCE))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--loose', help='also expand this participant ID into data/raw/kface/loose/<ID>')
    args = ap.parse_args()
    source = Path(args.source)
    ZIPS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as z:
        names = [i.filename for i in z.infolist() if i.filename.endswith('.zip')]
    start = time.time(); done = 0; status = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for name, state in pool.map(lambda n: copy_member(source, n), names):
            status[name] = state; done += 1
            if done % 20 == 0:
                print(f'{done}/{len(names)} {time.time()-start:.0f}s', flush=True)
    inventory = []
    for name in names:
        with zipfile.ZipFile(ZIPS / name) as z:
            members = [i for i in z.infolist() if not i.is_dir()]
        inventory.append({'subject_id': name[:-4], 'zip_bytes': (ZIPS / name).stat().st_size,
                          'jpg': sum(m.filename.endswith('.jpg') for m in members),
                          'txt': sum(m.filename.endswith('.txt') for m in members)})
    manifest = {'dataset': 'K-FACE High_Resolution', 'source_archive': source.name,
                'source_bytes': source.stat().st_size, 'participants': len(names),
                'jpg_total': sum(r['jpg'] for r in inventory), 'txt_total': sum(r['txt'] for r in inventory),
                'crc_verified': True, 'layout': 'data/raw/kface/zips/<subject>.zip -> S###/L##/E##/C##.jpg',
                'elapsed_seconds': round(time.time() - start, 1), 'participants_detail': inventory}
    (RAW / 'extract_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'participants_detail'}))
    if args.loose:
        with zipfile.ZipFile(ZIPS / f'{args.loose}.zip') as z:
            z.extractall(RAW / 'loose' / args.loose)

if __name__ == '__main__':
    main()
