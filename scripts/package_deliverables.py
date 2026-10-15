import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/face-dataset-v1';OUT.mkdir(exist_ok=True,parents=True)
# Keep human-facing outputs together; implementation remains at repository root.
for src in ['docs/구축보고서.md','docs/학습과확장.md','data/derived/annotations.csv','data/derived/training_manifest.csv','data/derived/measurements.csv','data/derived/literature.csv','data/derived/images.csv','data/derived/summary.json','models/pilot/training_report.json']:
    shutil.copy2(ROOT/src,OUT/Path(src).name)
archive=ROOT/'outputs/face_dataset_workbench_v1.zip'
files=[]
for folder in ['facework','scripts','tests','web','docs']:
    files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [ROOT/'README.md',ROOT/'requirements-lock.txt',ROOT/'start-workbench.ps1',ROOT/'.gitignore']
for folder in ['data/annotations']:
    files += [p for p in (ROOT/folder).rglob('*') if p.is_file()]
files += [p for p in (ROOT/'data/sources').iterdir() if p.is_file() and (p.name.startswith(('wikisource_','ctext')) or p.name=='manifest.json')]
files += list((ROOT/'data').glob('*.json'))+list((ROOT/'data').glob('*.tsv'))
# Original archives are included once; the downloader extracts verified archives on restore.
files += [p for p in (ROOT/'data/raw/frll').iterdir() if p.is_file()]
files += [p for p in (ROOT/'data/derived').iterdir() if p.is_file()]
files += [p for p in (ROOT/'data/derived/crops').rglob('*.jpg')]
files += [p for p in (ROOT/'data/derived/aligned').glob('*.jpg')]
files += [p for p in (ROOT/'data/derived/landmarks').glob('*.json')]
files += [p for p in (ROOT/'models/pilot').iterdir() if p.is_file()]
files += [p for p in (ROOT/'models').iterdir() if p.is_file()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.xlsx','.csv','.md','.json'] and p.name!='delivery_manifest.json']
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
manifest={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(set(files)),'restore':'Create Python 3.12 venv; install requirements; run scripts/download_frll.py to extract included verified ZIPs. Then start-workbench.ps1. Node artifact runtime is only required to rebuild Excel, not to use the workbook or web app.','limitations':['Korean AI Hub data not acquired','AI visual labels are not independent human ground truth','traditional named type ground truth not completed','local research prototype, not public production deployment']}
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest))
