"""Materialize a versioned ImageFolder tree without changing original files."""
import csv,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/derived/training_manifest.csv'
version=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
dest=ROOT/'data/derived/classified'/version;dest.mkdir(parents=True,exist_ok=True)
with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
for r in rows:
    target=dest/r['task']/r['split']/r['manual_label']/f"{r['image_id']}__{r['part']}.jpg"
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:os.link(ROOT/r['crop_path'],target)
        except OSError:shutil.copy2(ROOT/r['crop_path'],target)
shutil.copy2(source,dest/'training_manifest.csv')
(dest/'README.txt').write_text('Pilot morphology labels from a single AI visual rater. Unknown labels and smiling images excluded.\nUse subject-aware splits already provided. Left-eye and left-brow images are NOT mirrored in this export; facework.training mirrors them during loading.\nImages: DeBruine & Jones, Face Research Lab London Set, CC BY 4.0.\nhttps://doi.org/10.6084/m9.figshare.5047666.v5\n',encoding='utf-8')
(ROOT/'data/derived/training_set_location.json').write_text(json.dumps({'version':version,'path':dest.relative_to(ROOT).as_posix(),'images':len(rows)},indent=2),encoding='utf-8')
print(json.dumps({'path':str(dest),'images':len(rows)}))
