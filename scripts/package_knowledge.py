import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/knowledge-v2'
def main():
 files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
 manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 target=ROOT/'outputs/관상_문헌해석_측정정의_v2.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.iterdir()):
   if p.is_file():z.write(p,'knowledge-v2/'+p.name)
  for rel in ['data/interpretation_annotations.tsv','scripts/build_interpretation_catalog.py','scripts/build_type_requirements.py','scripts/build_measurement_pilot.py','scripts/build_knowledge_page.py','scripts/package_knowledge.py','facework/measurement_rules.py','web/knowledge-ui.js','tests/test_knowledge.py']:
   z.write(ROOT/rel,'source/'+rel)
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'output_files':len(files)+1},ensure_ascii=False))
if __name__=='__main__':main()
