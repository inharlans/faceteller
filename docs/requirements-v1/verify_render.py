"""Check document coverage and render every PDF page for visual review."""
from pathlib import Path
import csv, json, re
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/requirements-v1'
TMP=ROOT/'tmp/requirements-v1'
TMP.mkdir(parents=True,exist_ok=True)
content=json.loads((OUT/'설계서_구조화원본.json').read_text(encoding='utf8'))
readcsv=lambda name:list(csv.DictReader((OUT/name).open(encoding='utf-8-sig',newline='')))
req=readcsv('요구사항_추적대장.csv')
tests=readcsv('수용시험_계획대장.csv')
ids={r['id'] for r in req}
assert len(ids)==len(req)==len(tests)==104
assert ids=={r['id'] for r in content['requirements']}
assert ids=={r['requirement_id'] for r in tests}
assert all(t['status']=='미실행(계획)' and not t['evidence'] for t in tests)
assert next(r for r in req if r['id']=='FR-024')['release']=='R1/R1.1'
assert len(content['pages'])==62 and len(content['decisions'])==12
pdfpath=OUT/'상세_요구사항_설계서_v0.9.pdf'
reader=PdfReader(pdfpath)
pages=[p.extract_text() or '' for p in reader.pages]
full='\n'.join(pages)
missing=sorted(ids-set(re.findall(r'\b(?:FR|DR|LR|KR|ML|NF)-\d{3}\b',full)))
assert not missing,missing
assert '\ufffd' not in full
report={'pdf_pages':len(pages),'sections':62,'requirements':104,'feature_candidates':38,'decisions':12,'planned_tests':104,'product_tests_executed':0,'missing_requirement_ids':missing,'page_text_lengths':[len(t) for t in pages],'short_pages':[i+1 for i,t in enumerate(pages) if len(t)<150],'outline_entries':len(reader.outline),'text_extraction_replacement_characters':full.count('\ufffd')}
(TMP/'pdf_text.txt').write_text('\n\n'.join(f'--- PAGE {i+1} ---\n{t}' for i,t in enumerate(pages)),encoding='utf8')
pdf=pdfium.PdfDocument(str(pdfpath))
thumbs=[]
for i in range(len(pdf)):
    page=pdf[i]
    bitmap=page.render(scale=1.0)
    im=bitmap.to_pil().convert('RGB')
    im.save(TMP/f'page-{i+1:03}.png')
    tile=Image.new('RGB',(320,480),'#d9dfe5')
    small=ImageOps.contain(im,(304,440))
    tile.paste(small,((320-small.width)//2,8))
    ImageDraw.Draw(tile).text((12,456),f'PAGE {i+1} | {len(pages[i])} chars',fill='black')
    thumbs.append(tile)
    bitmap.close();page.close()
for start in range(0,len(thumbs),12):
    sheet=Image.new('RGB',(1280,1440),'white')
    for j,im in enumerate(thumbs[start:start+12]):sheet.paste(im,((j%4)*320,(j//4)*480))
    sheet.save(TMP/f'contact-{start//12+1:02}.png')
(OUT/'문서_검수결과.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False))
