from pathlib import Path
import csv,json,re
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/experiment-report-2026-10-01';TMP=ROOT/'tmp/experiment-report';TMP.mkdir(exist_ok=True)
readcsv=lambda n:list(csv.DictReader((OUT/n).open(encoding='utf-8-sig',newline='')))
features=readcsv('전체_38피처_실험성능.csv');models=readcsv('모델_50개_학습설정.csv');epochs=readcsv('모델_에포크별_학습이력.csv')
assert len(features)==38 and len(models)==50 and len(epochs)==275
assert len({r['feature_id'] for r in features})==38
assert sum(r['method']=='랜드마크' for r in features)==10
content=json.loads((OUT/'보고서_구조화원본.json').read_text(encoding='utf8'))
pdfpath=OUT/'실제_실험결과_보고서.pdf';reader=PdfReader(pdfpath)
texts=[p.extract_text() or '' for p in reader.pages];plain=re.sub(r'\s+','','\n'.join(texts))
assert all(re.sub(r'\s+','',r['feature'].split('(')[0]) in plain for r in features)
assert all(re.sub(r'\s+','',p['title']) in plain for p in content['pages'])
assert '\ufffd' not in plain
(TMP/'PDF_추출본문.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)),encoding='utf8')
pdf=pdfium.PdfDocument(str(pdfpath));thumbs=[]
for i in range(len(pdf)):
 page=pdf[i];bitmap=page.render(scale=1.1);im=bitmap.to_pil().convert('RGB');im.save(TMP/f'page-{i+1:03}.png')
 tile=Image.new('RGB',(320,480),'#d9dfe5');small=ImageOps.contain(im,(304,440));tile.paste(small,((320-small.width)//2,8))
 ImageDraw.Draw(tile).text((12,456),f'PAGE {i+1} | {len(texts[i])} chars',fill='black');thumbs.append(tile);bitmap.close();page.close()
for start in range(0,len(thumbs),12):
 sheet=Image.new('RGB',(1280,1440),'white')
 for j,im in enumerate(thumbs[start:start+12]):sheet.paste(im,((j%4)*320,(j//4)*480))
 sheet.save(TMP/f'contact-{start//12+1:02}.png')
q={'pages':len(texts),'sections':len(content['pages']),'features':38,'models':50,'epoch_records':275,'missing_feature_names':[],'short_pages':[i+1 for i,t in enumerate(texts) if len(t)<120],'page_text_lengths':[len(t) for t in texts],'scope':'문서와 보관 실험 집계 검증. 새 학습·추론 실행 아님.'}
(OUT/'보고서_검수결과.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(q,ensure_ascii=False))
