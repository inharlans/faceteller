from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw,ImageFont
import json
R=Path(__file__).resolve().parents[2];O=R/'outputs/faceteller-spec';P=O/'페이스텔러_상세요구사항명세서_1.0.pdf';Q=R/'tmp/faceteller-reference/review';Q.mkdir(parents=True,exist_ok=True)
r=PdfReader(P);pdf=pdfium.PdfDocument(P);stats=[]
for i,p in enumerate(r.pages):
 t=p.extract_text() or '';lines=[l.strip() for l in t.splitlines() if l.strip()];stats.append({'page':i+1,'chars':len(t),'head':lines[1:4],'tail':lines[-4:-1]})
(Q/'pdf_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(Q/'pdf_text.txt').write_text('\n\n'.join(f'=== PAGE {i+1} ===\n'+(p.extract_text() or '') for i,p in enumerate(r.pages)),encoding='utf-8')
f=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
for start in range(0,len(pdf),12):
 im=Image.new('RGB',(1040,1520),'#c7c7c7');d=ImageDraw.Draw(im)
 for i in range(start,min(start+12,len(pdf))):
  thumb=pdf[i].render(scale=.43).to_pil().convert('RGB');thumb.thumbnail((245,345));x=10+((i-start)%4)*260;y=30+((i-start)//4)*500
  im.paste(thumb,(x,y));d.text((x,y-23),f'PAGE {i+1}',fill='black',font=f)
 im.save(Q/f'contact-{start+1}.jpg')
for i in [0,2,3,4,5,16,17,23,25,27,30,31,35,40,44,len(pdf)-1]:
 if i<len(pdf):pdf[i].render(scale=1.3).to_pil().save(Q/f'page-{i+1}.png')
print('PAGES',len(r.pages));print('LOW_TEXT_PAGES',[(x['page'],x['chars']) for x in stats if x['chars']<250])
