import re, csv, json
from pathlib import Path
from urllib.parse import quote
ROOT=Path(__file__).resolve().parents[1]

def sections(volume):
    text=(ROOT/f'data/sources/wikisource_{volume}.txt').read_text(encoding='utf-8')
    matches=list(re.finditer(r'\n([^\n]+)\n\[\n编辑\n\]|\n===([^=]+)===',text))
    out={}
    for i,m in enumerate(matches):
        name=m[1] or m[2]
        end=matches[i+1].start() if i+1<len(matches) else text.find('此作品在全世界',m.end())
        body=re.sub(r'\s+|\u200b','',text[m.end():end])
        out[name]=body
    return out

def main():
    sec={633:sections(633),634:sections(634)}
    definitions=list(csv.DictReader((ROOT/'data/literature_morphology.tsv').open(encoding='utf-8'),delimiter='\t'))
    counts={};records=[]
    for d in definitions:
        name=d['name_zh'];part='brow' if name.endswith('眉') else 'nose' if name.endswith('鼻') else 'mouth' if name.endswith('口') else 'eye'
        v=633 if part in ['eye','brow'] else 634
        assert name in sec[v],name
        counts[part]=counts.get(part,0)+1
        records.append({'record_id':f'SXQP{v}-{part}-{counts[part]:03}',**d,'part':part,'original_section':sec[v][name],
          'source_url':f'https://zh.wikisource.org/wiki/欽定古今圖書集成/博物彙編/藝術典/第{v}卷#'+quote(name),
          'source_version':'2026-09-21 retrieved snapshot, SHA256 in data/sources/manifest.json',
          'translation_status':'assistant_morphology_gloss_not_expert_verified','personality_inference_allowed':False,
          'traditional_type_training_approved':False,
          'reuse_notice':'원저작 공공영역; Wikisource 편집·구두점 CC BY-SA 4.0, 원문·출처 보존. 한국어 형태 설명은 이 프로젝트의 초벌 설명.'})
    out=ROOT/'data/derived';out.mkdir(exist_ok=True,parents=True)
    (out/'literature.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'literature.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps({'literature_records':len(records),'parts':counts}))
if __name__=='__main__':main()
