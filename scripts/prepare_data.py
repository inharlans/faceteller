"""Reproducible extraction; measurements are estimates, not human truth labels."""
import sys, json, hashlib, csv, urllib.request, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw, ImageFont, ImageOps
from facework.features import ROOT, MODEL, MODEL_URL, detector, analyze

OUT=ROOT/'data/derived'
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    for folder in [OUT/'crops',OUT/'aligned',OUT/'landmarks',ROOT/'outputs/review']:
        folder.mkdir(parents=True,exist_ok=True)
    MODEL.parent.mkdir(exist_ok=True)
    if not MODEL.exists():urllib.request.urlretrieve(MODEL_URL,MODEL)
    (MODEL.parent/'face_landmarker_provenance.json').write_text(json.dumps({'url':MODEL_URL,'sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'date':'2026-09-21'}),encoding='utf-8')
    subjects=sorted({p.stem.split('_')[0] for p in (ROOT/'data/raw/frll/neutral_front').rglob('*.jpg')})
    ordered=sorted(subjects,key=lambda s:hashlib.sha256(('face-v1:'+s).encode()).hexdigest())
    annotation_path=ROOT/'data/annotations/visual_v1.tsv'
    if annotation_path.exists():
        # Split v2: choose only from labels, never from model scores. Require class coverage.
        annotations=list(csv.DictReader(annotation_path.open(encoding='utf-8'),delimiter='\t'))
        attrs={r['subject_id']:{(k,r[k]) for k in ['eye_right','brow_right','nose','mouth','face'] if r[k]!='unknown'} for r in annotations}
        all_attrs=set.union(*attrs.values());rng=random.Random(20260921)
        for attempt in range(100000):
            candidate=subjects.copy();rng.shuffle(candidate)
            groups=[candidate[:72],candidate[72:87],candidate[87:]]
            if all(all(sum(a in attrs[s] for s in group)>=minimum for a in all_attrs) for group,minimum in zip(groups,[3,2,2])):
                ordered=candidate;break
        else:raise RuntimeError('Cannot create subject splits with class coverage')
        (OUT/'split_protocol.json').write_text(json.dumps({'version':'v2_class_coverage','seed':20260921,'candidate_attempt':attempt+1,'selection_uses':'labels_only_no_model_outputs','minimum_subjects_per_label':{'train':3,'val':2,'test':2},'reason':'v1 random split had zero round and long face examples in test; v1 metrics retired.'},indent=2),encoding='utf-8')
    splits={s:('train' if i<72 else 'val' if i<87 else 'test') for i,s in enumerate(ordered)}
    records=[]; crops=[]; measures=[]
    with detector() as landmarker:
        for condition in ['neutral_front','smiling_front']:
            for path in sorted((ROOT/'data/raw/frll'/condition).rglob('*.jpg')):
                subject=path.stem.split('_')[0]; image_id=f'frll_{path.stem}'
                im=Image.open(path).convert('RGB')
                r=analyze(im,landmarker)
                record={'image_id':image_id,'subject_id':'frll_'+subject,'condition':condition,'split':splits[subject], 'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':im.width,'height':im.height,'landmark_status':r['status'],'dataset':'FRLL','license':'CC BY 4.0','purpose':'morphology' if condition=='neutral_front' else 'expression_stress_test'}
                records.append(record)
                if r['status']!='ok':continue
                r['aligned'].save(OUT/'aligned'/f'{image_id}.jpg',quality=95)
                for part,impart in r['crops'].items():
                    cp=OUT/'crops'/part/f'{image_id}.jpg';cp.parent.mkdir(exist_ok=True)
                    impart.save(cp,quality=95)
                    crops.append({'annotation_id':f'{image_id}:{part}','image_id':image_id,'subject_id':record['subject_id'],'condition':condition,'split':record['split'],'part':part,'crop_path':cp.relative_to(ROOT).as_posix(),'box_aligned':json.dumps(r['boxes_aligned'][part]),'status':'measured_unreviewed','manual_label':'','notes':''})
                measures.append({'image_id':image_id,**r['metrics']})
                saved={k:v for k,v in r.items() if k not in ['aligned','crops']}
                (OUT/'landmarks'/f'{image_id}.json').write_text(json.dumps(saved),encoding='utf-8')
            print(condition,'processed',len(records),flush=True)
    for name,rows in [('images',records),('crops',crops),('measurements',measures)]:
        (OUT/f'{name}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');write_csv(OUT/f'{name}.csv',rows)
    # All neutral individuals in readable numbered sheets. Each face was cropped from actual pixels.
    neutral=[r for r in records if r['condition']=='neutral_front' and r['landmark_status']=='ok']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for start in range(0,len(neutral),12):
        sheet=Image.new('RGB',(1600,1440),'#f2f4f7');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(neutral[start:start+12]):
            x=(j%4)*400;y=(j//4)*480
            im=Image.open(OUT/'crops/face'/f"{r['image_id']}.jpg")
            im=ImageOps.contain(im,(380,438))
            sheet.paste(im,(x+(400-im.width)//2,y+38))
            draw.text((x+10,y+7),r['image_id'],fill='#172b4d',font=font)
        sheet.save(ROOT/'outputs/review'/f'faces_{start//12+1:02}.jpg',quality=94)
    print(json.dumps({'images':len(records),'subjects':len(subjects),'crops':len(crops),'measurements':len(measures),'split_subjects':{s:list(splits.values()).count(s) for s in ['train','val','test']}}))

if __name__=='__main__':main()
