import csv,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/face-dataset-v1';OUT.mkdir(exist_ok=True,parents=True)
load=lambda n:json.loads((ROOT/f'data/derived/{n}.json').read_text(encoding='utf-8'))
images=load('images');ann=load('annotations');schema=json.loads((ROOT/'data/label_schema.json').read_text(encoding='utf-8'))
assert len(images)==204 and len(set(i['subject_id'] for i in images))==102
assert len(ann)==1428 and len({a['annotation_id'] for a in ann})==1428
assert len({i['sha256'] for i in images})==204
for image in images:
    p=ROOT/image['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==image['sha256']
    with Image.open(p) as im:assert im.size==(1350,1350);im.verify()
for row in ann:
    with Image.open(ROOT/row['crop_path']) as im:assert im.width>20 and im.height>20;im.verify()
    lm=json.loads((ROOT/f"data/derived/landmarks/{row['image_id']}.json").read_text())
    assert len(lm['landmarks_original'])==478
    box=json.loads(row['box_aligned']);assert 0<=box[0]<box[2]<=1350 and 0<=box[1]<box[3]<=1350
    assert row['manual_label'] in ['', 'unknown', *schema['tasks'][row['task']]['classes']]
subjects={s:{r['subject_id'] for r in images if r['split']==s} for s in ['train','val','test']}
assert not(subjects['train']&subjects['val'] or subjects['train']&subjects['test'] or subjects['val']&subjects['test'])
coverage={}
for task,definition in schema['tasks'].items():
    coverage[task]={}
    for split in subjects:
        coverage[task][split]={label:len({r['subject_id'] for r in ann if r['task']==task and r['manual_label']==label and r['split']==split and r['condition']=='neutral_front'}) for label in definition['classes']}
        assert min(coverage[task][split].values())>=2
train=list(csv.DictReader((ROOT/'data/derived/training_manifest.csv').open(encoding='utf-8-sig')))
assert len(train)==706 and all(r['manual_label'] not in ['','unknown'] and r['condition']=='neutral_front' for r in train)
lit=load('literature');assert len(lit)==103
assert all(r['original_section'] and r['morphology_ko'] and not r['traditional_type_training_approved'] for r in lit)
# Publisher template first two entries mark pupil centres; diagnostic agreement, not general validation.
errors=[]
for r in images:
    p=ROOT/r['path']
    if not p.with_suffix('.tem').exists():continue
    lines=p.with_suffix('.tem').read_text().splitlines();assert int(lines[0])==189
    source=np.array([[float(v) for v in line.split()] for line in lines[1:3]])
    source=source[np.argsort(source[:,0])]
    lm=json.loads((ROOT/f"data/derived/landmarks/{r['image_id']}.json").read_text())
    points=np.asarray(lm['landmarks_original'])[[468,473]];points=points[np.argsort(points[:,0])]
    error=np.linalg.norm(source-points,axis=1).mean()/np.linalg.norm(source[1]-source[0])
    errors.append({'image_id':r['image_id'],'pupil_nme':float(error)})
measure={r['image_id']:r for r in load('measurements')}
stress={}
for feature in ['mouth_open_ratio','lip_vermilion_mouth','eye_right_aperture_ratio','eye_left_aperture_ratio']:
    smiles={r['subject_id']:r['image_id'] for r in images if r['condition']=='smiling_front'}
    ds=[measure[smiles[r['subject_id']]][feature]-measure[r['image_id']][feature] for r in images if r['condition']=='neutral_front']
    stress[feature]={'median_change_smile_minus_neutral':float(np.median(ds)),'median_absolute_change':float(np.median(np.abs(ds)))}
result={'status':'passed','source_images_verified':204,'crops_verified':1428,'labeled_training_rows':706,'split_subject_counts':{s:len(x) for s,x in subjects.items()},'coverage_subjects':coverage,'template_pupil_agreement':{'compared_images':len(errors),'median_nme':float(np.median([e['pupil_nme'] for e in errors])),'p95_nme':float(np.quantile([e['pupil_nme'] for e in errors],.95)),'meaning':'mean pupil error / publisher interpupil distance; limited landmark diagnostic only; smiling archive has no templates'},'expression_stress_test':stress,'independent_human_review_count':0,'traditional_named_type_gold_labels':0}
(OUT/'data_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'pupil_comparison.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
# Crop montage for inspection, plus visible pupil overlays on original photographs.
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
ids=['001','003','005','025','037','143'];parts=['eye_right','eye_left','brow_right','brow_left','nose','mouth','face']
sheet=Image.new('RGB',(1400,6*220),'#f2f4f7');draw=ImageDraw.Draw(sheet)
for y,subject in enumerate(ids):
    for x,part in enumerate(parts):
        im=Image.open(ROOT/f'data/derived/crops/{part}/frll_{subject}_03.jpg');im=ImageOps.contain(im,(192,185));sheet.paste(im,(x*200+(200-im.width)//2,y*220+27));draw.text((x*200+4,y*220+4),subject+' '+part,fill='black',font=font)
sheet.save(OUT/'crop_qa.jpg',quality=95)
print(json.dumps(result,ensure_ascii=False))
