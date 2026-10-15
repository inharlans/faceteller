"""Read existing artifacts only; do not train or run image inference."""
from pathlib import Path
import csv,json,hashlib,collections,gc
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiment-report-2026-10-01';OUT.mkdir(exist_ok=True)
read=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
refs=list(csv.DictReader((ROOT/'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')))
byid={r['subject_id']:r for r in refs}
schema=read('data/label_schema_kface_v3.json')['features']
assignment=read('models/kface_v4/folds.json')['assignment']
frozen=read('data/derived/kface/split_frozen.json')['split']
assert all(frozen[s]==r['split'] for s,r in byid.items())
assert set(assignment)=={s for s,r in byid.items() if r['split']!='test'}
subject_counts={};totals=collections.defaultdict(collections.Counter)
for s,r in byid.items():
 with np.load(ROOT/'data/derived/kface/subjects'/f'{s}.npz') as d:
  keys=d['keys'];status=d['status'];pose=d['pose']
 c=collections.Counter(planned=len(keys))
 for i,key in enumerate(keys):
  acc,light,expr,cam,purpose=str(key).split('|')
  c['planned_'+purpose]+=1
  if status[i]!=1:continue
  c['detected']+=1;c['detected_'+purpose]+=1
  if purpose=='core' and abs(pose[i,1])<20 and abs(pose[i,0])<25:c['train_pool']+=1
  if purpose=='core' and cam in ('C6','C7','C8','C16') and light in ('L1','L2','L3','L8','L12','L19','L25','L4'):c['epoch_val_pool']+=1
  if purpose in ('core','expression','accessory') and cam in ('C6','C7','C8','C15','C16','C17'):c['full_eval_pool']+=1
  if purpose=='core' and cam=='C7' and light in ('L1','L2','L3','L8','L9','L12','L13','L22','L4','L14','L23'):c['service_pool']+=1
  if purpose=='core' and cam=='C7' and light in ('L4','L14','L23'):c['nonref_service_pool']+=1
 subject_counts[s]=c;totals[r['split']].update(c)
print('CAPTURES',dict(totals),flush=True)
folds=[]
for f in range(5):
 ids={sp:{s for s,r in byid.items() if ('test' if r['split']=='test' else 'val' if assignment[s]==f else 'train')==sp} for sp in ('train','val','test')}
 actual=list(csv.DictReader((ROOT/f'models/kface_v4/refs/fold{f}/reference_labels.csv').open(encoding='utf-8-sig')))
 assert all(r['subject_id'] in ids[r['split']] for r in actual)
 row={'fold':f,'train_people':len(ids['train']),'val_people':len(ids['val']),'test_people':len(ids['test']),
 'train_photo_pool':sum(subject_counts[s]['train_pool'] for s in ids['train']),
 'val_photo_pool':sum(subject_counts[s]['epoch_val_pool'] for s in ids['val']),
 'overlap_train_val':len(ids['train']&ids['val']),'overlap_train_test':len(ids['train']&ids['test'])}
 folds.append(row)
metas=[]
for path in sorted((ROOT/'models/kface_v4').glob('*/fold*/*.pt')):
 ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
 m=dict(ck['meta']);m['checkpoint']=path.relative_to(ROOT).as_posix();m['file_bytes']=path.stat().st_size
 f=int(path.parent.name[4:]);part=path.stem;mult=2 if part in ('eye','brow') else 1
 assert m['train_participants']==folds[f]['train_people']
 assert m['train_captures']==folds[f]['train_photo_pool']*mult
 for sp in ('train','val'):
  ids={s for s,r in byid.items() if r['split']!='test' and ((assignment[s]==f) if sp=='val' else (assignment[s]!=f))}
  h=hashlib.sha256(','.join(sorted(ids)).encode()).hexdigest()
  recorded=m.get(sp+'_participants_sha256')
  m[sp+'_hash_match']=recorded==h if recorded else 'not_recorded'
 m['val_people_recounted']=folds[f]['val_people'];m['val_crops_recounted']=folds[f]['val_photo_pool']*mult
 # Sampling is without replacement within each participant per epoch, then the last incomplete batch is dropped.
 trainids=[s for s,r in byid.items() if r['split']!='test' and assignment[s]!=f]
 sample=sum(min(36*mult,subject_counts[s]['train_pool']*mult) for s in trainids)
 m['epoch_sample_crops_before_drop_last']=sample;m['epoch_consumed_crops_batch64']=sample//64*64
 metas.append(m);del ck;gc.collect()
print('CHECKPOINTS',len(metas),flush=True)
labelcounts=[]
for k,f in schema.items():
 for sp in ('train','val','test'):
  rows=[r for r in refs if r['split']==sp];dist=collections.Counter(r.get(k,'') or 'missing' for r in rows)
  for cls in list(f['classes'])+sorted(set(dist)-set(f['classes'])):
   labelcounts.append({'feature_id':k,'feature':f['ko'],'part':f['part'],'source':f['source'],'split':sp,'class':cls,'people':dist.get(cls,0),'usable_class':cls in f['classes']})
result={'date':'2026-10-01','scope':'existing artifacts; no new training/inference','people':dict(collections.Counter(r['split'] for r in refs)),
 'capture_totals':dict(totals),'folds':folds,'checkpoint_count':len(metas),'checkpoints':metas,'label_counts':labelcounts,
 'split_disjoint':True,'frozen_split_matches':True,'test_outside_all_folds':True}
(OUT/'실험_재집계.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
for name,rows in [('피처별_클래스_인원.csv',labelcounts),('교차학습_분할.csv',folds)]:
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('DONE',flush=True)
