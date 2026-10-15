"""Controlled, isolated CNN backbone selection on the current 15 visual heads.

Frozen protocol first; no test inference until a validation choice is frozen.
Existing deployment checkpoints and participant split files are never changed.
"""
import argparse,csv,hashlib,io,json,math,os,random,sys,time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from facework.kface_training import PartNet,PART_CROPS,input_size

OUT=ROOT/'models/selection-20261014'
REPORT=ROOT/'outputs/model-selection-20261014'
SCHEMA=json.loads((ROOT/'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))['features']
PARTS=['eye','brow','nose','mouth','face']
ARCHES=['resnet18','efficientnet_b0','convnext_tiny']
SEEDS=[20261014,20261015,20261016]

def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def setup():
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    refs_path=ROOT/'models/kface_v4/refs/fold0/reference_labels.csv'
    refs={r['subject_id']:r for r in csv.DictReader(refs_path.open(encoding='utf-8-sig'))}
    val=sorted(k for k,r in refs.items() if r['split']=='val')
    rng=random.Random(20261014);rng.shuffle(val)
    groups={'train':sorted(k for k,r in refs.items() if r['split']=='train'),
            'selection':sorted(val[:34]),'calibration':sorted(val[34:]),
            'test':sorted(k for k,r in refs.items() if r['split']=='test')}
    assert [len(groups[k]) for k in groups]==[270,34,34,62]
    assert len(set().union(*map(set,groups.values())))==400
    heads={part:[k for k,f in SCHEMA.items() if f['part']==part and f['source']=='visual'] for part in PARTS}
    protocol={'version':'controlled-backbone-selection-v1','architectures':ARCHES,'seeds':SEEDS,'parts':PARTS,'heads':heads,
      'participants':{k:len(v) for k,v in groups.items()},'groups':groups,
      'pretraining':'torchvision official DEFAULT ImageNet-1K weights; compare deployable pretrained models, not architecture in isolation',
      'data':'existing normalized crop cache (face-width normalization); source S001 neutral E01, abs yaw<20 pitch<25, core',
      'train_captures_per_person':12,'evaluation':'C7 L1 E01 S001; bilateral crops mean BEFORE participant-level metrics',
      'epochs_max':12,'epochs_min':6,'early_stop_patience':3,'batch_size':32,
      'optimizer':'AdamW lr3e-4 weight_decay0.05','scheduler':'cosine per update, minimum lr3e-6',
      'augmentation':'per-image GPU affine ±3deg/±3% translation, brightness/contrast/saturation ±30%, grayscale10%, avgblur3x3 20%, horizontal flip50% onlynose/mouth/face; no scaling',
      'loss':'mean over categorical heads of train-frequency-capped weighted soft cross entropy; AI3-vote softlabels, hard-label fallback; no regression auxiliaries',
      'checkpoint':'selection34 mean per-head participant macro-F1; last epoch budget limit flagged',
      'calibration':'fit one positive temperature per head using calibration34 only, AFTER checkpoint fixed; never fit on selection/test',
      'candidates':'3 singles + all3 pairs + triple, each repeated across3seeds; ensemble arithmetic mean probabilities then own temperature',
      'decision_rule':'choose smallest measured warm CNN latency among candidates within0.01 of best mean selection macroF1; show paired uncertainty and seed variability; no claim of statistical equivalence',
      'test_policy':'freeze choice before reading test images or computing test predictions; test62 is historical reused audit, not external unseen population',
      'deployment_seed':SEEDS[0],'existing_deployment':'do not replace until separate integration decision',
      'reference_labels_sha256':sha(refs_path),'soft_labels_sha256':sha(ROOT/'data/annotations/kface_visual_v4_soft.json'),
      'independent_human_ground_truth':False,'scope_limit':'only5existingparts/15visualheads; no evidence for7untrainedregions or VGGT depth accuracy'}
    file=OUT/'protocol.json'
    if file.exists():assert json.loads(file.read_text(encoding='utf-8'))==protocol,'Protocol changed; use new experiment directory'
    else:dump(file,protocol)
    return refs,groups,heads,protocol

def prepare(part,refs,groups,heads,split):
    file=OUT/f'cache_{part}_{split}.pt'
    if file.exists():return torch.load(file,map_location='cpu',weights_only=True)
    soft=json.loads((ROOT/'data/annotations/kface_visual_v4_soft.json').read_text(encoding='utf-8'))
    images=[];owners=[];ys=[];vote=[];keys=[];crop_names=[]
    hh,ww=input_size(part)
    ids=groups[split]
    for si,sid in enumerate(ids):
        base=ROOT/'data/derived/kface/subjects'/sid
        with np.load(base.with_suffix('.npz')) as archive:
            data={k:archive[k] for k in ['keys','status','pose','parts','crop_offsets']}
        candidates=[]
        for index,key in enumerate(data['keys']):
            acc,light,expression,cam,purpose=str(key).split('|')
            if data['status'][index]!=1 or acc!='S001' or expression!='E01':continue
            if split=='train':
                if purpose!='core' or abs(float(data['pose'][index,1]))>=20 or abs(float(data['pose'][index,0]))>=25:continue
            elif cam!='C7' or light!='L1':continue
            candidates.append(index)
        if split=='train':
            rng=random.Random(20261014+int(sid));rng.shuffle(candidates);candidates=candidates[:12]
        else:candidates=candidates[:1]
        assert candidates,(sid,part,split)
        names=list(data['parts'])
        with base.with_suffix('.bin').open('rb') as blob:
            for index in candidates:
                for crop in PART_CROPS[part]:
                    offset,size=data['crop_offsets'][index,names.index(crop)];assert size>0
                    blob.seek(int(offset));raw=blob.read(int(size))
                    with Image.open(io.BytesIO(raw)) as im:
                        array=np.asarray(im.convert('RGB').resize((ww,hh),Image.Resampling.BILINEAR)).copy()
                    images.append(torch.from_numpy(array).permute(2,0,1));owners.append(si)
                    ys.append([SCHEMA[k]['classes'].index(refs[sid][k]) if refs[sid].get(k) in SCHEMA[k]['classes'] else -1 for k in heads])
                    v=np.zeros((len(heads),4),np.float32)
                    for j,k in enumerate(heads):
                        dist=soft.get(sid,{}).get(k)
                        if dist:
                            for c,label in enumerate(SCHEMA[k]['classes']):v[j,c]=dist.get(label,0)
                        elif ys[-1][j]>=0:v[j,ys[-1][j]]=1
                    vote.append(torch.from_numpy(v));keys.append(str(data['keys'][index]));crop_names.append(crop)
    result={'images':torch.stack(images).contiguous(),'owners':torch.tensor(owners),'truth':torch.tensor(ys),
            'votes':torch.stack(vote),'subject_ids':ids,'capture_keys':keys,'crop_names':crop_names}
    torch.save(result,file)
    print('CACHE',part,split,'people',len(ids),'crops',len(images),flush=True)
    return result

def feature_metrics(p,y):
    from sklearn.metrics import f1_score
    ok=y>=0
    if not ok.any():return {'n':0,'macro_f1':None,'accuracy':None}
    pred=p[ok].argmax(1);truth=y[ok]
    return {'n':int(ok.sum()),'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),
            'accuracy':float(np.mean(truth==pred))}

def normalize(x):
    return (x-torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]

def augment(x,g,part):
    n,_,h,w=x.shape
    def rand(*shape):return torch.rand(shape,device=x.device,generator=g)
    angle=(rand(n)*2-1)*(3*math.pi/180)
    matrix=torch.zeros(n,2,3,device=x.device)
    matrix[:,0,0]=angle.cos();matrix[:,0,1]=-angle.sin()*h/w
    matrix[:,1,0]=angle.sin()*w/h;matrix[:,1,1]=angle.cos();matrix[:,:,2]=(rand(n,2)*2-1)*.06
    grid=F.affine_grid(matrix,x.shape,align_corners=False)
    x=F.grid_sample(x-.5,grid,align_corners=False,padding_mode='zeros')+.5
    x=x*(.7+.6*rand(n,1,1,1));mean=x.mean((2,3),keepdim=True)
    x=(x-mean)*(.7+.6*rand(n,1,1,1))+mean
    gray=(x*torch.tensor([.299,.587,.114],device=x.device)[None,:,None,None]).sum(1,keepdim=True)
    x=gray+(x-gray)*(.7+.6*rand(n,1,1,1));x=torch.where(rand(n,1,1,1)<.1,gray,x)
    x=torch.where(rand(n,1,1,1)<.2,F.avg_pool2d(x,3,stride=1,padding=1),x)
    if part in ['nose','mouth','face']:x=torch.where(rand(n,1,1,1)<.5,x.flip(-1),x)
    return normalize(x.clamp(0,1))

def predict(model,cache,heads,device='cuda'):
    model.eval();ps=[[] for k in heads]
    with torch.inference_mode():
        for start in range(0,len(cache['images']),32):
            x=normalize(cache['images'][start:start+32].to(device).float()/255)
            _,out=model(x.contiguous(memory_format=torch.channels_last))
            for j,v in enumerate(out):ps[j].append(v.softmax(1).cpu().numpy())
    owner=cache['owners'].numpy();truth=cache['truth'].numpy();n=len(cache['subject_ids'])
    result={}
    for j,k in enumerate(heads):
        p=np.concatenate(ps[j]);averaged=np.stack([p[owner==i].mean(0) for i in range(n)])
        labels=np.array([truth[owner==i,j][0] for i in range(n)])
        result[k]={'prob':averaged.tolist(),'truth':labels.tolist(),'metrics':feature_metrics(averaged,labels)}
    return result

def train_one(arch,part,seed,refs,groups,heads,protocol):
    path=OUT/arch/str(seed);path.mkdir(parents=True,exist_ok=True)
    done=path/f'{part}.json'
    if done.exists():return
    start=time.perf_counter();training=prepare(part,refs,groups,heads,'train')
    selection=prepare(part,refs,groups,heads,'selection');calibration=prepare(part,refs,groups,heads,'calibration')
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=PartNet(arch,0,[len(SCHEMA[k]['classes']) for k in heads]).to('cuda',memory_format=torch.channels_last)
    n_params=sum(p.numel() for p in model.parameters())
    data=training['images'].to('cuda');votes=training['votes'].to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    steps=math.ceil(len(data)/protocol['batch_size'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,protocol['epochs_max']*steps,eta_min=3e-6)
    scaler=torch.amp.GradScaler('cuda');sampler=torch.Generator(device='cuda').manual_seed(seed+123)
    aug_rng=torch.Generator(device='cuda').manual_seed(seed+456)
    weights=[]
    for j,k in enumerate(heads):
        labels=[SCHEMA[k]['classes'].index(refs[sid][k]) for sid in groups['train'] if refs[sid].get(k) in SCHEMA[k]['classes']]
        counts=np.bincount(labels,minlength=len(SCHEMA[k]['classes']))
        weights.append(torch.tensor(np.minimum(counts.sum()/(len(counts)*np.maximum(counts,1)),5),device='cuda',dtype=torch.float32))
    best=-1.;best_epoch=0;history=[];best_state=None;best_predictions=None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,protocol['epochs_max']+1):
        model.train();order=torch.randperm(len(data),generator=sampler,device='cuda');loss_total=0.;batches=0;t=time.perf_counter()
        for start_index in range(0,len(data),protocol['batch_size']):
            idx=order[start_index:start_index+protocol['batch_size']]
            x=augment(data[idx].float()/255,aug_rng,part).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.float16):_,outputs=model(x)
            losses=[]
            for j,logits in enumerate(outputs):
                target=votes[idx,j,:logits.shape[1]];valid=target.sum(1)>0
                if valid.any():
                    w=(target[valid]*weights[j]).sum(1)
                    losses.append((-(target[valid]*logits[valid].float().log_softmax(1)).sum(1)*w).sum()/w.sum().clamp_min(1e-8))
            if not losses:continue
            loss=torch.stack(losses).mean();assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();scheduler.step()
            loss_total+=float(loss.detach());batches+=1
        torch.cuda.synchronize();train_seconds=time.perf_counter()-t;t=time.perf_counter()
        predictions=predict(model,selection,heads)
        score=float(np.mean([v['metrics']['macro_f1'] for v in predictions.values() if v['metrics']['n']]))
        history.append({'epoch':epoch,'train_loss':loss_total/max(1,batches),'selection_macro_f1':score,'train_seconds':train_seconds,'validation_seconds':time.perf_counter()-t})
        print('EPOCH',arch,part,seed,epoch,round(score,4),round(train_seconds,1),'seconds',flush=True)
        if score>best+1e-9:
            best=score;best_epoch=epoch;best_predictions=predictions
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        dump(path/f'{part}_progress.json',{'history':history,'best_epoch':best_epoch,'best_score':best})
        if epoch>=protocol['epochs_min'] and epoch-best_epoch>=protocol['early_stop_patience']:break
    model.load_state_dict(best_state);cal=predict(model,calibration,heads)
    metadata={'arch':arch,'part':part,'seed':seed,'classification_features':heads,'regression_features':[],
              'classes':{k:SCHEMA[k]['classes'] for k in heads},'crops':PART_CROPS[part],'input_size':input_size(part),
              'selected_epoch':best_epoch,'selection_macro_f1':best,'history':history,'parameters':n_params,
              'train_people':270,'selection_people':34,'calibration_people':34,'test_people_not_accessed':62,
              'train_crops':len(data),'elapsed_seconds':time.perf_counter()-start,
              'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
              'at_epoch_budget':best_epoch==protocol['epochs_max'],'protocol_sha256':sha(OUT/'protocol.json')}
    torch.save({'state_dict':best_state,'meta':metadata},path/f'{part}.pt')
    dump(done,{'meta':metadata,'selection':best_predictions,'calibration':cal})
    print('DONE',arch,part,seed,'best',round(best,4),'total_seconds',round(metadata['elapsed_seconds'],1),flush=True)
    del model,optimizer,scaler,data,votes,best_state,training,selection,calibration
    torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--one',nargs=3,metavar=('ARCH','PART','SEED'))
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.backends.cudnn.benchmark=True;torch.hub.set_dir(str(ROOT/'models/torch_cache/hub'))
    refs,groups,heads,protocol=setup()
    dump(REPORT/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                                 'python':sys.version,'cudnn_benchmark':True,'protocol_sha256':sha(OUT/'protocol.json')})
    if args.prepare_only:
        for part in PARTS:
            for split in ['train','selection','calibration']:prepare(part,refs,groups,heads[part],split)
        return
    if args.one:
        arch,part,seed=args.one;train_one(arch,part,int(seed),refs,groups,heads[part],protocol);return
    # Rotate architecture order across seeds to limit temporal/cache order effects.
    for si,seed in enumerate(SEEDS):
        order=ARCHES[si:]+ARCHES[:si]
        for part in PARTS:
            for arch in order:train_one(arch,part,seed,refs,groups,heads[part],protocol)
    print('ALL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
