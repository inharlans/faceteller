"""Train and audit the design-aligned release classifiers after backbone selection.

All outputs are separate from the controlled comparison. References are automated
morphology targets, not independent human ground truth. No test photo is used for
checkpoint selection or calibration. Test evaluation runs only after deployment
weights and temperatures have been frozen.
"""
import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.kface_training import PartNet
from scripts.model_selection_train import augment
from scripts.model_selection_metrics import metrics, fit_temperature, apply_temperature
from facework.release_preprocess import VERSION
from facework.release_parsing import EAR_TRANSFORM_VERSION
from facework.release_selection import inference_selection
from facework.release_thresholds import POLICY as THRESHOLD_POLICY, select_threshold

OUT=ROOT/'models/release'
DATA=ROOT/'data/derived/release'
REGISTRY=ROOT/'data/release/training_registry.json'
SEED=20261014


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def ear_protocol(config):
    source=DATA/'ears-v2/manifest.json'
    manifest=load(source)
    if (manifest.get('ear_transform_version')!=EAR_TRANSFORM_VERSION
            or manifest.get('preprocess_version')!=VERSION
            or manifest.get('participants')!=400 or manifest.get('complete') is not True
            or len(manifest.get('records',{}))!=400
            or manifest.get('parity',{}).get('pass') is not True
            or manifest['parity'].get('metadata_binary_hashes_verified')!=400
            or manifest['parity'].get('legacy_sources_unchanged')!=400):
        raise RuntimeError('Complete 400-person reference-scaled ear preparation required')
    value={'ear_transform_version':EAR_TRANSFORM_VERSION,'registry_sha256':config['registry_sha256'],
           'cache_manifest_sha256':sha(source),'input_size':[256,192],
           'normalization':'128 * cos(yaw) / eye-centre-distance; roll aligned; fixed window; no ear-bbox fitting'}
    path=OUT/'ear_preprocessing_frozen.json'
    if path.exists() and load(path)!=value:raise RuntimeError('Ear preprocessing protocol changed')
    if not path.exists():dump(path,value)
    return {'ear_transform_version':EAR_TRANSFORM_VERSION,'ear_preprocessing_sha256':sha(path)}


def protocol():
    choice_path=ROOT/'outputs/model-selection-20261014/choice_frozen.json'
    if not choice_path.exists():raise RuntimeError('Finish controlled backbone selection before release training')
    registry=load(REGISTRY);choice=load(choice_path)
    ready=load(ROOT/'data/release/extended_ready.json')
    if ready.get('ready') is not True or ready.get('registry_sha256')!=sha(REGISTRY):
        raise RuntimeError('Automatic training-data readiness differs from registry')
    if ready.get('source_hashes')!=registry['source_hashes']:
        raise RuntimeError('Readiness source list differs from registry')
    for path,digest in ready['source_hashes'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('Training target source changed after readiness')
    missing=[p for p,v in registry['parts'].items() if not v['features']]
    if missing:raise RuntimeError(f'Feature definitions still missing: {missing}')
    if not load(DATA/'crop_manifest.json')['complete']:raise RuntimeError('400-person crop preparation incomplete')
    config={'version':'release-classification-v1','registry_sha256':sha(REGISTRY),
            'backbone_choice_sha256':sha(choice_path),'architectures':choice['selected_architectures'],
            'preprocess_version':VERSION,'seed':SEED,'epochs_max':16,'epochs_min':6,
            'patience':4,'batch_size':16,'optimizer':'AdamW lr3e-4 wd0.05 cosine',
            'checkpoint':'selection34 mean per-head participant macroF1',
            'calibration':'calibration34 one temperature per head after averaging deployed models',
            'threshold':.6,'human_review_count':0,'test_policy':'historical62 after deployment manifest frozen',
            'split_sizes':registry['split_sizes']}
    path=OUT/'training_frozen.json'
    if path.exists() and load(path)!=config:raise RuntimeError('Release protocol changed')
    if not path.exists():dump(path,config)
    return registry,config


def cache(part,split,registry,config):
    path=OUT/'cache'/f'{part}_{split}.pt'
    ear_identity=ear_protocol(config) if part=='ear' else {}
    if path.exists():
        result=torch.load(path,map_location='cpu',weights_only=True)
        if result['registry_sha256']!=config['registry_sha256']:raise RuntimeError('Stale crop cache')
        if any(result.get(k)!=v for k,v in ear_identity.items()):raise RuntimeError('Stale ear crop cache')
        return result
    part_spec=registry['parts'][part];heads=part_spec['features'];features=registry['features']
    needs_eye_quality=part in ('eye','under_eye') or any(features[k].get('region',part)=='eye' for k in heads)
    ids=sorted(s for s,d in registry['subjects'].items() if d['split']==split)
    images=[];owners=[];truth=[];votes=[];available=[]
    max_classes=max(len(features[k]['classes']) for k in heads)
    for owner,sid in enumerate(ids):
        folder=DATA/('ears-v2' if part=='ear' else 'subjects')
        meta_path=folder/f'{sid}.json'
        if not meta_path.exists():continue
        meta=load(meta_path)
        if meta.get('preprocess_version',VERSION)!=VERSION:raise RuntimeError('Crop preprocessing mismatch')
        if part=='ear' and meta.get('ear_transform_version')!=EAR_TRANSFORM_VERSION:
            raise RuntimeError('Ear crop transform mismatch')
        spec=registry['subjects'][sid]
        with (folder/f'{sid}.bin').open('rb') as blob:
            for frame in meta['frames']:
                if frame['status']!='ready' or (split!='train' and frame['light']!='L1'):continue
                blend=frame.get('blend',[0]*5)
                if needs_eye_quality and (max(blend[1:3])>.5 or blend[0]>.3):continue
                if part in ('mouth','philtrum') and blend[0]>.3:continue
                for name in part_spec['crops']:
                    crop=frame['crops'].get(name)
                    if crop is None or crop.get('valid_fraction',1)<.9:continue
                    blob.seek(crop['offset']);raw=blob.read(crop['length'])
                    with Image.open(io.BytesIO(raw)) as image:
                        array=np.asarray(image.convert('RGB')).copy()
                    if list(array.shape[:2])!=part_spec['input_size']:raise RuntimeError('Crop shape mismatch')
                    y=[];v=np.zeros((len(heads),max_classes),np.float32)
                    for j,k in enumerate(heads):
                        classes=features[k]['classes'];label=spec['labels'].get(k)
                        target=classes.index(label) if label in classes else -1;y.append(target)
                        if target>=0:
                            if k in spec['soft_targets']:v[j,:len(classes)]=spec['soft_targets'][k]
                            else:v[j,target]=1
                    if split=='train' and max(y)<0:continue
                    images.append(torch.from_numpy(array).permute(2,0,1));owners.append(owner)
                    truth.append(y);votes.append(v)
                    available.append({'subject_id':sid,'capture':frame.get('capture',frame['light']),'crop':name})
    if not images:raise RuntimeError(f'No usable crops: {part}/{split}')
    result={'images':torch.stack(images),'owners':torch.tensor(owners),
            'truth':torch.tensor(truth),'votes':torch.from_numpy(np.stack(votes)),
            'ids':ids,'rows':available,'registry_sha256':config['registry_sha256'],**ear_identity}
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(result,path)
    return result


def predict(models,data,heads,features):
    outputs={k:[] for k in heads}
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    with torch.inference_mode():
        for start in range(0,len(data['images']),16):
            x=(data['images'][start:start+16].to('cuda').float()/255-mean)/std
            predictions=[m(x.contiguous(memory_format=torch.channels_last))[1] for m in models]
            for j,k in enumerate(heads):outputs[k].append(torch.stack([p[j].softmax(1) for p in predictions]).mean(0).cpu().numpy())
    owners=data['owners'].numpy();rows_truth=data['truth'].numpy();result={}
    for j,k in enumerate(heads):
        raw=np.concatenate(outputs[k]);classes=len(features[k]['classes'])
        prob=[];truth=[];present=[]
        for i in range(len(data['ids'])):
            mask=owners==i;present.append(bool(mask.any()))
            prob.append(raw[mask].mean(0) if mask.any() else np.full(classes,1/classes))
            truth.append(int(rows_truth[mask,j][0]) if mask.any() else -1)
        result[k]={'prob':np.asarray(prob).tolist(),'truth':truth,'available':present,
                   'metrics':metrics(np.asarray(prob),np.asarray(truth,dtype=int))}
    return result


def training_support(data,heads,features):
    """Count distinct people actually seen by each head, not repeated crops."""
    owners=data['owners'].numpy();truth=data['truth'].numpy();result={}
    for j,key in enumerate(heads):
        counts=[0]*len(features[key]['classes'])
        for owner in np.unique(owners):
            labels=np.unique(truth[owners==owner,j])
            if len(labels)!=1:raise RuntimeError('Inconsistent participant targets across crops')
            value=int(labels[0])
            if value>=0:counts[value]+=1
        result[key]=counts
    return result


def train_one(arch,part,registry,config):
    path=OUT/arch/f'{part}.json';weights_path=path.with_suffix('.pt')
    ear_identity=ear_protocol(config) if part=='ear' else {}
    if path.exists():
        if load(path)['config']!=config or not weights_path.exists():raise RuntimeError('Existing checkpoint invalid')
        if any(load(path)['meta'].get(k)!=v for k,v in ear_identity.items()):raise RuntimeError('Existing ear checkpoint uses obsolete input')
        return
    start_time=time.perf_counter();spec=registry['parts'][part];heads=spec['features'];features=registry['features']
    train=cache(part,'train',registry,config);selection=cache(part,'selection',registry,config)
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    model=PartNet(arch,0,[len(features[k]['classes']) for k in heads]).to('cuda',memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    batch=config['batch_size'];steps=math.ceil(len(train['images'])/batch)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs_max']*steps,eta_min=3e-6)
    scaler=torch.amp.GradScaler('cuda');rng=torch.Generator(device='cuda').manual_seed(SEED+1)
    data=train['images'].to('cuda');votes=train['votes'].to('cuda')
    actual_support=training_support(train,heads,features)
    weights=[]
    for k in heads:
        counts=np.asarray(actual_support[k],float)
        weights.append(torch.tensor(np.minimum(counts.sum()/(len(counts)*np.maximum(counts,1)),5),device='cuda',dtype=torch.float32))
    best=-1.;best_epoch=0;best_state=None;best_prediction=None;history=[]
    for epoch in range(1,config['epochs_max']+1):
        model.train();order=torch.randperm(len(data),device='cuda',generator=rng);losses=[];epoch_start=time.perf_counter();amp_skips=0
        for offset in range(0,len(data),batch):
            idx=order[offset:offset+batch];x=augment(data[idx].float()/255,rng,part).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda',dtype=torch.float16):_,predictions=model(x)
            per_head=[]
            for j,logits in enumerate(predictions):
                target=votes[idx,j,:logits.shape[1]];valid=target.sum(1)>0
                if not valid.any():continue
                weight=(target[valid]*weights[j]).sum(1)
                per_head.append((-(target[valid]*logits[valid].float().log_softmax(1)).sum(1)*weight).sum()/weight.sum().clamp_min(1e-8))
            if not per_head:continue
            loss=torch.stack(per_head).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);scaler.scale(loss).backward()
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale:scheduler.step()
            else:amp_skips+=1
            losses.append(float(loss.detach()))
        torch.cuda.synchronize();seconds=time.perf_counter()-epoch_start
        validation_start=time.perf_counter()
        model.eval();prediction=predict([model],selection,heads,features)
        validation_seconds=time.perf_counter()-validation_start
        scores=[v['metrics']['macro_f1'] for v in prediction.values() if v['metrics']['macro_f1'] is not None]
        if not scores:raise RuntimeError('No selection targets')
        score=float(np.mean(scores))
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_macro_f1':score,
                        'train_seconds':seconds,'validation_seconds':validation_seconds,'amp_skipped_steps':amp_skips})
        if score>best:
            best=score;best_epoch=epoch;best_prediction=prediction
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        dump(path.with_name(part+'_progress.json'),{'history':history,'best_epoch':best_epoch})
        print('RELEASE_EPOCH',arch,part,epoch,round(score,4),round(seconds,1),flush=True)
        if epoch>=config['epochs_min'] and epoch-best_epoch>=config['patience']:break
    metadata={'arch':arch,'part':part,'classification_features':heads,'classes':{k:features[k]['classes'] for k in heads},
              'crops':spec['crops'],'input_size':spec['input_size'],'preprocess_version':VERSION,
              'registry_sha256':config['registry_sha256'],'human_review_count':0,'train_support':actual_support,**ear_identity}
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':best_state,'meta':metadata},weights_path)
    dump(path,{'meta':metadata,'config':config,'history':history,'selected_epoch':best_epoch,
               'selection':best_prediction,'train_people_with_crops':len(set(train['owners'].tolist())),
               'train_people_by_feature':{k:len(set(train['owners'][train['truth'][:,j]>=0].tolist())) for j,k in enumerate(heads)},
               'train_support':actual_support,
               'train_crops':len(data),'elapsed_seconds':time.perf_counter()-start_time})
    del model,optimizer,scaler,data,votes,best_state,train,selection
    torch.cuda.empty_cache()


def load_models(part,config):
    models=[]
    ear_identity=ear_protocol(config) if part=='ear' else {}
    for arch in inference_selection(ROOT,config)[0]:
        ck=torch.load(OUT/arch/f'{part}.pt',map_location='cpu',weights_only=True);meta=ck['meta']
        if meta['registry_sha256']!=config['registry_sha256']:raise RuntimeError('Weight/registry mismatch')
        if any(meta.get(k)!=v for k,v in ear_identity.items()):raise RuntimeError('Ear checkpoint input mismatch')
        model=PartNet(arch,0,[len(meta['classes'][k]) for k in meta['classification_features']],pretrained=False)
        model.load_state_dict(ck['state_dict']);model.eval();models.append(model.to('cuda',memory_format=torch.channels_last))
    return models


def release_temperature(probabilities, truth, *, validation_ids, test_ids):
    """Do not authorize service confidence from one-class calibration data.

    A numerical optimizer can converge on positives alone by sharpening every
    prediction. That does not support a confidence threshold distinguishing the
    alternatives. Keep the raw model metrics, but leave this head unconfirmed.
    This release-only rule does not rewrite the historical backbone comparison.
    """
    result=fit_temperature(probabilities,truth,validation_ids=validation_ids,test_ids=test_ids)
    result['coverage_policy']='at_least_two_observed_classes_v1'
    if sum(count>0 for count in result['support'])==1:
        result.update(calibrated=False,temperature=None,log_temperature=None,
                      nll_after=None,at_bound=False,status='single_class_calibration_support')
    return result


def calibrate(registry,config):
    file=OUT/'deployment.json'
    architectures,selection_identity=inference_selection(ROOT,config)
    policy_path=OUT/'threshold_policy_frozen.json'
    if policy_path.exists() and load(policy_path)!=THRESHOLD_POLICY:raise RuntimeError('Threshold policy changed')
    if not policy_path.exists():dump(policy_path,THRESHOLD_POLICY)
    policy_identity={'threshold_policy_sha256':sha(policy_path)}
    if file.exists():
        frozen=load(file)
        if frozen['config']!=config:raise RuntimeError('Deployment config changed')
        inference_selection(ROOT,config,frozen)
        if any(frozen.get(k)!=v for k,v in policy_identity.items()):raise RuntimeError('Threshold calibration policy mismatch')
        for path,digest in frozen['weights'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Deployment weight changed')
        return frozen
    start_time=time.perf_counter();part_seconds={}
    features={};weights={};test_ids=[s for s,d in registry['subjects'].items() if d['split']=='test']
    for part,spec in registry['parts'].items():
        part_start=time.perf_counter()
        models=load_models(part,config);data=cache(part,'calibration',registry,config)
        support=[load(OUT/arch/f'{part}.json')['train_support'] for arch in architectures]
        if any(s!=support[0] for s in support[1:]):raise RuntimeError('Ensemble training support differs')
        prediction=predict(models,data,spec['features'],registry['features'])
        for k,result in prediction.items():
            p=np.asarray(result['prob']);y=np.asarray(result['truth'],dtype=int)
            cal=release_temperature(p,y,validation_ids=data['ids'],test_ids=test_ids)
            adjusted=apply_temperature(p,cal['temperature']) if cal['calibrated'] else p
            selection=select_threshold(adjusted,y,calibrated=cal['calibrated'],train_support=support[0][k],
                                       validation_ids=data['ids'],test_ids=test_ids)
            threshold=selection['threshold']
            features[k]={**cal,'threshold':threshold,'threshold_selection':selection,'train_support':support[0][k],
                         'raw':metrics(p,y,threshold=threshold),
                         'calibrated_metrics':metrics(adjusted,y,threshold=threshold) if cal['calibrated'] else None}
        for arch in architectures:
            path=OUT/arch/f'{part}.pt';weights[path.relative_to(ROOT).as_posix()]=sha(path)
        del models;torch.cuda.empty_cache()
        part_seconds[part]=time.perf_counter()-part_start
    frozen={'config':config,'weights':weights,'features':features,'test_used':False,**ear_protocol(config),**selection_identity,**policy_identity}
    frozen['calibration_timings']={'total_seconds':time.perf_counter()-start_time,'parts':part_seconds,
        'scope':'model loading, crop/cache preparation, inference and temperature fitting; excludes final JSON write'}
    dump(file,frozen);print('RELEASE_DEPLOYMENT_FROZEN',flush=True)
    return frozen


def evaluate_test(registry,config):
    start_time=time.perf_counter();part_seconds={}
    frozen=calibrate(registry,config);report={}
    for part,spec in registry['parts'].items():
        part_start=time.perf_counter()
        models=load_models(part,config);data=cache(part,'test',registry,config)
        prediction=predict(models,data,spec['features'],registry['features'])
        for k,result in prediction.items():
            cal=frozen['features'][k];p=np.asarray(result['prob']);y=np.asarray(result['truth'],dtype=int)
            report[k]={'raw':metrics(p,y,threshold=cal['threshold']),
                       'calibrated':metrics(apply_temperature(p,cal['temperature']),y,threshold=cal['threshold']) if cal['calibrated'] else None,
                       'photo_available':sum(result['available']),'participants':len(data['ids']),
                       'annotation_provenance':registry['features'][k]['annotation_provenance']}
        del models;torch.cuda.empty_cache()
        part_seconds[part]=time.perf_counter()-part_start
    result={'deployment_sha256':sha(OUT/'deployment.json'),
        'features':report,'human_review_count':0,'evaluation_meaning':'agreement with automated morphology targets on historical held-out participants'}
    result['timings']={'total_seconds':time.perf_counter()-start_time,'parts':part_seconds,
        'scope':'model loading, crop/cache preparation, inference, metrics and initial frozen verification; excludes final JSON write'}
    dump(ROOT/'outputs/release/test_metrics.json',result)
    print('RELEASE_TEST_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','calibrate','test','all']);args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=args.stage in ('train','all')
    torch.hub.set_dir(str(ROOT/'models/torch_cache/hub'))
    registry,config=protocol()
    if args.stage in ('train','all'):
        for part in registry['parts']:
            for arch in inference_selection(ROOT,config)[0]:train_one(arch,part,registry,config)
    if args.stage in ('calibrate','all'):calibrate(registry,config)
    if args.stage in ('test','all'):evaluate_test(registry,config)


if __name__=='__main__':main()
