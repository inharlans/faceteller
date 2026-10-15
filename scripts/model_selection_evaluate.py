"""Evaluate the frozen backbone experiment without using test to select models."""
import argparse,gc,itertools,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.model_selection_train import OUT,REPORT,SCHEMA,PARTS,ARCHES,SEEDS,setup,prepare,predict,normalize,dump
from scripts.model_selection_metrics import aggregate_metrics,fit_temperature,apply_temperature,paired_bootstrap
from facework.kface_training import PartNet

FAMILIES={'+'.join(c):list(c) for size in (1,2,3) for c in itertools.combinations(ARCHES,size)}

def load(path):return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def require(condition,message):
    if not condition:raise RuntimeError(f'Evaluation provenance: {message}')

def checked_setup():
    context=setup();refs,groups,heads,protocol=context
    require(protocol==load(OUT/'protocol.json'),'protocol differs from current setup')
    ids=[sid for group in groups.values() for sid in group]
    require(len(ids)==len(set(ids)),'participant groups overlap or contain duplicates')
    require(all(sid in refs for sid in ids),'participant missing from references')
    expected={part:[k for k,f in SCHEMA.items() if f['part']==part and f['source']=='visual'] for part in PARTS}
    require(heads==expected,'head definitions differ from current class schema')
    return context

def expected_truth(refs,ids,key):
    classes=SCHEMA[key]['classes']
    return np.array([classes.index(refs[sid][key]) if refs[sid].get(key) in classes else -1 for sid in ids],dtype=np.int64)

def validate_predictions(values,part,split,context,label):
    refs,groups,heads,_=context
    require(set(values)==set(heads[part]),f'{label}: incorrect {part} head set')
    for key in heads[part]:
        y=np.asarray(values[key]['truth']);p=np.asarray(values[key]['prob'],dtype=float)
        truth=expected_truth(refs,groups[split],key)
        require(y.dtype.kind in 'iu' and y.shape==truth.shape and np.array_equal(y,truth),
                f'{label}: {key} truth/participant order mismatch')
        require(p.shape==(len(truth),len(SCHEMA[key]['classes'])),f'{label}: {key} probability shape mismatch')
        require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
                and np.allclose(p.sum(1),1,atol=1e-6,rtol=0),f'{label}: {key} invalid probabilities')

def validate_meta(meta,arch,part,seed,context,label):
    _,groups,heads,_=context
    require(meta.get('protocol_sha256')==sha(OUT/'protocol.json'),f'{label}: wrong protocol')
    require((meta.get('arch'),meta.get('part'),meta.get('seed'))==(arch,part,seed),f'{label}: wrong model identity')
    require(meta.get('classification_features')==heads[part],f'{label}: wrong head order')
    require(meta.get('classes')=={k:SCHEMA[k]['classes'] for k in heads[part]},f'{label}: wrong class order')
    require(meta.get('regression_features')==[],f'{label}: unexpected regression heads')
    for split in ('train','selection','calibration'):
        require(meta.get(split+'_people')==len(groups[split]),f'{label}: wrong {split} participant count')

def training_record(arch,part,seed,context):
    path=OUT/arch/str(seed)/f'{part}.json'
    require(path.exists(),f'training incomplete: {path}')
    record=load(path);validate_meta(record['meta'],arch,part,seed,context,str(path))
    for split in ('selection','calibration'):
        validate_predictions(record[split],part,split,context,str(path))
    return record

def all_training(context=None):
    items={};context=context or checked_setup()
    for arch in ARCHES:
        for seed in SEEDS:
            items[arch,seed]={}
            for part in PARTS:
                record=training_record(arch,part,seed,context)
                items[arch,seed].update(record['selection'])
    return items

def training_manifest():
    manifest={}
    for arch in ARCHES:
        for seed in SEEDS:
            for part in PARTS:
                for suffix in ('.pt','.json'):
                    path=OUT/arch/str(seed)/(part+suffix)
                    require(path.is_file(),f'missing training artifact: {path}')
                    manifest[path.relative_to(OUT).as_posix()]=sha(path)
    return manifest

def check_manifest(recorded,current,label):
    require(isinstance(recorded,dict) and set(recorded)==set(current),f'{label}: incomplete training manifest')
    changed=[key for key,value in current.items() if recorded[key]!=value]
    require(not changed,f'{label}: changed training artifact(s): {", ".join(changed[:5])}')

def temperature_value(item,key):
    temperature=item.get('temperature')
    require(item.get('calibrated') is True and item.get('optimizer_success') is True
            and isinstance(temperature,(int,float)) and not isinstance(temperature,bool)
            and np.isfinite(temperature) and np.exp(-2)-1e-8<=temperature<=np.exp(2)+1e-8
            and isinstance(item.get('n_val'),int) and item['n_val']>0,
            f'{key}: temperature fit missing/failed or no valid calibration labels; cannot report calibrated results')
    return temperature

def validate_development(dev,context,manifest):
    refs,groups,heads,_=context;keys={k for values in heads.values() for k in values}
    require(dev.get('protocol_sha256')==sha(OUT/'protocol.json'),'development protocol mismatch')
    check_manifest(dev.get('training_manifest'),manifest,'development')
    require(set(dev['families'])==set(FAMILIES),'development family set mismatch')
    for name,arches in FAMILIES.items():
        family=dev['families'][name]
        require(family['architectures']==arches,f'{name}: architecture order mismatch')
        require(set(family['runs'])=={str(seed) for seed in SEEDS},f'{name}: seed set mismatch')
        for seed,run in family['runs'].items():
            require(set(run['temperatures'])==keys,f'{name}/{seed}: temperature head set mismatch')
            for key,item in run['temperatures'].items():
                temperature_value(item,key)
                require(item['n_val']==int((expected_truth(refs,groups['calibration'],key)>=0).sum()),
                        f'{name}/{seed}/{key}: calibration sample count mismatch')
                require(item.get('class_count')==len(SCHEMA[key]['classes']) and item.get('split')=='validation'
                        and item.get('test_used_for_fit') is False,f'{name}/{seed}/{key}: calibration protocol mismatch')
            for variant in ('raw','calibrated'):
                score=run[variant]
                require(score.get('complete') is True and score.get('feature_count')==len(keys)
                        and set(score['features'])==keys and score.get('primary_score') is not None
                        and np.isfinite(score['primary_score']),f'{name}/{seed}: incomplete selection score')

def validate_latency(lat):
    require(lat.get('protocol_sha256')==sha(OUT/'protocol.json'),'latency protocol mismatch')
    require(lat.get('development_sha256')==sha(REPORT/'development.json'),'latency/development mismatch')
    require(set(lat['families'])==set(FAMILIES),'latency family set mismatch')
    require(all(np.isfinite(row['mean_ms']) and row['mean_ms']>0 for row in lat['families'].values()),'invalid latency mean')

def selection_decision(dev,lat):
    scores={k:float(np.mean([v['calibrated']['primary_score'] for v in row['runs'].values()])) for k,row in dev['families'].items()}
    best=max(scores.values());eligible=[k for k,s in scores.items() if s>=best-.01-1e-12]
    selected=min(eligible,key=lambda k:lat['families'][k]['mean_ms'])
    return {'selected_family':selected,'selected_architectures':FAMILIES[selected],
            'selection_scores':scores,'best_selection_score':best,'tolerance_absolute':.01,'eligible':eligible,
            'deployment_seed':SEEDS[0]}

def validate_frozen():
    path=REPORT/'choice_frozen.json'
    require(path.is_file(),'choice must be frozen before test/summary')
    choice=load(path)
    require(choice.get('test_predictions_present_at_freeze') is False,'invalid pre-test freeze record')
    # Check all small report hashes BEFORE reading checkpoints, test caches, or doing statistics.
    for key,path in (('protocol_sha256',OUT/'protocol.json'),('development_sha256',REPORT/'development.json'),
                     ('latency_sha256',REPORT/'latency.json')):
        require(choice.get(key)==sha(path),f'frozen {key} mismatch')
    context=checked_setup();manifest=training_manifest()
    check_manifest(choice.get('training_manifest'),manifest,'frozen choice')
    all_training(context)
    dev=load(REPORT/'development.json');lat=load(REPORT/'latency.json')
    validate_development(dev,context,manifest);validate_latency(lat)
    for key,value in selection_decision(dev,lat).items():
        require(choice.get(key)==value,f'frozen decision mismatch: {key}')
    return choice,context

def validate_test_record(record,arch,part,seed,choice,context,choice_hash):
    base=f'{arch}/{seed}/{part}'
    require(record.get('choice_sha256')==choice_hash,f'{base}: stale test choice hash')
    require(record.get('checkpoint_sha256')==choice['training_manifest'][base+'.pt'],f'{base}: stale test checkpoint hash')
    require(record.get('metadata_sha256')==choice['training_manifest'][base+'.json'],f'{base}: stale test metadata hash')
    require(record.get('subject_ids')==context[1]['test'],f'{base}: test participant order mismatch')
    require(record.get('classes')=={k:SCHEMA[k]['classes'] for k in context[2][part]},f'{base}: test class order mismatch')
    validate_predictions(record['predictions'],part,'test',context,base)

def test_records(choice,context,require_all=False):
    records={};choice_hash=sha(REPORT/'choice_frozen.json')
    for arch in ARCHES:
        for seed in SEEDS:
            for part in PARTS:
                path=OUT/arch/str(seed)/f'{part}_test.json'
                if not path.exists():
                    require(not require_all,f'missing test predictions: {path}')
                    continue
                record=load(path)
                validate_test_record(record,arch,part,seed,choice,context,choice_hash)
                records[arch,seed,part]=record
    return records

def validate_capture_cache(cache,part,split,context):
    refs,groups,heads,_=context
    require(cache['subject_ids']==groups[split],f'{part}/{split}: crop cache participant order mismatch')
    owners=np.asarray(cache['owners']);truth=np.asarray(cache['truth'])
    require(owners.ndim==1 and owners.dtype.kind in 'iu' and len(owners)==len(cache['images'])
            and np.array_equal(np.unique(owners),np.arange(len(groups[split]))),f'{part}/{split}: crop owner mismatch')
    expected=np.stack([expected_truth(refs,groups[split],key)[owners] for key in heads[part]],axis=1)
    require(truth.shape==expected.shape and np.array_equal(truth,expected),f'{part}/{split}: crop truth mismatch')

def merged(arches,seed,split,context=None,validated_tests=None):
    if split=='test' and validated_tests is None:
        choice,context=validate_frozen();validated_tests=test_records(choice,context,require_all=True)
    context=context or checked_setup()
    gathered={}
    for arch in arches:
        for part in PARTS:
            record=validated_tests[arch,seed,part] if split=='test' else training_record(arch,part,seed,context)
            values=record[split] if split in ('selection','calibration') else record['predictions']
            for key,val in values.items():
                if key not in gathered:gathered[key]={'truth':np.array(val['truth']),'probs':[]}
                else:np.testing.assert_array_equal(gathered[key]['truth'],val['truth'])
                gathered[key]['probs'].append(np.array(val['prob'],dtype=float))
    y={k:v['truth'] for k,v in gathered.items()}
    p={k:np.mean(v['probs'],axis=0) for k,v in gathered.items()}
    return y,p

def calibrated(prob,temps):
    require(set(prob)==set(temps),'temperature head set mismatch')
    return {k:apply_temperature(p,temperature_value(temps[k],k)) for k,p in prob.items()}

def development():
    require(not (REPORT/'choice_frozen.json').exists(),'choice already frozen; resume with test/summary/all')
    context=checked_setup();all_training(context);_,groups,heads,protocol=context
    manifest=training_manifest()
    families={}
    for name,arches in FAMILIES.items():
        runs={}
        for seed in SEEDS:
            cy,cp=merged(arches,seed,'calibration',context)
            temps={k:fit_temperature(cp[k],cy[k],validation_ids=groups['calibration'],test_ids=groups['test']) for k in cp}
            sy,sp=merged(arches,seed,'selection',context)
            runs[str(seed)]={'temperatures':temps,'raw':aggregate_metrics(sy,sp),
                             'calibrated':aggregate_metrics(sy,calibrated(sp,temps))}
        families[name]={'architectures':arches,'runs':runs}
    result={'protocol_sha256':sha(OUT/'protocol.json'),'training_manifest':manifest,'families':families}
    validate_development(result,context,manifest)
    dump(REPORT/'development.json',result)
    print('DEVELOPMENT_COMPLETE',flush=True)

def load_model(arch,part,seed,context=None):
    context=context or checked_setup()
    cp=torch.load(OUT/arch/str(seed)/f'{part}.pt',map_location='cpu',weights_only=False)
    meta=cp['meta'];validate_meta(meta,arch,part,seed,context,'checkpoint')
    require(json.loads(json.dumps(meta))==training_record(arch,part,seed,context)['meta'],'checkpoint metadata differs from training JSON')
    model=PartNet(arch,0,[len(meta['classes'][k]) for k in meta['classification_features']],pretrained=False)
    model.load_state_dict(cp['state_dict']);return model.to('cuda',memory_format=torch.channels_last).eval(),meta

def latency():
    require(not (REPORT/'choice_frozen.json').exists(),'choice already frozen; latency cannot be overwritten')
    context=checked_setup();all_training(context);_,groups,heads,protocol=context
    dev=load(REPORT/'development.json');validate_development(dev,context,training_manifest())
    caches={p:torch.load(OUT/f'cache_{p}_selection.pt',map_location='cpu',weights_only=True) for p in PARTS}
    for part,cache in caches.items():validate_capture_cache(cache,part,'selection',context)
    requests=[{part:caches[part]['images'][caches[part]['owners']==i] for part in PARTS} for i in range(34)]
    results={}
    # Every repeat uses a rotated family order, independent of accuracy results.
    names=list(FAMILIES)
    for repeat in range(3):
        order=names[repeat:]+names[:repeat]
        for name in order:
            models={};params=0;bytes_disk=0
            for arch in FAMILIES[name]:
                models[arch]={}
                for part in PARTS:
                    model,meta=load_model(arch,part,SEEDS[0],context);models[arch][part]=model
                    params+=meta['parameters'];bytes_disk+=(OUT/arch/str(SEEDS[0])/f'{part}.pt').stat().st_size
            temps=dev['families'][name]['runs'][str(SEEDS[0])]['temperatures']
            def infer(request):
                result={}
                with torch.inference_mode():
                    for part in PARTS:
                        x=normalize(request[part].to('cuda').float()/255).contiguous(memory_format=torch.channels_last)
                        outputs=[models[a][part](x)[1] for a in FAMILIES[name]]
                        for j,key in enumerate(heads[part]):
                            p=torch.stack([out[j].softmax(1) for out in outputs]).mean(0).mean(0).cpu().numpy()
                            result[key]=apply_temperature(p,temperature_value(temps[key],key))
                return result
            for i in range(10):infer(requests[i])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for i in range(34):
                start=time.perf_counter();infer(requests[i]);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            row=results.setdefault(name,{'milliseconds':[],'parameters':params,'checkpoint_bytes':bytes_disk,'gpu_peak_allocated_gib':0})
            row['milliseconds'].extend(times);row['gpu_peak_allocated_gib']=max(row['gpu_peak_allocated_gib'],torch.cuda.max_memory_allocated()/2**30)
            row['mean_ms']=float(np.mean(row['milliseconds']));row['p50_ms']=float(np.median(row['milliseconds']));row['p95_ms']=float(np.percentile(row['milliseconds'],95))
            print('LATENCY',repeat,name,round(row['mean_ms'],2),flush=True)
            del model,models,infer;gc.collect();torch.cuda.empty_cache()
    dump(REPORT/'latency.json',{'protocol_sha256':sha(OUT/'protocol.json'),'development_sha256':sha(REPORT/'development.json'),
         'gpu':torch.cuda.get_device_name(0),'precision':'FP32','layout':'channels_last',
         'scope':'5part CNN only, 7 pre-resized uint8 crops -> GPU transfer/normalize/forward/bilateral and model mean/calibration/CPU probabilities; no decoding, MediaPipe, VGGT, HTTP, cold start',
         'warmup_per_repeat':10,'repeats':3,'trials_per_repeat':34,'families':results})

def freeze():
    if (REPORT/'choice_frozen.json').exists():
        choice,context=validate_frozen();test_records(choice,context)
        print('CHOICE_ALREADY_FROZEN');return
    require(not list(OUT.glob('*/*/*_test.json')),'test predictions already exist; refusing to write a pre-test choice')
    context=checked_setup();all_training(context);manifest=training_manifest()
    dev=load(REPORT/'development.json');lat=load(REPORT/'latency.json')
    validate_development(dev,context,manifest);validate_latency(lat)
    decision=selection_decision(dev,lat)
    choice={**decision,'training_manifest':manifest,'provenance_version':1,
        'criterion':'minimum measured warm CNN mean latency among families within0.01 of best3seed mean selection macroF1',
        'deployment_seed':SEEDS[0],'unix_time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),
        'development_sha256':sha(REPORT/'development.json'),'latency_sha256':sha(REPORT/'latency.json'),
        'test_predictions_present_at_freeze':False}
    require(not list(OUT.glob('*/*/*_test.json')),'test predictions appeared before freeze write')
    dump(REPORT/'choice_frozen.json',choice)
    print('CHOICE_FROZEN',decision['selected_family'],flush=True)

def test_inference():
    frozen,context=validate_frozen();refs,groups,heads,protocol=context
    existing=test_records(frozen,context);choice_hash=sha(REPORT/'choice_frozen.json')
    for part in PARTS:
        if all((arch,seed,part) in existing for arch in ARCHES for seed in SEEDS):continue
        cache=prepare(part,refs,groups,heads[part],'test')
        validate_capture_cache(cache,part,'test',context)
        for seed in SEEDS:
            for arch in ARCHES:
                dest=OUT/arch/str(seed)/f'{part}_test.json'
                if (arch,seed,part) in existing:continue
                model,meta=load_model(arch,part,seed,context);prediction=predict(model,cache,heads[part])
                base=f'{arch}/{seed}/{part}'
                record={'predictions':prediction,'choice_sha256':choice_hash,'subject_ids':groups['test'],
                        'checkpoint_sha256':frozen['training_manifest'][base+'.pt'],
                        'metadata_sha256':frozen['training_manifest'][base+'.json'],
                        'classes':{key:SCHEMA[key]['classes'] for key in heads[part]}}
                validate_test_record(record,arch,part,seed,frozen,context,choice_hash)
                dump(dest,record)
                del model;gc.collect();torch.cuda.empty_cache()
                print('TEST',arch,part,seed,flush=True)

def summary():
    choice,context=validate_frozen();records=test_records(choice,context,require_all=True)
    dev=load(REPORT/'development.json');lat=load(REPORT/'latency.json')
    refs,groups,heads,protocol=context;families={}
    for name,arches in FAMILIES.items():
        runs={}
        for seed in SEEDS:
            y,p=merged(arches,seed,'test',context,records);temps=dev['families'][name]['runs'][str(seed)]['temperatures']
            runs[str(seed)]={'raw':aggregate_metrics(y,p),'calibrated':aggregate_metrics(y,calibrated(p,temps))}
        row={'architectures':arches,'runs':runs,'latency':lat['families'][name]}
        for split,source in [('selection',dev['families'][name]['runs']),('test',runs)]:
            row[split]={}
            for key in ['macro_f1','accuracy','balanced_accuracy','nll','brier','ece','threshold_coverage','threshold_accuracy']:
                values=[r['calibrated']['mean_per_head'][key] for r in source.values()]
                present=[v for v in values if v is not None]
                row[split][key]={'mean':float(np.mean(present)) if present else None,
                                'std':float(np.std(present,ddof=1)) if len(present)>1 else None,
                                'values':values,'valid_seeds':len(present)}
        families[name]=row
    chosen=choice['selected_family'];comparison={}
    y,pchosen=merged(FAMILIES[chosen],SEEDS[0],'test',context,records)
    for name in FAMILIES:
        _,other=merged(FAMILIES[name],SEEDS[0],'test',context,records)
        comparison[name]=paired_bootstrap(y,pchosen,other,participant_ids=groups['test'])
    training=[]
    for arch in ARCHES:
        for seed in SEEDS:
            for part in PARTS:training.append(load(OUT/arch/str(seed)/f'{part}.json')['meta'])
    support={}
    for split,ids in groups.items():
        support[split]={}
        for part in PARTS:
            for key in heads[part]:
                labels=[SCHEMA[key]['classes'].index(refs[sid][key]) for sid in ids if refs[sid].get(key) in SCHEMA[key]['classes']]
                support[split][key]={'classes':SCHEMA[key]['classes'],'counts':np.bincount(labels,minlength=len(SCHEMA[key]['classes'])).tolist(),'missing':len(ids)-len(labels)}
    result={'choice':choice,'protocol':protocol,'latency_method':{k:v for k,v in lat.items() if k!='families'},
            'families':families,'chosen_vs_other_first_seed_paired_bootstrap':comparison,'training_runs':training,'class_support':support,
            'limitations':['AI3-rater visual consensus, not independent human ground truth','Historical test62 reused by earlier experiments; retrospective audit, not new external test',
                          'Only15visualheads in5parts; no new7regions, geometric regression auxiliaries or VGGTaccuracy claim',
                          'Selection34 is used for early stopping and family selection; development score is optimistic',
                          '3seeds share one participant split; seed dispersion is not generalization confidence interval',
                          'Same finetuning recipe and maxbudget, different official pretrained recipes; no architecture causal ranking',
                          'Warm CNN time excludes MediaPipe/VGGT and upload, not whole-service SLA',
                          'MacroF1 uses union of true and predicted classes perhead; absenttruth classes have no measured recall']}
    dump(REPORT/'results.json',result)
    print('SUMMARY_COMPLETE',chosen,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','latency','freeze','test','summary','all']);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cudnn.benchmark=True
    funcs={'development':development,'latency':latency,'freeze':freeze,'test':test_inference,'summary':summary}
    if a.stage=='all':
        if not (REPORT/'choice_frozen.json').exists():
            development();latency();freeze()
        test_inference();summary()
    else:funcs[a.stage]()

if __name__=='__main__':main()
