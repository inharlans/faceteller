"""CPU/local BiSeNet masks, actual visible-ear PNG crops, weak labels for 400.

Does not change the main frontal cache, existing labels, split, or training run.
All masks/crops are private authorized offline K-Face training artifacts.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.kface import Store
from facework.release_parsing import (ReleaseParser, VERSION, MODEL_SHA256, FEATURE_CLASSES,
    forehead_metrics, ear_metrics, stable_measurement, fit_thresholds, metric_label,
    visible_mark_candidates, repeated_mark_label, MARK_VERSION, MARK_SCOPE, EAR_TRANSFORM_VERSION)
from facework.release_preprocess import ear_crop, network_array, VERSION as CROP_VERSION, INPUT_SIZES

OUT=ROOT/'data/derived/release/ears'
LABEL_OUT=ROOT/'data/release'
LIGHTS=('L1','L2','L3')
CAMERAS={'C7':None,'C4':'ear_left_oblique','C10':'ear_right_oblique'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def append_png(blob,array):
    start=blob.tell()
    with Image.fromarray(array) as image:image.save(blob,format='PNG',compress_level=1)
    return {'offset':start,'length':blob.tell()-start,'size':[array.shape[1],array.shape[0]]}


def process(subject,split,store,parser,protocol_hash):
    path=OUT/f'{subject}.json';binary=OUT/f'{subject}.bin'
    source=ROOT/'data/derived/kface/subjects'/f'{subject}.npz'
    identity={'version':VERSION,'preprocess_version':CROP_VERSION,'protocol_sha256':protocol_hash,
              'model_sha256':MODEL_SHA256,'landmarks_sha256':digest(source),'lights':list(LIGHTS),
              'ear_transform_version':EAR_TRANSFORM_VERSION}
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if any(old.get(k)!=v for k,v in identity.items()):
            raise ValueError('existing parsing cache identity differs; do not overwrite')
        if binary.exists() and digest(binary)==old.get('sha256'):return old
        raise ValueError('existing parsing cache incomplete/corrupt')
    with np.load(source) as archive:
        data={k:archive[k] for k in ['keys','status','landmarks','pose']}
    # Several purposes can contain a capture. Prefer core, otherwise first exact capture.
    index={}
    for i,key in enumerate(data['keys']):
        capture='|'.join(str(key).split('|')[:4])
        if capture not in index or str(key).endswith('|core'):index[capture]=i
    frames=[];parsing_frames=[];blob=io.BytesIO()
    values={k:[] for k in ['forehead_height','forehead_width']}
    ear_values={camera:{k:[] for k in ['ear_length','ear_width']} for camera in ['C4','C10']}
    for light in LIGHTS:
        frame={'capture':f'S001|{light}|E01|C4,C10','light':light,'status':'unavailable','crops':{}}
        for camera,crop_name in CAMERAS.items():
            capture=f'S001|{light}|E01|{camera}'
            entry={'capture':capture,'light':light,'camera':camera,'status':'unavailable'}
            i=index.get(capture)
            if i is None or data['status'][i]!=1:
                entry['reasons']=['LANDMARKS_UNAVAILABLE'];parsing_frames.append(entry);continue
            points=data['landmarks'][i]
            pitch,yaw,roll=map(float,data['pose'][i])
            entry['pose']={'pitch':pitch,'yaw':yaw,'roll':roll}
            with store.image(subject,'S001',light,'E01',camera) as image:
                mask=parser.parse(image,points)
                entry['mask']=append_png(blob,mask)
                entry['mask']['coordinates']='original_image_pixels'
                entry['mask_classes']={'skin':1,'hair':17,'ears':[7,8]}
                entry['status']='parsed'
                if camera=='C7':
                    measured=forehead_metrics(mask,points)
                    if abs(yaw)>12 or abs(pitch+5)>16:
                        measured['values']={k:None for k in values};measured['reasons'].append('FRONTAL_POSE_INVALID')
                    for k in values:values[k].append(measured['values'][k])
                else:
                    measured=ear_metrics(mask,points,yaw)
                    for k in ear_values[camera]:ear_values[camera][k].append(measured['values'][k])
                    if measured['box'] is not None and not measured['reasons']:
                        crop=ear_crop(image,measured['box'],landmarks=points,yaw_degrees=yaw,mirror=camera=='C10')
                        try:
                            array=network_array(crop,'ear')
                            info=append_png(blob,array)
                            info.update(valid_fraction=crop.valid_fraction,source_capture=capture,
                                        box=measured['box'],mirror=camera=='C10')
                            frame['crops'][crop_name]=info
                        finally:crop.image.close()
                entry['measurement']=measured
            parsing_frames.append(entry)
        if frame['crops']:frame['status']='ready'
        else:frame['reason']='NO_USABLE_VISIBLE_EAR'
        frames.append(frame)
    aggregate={k:stable_measurement(v) for k,v in values.items()}
    sides={camera:{k:stable_measurement(v) for k,v in part.items()} for camera,part in ear_values.items()}
    for k in ['ear_length','ear_width']:
        pair=[sides[c][k]['value'] for c in ['C4','C10']]
        if any(v is None for v in pair):
            aggregate[k]={'value':None,'reason':'BOTH_EARS_NEED_STABLE_REPEATS','side_measurements':{c:sides[c][k] for c in sides}}
        else:
            # Person target is the bilateral median. Preserve each side; do not call asymmetry noise.
            aggregate[k]={'value':float(np.median(pair)),'reason':None,'side_measurements':{c:sides[c][k] for c in sides},
                          'bilateral_relative_difference':float(abs(pair[0]-pair[1])/np.mean(pair))}
    aggregate['pigmented_marks']={'value':None,'reason':'NO_VALIDATED_VISIBLE_MARK_DETECTOR; absence_not_inferred_from_no_detection'}
    payload=blob.getvalue();tmp=binary.with_suffix('.bin.tmp');tmp.write_bytes(payload);tmp.replace(binary)
    meta={**identity,'subject_id':subject,'split':split,'contains_private_training_images':True,
          'source':'K-Face original ZIP; images never transmitted','sha256':hashlib.sha256(payload).hexdigest(),
          'frames':frames,'parsing_frames':parsing_frames,'metrics':aggregate,
          'score_kind':'weak_automated_geometry_not_classifier_confidence'}
    write_json(path,meta)
    return meta


def finalize(protocol,protocol_hash):
    all_ids=sorted(sid for ids in protocol['groups'].values() for sid in ids)
    metadata={}
    for sid in all_ids:
        path=OUT/f'{sid}.json'
        if not path.exists():return {'status':'incomplete','people':len(metadata),'required':400}
        meta=json.loads(path.read_text(encoding='utf-8'))
        if (meta['version']!=VERSION or meta['protocol_sha256']!=protocol_hash or meta['model_sha256']!=MODEL_SHA256
                or meta['preprocess_version']!=CROP_VERSION or meta.get('ear_transform_version')!=EAR_TRANSFORM_VERSION):
            raise ValueError('cannot combine incompatible participant caches')
        metadata[sid]=meta
    metrics={sid:{k:v['value'] for k,v in meta['metrics'].items()} for sid,meta in metadata.items()}
    thresholds={k:fit_thresholds(metrics,protocol['groups']['train'],k) for k in FEATURE_CLASSES if k!='pigmented_marks'}
    marks_complete=all(m['metrics']['pigmented_marks'].get('version')==MARK_VERSION for m in metadata.values())
    thresholds['pigmented_marks']={'status':'fixed_algorithm' if marks_complete else 'unavailable',
        'reason':None if marks_complete else 'MARK_REPEAT_PASS_NOT_RUN','fit_split':None,
        'version':MARK_VERSION,'minimum_repeats':2,'absence_requires_clear_repeats':3,
        'scope':MARK_SCOPE,'population_threshold_fitting':False}
    subjects={sid:{k:metric_label(metrics[sid][k],thresholds[k],classes) if k!='pigmented_marks' else metadata[sid]['metrics'][k].get('label')
                   for k,classes in FEATURE_CLASSES.items()} for sid in all_ids}
    names={'forehead_height':'보이는 이마 높이','forehead_width':'보이는 이마 폭','ear_length':'보이는 귀 상대 길이',
           'ear_width':'보이는 귀 폭/길이','pigmented_marks':'보이는 색소 표식 유무(비의학적)'}
    names_ko={'forehead_height':['낮음','보통','높음'],'forehead_width':['좁음','보통','넓음'],
              'ear_length':['짧음','보통','김'],'ear_width':['좁음','보통','넓음'],'pigmented_marks':['없음','있음']}
    definitions={'forehead_height':'visible skin/hair transition to inner-brow baseline / eye-centre distance',
                 'forehead_width':'connected skin width halfway between hair boundary and inner-brow baseline / eye-centre distance',
                 'ear_length':'roll-aligned ear mask height / (projected eye-centre distance / cos(yaw)); bilateral median',
                 'ear_width':'roll-aligned ear mask width / height; bilateral median',
                 'pigmented_marks':MARK_SCOPE}
    provenance=json.loads((ROOT/'models/face_parsing/provenance.json').read_text(encoding='utf-8'))
    features={k:{'ko':names[k],'part':'forehead' if k.startswith('forehead') else 'ear' if k.startswith('ear') else 'mole',
                 'source':'visual','classes':classes,'classes_ko':names_ko[k],
                 'metric':definitions[k],'provenance':{'kind':'weak_automated','model_sha256':MODEL_SHA256,
                     'human_reviewed':False,'model':'yakhyo BiSeNet ResNet18/CelebAMask-HQ'},
                 'unknown':None,'confidence_available':False,
                 'label_status':'weak_geometry_label' if thresholds[k]['status']=='fitted' else 'weak_algorithm_label' if k=='pigmented_marks' and marks_complete else 'unavailable',
                 'thresholds':thresholds[k]} for k,classes in FEATURE_CLASSES.items()}
    support={split:{k:dict(Counter('unknown' if subjects[sid][k] is None else subjects[sid][k] for sid in ids))
                     for k in FEATURE_CLASSES} for split,ids in protocol['groups'].items()}
    summary={'people':len(subjects),'cpu_only':True,'split_support':support,
             'parsed_frames':sum(sum(f['status']=='parsed' for f in m['parsing_frames']) for m in metadata.values()),
             'ear_crops':sum(sum(len(f['crops']) for f in m['frames']) for m in metadata.values()),
             'L1_people_with_ear_crop':sum(bool(m['frames'][0]['crops']) for m in metadata.values()),
             'L1_people_with_both_ears':sum(len(m['frames'][0]['crops'])==2 for m in metadata.values()),
             'human_ground_truth':0,
             'all_train_classes_observed':{k:all(support['train'][k].get(c,0)>0 for c in classes) for k,classes in FEATURE_CLASSES.items()}}
    summary['support_flags']={k:{'minimum_train_class_count':min(support['train'][k].get(c,0) for c in classes),
        'selection_known':sum(support['selection'][k].get(c,0) for c in classes),
        'missing_calibration_classes':[c for c in classes if not support['calibration'][k].get(c,0)]}
        for k,classes in FEATURE_CLASSES.items()}
    if marks_complete:
        diagnostics=[f for m in metadata.values() for f in m['metrics']['pigmented_marks']['frames']]
        stable=[p['repeats'] for m in metadata.values() for p in m['metrics']['pigmented_marks']['stable_marks']]
        summary['mark_repeatability']={'frames':len(diagnostics),'quality_passed':sum(f['quality_passed'] for f in diagnostics),
            'candidate_count':sum(len(f['candidates']) for f in diagnostics),
            'stable_candidate_repeat_counts':dict(Counter(stable)),
            'subject_reasons':dict(Counter(m['metrics']['pigmented_marks']['reason'] or 'WEAK_LABEL_ASSIGNED' for m in metadata.values())),
            'interpretation':'repeat stability is not verified detection accuracy; stable candidates may occur in subjects withheld for other unstable candidates'}
    write_json(LABEL_OUT/'extended_schema.json',{'version':VERSION,'preprocess_version':CROP_VERSION,
              'ear_transform_version':EAR_TRANSFORM_VERSION,
              'network_input_sizes':{part:list(INPUT_SIZES[part]) for part in ['forehead','ear','mole']},
              'features':features,'model_provenance':provenance,
              'scope':'weak automated visible morphology; no medical, personality, intelligence or fate inference'})
    write_json(LABEL_OUT/'extended_labels.json',{'version':VERSION,'preprocess_version':CROP_VERSION,
              'ear_transform_version':EAR_TRANSFORM_VERSION,'protocol_sha256':protocol_hash,
              'processing_complete':{'masks_400':True,'mark_repeats_400':marks_complete},
              'training_ready':False,'training_readiness_note':'main must validate class support and weak-label policy; processing completion is not model readiness',
              'subjects':subjects,'metrics':{sid:m['metrics'] for sid,m in metadata.items()},
              'thresholds':thresholds,'groups':protocol['groups'],'summary':summary,
              'privacy':'private offline K-Face training annotations, never service request history'})
    return summary


def annotate_marks(subject,store):
    """Second CPU pass; reads saved masks, does not rerun/interfere with BiSeNet."""
    path=OUT/f'{subject}.json';meta=json.loads(path.read_text(encoding='utf-8'))
    if meta['metrics']['pigmented_marks'].get('version')==MARK_VERSION:return meta
    with np.load(ROOT/'data/derived/kface/subjects'/f'{subject}.npz') as a:
        keys=[str(k) for k in a['keys']];points=a['landmarks']
    diagnostics=[]
    with (OUT/f'{subject}.bin').open('rb') as binary:
        for entry in meta['parsing_frames']:
            if entry['camera']!='C7':continue
            if entry['status']!='parsed':
                diagnostics.append({'quality_passed':False,'reasons':['MASK_UNAVAILABLE'],'candidates':[],'light':entry['light']});continue
            i=next(i for i,k in enumerate(keys) if k.startswith(entry['capture']+'|'))
            info=entry['mask'];binary.seek(info['offset'])
            with Image.open(io.BytesIO(binary.read(info['length']))) as im:mask=np.asarray(im).copy()
            with store.image(subject,'S001',entry['light'],'E01','C7') as image:
                result=visible_mark_candidates(image,mask,points[i])
            result['light']=entry['light'];diagnostics.append(result)
    result=repeated_mark_label(diagnostics)
    meta['metrics']['pigmented_marks']={**result,'value':None,'version':MARK_VERSION,
        'scope':MARK_SCOPE,'frames':diagnostics,'source':'algorithm_generated_weak_target',
        'confidence':None,'clinical_truth_validated':False,'historical_mole_mapping':'unmapped'}
    write_json(path,meta)
    return meta


def migrate_ear_version(subject,store,protocol_hash):
    """Ear v2 changes pixels: legacy metadata-only migration is no longer valid."""
    raise ValueError('Ear transform changed; preserve ears and run prepare_ear_crops_v2.py into ears-v2')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--finalize-only',action='store_true')
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--marks-only',action='store_true')
    parser.add_argument('--migrate-ear-version',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(1)
    protocol_path=ROOT/'models/selection-20261014/protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));protocol_hash=digest(protocol_path)
    members=sorted((sid,split) for split,ids in protocol['groups'].items() for sid in ids)
    assert len(members)==400 and len({sid for sid,_ in members})==400 and len(protocol['groups']['train'])==270
    OUT.mkdir(parents=True,exist_ok=True)
    if args.migrate_ear_version:
        store=Store(2);checked=0
        try:
            for n,(sid,_) in enumerate(members[:args.limit],1):
                checked+=migrate_ear_version(sid,store,protocol_hash)
                if n%50==0:print('EAR_VERSION_PARITY',n,'crops',checked,flush=True)
        finally:store.close()
    elif args.marks_only:
        store=Store(2)
        try:
            for n,(sid,_) in enumerate(members[:args.limit],1):
                meta=annotate_marks(sid,store)
                if n%25==0:print('MARKS',n,'of',len(members[:args.limit]),flush=True)
        finally:store.close()
    elif not args.finalize_only:
        engine=ReleaseParser(threads=args.threads);store=Store(2);start=time.perf_counter()
        try:
            for n,(sid,split) in enumerate(members[:args.limit],1):
                meta=process(sid,split,store,engine,protocol_hash)
                print('PARSED',n,'of',len(members[:args.limit]),'ears',sum(len(f['crops']) for f in meta['frames']),
                      'elapsed',round(time.perf_counter()-start,1),flush=True)
        finally:store.close()
    print(json.dumps(finalize(protocol,protocol_hash),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
