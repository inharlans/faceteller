"""CPU rebuild of eye-referenced ears, isolated from frozen legacy cache/labels.

Reuses verified mask boxes and landmarks; reads original local photographs.
No parser/model loading, target fitting, registry writes, or GPU operations.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.kface import Store
from facework.release_parsing import EAR_TRANSFORM_VERSION, ear_metrics, eye_geometry
from facework.release_preprocess import VERSION, ear_crop, network_array

OLD=ROOT/'data/derived/release/ears'
OUT=ROOT/'data/derived/release/ears-v2'
LIGHTS=('L1','L2','L3')
CAMERAS={'C4':'ear_left_oblique','C10':'ear_right_oblique'}


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def preserve_manifest(path,value):
    """A repeated verification must not change a frozen manifest's byte hash."""
    if path.exists():
        previous=load(path)
        without_elapsed=lambda item:{k:v for k,v in item.items() if k!='elapsed_seconds'}
        if without_elapsed(previous)!=without_elapsed(value):
            raise ValueError('Existing manifest changed; inspect before replacing')
        return previous
    save(path,value)
    return value


def landmarks(sid):
    with np.load(ROOT/'data/derived/kface/subjects'/f'{sid}.npz',allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in ('keys','status','landmarks','pose')}
    index={}
    for i,key in enumerate(data['keys']):
        capture='|'.join(str(key).split('|')[:4])
        if capture not in index or str(key).endswith('|core'):index[capture]=i
    return data,index


def independent_transform(box,points,yaw,mirror):
    _,_,distance,roll=eye_geometry(points)
    theta=math.radians(roll);scale=128*math.cos(math.radians(yaw))/distance
    linear=scale*np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    centre=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
    matrix=np.c_[linear,np.array([96.,128.])-linear@centre]
    if mirror:matrix=(np.array([[-1,0,191],[0,1,0],[0,0,1]])@np.vstack([matrix,[0,0,1]]))[:2]
    return matrix


def process(sid,split,store,identity):
    old_path=OLD/f'{sid}.json';old=load(old_path);binary=OLD/f'{sid}.bin'
    lm_path=ROOT/'data/derived/kface/subjects'/f'{sid}.npz'
    old_hash=sha(old_path);old_binary_hash=sha(binary);lm_hash=sha(lm_path)
    if (old['subject_id']!=sid or old['split']!=split or old['protocol_sha256']!=identity['protocol_sha256']
            or old['sha256']!=old_binary_hash or old['landmarks_sha256']!=lm_hash
            or old['ear_transform_version']!='visible-ear-isotropic-v1'
            or old['preprocess_version']!=VERSION):raise ValueError('Legacy source identity mismatch')
    source={'legacy_metadata_sha256':old_hash,'legacy_binary_sha256':old_binary_hash,'landmarks_sha256':lm_hash}
    path=OUT/f'{sid}.json';target=OUT/f'{sid}.bin'
    if path.exists():
        meta=load(path)
        if any(meta.get(k)!=v for k,v in identity.items()) or meta['source_hashes']!=source or sha(target)!=meta['sha256']:
            raise ValueError('Existing v2 cache identity mismatch; never overwrite')
        return meta
    if target.exists():raise ValueError('Incomplete v2 binary exists; inspect rather than overwrite')
    data,index=landmarks(sid);frames=[];blob=io.BytesIO();rejections=[];attempts=0
    with binary.open('rb') as masks:
        for light in LIGHTS:
            frame={'capture':f'S001|{light}|E01|C4,C10','light':light,'status':'unavailable','crops':{}}
            for camera,name in CAMERAS.items():
                attempts+=1;capture=f'S001|{light}|E01|{camera}'
                entries=[f for f in old['parsing_frames'] if f['capture']==capture]
                if len(entries)!=1:raise ValueError('Cached mask capture identity mismatch')
                entry=entries[0];i=index.get(capture)
                if entry['status']!='parsed' or i is None or data['status'][i]!=1:
                    rejections.append({'capture':capture,'reasons':['LANDMARKS_OR_MASK_UNAVAILABLE']});continue
                points=data['landmarks'][i];yaw=float(data['pose'][i][1])
                if yaw!=entry['pose']['yaw']:raise ValueError('Cached pose changed')
                spec=entry['mask'];masks.seek(spec['offset']);payload=masks.read(spec['length'])
                with Image.open(io.BytesIO(payload)) as cached:mask=np.asarray(cached).copy()
                measured=ear_metrics(mask,points,yaw)
                if measured!=entry['measurement']:raise ValueError('Mask measurement changed; targets must remain fixed')
                if measured['box'] is None or measured['reasons']:
                    rejections.append({'capture':capture,'reasons':measured['reasons']});continue
                with store.image(sid,'S001',light,'E01',camera) as image:
                    if mask.shape!=(image.height,image.width):raise ValueError('Mask/source image dimensions differ')
                    try:crop=ear_crop(image,measured['box'],landmarks=points,yaw_degrees=yaw,mirror=camera=='C10')
                    except ValueError as exc:
                        if str(exc) not in ('EAR_WINDOW_TRUNCATED','EAR_SOURCE_TRUNCATED','EAR_VIEW_OUTSIDE_OBLIQUE_RANGE'):raise
                        rejections.append({'capture':capture,'reasons':[str(exc)]});continue
                    try:
                        array=network_array(crop,'ear');offset=blob.tell()
                        with Image.fromarray(array) as png:png.save(blob,format='PNG',compress_level=1)
                        frame['crops'][name]={'offset':offset,'length':blob.tell()-offset,'size':[192,256],
                            'valid_fraction':crop.valid_fraction,'source_capture':capture,'box':measured['box'],
                            'mirror':camera=='C10','yaw_degrees':yaw,'transform':crop.transform.tolist(),
                            'pixels_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                            'ear_transform_version':EAR_TRANSFORM_VERSION}
                    finally:crop.image.close()
            if frame['crops']:frame['status']='ready'
            else:frame['reason']='NO_USABLE_VISIBLE_EAR'
            frames.append(frame)
    raw=blob.getvalue();tmp=target.with_suffix('.bin.tmp');tmp.write_bytes(raw);tmp.replace(target)
    meta={**identity,'subject_id':sid,'split':split,'source_hashes':source,'sha256':hashlib.sha256(raw).hexdigest(),
          'contains_private_training_images':True,'source':'K-Face original local ZIP; cached mask geometry; no external transfer',
          'frames':frames,'attempted_ear_frames':attempts,'rejections':rejections}
    save(path,meta);return meta


def verify(sid,meta,store):
    """Reopen saved PNGs; independently reread raw photos and verify exact parity."""
    raw=(OUT/f'{sid}.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=meta['sha256']:raise ValueError('New binary hash mismatch')
    data,index=landmarks(sid);checked=0;cursor=0
    for frame in meta['frames']:
        for name,info in frame['crops'].items():
            capture=info['source_capture'];_,light,_,camera=capture.split('|');i=index[capture]
            if light!=frame['light'] or CAMERAS[camera]!=name or info['mirror']!=(camera=='C10'):
                raise ValueError('New crop capture mismatch')
            if info['offset']!=cursor or info['length']<=0:raise ValueError('PNG offset gap/overlap')
            cursor+=info['length']
            if cursor>len(raw):raise ValueError('PNG range outside binary')
            points=data['landmarks'][i];yaw=float(data['pose'][i][1])
            with Image.open(io.BytesIO(raw[info['offset']:cursor])) as png:
                if png.format!='PNG' or png.size!=(192,256):raise ValueError('Invalid ear PNG size/format')
                actual=np.asarray(png.convert('RGB')).copy()
            if hashlib.sha256(actual.tobytes()).hexdigest()!=info['pixels_sha256']:raise ValueError('PNG pixel hash mismatch')
            with store.image(sid,'S001',light,'E01',camera) as image:
                crop=ear_crop(image,info['box'],landmarks=points,yaw_degrees=yaw,mirror=info['mirror'])
                try:
                    np.testing.assert_array_equal(actual,network_array(crop,'ear'))
                    np.testing.assert_allclose(crop.transform,info['transform'],rtol=0,atol=1e-10)
                    np.testing.assert_allclose(crop.transform,independent_transform(info['box'],points,yaw,info['mirror']),rtol=0,atol=1e-10)
                    if crop.valid_fraction!=info['valid_fraction']:raise ValueError('Padding fraction mismatch')
                finally:crop.image.close()
            checked+=1
    if cursor!=len(raw):raise ValueError('Unexpected trailing binary data')
    # Reconfirm immutable legacy sources after reading/rebuilding.
    for path,key in [(OLD/f'{sid}.json','legacy_metadata_sha256'),(OLD/f'{sid}.bin','legacy_binary_sha256'),
                     (ROOT/'data/derived/kface/subjects'/f'{sid}.npz','landmarks_sha256')]:
        if sha(path)!=meta['source_hashes'][key]:raise ValueError('Source changed during preparation')
    return checked


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    cv2.setNumThreads(1);OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'models/selection-20261014/protocol.json';protocol=load(protocol_path)
    labels_path=ROOT/'data/release/extended_labels.json';labels=load(labels_path)
    members=sorted((str(sid),split) for split,ids in protocol['groups'].items() for sid in ids)
    if len(members)!=400 or len({sid for sid,_ in members})!=400:raise ValueError('400 unique participants required')
    identity={'ear_transform_version':EAR_TRANSFORM_VERSION,'preprocess_version':VERSION,
              'protocol_sha256':sha(protocol_path),'ear_function_sha256':hashlib.sha256(inspect.getsource(ear_crop).encode()).hexdigest()}
    if labels['protocol_sha256']!=identity['protocol_sha256'] or labels['groups']!=protocol['groups']:
        raise ValueError('Frozen label split mismatch')
    label_hash=sha(labels_path);store=Store(2);records={};counts=Counter();rejections=Counter();support={}
    start=time.perf_counter()
    try:
        for n,(sid,split) in enumerate(members[:args.limit],1):
            meta=process(sid,split,store,identity);checked=verify(sid,meta,store)
            counts['attempted_ear_frames']+=meta['attempted_ear_frames'];counts['png_crops_compared']+=checked
            counts['metadata_binary_hashes_verified']+=1;counts['legacy_sources_unchanged']+=1
            for rejection in meta['rejections']:rejections.update(rejection['reasons'])
            usable=[c for f in meta['frames'] if f['status']=='ready' and (split=='train' or f['light']=='L1')
                    for c in f['crops'].values() if c['valid_fraction']>=.9]
            first=[c for f in meta['frames'] if f['light']=='L1' for c in f['crops'].values() if c['valid_fraction']>=.9]
            counts['L1_people_with_usable_ear']+=bool(first);counts['L1_people_with_both_ears']+=len(first)==2
            group=support.setdefault(split,{'participants':0,'participants_with_usable_crop':0,'usable_crops':0,
                'ear_length':{},'ear_width':{}})
            group['participants']+=1;group['participants_with_usable_crop']+=bool(usable);group['usable_crops']+=len(usable)
            for key in ('ear_length','ear_width'):
                label=labels['subjects'][sid][key] if usable else None
                name=label if label is not None else 'unknown_or_no_input'
                group[key][name]=group[key].get(name,0)+1
            records[sid]={'split':split,'metadata_sha256':sha(OUT/f'{sid}.json'),'binary_sha256':meta['sha256'],
                          'parity_crops':checked,'source_hashes':meta['source_hashes']}
            if n%25==0 or n==len(members[:args.limit]):print('EAR_V2',n,'of',len(members[:args.limit]),'parity',counts['png_crops_compared'],'seconds',round(time.perf_counter()-start,1),flush=True)
    finally:store.close()
    if sha(labels_path)!=label_hash or sha(protocol_path)!=identity['protocol_sha256']:raise ValueError('Frozen input changed during preparation')
    manifest={**identity,'participants':len(records),'complete':len(records)==400,'records':records,
        'parity':{'pass':True,**dict(counts),'pixel_mismatches':0,'transform_mismatches':0,
                  'scope':'all stored ear PNGs from all three lights; raw-original regeneration, pixel/affine/padding equality; independent scale/roll formula'},
        'rejection_counts':dict(rejections),'support':support,'labels_sha256':label_hash,
        'input_size':[256,192],'reference':'128 / (eye-centre distance / cos(yaw)); eye-roll aligned; centred on visible ear box; never bbox resized',
        'support_policy':'train L1/L2/L3; other splits L1 only; valid_fraction>=0.9; counts distinct participants per class',
        'elapsed_seconds':time.perf_counter()-start,'gpu_used':False,'labels_modified':False}
    destination=OUT/('manifest.json' if manifest['complete'] else 'partial-manifest.json')
    manifest=preserve_manifest(destination,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
