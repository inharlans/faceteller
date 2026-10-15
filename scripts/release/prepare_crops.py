"""Rebuild design-aligned crops from original authorized K-FACE ZIPs, CPU only.

Never recrop an already-cropped legacy image. Participants remain in the frozen
270/34/34/62 groups. The private cache is a training artifact, not upload history.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.kface import Store
from facework.release_preprocess import frontal_crops, VERSION, PART_CROPS, network_array

OUT=ROOT/'data/derived/release/subjects'
LIGHTS=['L1','L2','L3','L8','L9','L12','L13','L14','L16','L19','L22','L23']


def write_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def process(subject,split,store,protocol_hash):
    meta_path=OUT/f'{subject}.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['preprocess_version']!=VERSION or meta['protocol_sha256']!=protocol_hash:
            raise ValueError('existing crop cache version differs')
        if (OUT/f'{subject}.bin').exists():return meta
    source=ROOT/'data/derived/kface/subjects'/f'{subject}.npz'
    with np.load(source) as archive:
        data={k:archive[k] for k in ('keys','status','landmarks','pose','blend')}
    index={str(k):i for i,k in enumerate(data['keys'])}
    # Keep several lighting repeats for label/repeatability audits in every split;
    # the trainer uses repeats only for train, L1 only for model evaluation.
    frames=[];blob=io.BytesIO()
    crop_to_part={crop:part for part,names in PART_CROPS.items() for crop in names}
    for light in LIGHTS:
        key=f'S001|{light}|E01|C7|core'
        i=index[key]
        frame={'capture':key,'light':light,'status':'unavailable','crops':{}}
        if data['status'][i]!=1:
            frame['reason']='FACE_NOT_FOUND';frames.append(frame);continue
        pitch,yaw,roll=map(float,data['pose'][i])
        frame.update(pose={'pitch':pitch,'yaw':yaw,'roll':roll},blend=data['blend'][i].tolist())
        if abs(yaw)>12 or abs(pitch+5)>16:
            frame['reason']='FRONTAL_POSE_INVALID';frames.append(frame);continue
        with store.image(subject,'S001',light,'E01','C7') as image:
            for name,crop in frontal_crops(image,data['landmarks'][i]).items():
                part=crop_to_part[name]
                # PNG is lossless: no second JPEG compression over the provider photo.
                from PIL import Image
                output=Image.fromarray(network_array(crop,part))
                start=blob.tell();output.save(blob,format='PNG',compress_level=1)
                frame['crops'][name]={'offset':start,'length':blob.tell()-start,
                                      'valid_fraction':crop.valid_fraction,'size':list(output.size)}
                crop.image.close();output.close()
        frame['status']='ready';frames.append(frame)
    payload=blob.getvalue()
    binary=OUT/f'{subject}.bin';temporary=binary.with_suffix('.bin.tmp')
    temporary.write_bytes(payload);temporary.replace(binary)
    meta={'subject_id':subject,'split':split,'preprocess_version':VERSION,
          'protocol_sha256':protocol_hash,'source':'K-FACE original participant ZIP',
          'contains_private_training_images':True,'frames':frames,
          'sha256':hashlib.sha256(payload).hexdigest()}
    write_json(meta_path,meta)
    return meta


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    protocol_path=ROOT/'models/selection-20261014/protocol.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    protocol_hash=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    subjects=sorted((sid,split) for split,ids in protocol['groups'].items() for sid in ids)
    assert len(subjects)==400 and len({s for s,_ in subjects})==400
    OUT.mkdir(parents=True,exist_ok=True)
    store=Store(2);summary=[]
    try:
        for i,(subject,split) in enumerate(subjects[:args.limit]):
            meta=process(subject,split,store,protocol_hash)
            n=sum(f['status']=='ready' for f in meta['frames'])
            summary.append({'subject_id':subject,'split':split,'ready_captures':n})
            print(f'{i+1}/{len(subjects[:args.limit])} {subject} {split} {n} captures',flush=True)
    finally:store.close()
    write_json(OUT.parent/'crop_manifest.json',{'preprocess_version':VERSION,
               'protocol_sha256':protocol_hash,'participants':summary,
               'complete':len(summary)==400,'evaluation_capture':'S001|L1|E01|C7|core'})


if __name__=='__main__':main()
