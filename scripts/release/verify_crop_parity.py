"""Check all400 original L1 images against the exact cached runtime tensors."""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.kface import Store
from facework.release_preprocess import frontal_crops,network_array,PART_CROPS,VERSION


def main():
    folder=ROOT/'data/derived/release/subjects'
    manifest=json.loads((folder.parent/'crop_manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and len(manifest['participants'])==400
    mapping={crop:part for part,names in PART_CROPS.items() for crop in names}
    store=Store(2);n=0;diffs=[];corrupt=[]
    try:
        for index,person in enumerate(manifest['participants']):
            sid=person['subject_id'];meta=json.loads((folder/f'{sid}.json').read_text(encoding='utf-8'))
            frame=next(f for f in meta['frames'] if f['light']=='L1')
            assert frame['status']=='ready'
            binary=(folder/f'{sid}.bin').read_bytes()
            if hashlib.sha256(binary).hexdigest()!=meta['sha256']:corrupt.append(sid)
            with np.load(ROOT/'data/derived/kface/subjects'/f'{sid}.npz') as archive:
                position=list(archive['keys']).index('S001|L1|E01|C7|core')
                points=archive['landmarks'][position]
            with store.image(sid,'S001','L1','E01','C7') as image:
                crops=frontal_crops(image,points)
                for name,crop in crops.items():
                    spec=frame['crops'][name]
                    with Image.open(io.BytesIO(binary[spec['offset']:spec['offset']+spec['length']])) as cached:
                        expected=network_array(crop,mapping[name]);actual=np.asarray(cached.convert('RGB'))
                        if not np.array_equal(actual,expected):diffs.append({'subject':sid,'crop':name})
                    n+=1;crop.image.close()
            if (index+1)%25==0:print('PARITY',index+1,400,flush=True)
    finally:store.close()
    result={'preprocess_version':VERSION,'participants':400,'representative_capture':'S001/L1/E01/C7',
            'image_crops_compared':n,'comparison':'exact decoded RGB equality of PNG cache and runtime preprocessing from original image',
            'mismatches':diffs,'cache_hash_failures':corrupt,'pass':not diffs and not corrupt,
            'scope':'all frontal crop families, original400 participants, L1 only; ears tested separately',
            'images_in_report':False}
    path=ROOT/'outputs/release/crop_parity.json';path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result['pass']:raise SystemExit(1)


if __name__=='__main__':main()
