"""CPU-only exact input parity: original photos through live preprocessing vs frozen selection cache.
Keeps photos/arrays in memory and saves scalar checks only. Does not fit models.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import torch
    from facework import analysis_v2
    from facework.kface import Store
    from facework.release_runtime import ReleaseService
    from facework.release_parsing import ReleaseParser,visible_ear_box,forehead_metrics,ear_metrics,visible_mark_candidates
    from facework.release_preprocess import network_array
    from facework.release_context import compose_context
    torch.set_num_threads(2)
    candidate=ROOT/'experiments/relabel-20261015/candidate-v3'
    load=lambda p:json.loads(p.read_text(encoding='utf8'))
    registry=load(ROOT/'data/release/training_registry.json')
    targets=load(candidate/'targets.json')
    ids=sorted(s for s,d in targets['subjects'].items() if d['split']=='selection')
    parser=ReleaseParser(threads=2);store=Store(2)
    resource=SimpleNamespace(registry=registry,parser=parser,visible_ear_box=visible_ear_box,
        forehead_metrics=forehead_metrics,ear_metrics=ear_metrics,visible_mark_candidates=visible_mark_candidates)
    actual={}
    try:
        for sid in ids:
            service=ReleaseService(resource,None)
            try:
                service.images=[store.image(sid,'S001','L1','E01',cam) for cam in ('C7','C4','C10')]
                service.detections=[analysis_v2.detect(im)[0] for im in service.images]
                if any(d is None for d in service.detections):raise RuntimeError('Missing face')
                service.align();service.prepare()
                for part,spec in registry['parts'].items():
                    whole=network_array(service.crops['face'],'face') if spec.get('context_version') else None
                    for name in spec['crops']:
                        crop=service.crops.get(name)
                        if crop is None or crop.valid_fraction<.9:continue
                        array=network_array(crop,part)
                        if whole is not None:array=compose_context(array,whole,part,mirror_face=name.endswith('_left'))
                        actual[(sid,part,name)]=array
            finally:service.close()
        compared=0;differences=[];missing=[]
        for part in registry['parts']:
            cache=torch.load(candidate/f'cache/{part}_selection.pt',map_location='cpu',weights_only=True)
            for index,row in enumerate(cache['rows']):
                key=(row['subject_id'],part,row['crop']);array=actual.get(key)
                if array is None:missing.append({'part':part,'crop':row['crop'],'subject_id':row['subject_id']});continue
                expected=cache['images'][index].permute(1,2,0).numpy();compared+=1
                if not np.array_equal(array,expected):
                    delta=np.abs(array.astype(np.int16)-expected.astype(np.int16))
                    differences.append({'subject_id':row['subject_id'],'part':part,'crop':row['crop'],
                        'max_channel_difference':int(delta.max()),'changed_channel_fraction':float(np.mean(delta!=0))})
            del cache
        result={'status':'passed' if compared and not differences and not missing else 'failed','selection_people':len(ids),
            'compared_crops':compared,'differences':differences,'missing_crops':missing,
            'photos_or_coordinates_saved':False,'gpu_used':False,'scope':'exact uint8 model inputs, not model accuracy'}
        (candidate/'input-parity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
    finally:
        store.close();analysis_v2.landmarker().close()
        del parser.session

if __name__=='__main__':main()
