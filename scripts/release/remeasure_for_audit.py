"""Remeasure all 400 cached landmark sets; no labels or models are changed."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.release.prepare_landmark_labels import measurements_from_points
from scripts.kface_labels import REF_LIGHTS, quality_ok, mouth_ok


def main():
    registry=json.loads((ROOT/'data/release/training_registry.json').read_text(encoding='utf-8'))
    features={k:f for k,f in registry['features'].items() if f['source']=='landmark'}
    result={'version':'audit-remeasure-20261015-v1','labels_changed':False,'fit_used':False,
            'registry_sha256':hashlib.sha256((ROOT/'data/release/training_registry.json').read_bytes()).hexdigest(),
            'subjects':{}}
    for number,(sid,subject) in enumerate(sorted(registry['subjects'].items()),1):
        source=ROOT/'data/derived/kface/subjects'/f'{sid}.npz'
        frames=[]
        with np.load(source) as archive:
            lookup={str(k):i for i,k in enumerate(archive['keys'])}
            for light in REF_LIGHTS:
                index=lookup.get(f'S001|{light}|E01|C7|core')
                if index is None or archive['status'][index]!=1:continue
                m=measurements_from_points(archive['landmarks'][index])
                pose=archive['pose'][index];blend=archive['blend'][index]
                frames.append({'light':light,'quality_ok':bool(quality_ok(pose,blend,m)),
                               'mouth_ok':bool(mouth_ok(m)),'pose':pose.tolist(),
                               'metrics':m})
        summaries={}
        for key,f in features.items():
            values=[x['metrics'][f['metric']] for x in frames if x['quality_ok'] and
                    (f['part'] not in ('mouth','philtrum') or x['mouth_ok'])]
            if values:
                summaries[key]={'n':len(values),'median':float(np.median(values)),
                                'sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                                'mad':float(np.median(np.abs(values-np.median(values)))),
                                'min':min(values),'max':max(values)}
        result['subjects'][sid]={'split':subject['split'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                 'frames':frames,'summaries':summaries}
        if number%100==0:print('REMEASURE',number,flush=True)
    result['train_summary']={}
    for key in features:
        rows=[d['summaries'][key] for d in result['subjects'].values() if d['split']=='train' and key in d['summaries']]
        values=[r['median'] for r in rows]
        result['train_summary'][key]={'people':len(values),'quantiles':dict(zip(['min','q10','q25','median','q75','q90','max'],
                        np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist())),
                        'median_repeat_sd':float(np.median([r['sd'] for r in rows if r['sd'] is not None]))}
    out=ROOT/'data/relabel-20261015/measurements.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'subjects':len(result['subjects']),'features':len(features),'summary':result['train_summary']},ensure_ascii=False))


if __name__=='__main__':main()
