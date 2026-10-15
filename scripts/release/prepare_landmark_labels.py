"""Freeze 2D morphology targets from repeated captures using train270 only.

Signed directions are referenced to zero, not to a population percentile.
This produces automated measurement targets, not human ground truth.
"""
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.kface_labels import REF_LIGHTS, quality_ok, mouth_ok
from facework.release_preprocess import eye_affine
from facework.morphometrics import BROW,landmark_metrics

CENTRES={'eye_tilt':0.,'brow_tail':0.,'mouth_corner':0.,'lip_balance':1.}


def measurements_from_points(points):
    q,_=eye_affine(points)
    metrics=landmark_metrics(q)
    # Compare outer eyebrow endpoint with inner endpoint. The old lateral-third
    # tangent mostly measured the descent from the arch, not overall tail height.
    slopes=[]
    for b in BROW.values():
        outer=(q[b['upper'][0]]+q[b['lower'][0]])/2
        inner=(q[b['upper'][-1]]+q[b['lower'][-1]])/2
        slopes.append(math.degrees(math.atan2(inner[1]-outer[1],abs(outer[0]-inner[0]))))
    metrics['brow_outer_inner_tilt']=float(np.mean(slopes))
    return metrics


def fit_feature(samples,train_ids,feature,key):
    train=[samples[s] for s in train_ids if len(samples.get(s,[]))>=3]
    if len(train)<3:raise ValueError('Insufficient training measurements')
    values=[statistics.median(v) for v in train]
    repeat_sd=float(statistics.median(statistics.stdev(v) for v in train))
    if not math.isfinite(repeat_sd) or repeat_sd<=0:raise ValueError('Invalid repeat variation')
    if key in CENTRES:
        centre=CENTRES[key];tolerance=1.96*repeat_sd
        cuts=[centre-tolerance,centre+tolerance]
        method='signed_direction_or_unit_ratio_with_train_repeat_tolerance'
    else:
        cuts=np.quantile(values,feature.get('quantiles',[.25,.75])).tolist()
        method='training_participant_quantiles'
    if any(a>=b for a,b in zip(cuts,cuts[1:])):raise ValueError('Degenerate class boundaries')
    result={'cuts':cuts,'method':method,'train_subjects':len(train),'fit_split':'train',
            'median_within_subject_sd':repeat_sd,'classes':feature['classes'],
            'metric':feature['metric'],'boundary_guard':'1.96 * train median repeat SD / sqrt(valid captures); heuristic guard, not a validated confidence interval'}
    if key in CENTRES:result.update(physical_centre=CENTRES[key],single_capture_tolerance=1.96*repeat_sd)
    else:result['quantiles']=feature.get('quantiles',[.25,.75])
    return result


def label(values,definition):
    if len(values)<3:return None
    value=statistics.median(values)
    band=1.96*definition['median_within_subject_sd']/math.sqrt(len(values))
    if any(abs(value-cut)<band for cut in definition['cuts']):return None
    return definition['classes'][int(np.searchsorted(definition['cuts'],value,side='right'))]


def main():
    schema_path=ROOT/'data/label_schema_kface_v3.json'
    protocol_path=ROOT/'models/selection-20261014/protocol.json'
    schema=json.loads(schema_path.read_text(encoding='utf-8'))['features']
    features={k:f for k,f in schema.items() if f['source']=='landmark'}
    features['brow_tail']={**features['brow_tail'],'metric':'brow_outer_inner_tilt'}
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    ids=sorted(s for group in protocol['groups'].values() for s in group)
    samples={k:{} for k in features};hashes={};folder=ROOT/'data/derived/kface/subjects'
    for number,sid in enumerate(ids,1):
        path=folder/f'{sid}.npz'
        hashes[sid]={'npz':hashlib.sha256(path.read_bytes()).hexdigest()}
        with np.load(path) as data:
            indices={str(k):i for i,k in enumerate(data['keys'])}
            for light in REF_LIGHTS:
                index=indices.get(f'S001|{light}|E01|C7|core')
                if index is None or data['status'][index]!=1:continue
                m=measurements_from_points(data['landmarks'][index])
                if not quality_ok(data['pose'][index],data['blend'][index],m):continue
                for key,f in features.items():
                    if f['part']=='mouth' and not mouth_ok(m):continue
                    value=m.get(f['metric'])
                    if value is not None and math.isfinite(value):samples[key].setdefault(sid,[]).append(float(value))
        if number%50==0:print('LANDMARK_TARGETS',number,len(ids),flush=True)
    definitions={k:fit_feature(samples[k],protocol['groups']['train'],f,k) for k,f in features.items()}
    subjects={sid:{k:label(samples[k].get(sid,[]),definitions[k]) for k in features} for sid in ids}
    support={split:{k:{'counts':[sum(subjects[s][k]==c for s in members) for c in f['classes']],
                         'unknown':sum(subjects[s][k] is None for s in members)} for k,f in features.items()}
             for split,members in protocol['groups'].items()}
    result={'version':'train270-landmark-targets-v1','participants':len(ids),'human_review_count':0,
            'reference_lights':REF_LIGHTS,'features':definitions,'subjects':subjects,'class_support':support,
            'source_hashes':{'schema':hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                             'protocol':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                             'implementation':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'capture_source_hashes':hashes,'train_ids_sha256':hashlib.sha256('\n'.join(sorted(protocol['groups']['train'])).encode()).hexdigest(),
            'meaning':'agreement with automated visible geometry, not human judgments or metric 3D accuracy'}
    out=ROOT/'data/release/landmark_labels.json'
    if (ROOT/'models/release/training_frozen.json').exists():raise RuntimeError('Release targets already frozen')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'participants':len(ids),'features':len(features),'train_support':support['train']},ensure_ascii=False))


if __name__=='__main__':main()
