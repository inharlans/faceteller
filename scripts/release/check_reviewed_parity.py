"""Compare real HTTP CNN outputs with the selected checkpoint's cached predictions.

Uses the 34 selection people solely for implementation parity, not new model
selection. No images or coordinates are saved, and no model/GPU is loaded.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.release_runtime import temperature_probability


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))


def verify(candidate, service_path, tolerance=1e-4):
    targets=load(candidate/'targets.json');manifest=load(candidate/'deployment_candidate.json')
    ids=sorted(s for s,d in targets['subjects'].items() if d['split']=='selection')
    service=load(service_path)
    if len(ids)!=34 or not service['complete'] or not service['all400']:
        raise RuntimeError('Requires completed real400 run and fixed selection34')
    rows={r['subject_id']:r for r in service['participants'] if r['split']=='selection'}
    if set(rows)!=set(ids):raise RuntimeError('Selection participants differ')
    registry=load(ROOT/'data/release/training_registry.json')
    checked=[];differences=[];skipped=0;missing_input=0
    for part,spec in registry['parts'].items():
        record=load(candidate/f'models/resnet18/{part}.json')
        for key in spec['features']:
            cal=manifest['features'][key];prediction=record['selection'][key]
            if len(prediction['prob'])!=len(ids):raise RuntimeError('Cached prediction row count differs')
            for index,sid in enumerate(ids):
                item=next(i for i in rows[sid]['items'] if i['item_id']==key)
                if item['status']!='CONFIRMED':skipped+=1;continue
                if not prediction['available'][index]:
                    missing_input+=1;continue
                if not cal['calibrated']:raise RuntimeError('Service confirmed an uncalibrated head')
                p=temperature_probability(prediction['prob'][index],cal['temperature'])
                top=int(np.argmax(p));label=f'{key}:{cal["classes"][top]}'
                delta=abs(float(p[top])-item['confidence'])
                entry={'subject_id':sid,'item_id':key,'label_matches':label==item['label_id'],
                       'confidence_absolute_difference':delta}
                checked.append(entry)
                if not entry['label_matches'] or delta>tolerance:differences.append(entry)
    result={'status':'passed' if checked and not differences else 'failed',
            'meaning':'cached selected-checkpoint inputs vs real original-photo HTTP inputs; implementation parity, not classification accuracy',
            'selection_people':len(ids),'checked_confirmed_predictions':len(checked),'skipped_unconfirmed_predictions':skipped,
            'skipped_missing_cached_input':missing_input,
            'tolerance':tolerance,'maximum_confidence_difference':max((r['confidence_absolute_difference'] for r in checked),default=None),
            'differences':differences,'photos_or_coordinates_saved':False,'new_model_fitting':False}
    output=candidate/'runtime-parity.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='differences'},ensure_ascii=False))
    if result['status']!='passed':raise RuntimeError(f'CNN runtime parity failed; inspect {output}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path);parser.add_argument('service_report',type=Path)
    args=parser.parse_args();verify(ROOT/args.candidate,args.service_report)
