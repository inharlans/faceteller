"""Exercise the running local service with actual photos, verify returned mixed G.

Photographs stay in memory. Only the offline evaluator stores numeric responses.
"""
import csv
import io
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.pilot.run_vggt import triplet
from facework.pilot_contract import DIRECTIONS, Decision, PassageSelector, load_real_repository

OUT=ROOT/'outputs/pilot-20261014/mixed-service'
OUT.mkdir(exist_ok=True)

def encode(sid):
    images=triplet(sid);blobs=[]
    try:
        for image in images:
            b=io.BytesIO()
            image.resize((640,640),Image.Resampling.LANCZOS).save(b,format='JPEG',quality=90)
            blobs.append(b.getvalue())
    finally:
        for image in images:image.close()
    return blobs

def main():
    repo,_=load_real_repository(ROOT,allow_draft=True)
    ids=sorted(r['subject_id'] for r in csv.DictReader((ROOT/'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')) if r['split']=='test')
    rows=[];responses=[];count=Counter();examples={}
    with httpx.Client(base_url='http://127.0.0.1:8767',timeout=45,trust_env=False) as client:
        info=client.get('/api/v1/service-info').raise_for_status().json()
        assert info['score_mode']=='include_measurement_scores'
        def post(blobs):
            start=time.perf_counter()
            response=client.post('/api/v1/analysis',files={d:(d+'.jpg',b,'image/jpeg') for d,b in zip(DIRECTIONS,blobs)},data={'input_mode':'upload'})
            elapsed=time.perf_counter()-start
            events=[json.loads(line) for line in response.text.splitlines()]
            return response,events,elapsed
        for i,sid in enumerate(ids,1):
            response,events,elapsed=post(encode(sid))
            assert response.status_code==200,(i,response.status_code,events[-1])
            assert events[-1]['type']=='result',(i,events[-1])
            result=events[-1]['data']
            assert result['ranking_policy']['score_mode']=='include_measurement_scores'
            confirmed={r['item_id']:r for r in result['items'] if r['status']=='CONFIRMED'}
            labels={r['label_id'] for r in confirmed.values()}
            candidates=repo.load_candidates(labels)
            index={p.passage_id:p for p in candidates}
            per_region=Counter();last_score=math.inf
            assert len(result['passages'])<=5
            for rank,p in enumerate(result['passages'],1):
                assert p['score_mode']=='include_measurement_scores'
                assert p['rank']==rank
                actual=index[p['passage_id']]
                assert actual.required_labels<=labels
                evidence=[confirmed[k] for k in p['evidence_item_ids']]
                assert {x['label_id'] for x in evidence}==actual.required_labels
                values=[e['confidence'] if e['kind']=='classification' else e['measurement_score'] for e in evidence]
                expected=math.prod(values)**(1/len(values))
                assert math.isclose(p['G'],expected,rel_tol=1e-12,abs_tol=1e-12)
                assert p['score_feature_count']==len(values)==len(p['score_components'])
                assert {c['item_id'] for c in p['score_components']}=={e['item_id'] for e in evidence}
                for component in p['score_components']:
                    item=confirmed[component['item_id']]
                    assert component['score']==(item['confidence'] if item['kind']=='classification' else item['measurement_score'])
                    if item['kind']=='depth':assert item['confidence'] is None
                assert p['G']<=last_score+1e-15;last_score=p['G']
                per_region.update({e['region'] for e in evidence})
                assert max(per_region.values())<=3
                kinds={e['kind'] for e in evidence}
                category='mixed' if len(kinds)==2 else 'depth_only' if 'depth' in kinds else 'cnn_only'
                count['selected_'+category]+=1
                if category not in examples:examples[category]={'case':i,'result':result}
                count['verified_candidate_formulas']+=1
            old=PassageSelector('classification_only').select([Decision(**r) for r in result['items']],candidates)
            old_ids=[p['passage_id'] for p in old];new_ids=[p['passage_id'] for p in result['passages']]
            changed=old_ids!=new_ids
            if changed and 'ranking_changed' not in examples:examples['ranking_changed']={'case':i,'result':result,'previous_passages':old}
            count['order_changed_people']+=changed
            count['selection_changed_people']+=set(old_ids)!=set(new_ids)
            count['people_with_passages']+=bool(new_ids)
            count['within5_seconds']+=elapsed<=5
            record={'case':i,'http_seconds':elapsed,'status':result['status'],'selected_count':len(new_ids),
                    'confirmed':len(confirmed),'held':len(result['items'])-len(confirmed),
                    'previous_ids':old_ids,'mixed_ids':new_ids,'order_changed':changed,
                    'stage_seconds':result['pilot_timings']}
            rows.append(record);responses.append({'case':i,'result':result})
            if i%10==0 or changed:print('ACTUAL_HTTP',i,'seconds',round(elapsed,3),'passages',len(new_ids),'rank_changed',changed,flush=True)
        # Bad view directions withhold depth; CNN-only candidates keep working.
        blobs=encode(ids[0])
        response,events,_=post([blobs[0]]*3)
        bad=events[-1]['data'];assert response.status_code==200
        assert all(r['status']=='HELD' for r in bad['items'] if r['kind']=='depth')
        assert all(c['kind']=='classification' for p in bad['passages'] for c in p['score_components'])
        blank=io.BytesIO();Image.new('RGB',(640,640),'white').save(blank,format='JPEG')
        response,events,_=post([blobs[0],blank.getvalue(),blobs[2]])
        assert response.status_code==422 and events[-1]['direction']=='left_oblique'
    times=np.array([r['http_seconds'] for r in rows])
    report={'status':'passed','actual_people':len(rows),'actual_photos':len(rows)*3,
            'service_info':info,'counts':dict(count),
            'timing':{'mean':float(times.mean()),'median':float(np.median(times)),'p95':float(np.quantile(times,.95)),'max':float(times.max())},
            'timing_scope':'real local TCP HTTP multipart request -> CNN+VGGT+memory draft repository -> NDJSON; model loading/startup excluded; image ZIP reading/cropping/JPEG encoding before timer',
            'extra_checks':{'wrong_directions_hold_depth_keep_CNN':True,'blank_side_directional422':True},
            'examples':examples,'records':rows,'privacy':'API has no storage; offline evaluator saves numeric responses only; no photographs or landmark arrays',
            'limitations':['new7regionmodels still absent','depth boundaries exploratory without3Dgroundtruth','literature drafts; actual runtime repository in memory','physical camera capture not tested']}
    (OUT/'live_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'actual_responses.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in responses)+'\n',encoding='utf-8')
    print('PASS',json.dumps({k:v for k,v in report.items() if k not in ['examples','records']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
