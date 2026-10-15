"""Actual K-FACE triplets through the memory-only NDJSON API and real models.

Offline research evaluator stores aggregate timings and counts, never request photos.
Run after cohort analysis and fold0 validation calibration have completed.
"""
import csv,io,json,sys,time,collections,gc
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.pilot.run_vggt import triplet
from facework.pilot.runtime import RuntimeResources,make_app
from facework.pilot_contract import DIRECTIONS
from fastapi.testclient import TestClient

OUT=ROOT/'outputs/pilot-20261014'
D=ROOT/'data/derived/pilot-20261014'

def encode(ims):
    blobs=[]
    for im in ims:
        b=io.BytesIO();im.resize((640,640),Image.Resampling.LANCZOS).save(b,format='JPEG',quality=90);blobs.append(b.getvalue())
    return blobs

def main():
    r=RuntimeResources();testrows=[x for x in csv.DictReader((ROOT/'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')) if x['split']=='test']
    payloads=[encode(triplet(x['subject_id'])) for x in testrows[:10]]
    client=TestClient(make_app(r,exploratory_depth=True))
    def request(blobs):
        files={k:(k+'.jpg',b,'image/jpeg') for k,b in zip(DIRECTIONS,blobs)}
        t=time.perf_counter();response=client.post('/api/v1/analysis',files=files,data={'input_mode':'upload'})
        elapsed=time.perf_counter()-t
        events=[json.loads(l) for l in response.text.splitlines()]
        return response,events,elapsed
    first=request(payloads[0]);records=[];raw_results=[]
    if first[1][-1]['type']=='error':
        import asyncio
        from facework.pilot.runtime import ActualService
        from facework.pilot_contract import InputSet
        async def debug():
            inp=InputSet({k:bytearray(b) for k,b in zip(DIRECTIONS,payloads[0])},'upload')
            svc=ActualService(r,exploratory_depth=True)
            try:
                await svc.validate(inp)
                async for event in svc.analyze(inp):print('DEBUG_STAGE',event.get('stage',event['type']),flush=True)
            finally:svc.close();inp.clear()
        asyncio.run(debug())
        raise RuntimeError('HTTP adapter failed although direct call passed')
    for n,blobs in enumerate(payloads):
        resp,events,elapsed=request(blobs)
        assert resp.status_code==200,(resp.status_code,events)
        assert events[-1]['type']=='result',events[-1]
        result=events[-1]['data'];items=result['items'];raw_results.append(result)
        assert [v['stage'] for v in events if v['type']=='progress']==['얼굴 정렬','부위 추출','형태 분류','깊이 분석','문헌 선정']
        assert len(result['passages'])<=5
        assert all(i['label_id'] is None for i in items if i['status']=='HELD')
        assert all(i['confidence'] is None for i in items if i['kind']=='depth')
        def response_keys(value):
            if isinstance(value,dict):
                return set(value).union(*(response_keys(v) for v in value.values()))
            if isinstance(value,list):return set().union(*(response_keys(v) for v in value))
            return set()
        assert not response_keys(result) & {'world_points','landmarks','base64','image_bytes','photos'}
        rec={'case':n+1,'seconds_local_testclient':elapsed,'status':result['status'],'confirmed':sum(i['status']=='CONFIRMED' for i in items),'held':sum(i['status']=='HELD' for i in items),'depth_confirmed':sum(i['status']=='CONFIRMED' and i['kind']=='depth' for i in items),'passages':len(result['passages']),'stage_seconds':result['pilot_timings']}
        records.append(rec);print('ACTUAL_API',n+1,json.dumps(rec),flush=True)
    # Same photographs in all three slots must withhold depth, while keeping frontal CNN results.
    resp,events,_=request([payloads[0][0]]*3);wrong=events[-1]['data']
    assert all(i['status']=='HELD' for i in wrong['items'] if i['kind']=='depth')
    blank=io.BytesIO();Image.new('RGB',(640,640),'white').save(blank,format='JPEG')
    resp,events,_=request([payloads[0][0],blank.getvalue(),payloads[0][2]])
    assert resp.status_code==422 and events[-1]['code']=='FACE_NOT_FOUND' and events[-1]['direction']=='left_oblique'
    strict=TestClient(make_app(r,exploratory_depth=False))
    resp=strict.post('/api/v1/analysis',files={k:(k+'.jpg',b,'image/jpeg') for k,b in zip(DIRECTIONS,payloads[0])},data={'input_mode':'upload'})
    strict_result=json.loads(resp.text.splitlines()[-1])['data']
    assert all(i['status']=='HELD' for i in strict_result['items'] if i['kind']=='depth')
    # Design-preprocessing comparison: eyes normalized to 128px, no model retraining.
    from facework.pilot.design_preprocess import design_crops
    from facework.morphometrics import normalized_crops
    comparisons=[]
    for blobs in payloads[:5]:
        image=Image.open(io.BytesIO(blobs[0])).convert('RGB');d,_=r.a.detect(image)
        legacy=normalized_crops(image,d['landmarks'][:,:2]);new,meta=design_crops(image,d['landmarks'])
        aa={o.item_id:o for o in r.cnn(image,d,legacy) if o.label_id};bb={o.item_id:o for o in r.cnn(image,d,new) if o.label_id}
        comparisons.append({'n':len(aa),'labels_equal':sum(aa[k].label_id==bb[k].label_id for k in aa),'mean_abs_confidence_delta':float(np.mean([abs(aa[k].confidence-bb[k].confidence) for k in aa])),'transform':meta})
    times=np.array([x['seconds_local_testclient'] for x in records])
    summary={'actual_subjects':10,'images_per_request':3,'source':'K-FACE S001/L1/E01 C7,C4,C10; provider guide crop then640pxJPEG','cnn_models':10,'cnn_selection':'fold0 ConvNeXt+EfficientNet per5parts','cnn_calibration_loaded':bool(r.calibration.get('features')),'vggt_size':518,'gpu':r.engine.torch.cuda.get_device_name(0),'warmup_seconds':first[2],'warm_timing':{'mean':float(times.mean()),'median':float(np.median(times)),'p95':float(np.quantile(times,.95)),'max':float(times.max()),'within_5s_count':int((times<=5).sum()),'n':len(times)},'timing_scope':'multipart encoding + local ASGI TestClient HTTP lifecycle + decode/MediaPipe +10CNN +VGGT +reference lookup +NDJSON serialize; models preloaded; no WAN/TLS/real browser. Simulated crops prepared before timer.','repository':'real draft JSON content in memory; actual PostgreSQL adapter tested separately with synthetic decisions','records':records,'actual_error_tests':{'wrong_directions_withhold_only_depth':True,'blank_side_422_direction':True,'strict_unvalidated_depth_held':True},'design_eye_normalized_vs_training_crop':comparisons,'privacy':'no image files saved by API; offline evaluator stores aggregate research metrics','performance_is_not_accuracy':True}
    (OUT/'integration_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Research-only saved results for reproducible UI integration; no photographs/landmarks/identity.
    (D/'api_result_example.json').write_text(json.dumps(next((v for v in raw_results if v['passages']),raw_results[0]),ensure_ascii=False,indent=2),encoding='utf-8')
    print('SUMMARY',json.dumps(summary['warm_timing']),flush=True)
if __name__=='__main__':main()
