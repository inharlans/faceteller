"""Actual-image adapter for the isolated design pilot (never replaces server.py).

Preload VGGT and fold-0 CNNs once. The full 50-model legacy ensemble is audited
separately. Only categorical CNN heads can acquire calibrated confidence here.
"""
import asyncio,io,json,threading,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from scipy.special import softmax
from facework.pilot_contract import (Observation,DecisionPolicy,ContractError,DIRECTIONS,STAGES,
                                     build_result,load_real_repository,create_app)
from facework.pilot.depth import ROOT,VGGTEngine,measure_geometry,FEATURES
from facework.pilot.scoring import boundary_scores

from concurrent.futures import ThreadPoolExecutor
RUN_LOCK=threading.Lock()
# Reuse a single inference worker across requests (and CUDA library handles).
INFERENCE_WORKER=ThreadPoolExecutor(max_workers=1,thread_name_prefix='faceteller-inference')

async def nonpreemptible(fn):
    # A CUDA kernel cannot be interrupted. Wait for the request's worker before
    # releasing its image references, including when the client disconnects.
    task=asyncio.ensure_future(asyncio.get_running_loop().run_in_executor(INFERENCE_WORKER,fn))
    try:return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:await task
        finally:raise

REGIONS={'eye':'눈','brow':'눈썹','nose':'코','mouth':'입','face':'얼굴형'}
NEW_REGIONS={'forehead':'이마','glabella':'미간','under_eye':'애교살','cheekbone':'광대','ear':'귀','philtrum':'인중','mole':'점'}

class RuntimeResources:
    def __init__(self,cnn_device='cuda'):
        import torch
        from facework import analysis_v2 as a
        from facework.kface_ensemble import checkpoints,load,combined_meta
        from facework.kface_training import eval_transform
        self.a=a;self.engine=VGGTEngine(518);self.cnn_device=cnn_device
        self.models={};self.load_seconds=self.engine.load_seconds
        for part in ['eye','brow','nose','mouth','face']:
            paths=checkpoints(part,ROOT/'models/kface_v4',fold=0)
            models=[load(p,device=cnn_device) for p in paths]
            self.models[part]=([v[0] for v in models],combined_meta([v[1] for v in models]),eval_transform(part))
        cp=ROOT/'outputs/pilot-20261014/cnn_fold0_calibration.json'
        self.calibration=json.loads(cp.read_text(encoding='utf-8')) if cp.exists() else {'features':{}}
        self.boundaries=json.loads((ROOT/'data/derived/pilot-20261014/boundary_config.json').read_text(encoding='utf-8'))
        # Existing texts are drafts. The pilot explicitly exposes that status, never calls them reviewed.
        self.repository,self.repository_audit=load_real_repository(ROOT,allow_draft=True)
        self.repository_audit['usage']='local exploratory model integration; all returned sources retain draft status'
        self.a.landmarker()

    def cnn(self, image, detection, crops):
        import torch
        predictions={}
        for part,(models,meta,tf) in self.models.items():
            x=torch.stack([tf(crops[c]) for c in meta['crops']]).to(self.cnn_device)
            with torch.inference_mode():
                outputs=[m(x) for m in models]
                for j,k in enumerate(meta['classification_features']):
                    p=torch.stack([o[1][j].softmax(1) for o in outputs]).mean(0).mean(0).cpu().numpy()
                    conf=self.calibration.get('features',{}).get(k,{})
                    T=conf.get('temperature')
                    calibrated=T is not None and T>0 and conf.get('calibrated',True)
                    if calibrated:p=softmax(np.log(np.clip(p,1e-12,1))/T)
                    idx=int(np.argmax(p));f=self.a.SCHEMA['features'][k]
                    predictions[k]=Observation(k,part,'classification',f'{k}:{meta["classes"][k][idx]}',f['classes_ko'][idx],confidence=float(p[idx]),class_count=len(p),calibrated=bool(calibrated),basis=f'fold0 ConvNeXt/EfficientNet 정면 부위 분류; T={T}; AI 초벌 레이블 기준 파일럿')
                # Legacy continuous CNN heads do not have class confidence. Do not fabricate it.
                for j,k in enumerate(meta['regression_features']):
                    if k in FEATURES:continue
                    f=self.a.SCHEMA['features'][k]
                    predictions[k]=Observation(k,part,'classification',None,None,basis='기존 회귀형 CNN 출력: 설계서의 보정된 분류 신뢰도를 아직 제공하지 않음',hold_reasons=('REGRESSION_HEAD_HAS_NO_CLASS_CONFIDENCE',))
        for key,ko in NEW_REGIONS.items():
            predictions['region_'+key]=Observation('region_'+key,key,'classification',None,None,basis=f'{ko} 전용 분류 모델 미학습',hold_reasons=('MODEL_NOT_TRAINED',))
        return list(predictions.values())

class ActualService:
    def __init__(self,resources,*,exploratory_depth=False):
        self.r=resources;self.images=[];self.detections=[];self.exploratory_depth=exploratory_depth
        self.timings={}

    def close(self):
        for image in self.images:
            image.close()
        self.images.clear();self.detections.clear()

    async def validate(self,inputs):
        def check():
            with RUN_LOCK:
                for direction in DIRECTIONS:
                    im=ImageOps.exif_transpose(Image.open(io.BytesIO(inputs.images[direction]))).convert('RGB')
                    d,err=self.r.a.detect(im)
                    if d is None:raise ContractError('FACE_NOT_FOUND' if err=='no_face' else 'PREPROCESS_FAILED','얼굴이 한 개 검출되어야 합니다.',422,direction)
                    self.images.append(im);self.detections.append(d)
        t=time.perf_counter()
        try:await nonpreemptible(check)
        except BaseException:
            self.close()
            raise
        self.timings['validation_and_landmarks']=time.perf_counter()-t

    async def analyze(self,inputs):
        from facework.morphometrics import normalized_crops,measure
        from dataclasses import replace
        observations=[];g=None
        try:
            t0=time.perf_counter();yield {'type':'progress','stage':STAGES[0]}
            # Existing crop function combines in-plane alignment and scale normalization.
            yield {'type':'progress','stage':STAGES[1]}
            t=time.perf_counter();crops=normalized_crops(self.images[0],self.detections[0]['landmarks'][:,:2])
            metrics=measure(self.images[0],self.detections[0]['landmarks'][:,:2])
            flags=self.r.a.quality_flags(self.detections[0],metrics,self.images[0]);self.timings['crop_and_quality']=time.perf_counter()-t
            yield {'type':'progress','stage':STAGES[2]}
            def run_cnn():
                with RUN_LOCK:return self.r.cnn(self.images[0],self.detections[0],crops)
            t=time.perf_counter();observations=await nonpreemptible(run_cnn);self.timings['cnn']=time.perf_counter()-t
            for n,o in enumerate(observations):
                reasons=tuple(msg for msg,scope in flags if scope in ('all',o.region) and msg)
                if reasons:observations[n]=replace(o,hold_reasons=o.hold_reasons+reasons)
            yield {'type':'progress','stage':STAGES[3]}
            def run_depth():
                with RUN_LOCK:return self.r.engine.infer(self.images)
            t=time.perf_counter();g=await nonpreemptible(run_depth)
            # Use the SAME landmark input resolution as cohort calibration.
            # Merely rescaling landmarks detected at another resolution introduces
            # a detector-distribution difference at the small measurement sites.
            with RUN_LOCK:
                mapped=[self.r.a.detect(im)[0] for im in g['prepared_images']]
            dep=measure_geometry(g,mapped);self.timings['depth_and_measurement']=time.perf_counter()-t
            yaw=[d['yaw'] for d in self.detections]
            directions_ok=abs(yaw[0])<=15 and 20<=abs(yaw[1])<=60 and 20<=abs(yaw[2])<=60 and yaw[1]*yaw[2]<0
            for k,region in [('nose_projection','nose'),('nose_bridge','nose'),('lip_projection','mouth'),('chin_projection','face')]:
                c=self.r.boundaries['features'][k];value=dep.get('values',{}).get(k)
                reasons=[]
                if not directions_ok:reasons.append('DEPTH_VIEW_DIRECTIONS_INVALID')
                if dep['status']!='measured':reasons.append(dep.get('reason','DEPTH_FAILED'))
                # Keep production/strict policy separate from an explicit exploratory cohort-reference run.
                if not self.exploratory_depth:reasons.append('GEOMETRY_QUALITY_NOT_INDEPENDENTLY_VALIDATED')
                score=None;label=None;label_ko=None
                if value is not None:
                    s=c.get('sigma_repeat_train') or c.get('sigma_repeat_all') or c['sigma_population']
                    z=boundary_scores(value,c['lower'],c['upper'],s);score=z['score']
                    f=self.r.a.SCHEMA['features'][k];idx=['low','medium','high'].index(z['label']);label=f'{k}:{f["classes"][idx]}';label_ko=f['classes_ko'][idx]
                observations.append(Observation(k,region,'depth',label,label_ko,measurement=None if value is None else {'indicator':k,'value':value,'unit':'ratio'},measurement_score=score,quality_passed=dep['status']=='measured' and directions_ok,boundaries_validated=self.exploratory_depth,basis='VGGT 추정 형상 / 외안각 거리; 400명 경험 P25/P75; 반복측정 σ; 탐색적 기준(실측 3D 정답 없음)',hold_reasons=tuple(reasons)))
            items=DecisionPolicy().apply(observations)
            yield {'type':'progress','stage':STAGES[4]}
            t=time.perf_counter();result=build_result(items,self.r.repository);self.timings['literature']=time.perf_counter()-t
            self.timings['analysis_after_validation']=time.perf_counter()-t0
            result['pilot']={'mode':'exploratory_cohort' if self.exploratory_depth else 'strict_unvalidated_depth_held','cnn':'fold0 two architectures, 10 models total','depth':'VGGT-1B; three RGB views','independent_human_labels':0,'new_region_models':'not trained; held','literature_review':'draft','score_meaning':'equal-feature geometric mean of CNN confidence and depth membership; not calibrated correct-answer probability'}
            # Timings are a pilot-only extension; no photos or raw landmarks are returned.
            result['pilot_timings']=self.timings
            yield {'type':'result','data':result}
        finally:
            self.close();observations.clear()
            g=None

def make_app(resources,exploratory_depth=False):
    from fastapi.responses import FileResponse
    app=create_app(lambda:ActualService(resources,exploratory_depth=exploratory_depth))
    @app.get('/pilot')
    def page():return FileResponse(ROOT/'web/pilot.html')
    @app.get('/pilot.js')
    def js():return FileResponse(ROOT/'web/pilot.js')
    @app.get('/pilot.css')
    def css():return FileResponse(ROOT/'web/pilot.css')
    @app.get('/api/v1/service-info')
    def service_info():
        from facework.pilot_contract import SCORING_VERSION, SERVICE_SCORE_MODE
        return {'ready': True, 'scoring_version': SCORING_VERSION, 'score_mode': SERVICE_SCORE_MODE,
                'depth_mode': 'exploratory_cohort' if exploratory_depth else 'strict',
                'max_passages': 5, 'max_per_region': 3}
    reports = {
        'validation.pdf': 'FaceTeller_파일럿_검증보고서.pdf',
        'depth-method.pdf': 'FaceTeller_깊이_3단계분류와점수_설계안.pdf',
        'design-changes.txt': 'FaceTeller_설계서_수정문구.txt',
        'depth-distributions.pdf': 'depth_distributions.pdf',
        'result-example.pdf': 'printpilot-recorded-inference.pdf',
    }
    @app.get('/pilot-reports/{name}')
    def report(name: str):
        from fastapi import HTTPException
        if name not in reports:
            raise HTTPException(status_code=404, detail='Report not found')
        path = ROOT/'outputs/pilot-20261014'/reports[name]
        return FileResponse(path, media_type='application/pdf' if name.endswith('.pdf') else 'text/plain; charset=utf-8')
    return app
