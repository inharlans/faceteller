import io,json,threading,os,time
from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI,UploadFile,File,HTTPException,Request
from fastapi.responses import FileResponse,HTMLResponse
from pydantic import BaseModel,Field
from PIL import Image,ImageOps,UnidentifiedImageError
from .features import ROOT,detector,analyze
from .measurement_rules import observations

app=FastAPI(title='얼굴 형태 연구 작업대',docs_url='/api/docs')
LOCK=threading.Lock();ANALYSIS_LOCK=threading.Lock()
DATA=ROOT/'data/derived';OVERRIDES=ROOT/'data/annotations/overrides.json'
SCHEMA=json.loads((ROOT/'data/label_schema.json').read_text(encoding='utf-8'))
MEASUREMENTS=json.loads((ROOT/'data/measurement_definitions.json').read_text(encoding='utf-8'))
def read(name):return json.loads((DATA/f'{name}.json').read_text(encoding='utf-8'))
def override_data():return json.loads(OVERRIDES.read_text(encoding='utf-8')) if OVERRIDES.exists() else {}

@app.middleware('http')
async def local_mutations(request:Request,call_next):
    if request.method in ['POST','PUT','DELETE','PATCH']:
        origin=request.headers.get('origin','')
        if request.headers.get('x-face-workbench')!='1' or (origin and origin not in ['http://127.0.0.1:8765','http://localhost:8765',f'{request.url.scheme}://{request.url.netloc}']):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':'이 작업대에서 보낸 요청만 허용합니다.'},status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Cache-Control']='no-store'
    return response

@app.get('/')
def home():return HTMLResponse((ROOT/'web/index.html').read_text(encoding='utf-8'))

@app.get('/knowledge-ui.js')
def knowledge_script():return FileResponse(ROOT/'web/knowledge-ui.js',media_type='application/javascript')

@app.get('/api/knowledge')
def knowledge():
    return {'interpretations':json.loads((ROOT/'data/interpretation_catalog_v1.json').read_text(encoding='utf-8')),
            'type_requirements':json.loads((ROOT/'data/type_requirements.json').read_text(encoding='utf-8')),
            'measurements':MEASUREMENTS}

@app.get('/api/knowledge/export/{kind}')
def knowledge_export(kind:str):
    files={'interpretations':'문헌해석_103항목.csv','requirements':'유형별_레이블링_요건.csv'}
    if kind not in files:raise HTTPException(404)
    return FileResponse(ROOT/'outputs/knowledge-v2'/files[kind],filename=files[kind],media_type='text/csv; charset=utf-8')

@app.get('/api/project')
def project():
    report=ROOT/'models/pilot/training_report.json'
    return {'summary':read('summary'),'schema':SCHEMA,'images':read('images'),'literature':read('literature'),'literature_links':json.loads((ROOT/'data/literature_links.json').read_text(encoding='utf-8')),'training':json.loads(report.read_text(encoding='utf-8')) if report.exists() else None}

@app.get('/api/image/{image_id}')
def image_details(image_id:str):
    r=next((r for r in read('images') if r['image_id']==image_id),None)
    if r is None:raise HTTPException(404,'이미지가 없습니다.')
    overrides=override_data()
    rows=[r|{'revision':overrides.get(r['annotation_id'],{}).get('revision',0)} for r in read('annotations') if r['image_id']==image_id]
    landmarks=DATA/'landmarks'/f'{image_id}.json'
    return {'image':r,'annotations':rows,'metrics':json.loads(landmarks.read_text())['metrics'] if landmarks.exists() else {}}

@app.get('/media/{image_id}/{part}')
def media(image_id:str,part:str):
    images={r['image_id']:r for r in read('images')}
    if image_id not in images:raise HTTPException(404)
    if part=='original':path=ROOT/images[image_id]['path']
    elif part=='aligned':path=DATA/'aligned'/f'{image_id}.jpg'
    elif part in [p for t in SCHEMA['tasks'].values() for p in t['parts']]:path=DATA/'crops'/part/f'{image_id}.jpg'
    else:raise HTTPException(404)
    return FileResponse(path,media_type='image/jpeg')

class Review(BaseModel):
    annotation_id:str
    label:str
    notes:str=Field(default='',max_length=2000)
    revision:int=0

@app.post('/api/review')
def review(change:Review):
    with LOCK:
        row=next((r for r in read('annotations') if r['annotation_id']==change.annotation_id),None)
        if row is None:raise HTTPException(404)
        if change.label not in ['unknown',*SCHEMA['tasks'][row['task']]['classes']]:raise HTTPException(422,'허용되지 않은 레이블')
        values=override_data();prior=values.get(change.annotation_id,{});revision=prior.get('revision',0)
        if change.revision!=revision:raise HTTPException(409,'다른 수정이 먼저 저장되었습니다. 새로고침 후 확인하세요.')
        values[change.annotation_id]={'label':change.label,'notes':change.notes,'revision':revision+1,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        temp=OVERRIDES.with_suffix('.tmp');temp.write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,OVERRIDES)
        audit=ROOT/'data/annotations/review_history.jsonl'
        with audit.open('a',encoding='utf-8') as f:f.write(json.dumps({'annotation_id':change.annotation_id,'before':prior,'after':values[change.annotation_id]},ensure_ascii=False)+'\n')
        from scripts.finalize_labels import main
        main()
        return {'saved':True,'revision':revision+1,'training_checkpoints_stale':True}

@app.get('/api/export/{kind}')
def export(kind:str):
    if kind not in ['annotations','training_manifest','literature','measurements','images']:raise HTTPException(404)
    return FileResponse(DATA/f'{kind}.csv',filename=f'{kind}.csv',media_type='text/csv; charset=utf-8')

@lru_cache(maxsize=1)
def get_detector():return detector()

@lru_cache(maxsize=1)
def predictors():
    import torch
    from .training import model_for,preprocess
    torch.set_num_threads(4)
    loaded={}
    for task in SCHEMA['tasks']:
        path=ROOT/f'models/pilot/{task}.pt'
        if path.exists():
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            m=model_for(checkpoint['classes'],pretrained=False);m.load_state_dict(checkpoint['state_dict']);m.eval()
            loaded[task]=(m,checkpoint['classes'])
    return loaded,preprocess()

@app.post('/api/analyze')
async def analyze_upload(file:UploadFile=File(...)):
    body=await file.read(10*1024*1024+1)
    if len(body)>10*1024*1024:raise HTTPException(413,'10MB 이하 이미지를 사용하세요.')
    try:
        im=Image.open(io.BytesIO(body))
        if im.width*im.height>16_000_000:raise HTTPException(413,'1600만 화소 이하 이미지를 사용하세요.')
        im=ImageOps.exif_transpose(im).convert('RGB');im.load()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError):raise HTTPException(422,'이미지 파일을 읽을 수 없습니다.')
    with ANALYSIS_LOCK:
        result=analyze(im,get_detector())
        if result['status']!='ok':return result
        metrics=result['metrics'];flags=[]
        if abs(metrics.get('mesh_yaw_deg',0))>15 or abs(metrics.get('mesh_pitch_deg',0))>20:flags.append('정면에서 다시 촬영해 주세요.')
        if metrics['face_width_px']<150:flags.append('얼굴을 더 크게 촬영해 주세요.')
        if metrics['smile_score']>.45 or metrics['mouth_open_ratio']>.08:flags.append('입을 편하게 다물고 무표정으로 촬영해 주세요.')
        import torch,hashlib
        models,transform=predictors();predictions=[]
        for task,(model,classes) in models.items():
            for part in SCHEMA['tasks'][task]['parts']:
                crop=result['crops'][part]
                if part.endswith('_left'):crop=ImageOps.mirror(crop)
                with torch.inference_mode():prob=model(transform(crop).unsqueeze(0)).softmax(1)[0]
                value,index=prob.max(0);candidate=classes[int(index)]
                predictions.append({'part':part,'task':task,'label':candidate if float(value)>=.65 and not flags else 'uncertain','candidate':candidate,'candidate_ko':SCHEMA['tasks'][task]['classes'][candidate],'max_softmax':round(float(value),4),'description':SCHEMA['tasks'][task]['description']})
        report_path=ROOT/'models/pilot/training_report.json'
        report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        current_hash=hashlib.sha256((DATA/'training_manifest.csv').read_bytes()).hexdigest()
        return {'status':'ok','quality_flags':flags,'quality_policy_status':'pilot_heuristics_not_validated','metrics':metrics,'measurements':observations(metrics,MEASUREMENTS,flags),'predictions':predictions,'model_status':'pilot','checkpoints_stale':report.get('training_manifest_sha256')!=current_hash,'notice':'외형 분류 실험 결과입니다. 점수는 정답 확률로 검증되지 않았습니다. 크기 구간 기준은 미확정입니다. 원본 업로드는 디스크에 저장하지 않습니다.'}

# ---- v2: K-FACE part models, literature type matching and readings ----
async def _read_upload(file):
    body = await file.read(10 * 1024 * 1024 + 1)
    if len(body) > 10 * 1024 * 1024: raise HTTPException(413, '10MB 이하 이미지를 사용하세요.')
    try:
        im = Image.open(io.BytesIO(body))
        if im.width * im.height > 16_000_000: raise HTTPException(413, '1600만 화소 이하 이미지를 사용하세요.')
        im = ImageOps.exif_transpose(im).convert('RGB'); im.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError): raise HTTPException(422, '이미지 파일을 읽을 수 없습니다.')
    return im

@app.get('/v2')
def v2_page(): return HTMLResponse((ROOT / 'web/v2.html').read_text(encoding='utf-8'))

@app.get('/api/v2/status')
def v2_status():
    base = ROOT / 'models/kface_v2'; derived = ROOT / 'data/derived/kface'
    ens = ROOT / 'models/kface_v4'; evdir = ens if (ens / 'evaluation.json').exists() else base
    load = lambda p: json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    summary = load(derived / 'reference_summary.json')
    return {'models': sorted(p.stem for p in base.glob('*.pt')), 'evaluation': load(evdir / 'evaluation.json'),
            'ensemble': str(evdir.relative_to(ROOT)), 'rater_agreement': load(ROOT / 'data/annotations/kface_rater_agreement.json'),
            'fewview_depth': load(base / 'fewview_depth.json'), 'depth_reliability': (load(derived / 'depth.json') or {}).get('summary'),
            'cohort': None if summary is None else {'participants': summary['participants'], 'split': summary['split'],
                      'features': {k: {kk: v[kk] for kk in ('cuts', 'icc_1_1_train', 'median_within_subject_sd', 'between_subject_sd')} for k, v in summary['features'].items()}},
            'schema': json.loads((ROOT / 'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))}

@app.get('/api/v2/types')
def v2_types():
    service = json.loads((ROOT / 'data/interpretation_service_v1.json').read_text(encoding='utf-8'))
    rules = {r['record_id']: r for r in json.loads((ROOT / 'data/type_rules_v3.json').read_text(encoding='utf-8'))['records']}
    catalog = {r['record_id']: r for r in json.loads((ROOT / 'data/interpretation_catalog_v1.json').read_text(encoding='utf-8'))['records']}
    return {'tone_policy': service['tone_policy'], 'records': [r | {'rule': rules.get(r['record_id']), 'original': catalog[r['record_id']]['original_section'], 'source_url': catalog[r['record_id']]['source_url']} for r in service['records']]}

@app.post('/api/v2/analyze')
async def v2_analyze(front: UploadFile = File(...), left: UploadFile | None = File(None), right: UploadFile | None = File(None)):
    from .analysis_v2 import analyze as analyze_v2
    image = await _read_upload(front)
    sides = [await _read_upload(f) for f in (left, right) if f is not None and f.filename]
    with ANALYSIS_LOCK:
        return analyze_v2(image, sides)

# ---- v2 blind review of AI visual labels (local server only; K-FACE crops never leave this PC) ----
REVIEW_FORM = ROOT / 'outputs/kface-v2/독립검수_시각레이블_양식.csv'
REVIEW_ANSWERS = ROOT / 'outputs/kface-v2/독립검수_응답.csv'
REVIEW_TASKS = {'brow_shape': ['straight', 'arched', 'angled', 'unknown'], 'brow_thickness': ['thin', 'medium', 'thick', 'unknown'],
                'face_shape': ['oval', 'round', 'angular', 'long', 'unknown']}  # web /v2/review: quick 3-item form; full 16-item form is tmp/kface_review/kface_labeling_v3_*.html

@app.get('/v2/review')
def v2_review_page(): return HTMLResponse((ROOT / 'web/v2_review.html').read_text(encoding='utf-8'))

@app.get('/api/v2/review/items')
def v2_review_items():
    import csv as _csv
    if not REVIEW_FORM.exists(): raise HTTPException(404, 'scripts/kface_agreement.py form 으로 양식을 먼저 만드세요.')
    ids = [r['subject_id'] for r in _csv.DictReader(REVIEW_FORM.open(encoding='utf-8-sig'))]
    done = {}
    if REVIEW_ANSWERS.exists():
        for r in _csv.DictReader(REVIEW_ANSWERS.open(encoding='utf-8-sig')): done.setdefault(r['reviewer'], []).append(r['subject_id'])
    return {'subjects': ids, 'tasks': REVIEW_TASKS, 'done': done}

@app.get('/api/v2/review/crop/{subject}/{part}')
def v2_review_crop(subject: str, part: str):
    import numpy as _np
    from fastapi.responses import Response
    d_path = ROOT / 'data/derived/kface/subjects' / f'{subject}.npz'
    if not subject.isdigit() or part not in ('brow_right', 'brow_left', 'face') or not d_path.exists(): raise HTTPException(404)
    d = _np.load(d_path); names = list(d['parts']); keys = list(d['keys'])
    for light in ('L1', 'L2', 'L8', 'L12'):
        i = keys.index(f'S001|{light}|E01|C7|core')
        if d['status'][i] == 1:
            o, n = d['crop_offsets'][i, names.index(part)]
            with (ROOT / 'data/derived/kface/subjects' / f'{subject}.bin').open('rb') as f:
                f.seek(int(o)); return Response(f.read(int(n)), media_type='image/jpeg')
    raise HTTPException(404)

class BlindReview(BaseModel):
    subject_id: str
    reviewer: str = Field(min_length=1, max_length=40)
    brow_shape: str
    brow_thickness: str
    face_shape: str
    notes: str = Field(default='', max_length=500)

@app.post('/api/v2/review')
def v2_review_save(item: BlindReview):
    import csv as _csv
    for t, allowed in REVIEW_TASKS.items():
        if getattr(item, t) not in allowed: raise HTTPException(422, f'{t}: 허용되지 않은 값')
    if item.subject_id not in v2_review_items()['subjects']: raise HTTPException(404)
    with LOCK:
        new = not REVIEW_ANSWERS.exists()
        with REVIEW_ANSWERS.open('a', newline='', encoding='utf-8-sig') as f:
            w = _csv.DictWriter(f, fieldnames=['subject_id', 'reviewer', *REVIEW_TASKS, 'notes', 'saved_at'])
            if new: w.writeheader()
            w.writerow(item.model_dump() | {'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    return {'saved': True}
