"""Local-only HTTP verification using original K-FACE C7/C4/C10 uploads.

Run only after the main agent releases the GPU and starts the real service.
The selection oracle applies the pinned semantic exclusions (ReadingCatalog)
before the independent AND/G/ranking check, compares original text/source with
PostgreSQL and editorial text/readings/named types with the pinned files, and
records scalar-only face-overlay QA (no coordinates or images).
No provider annotations, face framing or part crops are used by this client.
Only aggregate/scalar measurements are written; request images stay in memory.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from PIL import Image,ImageOps
import httpx
from facework.pilot_contract import (DIRECTIONS,InputLimits,Decision,PassageSelector,
                                    SCORING_VERSION,SERVICE_SCORE_MODE)
from facework.release_repository import ReleaseRepository


def local_url(value):
    parsed=urlsplit(value)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query or parsed.fragment:
        raise ValueError('Verification only sends images to an explicit loopback HTTP origin')
    return value.rstrip('/')


def upload_bytes(raw,max_side=None):
    """Original encoded JPEG/PNG, or whole-image isotropic resize; never a crop."""
    if max_side is None:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ('PNG','JPEG'): raise ValueError('Original upload must be JPEG or PNG')
            mime='image/png' if image.format=='PNG' else 'image/jpeg'
        return raw,mime
    with Image.open(io.BytesIO(raw)) as image:
        with ImageOps.exif_transpose(image) as oriented:
            with oriented.convert('RGB') as rgb:
                rgb.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
                stream=io.BytesIO(); rgb.save(stream,format='PNG')
                return stream.getvalue(),'image/png'


def image_files(store,subject,max_side):
    from facework.kface import member
    result=[]
    for direction,camera in zip(DIRECTIONS,('C7','C4','C10')):
        raw,mime=upload_bytes(store.read(subject,member('S001','L1','E01',camera)),max_side)
        result.append((direction,(direction+('.png' if mime=='image/png' else '.jpg'),raw,mime)))
    return result


def request_analysis(client,files):
    start=time.perf_counter(); stages=[]; terminal=None; code=None
    with client.stream('POST','/api/v1/analysis',data={'input_mode':'upload'},files=files) as response:
        code=response.status_code
        for line in response.iter_lines():
            if not line: continue
            event=json.loads(line)
            if terminal is not None: raise RuntimeError('Event after terminal result')
            if event.get('type')=='progress': stages.append(time.perf_counter()-start)
            elif event.get('type') in ('result','error'): terminal=event
            else: raise RuntimeError('Unknown NDJSON event')
    if terminal is None: raise RuntimeError('Stream ended without result/error')
    return code,terminal,time.perf_counter()-start,stages


def verify_selection(items,candidates,passages):
    """Independent AND/G/ranking oracle plus a production-selector cross-check.

    Inputs are response scalars and the complete DB candidate set at its pinned
    revision. This checks retrieval arithmetic, not model accuracy or human truth.
    """
    def require(ok,message):
        if not ok: raise RuntimeError('Selection verification: '+message)
    def score(value):
        return type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
    confirmed={}; seen_ids=set(); seen_features=set()
    for item in items:
        require(item['item_id'] not in seen_ids,'duplicate item_id')
        seen_ids.add(item['item_id'])
        require(item['status'] in ('CONFIRMED','HELD'),'unknown item status')
        if item['status']!='CONFIRMED': continue
        label=item['label_id']
        require(isinstance(label,str) and label.count(':')==1,'invalid confirmed label')
        feature=label.split(':')[0]
        require(feature==item['item_id'] and feature not in seen_features,'ambiguous confirmed feature')
        require(label not in confirmed and not item['hold_reasons'],'ambiguous or held evidence')
        require(item['kind'] in ('classification','depth'),'unknown score kind')
        value=item['confidence'] if item['kind']=='classification' else item['measurement_score']
        require(score(value),'invalid confirmed feature score')
        confirmed[label]=item;seen_features.add(feature)
    ranked=[];by_id={}
    for candidate in candidates:
        pid=str(candidate.passage_id)
        require(pid not in by_id,'duplicate DB passage ID')
        require(candidate.required_labels and candidate.required_labels<=confirmed.keys(),
                'DB candidate has unconfirmed required condition')
        evidence=[confirmed[label] for label in sorted(candidate.required_labels)]
        components=[]
        for item in evidence:
            cnn=item['kind']=='classification'
            components.append({'item_id':item['item_id'],'label_id':item['label_id'],
                               'label_name':item['label_name'],'kind':item['kind'],
                               'score':item['confidence'] if cnn else item['measurement_score'],
                               'score_kind':'calibrated_classification_confidence' if cnn
                               else 'geometry_membership_not_calibrated_accuracy'})
        values=[c['score'] for c in components]
        # Log-domain computation also handles tiny positive scores without a
        # product underflow. An exact zero forces an exact zero, never epsilon.
        g=0.0 if any(v==0 for v in values) else math.exp(math.fsum(map(math.log,values))/len(values))
        row=(candidate,evidence,components,g);by_id[pid]=row;ranked.append(row)
    def id_key(pid):
        text=str(pid)
        return (0,int(text)) if text.isdecimal() else (1,text)
    ranked.sort(key=lambda r:(-r[3],r[0].literature_priority,r[0].passage_order,id_key(r[0].passage_id)))
    counts=Counter();expected=[]
    for candidate,evidence,components,g in ranked:
        regions={item['region'] for item in evidence}
        if any(counts[region]>=3 for region in regions): continue
        expected.append(str(candidate.passage_id));counts.update(regions)
        if len(expected)==5: break
    actual=[str(p['passage_id']) for p in passages]
    require(actual==expected,'selected IDs/order differ from full DB candidates (rank/tie/caps)')
    require(len(actual)<=5 and len(actual)==len(set(actual)),'global cap or duplicate selection')
    actual_regions=Counter();component_count=0
    for rank,passage in enumerate(passages,1):
        candidate,evidence,components,g=by_id[str(passage['passage_id'])]
        require(type(passage['rank']) is int and passage['rank']==rank,'incorrect rank')
        require(passage['evidence_item_ids']==[i['item_id'] for i in evidence],'evidence does not exactly match all required labels')
        require(type(passage['score_feature_count']) is int and passage['score_feature_count']==len(components),
                'score component count differs from required feature count')
        supplied=passage['score_components']
        require(isinstance(supplied,list) and all(isinstance(c,dict) and score(c.get('score')) for c in supplied)
                and supplied==components,'score components differ from CNN confidence/depth membership')
        require(score(passage['G']) and math.isclose(passage['G'],g,rel_tol=1e-12,abs_tol=0),
                'incorrect geometric G (including log/zero rule)')
        require(passage['score_mode']==SERVICE_SCORE_MODE and passage['score_kind']=='geometric_feature_match',
                'incorrect passage score semantics')
        actual_regions.update({i['region'] for i in evidence});component_count+=len(components)
    require(all(n<=3 for n in actual_regions.values()),'region cap exceeded')
    decisions=[Decision(**{k:item[k] for k in Decision.__dataclass_fields__}) for item in items]
    production=PassageSelector(SERVICE_SCORE_MODE).select(decisions,candidates)
    require([str(p['passage_id']) for p in production]==expected,'production selector disagrees with independent ranking')
    for expected_row in production:
        require(math.isclose(expected_row['G'],by_id[str(expected_row['passage_id'])][3],rel_tol=1e-12,abs_tol=0),
                'production selector disagrees with independent G')
    return {'score_formula_verified':True,'score_formula_verified_count':len(passages),
            'score_components_verified_count':component_count,'selection_verified':True}


READING_FIELDS=('title','meaning','valence','valence_label','topics')
EDITORIAL_SOURCE_KEYS=('editorial_version','editorial_status')
OVERLAY_ITEMS=('eye_size','eye_roundness','eye_tilt','eye_asymmetry','brow_length','brow_tail','brow_eye_gap',
               'brow_gap','nose_width','nose_length','mouth_width','lip_thickness','lip_balance','mouth_corner',
               'face_length','jaw_width','eye_spacing','philtrum_length','lower_face_ratio')
SEGMENTATION_ITEMS=('brow_length','brow_tail')


def verify_reading_identity(runtime,readings):
    """Pinned editorial/semantic files must be exactly the ones the service loaded."""
    identity=getattr(readings,'identity',None)
    if not isinstance(identity,dict):raise RuntimeError('Editorial verification: reading catalog identity missing')
    for key in ('readings_sha256','semantic_review_sha256','readings_version','semantic_review_version'):
        if not identity.get(key) or runtime.get(key)!=identity[key]:
            raise RuntimeError('Editorial verification: service reading identity differs from pinned files: '+key)
    eligible=sum(bool(p['service_eligible']) for p in readings.passages.values())
    if 'eligible_passages' in runtime and runtime['eligible_passages']!=eligible:
        raise RuntimeError('Editorial verification: eligible passage count differs from pinned corpus')
    if 'active_mapping_count' in runtime and runtime['active_mapping_count']!=eligible-len(readings.excluded):
        raise RuntimeError('Editorial verification: active mapping count differs from semantic review')


def expected_named_types(items,readings):
    """Independent recomputation from the pinned reading document."""
    confirmed={i['label_id']:i for i in items if i['status']=='CONFIRMED'}
    expected=[]
    for kind in readings.document.get('named_types',[]):
        if not set(kind['required_labels'])<=confirmed.keys():continue
        if any(p in readings.excluded for p in kind['source_passage_ids']):continue
        value={k:v for k,v in kind.items() if k!='required_labels'}
        value['evidence_item_ids']=[confirmed[k]['item_id'] for k in kind['required_labels']]
        expected.append(value)
    return expected


def verify_editorial(data,candidates,readings):
    """Editorial text/reading must match pinned files; original text/source must match PostgreSQL."""
    def require(ok,message):
        if not ok:raise RuntimeError('Editorial verification: '+message)
    document=readings.document
    require(data['ranking_policy'].get('semantic_review_version')==readings.policy['version'],'semantic review version missing from ranking policy')
    for passage in data['passages']:
        pid=str(passage['passage_id']);original=candidates[pid];archived=readings.passages.get(pid)
        require(pid not in readings.excluded,'semantically excluded passage was returned')
        require(archived is not None and archived['service_eligible'],'returned passage is not an archived eligible passage')
        require(passage['original_text']==original.original_text,'original text differs from PostgreSQL')
        require(archived['original_text']==original.original_text,'PostgreSQL original text differs from the reading-bound archived corpus')
        source=dict(passage['source'])
        require(source.get('editorial_version')==document['version'] and source.get('editorial_status')==document['authoring'],
                'editorial provenance differs from pinned reading document')
        for key in EDITORIAL_SOURCE_KEYS:source.pop(key,None)
        require(source==original.source,'source differs from PostgreSQL')
        reading=document['readings'].get(pid)
        require(reading is not None,'no pinned reading for returned passage')
        require(passage['translation']==reading['translation'],'translation differs from pinned reading')
        require(passage['commentary']==reading['meaning'],'commentary differs from pinned reading')
        require(passage.get('reading')=={k:reading[k] for k in READING_FIELDS},'reading differs from pinned reading')
    expected=expected_named_types(data['items'],readings)
    require(data.get('named_types')==expected,'named types differ from pinned definitions')
    for kind in expected:
        if 'original_text' in kind:
            texts=[readings.passages[pid]['original_text'] for pid in kind['source_passage_ids']]
            require(kind['original_text'] in texts,'named type original text differs from its archived source')
    return {'editorial_verified':True,'editorial_verified_passages':len(data['passages']),'named_type_count':len(expected),
            'named_type_ids':[k['id'] for k in expected]}


def overlay_summary(overlay):
    """Scalar QA of measurement paths: counts, finiteness and image bounds only.

    No coordinates, masks or photos are returned or saved. Passing checks that the
    drawn geometry is well formed in original pixels, not that it is anatomically
    correct or agrees with any human annotation.
    """
    def require(ok,message):
        if not ok:raise RuntimeError('Overlay verification: '+message)
    require(isinstance(overlay,dict),'face_overlay missing')
    width,height=overlay.get('width'),overlay.get('height')
    require(type(width) is int and type(height) is int and width>0 and height>0,'invalid image size')
    features=overlay.get('features');require(isinstance(features,list),'features missing')
    if not features:
        # face_overlay deliberately returns no paths for degenerate/invalid landmarks.
        # Recorded as not verified (never counted as a pass), not as an exception.
        return {'overlay_verified':False,'overlay_feature_count':0,'overlay_path_count':0,'overlay_point_count':0,
                'overlay_sources':{},'overlay_missing_items':list(OVERLAY_ITEMS),'overlay_reason':'OVERLAY_EMPTY',
                'overlay_meaning':'no measurement paths returned; not verified'}
    ids=[f.get('item_id') for f in features if isinstance(f,dict)]
    require(len(ids)==len(features) and len(set(ids))==len(ids) and set(ids)<=set(OVERLAY_ITEMS),'unknown or duplicate overlay item')
    missing=set(OVERLAY_ITEMS)-set(ids)
    require(missing<=set(SEGMENTATION_ITEMS) and 17<=len(ids)<=19,'overlay feature count outside 17-19 or non-brow item missing')
    sources=Counter();paths=points=0
    for feature in features:
        source=feature.get('source');sources[source]+=1
        require(source=='landmark' or (source=='segmentation' and feature['item_id'] in SEGMENTATION_ITEMS),'invalid path source type')
        segments=feature.get('paths')
        require(isinstance(segments,list) and segments,'empty path list')
        for segment in segments:
            require(isinstance(segment,list) and len(segment)>=2,'path needs at least two points')
            for point in segment:
                require(isinstance(point,list) and len(point)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in point),
                        'non-finite or malformed coordinate')
                require(0<=point[0]<=width and 0<=point[1]<=height,'coordinate outside original image')
                points+=1
            paths+=1
        if 'value' in feature:
            require(type(feature['value']) in (int,float) and math.isfinite(feature['value']) and isinstance(feature.get('unit'),str),'invalid overlay value')
    return {'overlay_verified':True,'overlay_feature_count':len(features),'overlay_path_count':paths,'overlay_point_count':points,
            'overlay_sources':dict(sorted(sources.items())),'overlay_missing_items':sorted(missing),
            'overlay_meaning':'well-formed finite original-pixel paths inside the image; not anatomical or human-verified accuracy'}


def summarize_result(event,repository,readings=None):
    """Verify one service result. ``readings`` (ReadingCatalog) enables the current
    editorial/semantic oracle; None keeps the legacy direct-PostgreSQL text check
    and refuses responses that claim an editorial layer."""
    if event['type']=='error':
        return {'status':'error','error_code':event['code'],'features':0,'confirmed':0,'held':0,
                'items':[],'passages':[],'passage_count':0,'timings':{},'regions':{},
                'corpus_sha256':None,'db_source_verified':False,'score_formula_verified':False,
                'score_formula_verified_count':0,'score_components_verified_count':0,'selection_verified':False,
                **({'editorial_verified':False} if readings is not None else {})}
    data=event['data']; items=data['items']; labels={i['label_id'] for i in items if i['status']=='CONFIRMED'}
    runtime=data.get('runtime',{})
    if readings is None and (runtime.get('readings_sha256') or 'named_types' in data
                             or 'semantic_review_version' in data.get('ranking_policy',{})):
        raise RuntimeError('Editorial verification: service applies readings but no pinned reading catalog was supplied')
    if readings is not None: verify_reading_identity(runtime,readings)
    with repository.snapshot() as snapshot:
        revision=snapshot.corpus_revision()
        if revision is None or runtime.get('corpus_sha256')!=revision:
            raise RuntimeError('PostgreSQL corpus revision differs from the service result')
        all_candidates=snapshot.load_candidates(labels) if labels else []
        candidates={str(p.passage_id):p for p in all_candidates}
    policy=data['ranking_policy']
    if (policy.get('version')!=SCORING_VERSION or policy.get('score_mode')!=SERVICE_SCORE_MODE
            or policy.get('formula')!='exp(sum(log(s_i))/n)' or policy.get('weighting')!='equal_per_required_feature'
            or policy.get('required_conditions')!='all_confirmed' or policy.get('max_passages')!=5
            or policy.get('max_per_region')!=3 or policy.get('is_correctness_probability') is not False
            or policy.get('tie_break')!=['literature_priority','passage_order','passage_id']):
        raise RuntimeError('Selection verification: incorrect ranking policy')
    # Semantic exclusions apply BEFORE the independent AND/G/ranking oracle and caps.
    eligible=[p for p in all_candidates if readings is None or str(p.passage_id) not in readings.excluded]
    selection=verify_selection(items,eligible,data['passages'])
    for passage in data['passages']:
        if candidates.get(str(passage['passage_id'])) is None:
            raise RuntimeError('Returned passage is absent from eligible AND-matched PostgreSQL candidates')
    if readings is None:
        for passage in data['passages']:
            original=candidates[str(passage['passage_id'])]
            for field in ('original_text','translation','commentary','source'):
                if passage[field]!=getattr(original,field): raise RuntimeError('Returned source/text differs from PostgreSQL')
        editorial={}
    else:
        editorial={**verify_editorial(data,candidates,readings),'semantic_excluded_candidates':len(all_candidates)-len(eligible),
                   'readings_sha256':readings.identity['readings_sha256'],'semantic_review_sha256':readings.identity['semantic_review_sha256']}
    if readings is not None and 'face_overlay' not in data:
        raise RuntimeError('Overlay verification: current release omitted face_overlay')
    overlay=overlay_summary(data['face_overlay']) if 'face_overlay' in data else {}
    regions={r:{'confirmed':0,'held':0} for r in sorted({i['region'] for i in items})}
    for item in items: regions[item['region']]['confirmed' if item['status']=='CONFIRMED' else 'held']+=1
    compact_items=[{key:item.get(key) for key in ('item_id','region','kind','status','confidence',
                   'measurement_score','label_id','hold_reasons')} for item in items]
    compact_passages=[{'passage_id':p['passage_id'],'G':p['G'],'rank':p['rank'],
                       'score_feature_count':p['score_feature_count'],
                       'evidence_item_ids':p['evidence_item_ids']} for p in data['passages']]
    return {'status':data['status'],'features':len(items),
            'confirmed':sum(i['status']=='CONFIRMED' for i in items),'held':sum(i['status']=='HELD' for i in items),
            'items':compact_items,'passages':compact_passages,'passage_count':len(data['passages']),
            'db_source_verified':True,'db_verified_passages':len(data['passages']),**selection,**editorial,**overlay,
            'corpus_sha256':revision,
            'db_candidates':len(candidates),'regions':regions,'timings':runtime.get('timings',{}),
            'deployment_sha256':runtime.get('deployment_sha256'),'depth_config_sha256':runtime.get('depth_config_sha256')}


def verify_identity(summary,health):
    if summary['status']=='error': return
    for key in ('deployment_sha256','depth_config_sha256','corpus_sha256'):
        if summary.get(key)!=health.get(key):
            raise RuntimeError('Service deployment changed during verification')


def error_checks(client):
    with Image.new('RGB',(128,128),'black') as image:
        stream=io.BytesIO(); image.save(stream,format='PNG'); black=stream.getvalue()
    files=[(d,(d+'.png',black,'image/png')) for d in DIRECTIONS]
    cases=[('missing_view',files[:2],400),
           ('corrupt_image',[(d,(d+'.png',b'not an image','image/png')) for d in DIRECTIONS],415),
           ('mime_mismatch',[(d,(d+'.jpg',black,'image/jpeg')) for d in DIRECTIONS],415),
           ('no_face',files,422),
           ('file_too_large',[(DIRECTIONS[0],('oversize.png',b'0'*(InputLimits().file_bytes+1),'image/png'))]+files[1:],413)]
    result=[]
    for name,uploads,expected in cases:
        status,event,seconds,_=request_analysis(client,uploads)
        passed=status==expected and event['type']=='error'
        result.append({'case':name,'http_status':status,'error_code':event.get('code'),
                       'seconds':seconds,'passed':passed})
    return result


def aggregate(rows):
    result={}
    for split in sorted({row['split'] for row in rows}):
        group=[row for row in rows if row['split']==split]
        result[split]={'participants':len(group),'errors':sum(row['status']=='error' for row in group),
                      'mean_seconds':statistics.mean(row['seconds'] for row in group),
                      'mean_confirmed':statistics.mean(row['confirmed'] for row in group),
                      'mean_held':statistics.mean(row['held'] for row in group),
                      'db_verified_passages':sum(row.get('db_verified_passages',0) for row in group),
                      'score_formula_verified':all(row.get('score_formula_verified',False) for row in group),
                      'score_formula_verified_count':sum(row.get('score_formula_verified_count',0) for row in group),
                      'score_components_verified_count':sum(row.get('score_components_verified_count',0) for row in group),
                      'selection_verified_participants':sum(row.get('selection_verified',False) for row in group),
                      'editorial_verified_participants':sum(row.get('editorial_verified',False) for row in group),
                      'semantic_excluded_candidates':sum(row.get('semantic_excluded_candidates',0) for row in group),
                      'named_type_count':sum(row.get('named_type_count',0) for row in group),
                      'overlay_verified_participants':sum(row.get('overlay_verified',False) for row in group),
                      'overlay_feature_counts':dict(sorted(Counter(row['overlay_feature_count'] for row in group if 'overlay_feature_count' in row).items())),
                      'overlay_segmentation_participants':sum(row.get('overlay_sources',{}).get('segmentation',0)>0 for row in group)}
    return result


def save_report(path,report):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers/antivirus can briefly deny replacement of an open target.
    # Keep the complete temp artifact on failure, never truncate the old report.
    for attempt in range(40):
        try:
            temp.replace(path);break
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',type=local_url,default='http://127.0.0.1:8767')
    parser.add_argument('--max-side',type=int,choices=[1280],default=None)
    parser.add_argument('--limit',type=int,default=None,help='Explicit subset; default verifies all400')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/release/service-verification.json')
    parser.add_argument('--resume',action='store_true',help='Continue a verified incomplete prefix with identical runtime/input identity')
    args=parser.parse_args()
    from facework.kface import Store
    registry=json.loads((ROOT/'data/release/training_registry.json').read_text(encoding='utf-8'))
    subjects=sorted(registry['subjects'])
    if len(subjects)!=400: raise RuntimeError('Expected the fixed400 participant registry')
    if args.limit is not None and not 1<=args.limit<=400: parser.error('--limit must be 1..400')
    if args.output.exists() and not args.resume: raise RuntimeError('Verification output exists; choose a new output to preserve provenance')
    if args.resume and not args.output.exists():raise RuntimeError('Resume output is missing')
    repository=ReleaseRepository(); store=Store(2)
    from facework.release_readings import ReadingCatalog
    readings=ReadingCatalog.load(ROOT)  # pinned editorial/semantic files; compared with service identity below
    report={'schema_version':'release-service-verification-v2',
            'input':'original S001/L1/E01 C7,C4,C10; no provider annotations or guide crops',
            'max_side':args.max_side,'independent_human_labels':0,'warmup':[],'errors_tested':[],
            'participants':[],'complete':False,'all400':False,
            'score_formula_verified':False,'score_formula_verified_count':0,
            'score_components_verified_count':0,'selection_verified_participants':0,
            'model_metrics_path':('outputs/release/reviewed-test-metrics-20261015.json'
                                  if registry.get('reviewed_candidate_version') else 'outputs/release/test_metrics.json'),
            'readings_identity':dict(readings.identity),'semantic_excluded_passages':len(readings.excluded),
            'overlay_qa':'scalar counts only (finite original-pixel paths inside image); no photos or coordinates saved; not accuracy',
            'license_note':'Original VGGT research weights are noncommercial; not commercial-ready.'}
    if args.resume:
        previous=json.loads(args.output.read_text(encoding='utf8'))
        if (previous.get('complete') is not False or previous.get('schema_version')!=report['schema_version']
                or previous.get('input')!=report['input'] or previous.get('max_side')!=args.max_side
                or previous.get('independent_human_labels')!=0 or previous.get('readings_identity')!=report['readings_identity']):
            raise RuntimeError('Resume input protocol or completion state mismatch')
        rows=previous['participants'];expected=subjects[:args.limit]
        if [row['subject_id'] for row in rows]!=expected[:len(rows)] or len(rows)>=len(expected):
            raise RuntimeError('Resume participants are not a strict prefix of requested cohort')
        for row in rows:
            if row['split']!=registry['subjects'][row['subject_id']]['split']:
                raise RuntimeError('Resume split changed')
            verify_identity(row,previous['service'])
        report=previous
    try:
        with httpx.Client(base_url=args.url,timeout=600,trust_env=False,follow_redirects=False) as client:
            health=client.get('/api/v1/health'); health.raise_for_status()
            if args.resume and report['service']!=health.json():raise RuntimeError('Resume runtime identity changed')
            report['service']=health.json()
            if args.resume:report.setdefault('resume_events',[]).append({'completed_before_resume':len(report['participants']),
                'reason':'continue preserved real observations; fresh server warmup and error checks recorded separately'})
            if not report['service'].get('ready') or report['service'].get('repository')!='postgresql_readonly':
                raise RuntimeError('Real PostgreSQL release service is not ready')
            report['corpus_sha256']=report['service'].get('corpus_sha256')
            verify_reading_identity(report['service'],readings)
            if not report['corpus_sha256'] or repository.corpus_revision()!=report['corpus_sha256']:
                raise RuntimeError('Active PostgreSQL corpus revision does not match service readiness')
            # One actual warmup, also exercise bounded admission without a second inference.
            uploads=image_files(store,subjects[0],args.max_side)
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending=executor.submit(request_analysis,client,uploads)
                busy=False; start=time.perf_counter()
                while not pending.done() and time.perf_counter()-start<30:
                    time.sleep(.05)
                    probe=client.post('/api/v1/analysis',content=b'bad',headers={'Content-Type':'text/plain'})
                    if probe.status_code==503:
                        busy=True; break
                status,event,seconds,stages=pending.result()
            uploads.clear()
            summary=summarize_result(event,repository,readings);verify_identity(summary,report['service'])
            report['warmup'].append({'subject_id':subjects[0],'split':registry['subjects'][subjects[0]]['split'],
                                    'http_status':status,'seconds':seconds,'progress_event_seconds':stages,
                                    **summary})
            report['errors_tested'].append({'case':'busy_admission','http_status':503 if busy else None,'passed':busy})
            report['errors_tested'].extend(error_checks(client))
            save_report(args.output,report)
            for subject in subjects[:args.limit][len(report['participants']):]:
                uploads=image_files(store,subject,args.max_side)
                try:
                    status,event,seconds,stages=request_analysis(client,uploads)
                    summary=summarize_result(event,repository,readings);verify_identity(summary,report['service'])
                    row={'subject_id':subject,'split':registry['subjects'][subject]['split'],
                         'http_status':status,'seconds':seconds,'progress_event_seconds':stages,
                         **summary}
                finally: uploads.clear()
                report['participants'].append(row)
                report['split_summary']=aggregate(report['participants'])
                report['score_formula_verified']=all(r['score_formula_verified'] for r in report['participants'])
                for key in ('score_formula_verified_count','score_components_verified_count'):
                    report[key]=sum(r[key] for r in report['participants'])
                report['selection_verified_participants']=sum(r['selection_verified'] for r in report['participants'])
                report['editorial_verified_participants']=sum(r.get('editorial_verified',False) for r in report['participants'])
                report['overlay_verified_participants']=sum(r.get('overlay_verified',False) for r in report['participants'])
                save_report(args.output,report)
                print(subject,row['split'],row['status'],round(seconds,3),flush=True)
            report['complete']=True; report['all400']=len(report['participants'])==400
            report['error_checks_passed']=all(row['passed'] for row in report['errors_tested'])
            save_report(args.output,report)
            if not report['error_checks_passed']: raise RuntimeError('At least one actual error-path check failed; inspect scalar report')
    finally: store.close()


if __name__=='__main__': main()
