"""Release adapters. Import is inert: no model, GPU, DB connection or network.

Production starts only with a complete, matching release deployment and new
400-person depth calibration. Test doubles are injected explicitly by tests.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from facework.pilot_contract import (
    ContractError, DecisionPolicy, DIRECTIONS, InputLimits, Observation, STAGES,
    build_result, create_app, ndjson, SCORING_VERSION, SERVICE_SCORE_MODE,
)
from facework.release_repository import ReleaseRepository, database_dsn
from facework.release_selection import inference_selection
from facework.release_readings import ReadingCatalog

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ('eye','brow','nose','mouth','face','forehead','glabella','under_eye',
           'cheekbone','ear','philtrum','mole')
LIMITS = InputLimits()
QUALITY_VERSION = 'face-roi-and-signed-poses-v3'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as file:
        return json.load(file)


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _number(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate_deployment(root=ROOT):
    """All file/schema/hash checks precede model loading, including CUDA access."""
    from facework.release_preprocess import VERSION, PART_CROPS, INPUT_SIZES
    from facework.release_depth import VERSION as DEPTH_VERSION, FEATURES as DEPTH_FEATURES
    from facework.release_parsing import EAR_TRANSFORM_VERSION
    root=Path(root)
    registry_path=root/'data/release/training_registry.json'
    manifest_path=root/'models/release/deployment.json'
    depth_path=root/'data/release/depth_config.json'
    for path in (registry_path,manifest_path,depth_path):
        _require(path.is_file(),f'Required release artifact missing: {path.relative_to(root)}')
    registry, manifest, depth = map(read_json,(registry_path,manifest_path,depth_path))
    config=manifest['config']; parts=registry['parts']; features=registry['features']
    _require(config['registry_sha256']==sha256(registry_path),'Deployment registry hash mismatch')
    _require(config['preprocess_version']==registry['preprocess_version']==VERSION,'Preprocessing version mismatch')
    ear_path=root/'models/release/ear_preprocessing_frozen.json'
    _require(ear_path.is_file(),'Reference-scaled ear preprocessing protocol missing')
    ear=read_json(ear_path)
    _require(ear.get('ear_transform_version')==manifest.get('ear_transform_version')==EAR_TRANSFORM_VERSION
             and ear.get('registry_sha256')==config['registry_sha256']
             and manifest.get('ear_preprocessing_sha256')==sha256(ear_path),
             'Ear preprocessing version/hash mismatch')
    _require(set(parts)==set(REGIONS),'Release requires all 12 region definitions')
    heads=[]
    for part,spec in parts.items():
        _require(spec['features'],f'Region has no trained feature definitions: {part}')
        _require(spec['crops']==list(PART_CROPS[part]) and spec['input_size']==list(INPUT_SIZES[part]),f'Crop contract mismatch: {part}')
        if spec.get('context_version'):
            from facework.release_context import CONTEXT_VERSION, CONTEXT_PARTS, model_input_size
            _require(part in CONTEXT_PARTS and spec['context_version']==CONTEXT_VERSION
                     and spec.get('model_input_size')==model_input_size(part),f'Context input contract mismatch: {part}')
        elif 'model_input_size' in spec:
            _require(spec['model_input_size']==list(INPUT_SIZES[part]),f'Unexpected model input shape: {part}')
        for key in spec['features']:
            f=features[key]; classes=f['classes']
            _require(f['part']==part and f['kind']=='classification',f'Noncategorical or mismatched head: {key}')
            _require(f.get('region',part) in REGIONS,f'Invalid observation region: {key}')
            _require(len(classes)>=2 and len(set(classes))==len(classes) and all(isinstance(c,str) and c and ':' not in c for c in classes),f'Invalid class order: {key}')
            _require(len(f['classes_ko'])==len(classes),f'Missing class display names: {key}')
            if f.get('source')=='brow_mask_soft':
                from facework.release_brows import VERSION as BROW_VERSION, GATES as BROW_GATES
                definition=f.get('reviewed_definition',{})
                _require(definition.get('mask_definition')==BROW_VERSION
                         and definition.get('mask_gates')==BROW_GATES,f'Brow measurement version mismatch: {key}')
            heads.append(key)
    _require(len(heads)==len(set(heads)) and set(heads)==set(features)==set(manifest['features']),'Deployment head inventory mismatch')
    architectures,_=inference_selection(root,config,manifest)
    policy_path=root/'models/release/threshold_policy_frozen.json'
    if policy_path.exists() or 'threshold_policy_sha256' in manifest:
        from facework.release_thresholds import POLICY as THRESHOLD_POLICY
        _require(policy_path.is_file() and manifest.get('threshold_policy_sha256')==sha256(policy_path)
                 and read_json(policy_path)==THRESHOLD_POLICY,'Threshold policy hash mismatch')
        for key,c in manifest['features'].items():
            choice=c.get('threshold_selection',{})
            _require(choice.get('threshold')==c['threshold'] and choice.get('test_used') is False
                     and choice.get('source_split')=='calibration'
                     and choice.get('policy_version')==THRESHOLD_POLICY['version'],
                     f'Threshold selection provenance mismatch: {key}')
    _require(isinstance(architectures,list) and architectures and len(set(architectures))==len(architectures),'Invalid ensemble architecture list')
    expected={f'models/release/{arch}/{part}.pt' for arch in architectures for part in parts}
    _require(set(manifest['weights'])==expected,'Deployment weights do not exactly cover all models')
    for relative,digest in manifest['weights'].items():
        path=(root/relative).resolve()
        _require(path.is_relative_to(root.resolve()/'models/release') and path.is_file(),f'Missing or invalid release weight: {relative}')
        _require(sha256(path)==digest,f'Deployment weight hash mismatch: {relative}')
    for key,c in manifest['features'].items():
        count=len(features[key]['classes']); t=c.get('temperature'); support=c.get('train_support')
        from facework.release_reviewed_policy import COVERAGE_POLICY, SUPPORT_POLICY, validate_reviewed_calibration
        reviewed=c.get('coverage_policy')==COVERAGE_POLICY
        if reviewed:
            exposure=registry.get('training_exposure',{})
            _require(exposure.get('policy')==SUPPORT_POLICY,'Missing reviewed training exposure policy')
            validate_reviewed_calibration(c,features[key]['classes'],exposure.get('counts',{}).get(key))
        calibration_support=c.get('support')
        valid_coverage=(isinstance(calibration_support,list) and len(calibration_support)==count
                        and all(type(n) is int and n>=0 for n in calibration_support)
                        and type(c.get('n_val')) is int and sum(calibration_support)==c['n_val'])
        observed_classes=sum(n>0 for n in calibration_support) if valid_coverage else 0
        if c.get('calibrated') is True:
            _require(_number(t) and t>0 and c.get('optimizer_success') is True
                     and type(c.get('n_val')) is int and c['n_val']>0 and c.get('test_used_for_fit') is False,
                     f'Missing or unsuccessful validation calibration: {key}')
            if 'coverage_policy' in c and not reviewed:
                _require(c['coverage_policy']=='at_least_two_observed_classes_v1'
                         and valid_coverage and observed_classes>=2,
                         f'Insufficient calibration class coverage: {key}')
        else:
            no_labels=(c.get('n_val')==0 and c.get('status')=='no_valid_validation_labels')
            single_class=(valid_coverage and observed_classes==1
                          and c.get('status')=='single_class_calibration_support'
                          and c.get('coverage_policy')=='at_least_two_observed_classes_v1'
                          and c.get('test_used_for_fit') is False)
            _require(c.get('calibrated') is False and t is None and type(c.get('n_val')) is int
                     and (reviewed or no_labels or single_class),
                     f'Invalid disabled calibration state: {key}')
        _require(c.get('class_count',count)==count,f'Calibration class count mismatch: {key}')
        _require(_number(c.get('threshold')) and (.6 if count==2 else .5)<=c['threshold']<=1,f'Invalid confidence threshold: {key}')
        _require(isinstance(support,list) and len(support)==count and all(type(n) is int and n>=0 for n in support),f'Invalid training support: {key}')
        raw_support=(registry['training_exposure']['counts'][key] if reviewed
                     else registry['class_support']['train'][key]['counts'])
        _require(isinstance(raw_support,list) and len(raw_support)==count
                 and all(type(n) is int and n>=0 for n in raw_support)
                 and all(actual<=raw for actual,raw in zip(support,raw_support)),
                 f'Training support exceeds registry counts or registry support invalid: {key}')
        if 'classes' in c:
            _require(c['classes']==features[key]['classes'],f'Calibration class order mismatch: {key}')
    _require(depth.get('framing_version')==DEPTH_VERSION and depth.get('cohort_n')==400,'New 400-person release depth calibration required')
    _require(set(depth['features'])==set(registry['depth_features'])==set(DEPTH_FEATURES),'Depth head inventory mismatch')
    for key,c in depth['features'].items():
        _require(all(_number(c.get(n)) for n in ('lower','upper','sigma_repeat_train'))
                 and c['lower']<c['upper'] and c['sigma_repeat_train']>0
                 and c.get('n')==400 and c.get('repeat_train_people',0)>=5,f'Invalid new depth boundaries: {key}')
    q=depth['quality']
    _require(_number(q.get('min_confidence')) and q['min_confidence']>=0
             and _number(q.get('max_cross_view_error')) and q['max_cross_view_error']>=0,'Invalid depth quality settings')
    checkpoint=Path(depth.get('checkpoint_path','models/vggt/model.pt'))
    _require(not checkpoint.is_absolute(),'Depth checkpoint must be project-relative')
    checkpoint=(root/checkpoint).resolve()
    _require(checkpoint.is_relative_to(root.resolve()) and checkpoint.is_file(),'Configured depth checkpoint missing')
    _require(sha256(checkpoint)==depth.get('checkpoint_sha256'),'Depth checkpoint hash mismatch')
    source=root/'third_party/vggt'; provenance=read_json(source/'provenance.json')
    _require(depth.get('code_commit')==provenance['commit'],'Depth code commit mismatch')
    for relative,digest in provenance['files'].items():
        path=(source/relative).resolve()
        _require(path.is_relative_to(source.resolve()) and path.is_file() and sha256(path)==digest,
                 f'Pinned VGGT source checksum mismatch: {relative}')
    return registry,manifest,depth


def validate_checkpoint(meta, arch, part, registry, manifest):
    spec=registry['parts'][part]
    expected={'arch':arch,'part':part,'classification_features':spec['features'],
              'classes':{k:registry['features'][k]['classes'] for k in spec['features']},
              'crops':spec['crops'],'input_size':spec['input_size'],
              'registry_sha256':manifest['config']['registry_sha256'],
              'preprocess_version':registry['preprocess_version'],
              'train_support':{k:manifest['features'][k]['train_support'] for k in spec['features']}}
    if part=='ear':
        from facework.release_parsing import EAR_TRANSFORM_VERSION
        expected.update(ear_transform_version=EAR_TRANSFORM_VERSION,
                        ear_preprocessing_sha256=manifest.get('ear_preprocessing_sha256'))
        _require(isinstance(expected['ear_preprocessing_sha256'],str), 'Ear preprocessing fingerprint missing')
    if spec.get('context_version'):
        expected.update(context_version=spec['context_version'],model_input_size=spec['model_input_size'])
    else:
        _require(not meta.get('context_version'),f'Checkpoint requires undeclared context: {part}')
    _require(all(meta.get(k)==v for k,v in expected.items()),f'Checkpoint metadata/head/class order mismatch: {arch}/{part}')
    # JSON equality alone accepts booleans/floats as integers. These counts must
    # describe actual unique training people with usable crops for each head.
    _require(all(isinstance(counts,list) and all(type(n) is int and n>=0 for n in counts)
                 for counts in meta['train_support'].values()),
             f'Checkpoint training support invalid: {arch}/{part}')


def temperature_probability(probabilities, temperature):
    """Same exact p**(1/T) transform as release calibration, after both means."""
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
        raise ValueError('Invalid classifier probabilities')
    if not _number(temperature) or temperature<=0:
        raise ValueError('Invalid temperature')
    # Preserve exact zero probability; no smoothing changes the deployed formula.
    with np.errstate(divide='ignore'):
        logits=np.log(p)/temperature
    p=np.exp(logits-np.max(logits))
    return p/p.sum()


class InferenceWorker:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='release-inference')

    async def run(self, function):
        task=asyncio.get_running_loop().run_in_executor(self.executor,function)
        cancelled=False
        # Shield repeatedly: ASGI cancellation scopes may cancel more than once.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled=True
            except Exception:
                break
        if cancelled:
            if not task.cancelled():
                task.exception()  # consume worker failure; cancellation still wins
            raise asyncio.CancelledError
        return task.result()

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=False)


def quality_reasons(images,detections):
    """Explicit scope for every region; thresholds are gates, not accuracy claims."""
    flags={part:[] for part in REGIONS}; d=detections[0]; points=np.asarray(d['landmarks'])
    def add(parts,reason):
        for part in parts: flags[part].append(reason)
    if not np.isfinite(points).all():
        add(REGIONS,'LANDMARKS_INVALID')
        return flags
    if abs(d['yaw'])>12 or abs(d['pitch']+5)>16 or abs(d.get('roll',0))>12: add(REGIONS,'FRONTAL_POSE_INVALID')
    # Match acquisition: evaluate the visible face, not dark background/clothes.
    image=images[0]
    low=np.maximum(points[:,:2].min(axis=0),[0,0])
    high=np.minimum(points[:,:2].max(axis=0),[image.width,image.height])
    if high[0]-low[0]<150: add(REGIONS,'FACE_TOO_SMALL')
    if np.any(high<=low):
        add(REGIONS,'LANDMARKS_INVALID')
    else:
        with image.crop((*low,*high)) as face:
            with face.resize((48,48),Image.Resampling.BILINEAR) as reduced:
                pixels=np.asarray(reduced,dtype=np.float32)
                brightness=float((pixels@np.asarray([.2126,.7152,.0722])).mean())
        if brightness<45: add(REGIONS,'ILLUMINATION_TOO_DARK')
        if brightness>235: add(REGIONS,'ILLUMINATION_TOO_BRIGHT')
    if d['smile']>.3: add(('eye','under_eye','mouth','philtrum','cheekbone'),'EXPRESSION_SMILE')
    if d['blink']>.5: add(('eye','under_eye'),'EXPRESSION_BLINK')
    if d.get('brow_down',0)>.2: add(('brow','forehead','glabella'),'EXPRESSION_FROWN')
    width=np.linalg.norm(points[61,:2]-points[291,:2])
    if width<=0 or np.linalg.norm(points[13,:2]-points[14,:2])/max(width,1)>.06:
        add(('mouth','philtrum'),'MOUTH_OPEN')
    if (not -60<=detections[1]['yaw']<=-20 or not 20<=detections[2]['yaw']<=60
            or any(abs(d['pitch']+5)>16 or abs(d.get('roll',0))>12 for d in detections[1:])):
        add(('ear',),'EAR_VIEW_DIRECTIONS_INVALID')
    return flags


class RuntimeResources:
    @classmethod
    def load(cls, root=ROOT, device='cuda'):
        root=Path(root); registry,manifest,depth=validate_deployment(root)
        _require(device in ('cpu','cuda'),'Device must be cpu or cuda')
        # Check local dependencies and DB before allocating any CNN/VGGT models.
        _require((root/'models/face_landmarker.task').is_file(),'Local MediaPipe model missing')
        from facework.release_parsing import (ReleaseParser, visible_ear_box, forehead_metrics,
                                              ear_metrics, visible_mark_candidates)
        import psycopg
        dsn=database_dsn()
        with psycopg.connect(dsn,connect_timeout=5) as conn:
            role=conn.execute('SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user').fetchone()
            _require(not any(role),'Runtime must use an unprivileged DB reader')
            _require(conn.execute('SHOW default_transaction_read_only').fetchone()[0]=='on','Runtime DB role is not read-only')
            for table in ('literature','passage','label','label_passage','active_corpus'):
                writable=conn.execute("SELECT has_table_privilege(current_user,%s,'INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER')",
                                      ('face_release.'+table,)).fetchone()[0]
                _require(not writable,'Runtime DB role has write privileges')
            eligible=conn.execute('SELECT count(*) FROM face_release.passage WHERE service_eligible').fetchone()[0]
            _require(eligible>0,'Release literature database has no eligible passages')
        corpus_sha256=ReleaseRepository(dsn).corpus_revision()
        _require(corpus_sha256 is not None,'Active corpus revision missing; re-import the intended corpus')
        import torch
        torch.set_num_threads(2)
        from facework.kface_training import PartNet
        from facework import analysis_v2
        from facework.pilot.depth import VGGTEngine,VGGT_SOURCE
        _require(VGGT_SOURCE.resolve()==(root/'third_party/vggt').resolve(),'VGGT must load pinned third_party source, not temporary files')
        from facework import release_depth
        self=cls(); self.registry=registry; self.manifest=manifest; self.depth_config=depth
        self.readings=ReadingCatalog.load(root)
        self.device=device; self.repository=ReleaseRepository(dsn); self.models={}
        self.detect=analysis_v2.detect; self.detector=None
        self.visible_ear_box=visible_ear_box; self.forehead_metrics=forehead_metrics; self.ear_metrics=ear_metrics
        self.visible_mark_candidates=visible_mark_candidates
        self.depth_estimate=release_depth.estimate; self.depth_observations=release_depth.observations
        self.ready=False
        try:
            self.detector=analysis_v2.landmarker()
            self.parser=ReleaseParser(threads=2)
            for part,spec in registry['parts'].items():
                self.models[part]=[]
                for arch in inference_selection(root,manifest['config'],manifest)[0]:
                    path=root/'models/release'/arch/f'{part}.pt'
                    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                    validate_checkpoint(checkpoint['meta'],arch,part,registry,manifest)
                    model=PartNet(arch,0,[len(registry['features'][k]['classes']) for k in spec['features']],pretrained=False)
                    model.load_state_dict(checkpoint['state_dict'],strict=True)
                    model.eval().requires_grad_(False)
                    self.models[part].append(model.to(device))
                    del checkpoint
            class ManagedEngine(VGGTEngine):
                def prepare(engine,*args,**kwargs):
                    result=super().prepare(*args,**kwargs)
                    engine.pending_images.extend(result[1]); return result
                def infer(engine,*args,**kwargs):
                    engine.pending_images=[]; success=False
                    try:
                        result=super().infer(*args,**kwargs); success=True; return result
                    finally:
                        if not success:
                            for image in engine.pending_images: image.close()
                        engine.pending_images=[]
            # The existing VGGT engine is CUDA-specific; CPU mode is for CNN tests,
            # never a silent substitute for the full deployed depth service.
            _require(device=='cuda','Full release VGGT service requires cuda; use fake resources for CPU tests')
            self.engine=ManagedEngine(518,device=device,weights_path=root/depth.get('checkpoint_path','models/vggt/model.pt'))
            self.identity={'deployment_sha256':sha256(root/'models/release/deployment.json'),
                           'classifier_architectures':inference_selection(root,manifest['config'],manifest)[0],
                           'classifier_model_count':sum(len(models) for models in self.models.values()),
                           'quality_version':QUALITY_VERSION,
                           'registry_sha256':manifest['config']['registry_sha256'],
                           'depth_config_sha256':sha256(root/'data/release/depth_config.json'),
                           'preprocess_version':registry['preprocess_version'],
                           'ear_transform_version':manifest['ear_transform_version'],
                           'depth_framing_version':depth['framing_version'],
                           'depth_checkpoint_sha256':depth['checkpoint_sha256'],
                           'depth_checkpoint':depth.get('checkpoint','original research checkpoint'),
                           'depth_code_commit':depth['code_commit'],
                           'depth_weights_license':'research_noncommercial; not commercial-ready',
                           'uncalibrated_heads':{key:{'status':c['status'],'n_val':c['n_val'],'calibrated':False}
                                                 for key,c in manifest['features'].items() if not c['calibrated']},
                           'calibrated_head_count':sum(c['calibrated'] for c in manifest['features'].values()),
                           'repository':'postgresql_readonly','eligible_passages':eligible,
                           'corpus_sha256':corpus_sha256}
            self.identity.update(self.readings.identity)
            self.identity['active_mapping_count']=eligible-len(self.readings.excluded)
            self.identity['face_overlay_version']='original-pixel-measurements-v1'
            self.identity['brow_measurement_version']=registry['features'].get('brow_length',{}).get(
                'reviewed_definition',{}).get('mask_definition','landmark-guide')
            self.ready=True
            return self
        except BaseException:
            self.close(); raise

    def predict_part(self,part,crops,*,face_context=None,crop_names=None):
        import torch
        from facework.release_preprocess import network_array
        arrays=[network_array(crop,part) for crop in crops]
        if self.registry['parts'][part].get('context_version'):
            from facework.release_context import compose_context
            if face_context is None or crop_names is None or len(crop_names)!=len(crops):
                raise ValueError('Whole-face context and crop side are required by this checkpoint')
            whole=network_array(face_context,'face')
            arrays=[compose_context(a,whole,part,mirror_face=name.endswith('_left'))
                    for a,name in zip(arrays,crop_names)]
        x=torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).to(self.device).float()/255
        mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
        x=(x-mean)/std
        with torch.inference_mode():
            outputs=[model(x)[1] for model in self.models[part]]
            return {key:torch.stack([output[j].float().softmax(1) for output in outputs]).mean(0).mean(0).cpu().numpy()
                    for j,key in enumerate(self.registry['parts'][part]['features'])}

    def close(self):
        self.ready=False
        if hasattr(self,'models'): self.models.clear()
        self.engine=None
        if hasattr(self,'parser') and hasattr(self.parser,'close'): self.parser.close()
        self.parser=None
        if getattr(self,'detector',None) is not None:
            self.detector.close(); self.detector=None
            from facework import analysis_v2
            analysis_v2.landmarker.cache_clear()

    def check_database(self):
        import psycopg
        with psycopg.connect(self.repository._dsn,connect_timeout=3) as conn:
            conn.execute("SET LOCAL statement_timeout='3s'")
            return conn.execute('SELECT EXISTS(SELECT 1 FROM face_release.passage WHERE service_eligible)').fetchone()[0]


class ReleaseService:
    def __init__(self,resources,worker):
        self.r=resources; self.worker=worker; self.images=[]; self.detections=[]
        self.crops={}; self.flags={part:[] for part in REGIONS}; self.timings={}
        self.visible_brows=({} if self.r.registry['features'].get('brow_length',{}).get('source')=='brow_mask_soft' else None)

    def close(self):
        for crop in self.crops.values(): crop.image.close()
        self.crops.clear()
        for image in self.images: image.close()
        self.images.clear(); self.detections.clear(); self.flags.clear()
        if self.visible_brows is not None: self.visible_brows.clear()

    async def validate(self,inputs):
        def check():
            for direction in DIRECTIONS:
                with Image.open(io.BytesIO(inputs.images[direction])) as raw:
                    with ImageOps.exif_transpose(raw) as oriented:
                        image=oriented.convert('RGB')
                self.images.append(image)  # own before detect can fail
                detection,error=self.r.detect(image)
                if detection is None:
                    code='FACE_NOT_FOUND' if error=='no_face' else 'MULTIPLE_FACES' if error=='multiple_faces' else 'PREPROCESS_FAILED'
                    raise ContractError(code,'각 사진에서 얼굴 한 개를 확인할 수 있어야 합니다.',422,direction)
                self.detections.append(detection)
        start=time.perf_counter()
        try: await self.worker.run(check)
        except BaseException:
            self.close(); raise
        self.timings['validation_seconds']=time.perf_counter()-start

    def align(self):
        from facework.release_preprocess import eye_affine
        eye_affine(self.detections[0]['landmarks'])
        self.flags=quality_reasons(self.images,self.detections)

    def prepare(self):
        from facework.release_preprocess import frontal_crops, ear_crop
        self.crops.update(frontal_crops(self.images[0],self.detections[0]['landmarks']))
        # Segmentation only supplies visibility/ear geometry. Never mask CNN RGB.
        try:
            mask=self.r.parser.parse(self.images[0],self.detections[0]['landmarks'])
        except Exception:
            self.flags['forehead'].append('PARSING_FAILED')
            self.flags['mole'].append('PARSING_FAILED')
        else:
            if self.visible_brows is not None and not self.flags['brow']:
                try:
                    from facework.release_brows import measure_brows
                    d=self.detections[0]
                    measured=measure_brows(mask,d['landmarks'],
                        pose=(d['pitch'],d['yaw'],d.get('roll',0)),
                        blend=(d['smile'],d['blink'],d['blink']))
                    self.visible_brows={side:{**brow,'path':brow['path_original']}
                                        for side,brow in measured['sides'].items()}
                except Exception:
                    self.visible_brows={}
            try:
                values=self.r.forehead_metrics(mask,self.detections[0]['landmarks'])
                self.flags['forehead'].extend(values.get('reasons',[]))
            except Exception: self.flags['forehead'].append('FOREHEAD_VISIBILITY_UNAVAILABLE')
            try:
                marks=self.r.visible_mark_candidates(self.images[0],mask,self.detections[0]['landmarks'])
                if not marks['quality_passed']:
                    self.flags['mole'].extend(marks.get('reasons') or ['SKIN_MARK_VISIBILITY_INSUFFICIENT'])
                del marks
            except Exception: self.flags['mole'].append('SKIN_MARK_VISIBILITY_UNAVAILABLE')
            del mask
        for index,name in ((1,'ear_left_oblique'),(2,'ear_right_oblique')):
            try:
                detection=self.detections[index]
                mask=self.r.parser.parse(self.images[index],detection['landmarks'])
                box=self.r.visible_ear_box(mask,detection['landmarks'])
                metrics=self.r.ear_metrics(mask,detection['landmarks'],detection['yaw'])
                del mask
                if box is None or metrics.get('reasons'): continue
                self.crops[name]=ear_crop(self.images[index],box,landmarks=detection['landmarks'],
                                          yaw_degrees=detection['yaw'],mirror=(index==2))
            except Exception:
                # The other visible ear remains independently usable.
                continue

    def classify(self):
        out=[]
        for part,spec in self.r.registry['parts'].items():
            names=[k for k in spec['crops'] if k in self.crops and self.crops[k].valid_fraction>=.9]
            available=[self.crops[k] for k in names]
            reasons=list(self.flags[part]); predictions={}
            if not available: reasons.append('CROP_UNAVAILABLE')
            context=self.crops.get('face') if spec.get('context_version') else None
            if spec.get('context_version') and (context is None or context.valid_fraction<.9):
                reasons.append('CROP_UNAVAILABLE')
            if available and not reasons:
                try:
                    predictions=(self.r.predict_part(part,available,face_context=context,crop_names=names)
                                 if spec.get('context_version') else self.r.predict_part(part,available))
                except Exception: reasons.append('CNN_PART_FAILED')
            for key in spec['features']:
                f=self.r.registry['features'][key]; cal=self.r.manifest['features'][key]
                region=f.get('region',part)
                item_reasons=list(reasons); confidence=None; label=None; name=None
                if region!=part:
                    item_reasons.extend(reason for reason in self.flags[region] if reason not in item_reasons)
                if f.get('source')=='brow_mask_soft' and len(self.visible_brows or {})!=2:
                    item_reasons.append('BROW_MEASUREMENT_UNAVAILABLE')
                if not cal['calibrated']: item_reasons.append('CONFIDENCE_NOT_CALIBRATED')
                if not item_reasons:
                    try:
                        p=temperature_probability(predictions[key],cal['temperature'])
                        if len(p)!=len(f['classes']): raise ValueError('Wrong class count')
                        idx=int(np.argmax(p)); confidence=float(p[idx]); label=f'{key}:{f["classes"][idx]}'
                        name=f['classes_ko'][idx]
                        if cal['train_support'][idx]==0: item_reasons.append('PREDICTED_CLASS_NOT_TRAINED')
                        from facework.release_reviewed_policy import confirmation_class_allowed
                        if not confirmation_class_allowed(cal,f['classes'][idx]):
                            item_reasons.append('CLASS_CALIBRATION_INSUFFICIENT')
                    except Exception: item_reasons.append('CNN_HEAD_FAILED')
                basis='자동 형태 레이블로 학습한 모델이 사진의 모양을 분류합니다.'
                if f.get('source')=='brow_mask_soft':
                    basis='사진에 보이는 눈썹 털을 기준으로 학습한 분류입니다. 길이는 눈 가로 대비 비율을 비교 자료와 비교합니다.'
                if key=='eye_asymmetry':
                    basis='두 눈이 함께 보이는 얼굴 사진으로 좌우 눈의 개방 차이를 분류합니다. 자동 형태 레이블을 학습한 모델의 추정입니다.'
                basis+=(' 분류 신뢰도는 별도 자료로 보정한 값입니다.' if cal['calibrated']
                        else ' 신뢰도 보정 자료가 부족해 이 항목을 확정하지 않습니다.')
                if key=='pigmented_marks':
                    basis+=' 머리카락·눈썹·눈·코·입과 아랫얼굴을 제외한 이마·윗볼의 작은 어두운 표시만 판정합니다. 없음은 이 관찰 범위에 대한 결과입니다.'
                out.append(Observation(key,region,'classification',label,name,confidence=confidence,
                    class_count=len(f['classes']),calibrated=cal['calibrated'] is True,
                    basis=basis,
                    hold_reasons=tuple(item_reasons)))
        return out

    def depth(self):
        try:
            measured=self.r.depth_estimate(self.r.engine,self.images,self.detections,self.r.detect)
            observations=self.r.depth_observations(measured,self.detections,self.r.depth_config,self.r.registry['depth_features'])
            return [replace(o,hold_reasons=o.hold_reasons+tuple(self.flags[o.region])) for o in observations]
        except Exception:
            return [Observation(key,f['part'],'depth',None,None,
                                basis='다중 시점 깊이 측정 실패로 이 항목을 보류했습니다.',
                                hold_reasons=('DEPTH_ESTIMATION_FAILED',))
                    for key,f in self.r.registry['depth_features'].items()]

    async def analyze(self,inputs):
        observations=[]
        try:
            yield {'type':'progress','stage':STAGES[0]}
            start=time.perf_counter(); await self.worker.run(self.align)
            self.timings['alignment_seconds']=time.perf_counter()-start
            yield {'type':'progress','stage':STAGES[1]}
            start=time.perf_counter(); await self.worker.run(self.prepare)
            self.timings['preprocessing_seconds']=time.perf_counter()-start
            yield {'type':'progress','stage':STAGES[2]}
            start=time.perf_counter(); observations.extend(await self.worker.run(self.classify))
            self.timings['cnn_seconds']=time.perf_counter()-start
            yield {'type':'progress','stage':STAGES[3]}
            start=time.perf_counter(); observations.extend(await self.worker.run(self.depth))
            self.timings['depth_seconds']=time.perf_counter()-start
            policy=DecisionPolicy({k:c['threshold'] for k,c in self.r.manifest['features'].items()})
            items=policy.apply(observations)
            yield {'type':'progress','stage':STAGES[4]}
            def literature():
                with self.r.repository.snapshot() as repository:
                    revision=repository.corpus_revision()
                    _require(revision is not None,'Active corpus revision is unavailable')
                    readings=getattr(self.r,'readings',None)
                    result=build_result(items,readings.repository(repository) if readings else repository)
                    return readings.apply(result) if readings else result,revision
            start=time.perf_counter(); result,revision=await self.worker.run(literature)
            self.timings['literature_seconds']=time.perf_counter()-start
            result['runtime']={**self.r.identity,'corpus_sha256':revision,
                               'timings':dict(self.timings),'independent_human_labels':0}
            if getattr(self.r,'readings',None):
                result['runtime'].update(self.r.readings.identity)
            from facework.release_overlay import face_overlay
            result['face_overlay']=face_overlay(self.detections[0]['landmarks'],*self.images[0].size,
                                               visible_brows=self.visible_brows)
            yield {'type':'result','data':result}
        finally:
            self.close(); observations.clear()


class AdmissionMiddleware:
    """Bound admission before reading uploads; hold slot through response cleanup."""
    def __init__(self,app,*,capacity=1):
        self.app=app; self.capacity=capacity; self.active=0

    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('path')!='/api/v1/analysis':
            return await self.app(scope,receive,send)
        if self.active>=self.capacity:
            payload=ndjson(ContractError('SERVICE_BUSY','분석 중입니다. 잠시 후 다시 시도해 주세요.',503).event())
            await send({'type':'http.response.start','status':503,'headers':[(b'content-type',b'application/x-ndjson'),(b'cache-control',b'no-store'),(b'retry-after',b'1')]})
            await send({'type':'http.response.body','body':payload})
            return
        self.active+=1
        try: await self.app(scope,receive,send)
        finally: self.active-=1


def make_app(resources=None,*,root=ROOT,device='cuda',worker=None,limits=LIMITS):
    """Inject resources only in tests; production lifespan loads verified artifacts."""
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    root=Path(root); owned_worker=worker is None; worker=worker or InferenceWorker()
    state={'resources':resources}
    def factory():
        r=state['resources']
        if r is None or not r.ready: raise ContractError('SERVICE_NOT_READY','서비스 준비가 완료되지 않았습니다.',503)
        return ReleaseService(r,worker)
    app=create_app(factory,limits)
    @asynccontextmanager
    async def lifespan(app):
        try:
            if state['resources'] is None:
                def load():
                    state['resources']=RuntimeResources.load(root,device)
                await worker.run(load)
            yield
        finally:
            if resources is None and state['resources'] is not None:
                await worker.run(state['resources'].close)
            if owned_worker: worker.close()
    app.router.lifespan_context=lifespan
    app.add_middleware(AdmissionMiddleware,capacity=1)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','[::1]','testserver'])
    @app.get('/api/v1/health')
    @app.get('/api/v1/service-info')
    async def health():
        r=state['resources']; ready=r is not None and r.ready
        revision=None
        if ready and hasattr(r,'check_database'):
            try: ready=bool(await asyncio.to_thread(r.check_database))
            except Exception: ready=False
        if ready:
            try:
                revision=await asyncio.to_thread(r.repository.corpus_revision)
                ready=revision is not None
            except Exception: ready=False
        return JSONResponse({'ready':ready,'scoring_version':SCORING_VERSION,'score_mode':SERVICE_SCORE_MODE,
            'max_passages':5,'max_per_region':3,'limits':vars(limits),
            **(r.identity if ready else {}),'corpus_sha256':revision},
            status_code=200 if ready else 503,headers={'Cache-Control':'no-store'})
    dist=root/'frontend/dist'
    if (dist/'assets').is_dir(): app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/')
    async def index():
        if not (dist/'index.html').is_file():
            return JSONResponse({'error':'FRONTEND_BUILD_MISSING'},status_code=503)
        return FileResponse(dist/'index.html',headers={'Cache-Control':'no-store'})
    return app
