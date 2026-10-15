"""Run raw K-FACE triplets through the exact service framing, then fit CDF settings."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.release_depth import VERSION,estimate
from facework.pilot.depth import VGGTEngine,FEATURES
from scripts.release.audit_depth import validate_records

OUT=ROOT/'data/derived/release'


def publish_config(config):
    """Publish a complete config atomically; completion requires its bound audit."""
    target=ROOT/'data/release/depth_config.json';target.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)
    temporary=target.with_suffix('.json.tmp')
    try:
        temporary.write_text(payload,encoding='utf-8')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    from scripts.release.audit_depth import main as audit_depth
    audit_depth()
    print('RELEASE_DEPTH_CONFIG_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-only',action='store_true');args=parser.parse_args()
    protocol=json.loads((ROOT/'models/selection-20261014/protocol.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True,exist_ok=True);records_path=OUT/'depth_records.jsonl'
    attempts=[]
    if records_path.exists():
        for line in records_path.read_text(encoding='utf-8').splitlines():
            if line.strip():attempts.append(json.loads(line))
    split,rows=validate_records(attempts,protocol['groups'],VERSION);ids=sorted(split)
    if not args.fit_only:
        from facework.kface import Store
        from facework.analysis_v2 import detect
        engine=VGGTEngine(518);store=Store(2)
        jobs=[(sid,'L1') for sid in ids]+[(sid,light) for sid in ids[::10] for light in ['L2','L3']]
        try:
            for index,(sid,light) in enumerate(jobs):
                key=f'{sid}:{light}'
                if key in rows and rows[key]['status']!='error':continue
                images=[];t=time.perf_counter()
                row={'job':key,'subject_id':sid,'split':split[sid],'light':light,'framing_version':VERSION}
                try:
                    images=[store.image(sid,'S001',light,'E01',camera) for camera in ('C7','C4','C10')]
                    detections=[detect(im)[0] for im in images]
                    if any(d is None for d in detections):row.update(status='held',reason='FACE_NOT_FOUND')
                    else:
                        row.update(estimate(engine,images,detections,detect))
                        row['poses']=[{k:d[k] for k in ['yaw','pitch','smile','blink']} for d in detections]
                except Exception as exc:
                    row.update(status='error',error_type=type(exc).__name__)
                    if 'out of memory' in str(exc).lower():engine.torch.cuda.empty_cache()
                finally:
                    for im in images:im.close()
                row['wall_seconds']=time.perf_counter()-t
                with records_path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                rows[key]=row
                print('RELEASE_DEPTH',index+1,len(jobs),sid,light,row['status'],round(row['wall_seconds'],2),flush=True)
        finally:store.close()
    cohort=[rows.get(f'{sid}:L1',{}) for sid in ids]
    if len(cohort)!=400 or any(r.get('status')!='measured' for r in cohort):
        raise RuntimeError('400 valid base measurements required; inspect explicit failed/held records')
    train=[r for r in cohort if split[r['subject_id']]=='train']
    cross=[r['cross_view_midline_error_ratio'] for r in train if r.get('cross_view_midline_error_ratio') is not None]
    confidence=[r['depth_confidence_min'] for r in train]
    with (ROOT/'models/vggt/model.pt').open('rb') as weight_file:
        weight_hash=hashlib.file_digest(weight_file,'sha256').hexdigest()
    source_info=json.loads((ROOT/'third_party/vggt/provenance.json').read_text(encoding='utf-8'))
    config={'framing_version':VERSION,'cohort_n':400,'checkpoint_sha256':weight_hash,
            'code_commit':source_info['commit'],'checkpoint':'facebook/VGGT-1B (original research checkpoint)',
            'boundary_scope':'all400 user-defined descriptive reference cohort; not independent depth accuracy',
            'quality':{'min_confidence':float(np.quantile(confidence,.01)),
                       'max_cross_view_error':float(np.quantile(cross,.99)),
                       'fit_scope':'train270 empirical 1st/99th percentiles; not metric 3D ground truth'},
            'features':{},'records_sha256':hashlib.sha256(records_path.read_bytes()).hexdigest()}
    for key in FEATURES:
        values=np.array([r['values'][key] for r in cohort]);lower,upper=np.quantile(values,[.25,.75])
        repeats=[]
        for sid in ids[::10]:
            if split[sid]!='train':continue
            rr=[rows.get(f'{sid}:{light}') for light in ['L1','L2','L3']]
            if all(r and r['status']=='measured' for r in rr):repeats.append([r['values'][key] for r in rr])
        if len(repeats)<5:raise RuntimeError('insufficient training-only repeat captures')
        sigma=float(np.sqrt(np.var(repeats,axis=1,ddof=1).mean()))
        if sigma<=0 or not np.isfinite(sigma):raise RuntimeError('invalid repeat sigma')
        config['features'][key]={'lower':float(lower),'upper':float(upper),'sigma_repeat_train':sigma,
            'repeat_train_people':len(repeats),'population_sd':float(values.std(ddof=1)),
            'n':400,'quantiles':[.25,.75]}
    publish_config(config)


if __name__=='__main__':main()
