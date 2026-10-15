"""Repeatability and distribution audit of actual release-framed VGGT measurements.

CPU only. No changes to thresholds, models or labels; no metric-depth accuracy claim.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.pilot.depth import FEATURES
from facework.pilot.scoring import boundary_scores


SPLIT_COUNTS={'train':270,'selection':34,'calibration':34,'test':62}


def validate_groups(groups):
    if not isinstance(groups,dict) or set(groups)!=set(SPLIT_COUNTS):
        raise ValueError('Expected train/selection/calibration/test splits')
    owners={}
    for split,n in SPLIT_COUNTS.items():
        ids=groups[split]
        if not isinstance(ids,list) or len(ids)!=n:
            raise ValueError(f'Expected {n} participants in {split}')
        for sid in ids:
            if not isinstance(sid,str) or not sid.strip() or sid in owners:
                raise ValueError('Participant IDs must be nonempty strings and splits must be disjoint without duplicates')
            owners[sid]=split
    return owners


def validate_records(rows,groups,framing_version):
    """Validate every attempt before retaining the latest attempt for each job."""
    owners=validate_groups(groups)
    repeat_people=set(sorted(owners)[::10])
    jobs={}
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Measurement record must be an object')
        sid,light=row.get('subject_id'),row.get('light')
        if not isinstance(sid,str) or sid not in owners or light not in ('L1','L2','L3'):
            raise ValueError('Unknown record participant or light')
        if row.get('job')!=f'{sid}:{light}':
            raise ValueError('Record job does not match subject_id:light')
        if light!='L1' and sid not in repeat_people:
            raise ValueError('Repeat capture is outside the scheduled participant set')
        if row.get('split')!=owners[sid] or row.get('framing_version')!=framing_version:
            raise ValueError('Record split or framing mismatch')
        jobs[row['job']]=row
    return owners,jobs


def label(value,config):
    return int(np.searchsorted([config['lower'],config['upper']],value,side='right'))


def repeat_summary(matrix,config):
    a=np.asarray(matrix,dtype=float)
    if a.size==0:return {'people':0,'pairs':0}
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():raise ValueError('Three finite captures per repeat participant required')
    n,k=a.shape
    groups=a.mean(axis=1)
    between=float(k*np.var(groups,ddof=1)) if n>1 else None
    within=float(np.var(a,axis=1,ddof=1).mean())
    denominator=between+(k-1)*within if between is not None else 0
    icc=(between-within)/denominator if denominator else None
    labels=np.searchsorted([config['lower'],config['upper']],a,side='right')
    return {'people':n,'pairs':2*n,'mean_absolute_change_from_L1':float(np.abs(a[:,1:]-a[:,:1]).mean()),
            'mean_shift_L2':float((a[:,1]-a[:,0]).mean()),'mean_shift_L3':float((a[:,2]-a[:,0]).mean()),
            'label_agreement_L1_L2':float((labels[:,0]==labels[:,1]).mean()),
            'label_agreement_L1_L3':float((labels[:,0]==labels[:,2]).mean()),
            'all_three_labels_agree':float((labels==labels[:,:1]).all(axis=1).mean()),
            'pooled_repeat_sd':within**.5,'icc_1_1':icc}


def analyze(rows,config,groups):
    owners,jobs=validate_records(rows,groups,config['framing_version'])
    cohort=[jobs.get(f'{sid}:L1') for sid in sorted(owners)]
    if any(not r or r.get('status')!='measured' for r in cohort):raise ValueError('400 measured L1 records required')
    features={}
    for key in FEATURES:
        c=config['features'][key];values=np.array([r['values'][key] for r in cohort],float)
        if not np.isfinite(values).all() or not np.allclose(np.quantile(values,[.25,.75]),[c['lower'],c['upper']],rtol=1e-12,atol=1e-12):
            raise ValueError('Depth values do not reproduce frozen boundaries')
        scores=[boundary_scores(float(v),c['lower'],c['upper'],c['sigma_repeat_train']) for v in values]
        repeats={s:[] for s in ('all','train','selection','calibration','test')}
        for sid,split in owners.items():
            triplet=[jobs.get(f'{sid}:{light}') for light in ('L1','L2','L3')]
            if all(r and r['status']=='measured' for r in triplet):
                row=[r['values'][key] for r in triplet];repeats['all'].append(row);repeats[split].append(row)
        summaries={s:repeat_summary(v,c) for s,v in repeats.items()}
        train=summaries['train']
        if train['people']!=c['repeat_train_people'] or not np.isclose(train['pooled_repeat_sd'],c['sigma_repeat_train'],rtol=1e-12,atol=1e-12):
            raise ValueError('Training repeats do not reproduce frozen sigma')
        w,p=stats.shapiro(values)
        counts=np.bincount([label(v,c) for v in values],minlength=3).tolist()
        features[key]={'participants':400,'mean':float(values.mean()),'population_sd':float(values.std(ddof=1)),
                       'skewness':float(stats.skew(values,bias=False)),'excess_kurtosis':float(stats.kurtosis(values,bias=False)),
                       'shapiro_w':float(w),'shapiro_p':float(p),'normal_distribution_required':False,
                       'class_counts':dict(zip(('low','medium','high'),counts)),
                       'mean_assigned_score':float(np.mean([s['score'] for s in scores])),
                       'repeat':summaries}
    return {'features':features,'base_participants':400,
            'repeat_jobs_attempted':sum(r['light']!='L1' for r in jobs.values()),
            'repeat_jobs_measured':sum(r['light']!='L1' and r['status']=='measured' for r in jobs.values())}


def main():
    config_path=ROOT/'data/release/depth_config.json';record_path=ROOT/'data/derived/release/depth_records.jsonl'
    config_raw=config_path.read_bytes();config=json.loads(config_raw);raw=record_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['records_sha256']:raise ValueError('Measurement records changed after calibration')
    protocol=json.loads((ROOT/'models/selection-20261014/protocol.json').read_text(encoding='utf-8'))
    report=analyze([json.loads(line) for line in raw.decode().splitlines() if line],config,protocol['groups'])
    report.update(version='release-depth-repeatability-v1',depth_config_sha256=hashlib.sha256(config_raw).hexdigest(),
                  records_sha256=config['records_sha256'],framing_version=config['framing_version'],
                  human_ground_truth_count=0,metric_3d_ground_truth_count=0,
                  interpretation='Repeat lighting stability of reconstructed ratios, not depth accuracy. Boundaries describe all400 by user specification; sigma fits training repeats only. No model or boundary changed using this audit.')
    out=ROOT/'outputs/release/depth-stability.json';out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(out)
    print('DEPTH_REPEATABILITY_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
