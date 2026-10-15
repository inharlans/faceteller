"""Continue authorized local annotation after the serial GPU queue yields.

Readiness means nonempty automated training/calibration data, not verified truth.
Stops on insufficient support; never fills unknown targets or fabricates weights.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.release_preprocess import PART_CROPS
from scripts.release.annotate_marks_local import selected_subjects,digest
OUT=ROOT/'outputs/release';STATE=OUT/'input_preparation.json'
ANNOTATIONS=ROOT/'data/release/local_marks/qwen3-vl-2b-v2'
LABEL_PYTHON=ROOT/'.venv-labeling/Scripts/python.exe'


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def state(stage,status,**extra):
    data={'stage':stage,'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    save(STATE,data);print(json.dumps(data,ensure_ascii=False),flush=True)
def run(stage,script,*args,label_python=False):
    state(stage,'running')
    with (OUT/f'inputs-{stage}.log').open('ab',buffering=0) as log:
        subprocess.run([str(LABEL_PYTHON) if label_python else sys.executable,'-X','utf8',script,*args],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def check_ready(registry,overlay):
    if len(registry['subjects'])!=400 or set(registry['parts'])!=set(PART_CROPS) or any(not p['features'] for p in registry['parts'].values()):
        raise RuntimeError('400 participants and all12 nonempty parts required')
    counts=overlay['class_support']
    people=registry['subjects']
    if set(overlay['subjects'])!=set(people) or set(overlay['source_hashes']['records'])!=set(people):
        raise RuntimeError('Every participant must have an explicit matching annotation record')
    for split,ids in overlay['groups'].items():
        if set(ids)!={s for s,p in people.items() if p['split']==split}:raise RuntimeError('Overlay split mismatch')
        labels=[overlay['subjects'][s]['pigmented_marks'] for s in ids]
        if any(people[s]['labels']['pigmented_marks']!=overlay['subjects'][s]['pigmented_marks'] for s in ids):
            raise RuntimeError('Registry mark targets differ from overlay')
        expected={name:labels.count(value) for name,value in [('present','present'),('absent','absent'),('unknown',None)]}
        if counts[split]!=expected:raise RuntimeError('Overlay class support differs from actual labels')
    if min(counts['train']['present'],counts['train']['absent'])<2:
        raise RuntimeError('Insufficient positive/negative training mark examples')
    for split in ('selection','calibration'):
        if counts[split]['present']+counts[split]['absent']<2:
            raise RuntimeError(f'Insufficient known mark targets for {split}')
    if len(overlay['source_hashes']['records'])!=400:raise RuntimeError('Every participant must have an explicit annotation record')
    return {'train_per_class_minimum':2,'selection_known_minimum':2,'calibration_known_minimum':2,
            'test_used_for_readiness':False,'independent_accuracy_verified':False,
            'class_support':counts,'meaning':'nonempty weak-target training support only; per-class limitations reported separately'}


def check_smoke(smoke,groups,manifest):
    expected=dict(selected_subjects(groups,True));rows=smoke['participants']
    if len(rows)!=20 or len({r['subject_id'] for r in rows})!=20 or {r['subject_id']:r['split'] for r in rows}!=expected:
        raise RuntimeError('Smoke must contain the fixed20 unique training participants')
    if smoke['config_sha256']!=digest(manifest):raise RuntimeError('Smoke configuration changed')
    good=sum(r['status']=='ok' for r in rows)
    if good<18:raise RuntimeError('Smoke response/input validity below18/20; inspect before extending')
    return good


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    marker=ROOT/'data/release/extended_ready.json'
    if marker.exists():raise RuntimeError('Readiness marker already exists; inspect current run instead of duplicating it')
    state('gpu-ownership','waiting_for_release_labels')
    while True:
        queue=load(OUT/'training_queue.json')
        if queue['status']=='failed':raise RuntimeError('GPU queue failed before annotation')
        if queue['stage']=='release-labels' and queue['status']=='waiting_for_reviewed_automatic_label_artifacts':break
        if not args.wait:raise RuntimeError('GPU belongs to comparison/depth/training; annotation not started')
        time.sleep(10)
    run('smoke','scripts/release/annotate_marks_local.py','--smoke',label_python=True)
    smoke=load(ANNOTATIONS/'smoke-run.json')
    groups=load(ROOT/'models/selection-20261014/protocol.json')['groups']
    good=check_smoke(smoke,groups,load(ANNOTATIONS/'manifest.json'))
    state('smoke','complete',valid_json_records=good,semantic_accuracy_verified=False)
    run('all400','scripts/release/annotate_marks_local.py','--all',label_python=True)
    run('merge-strict','scripts/release/merge_mark_labels.py','--negative-policy','strict')
    run('merge-agreement','scripts/release/merge_mark_labels.py','--negative-policy','vlm_three_absent_no_stable')
    run('registry','scripts/release/build_training_registry.py')
    registry_path=ROOT/'data/release/training_registry.json'
    overlay_path=ROOT/'data/release/local_marks/merged_mark_labels_vlm_agreement.json'
    registry,overlay=load(registry_path),load(overlay_path)
    validation=check_ready(registry,overlay)
    parity=load(OUT/'crop_parity.json')
    if not parity['pass'] or parity['participants']!=400 or parity['preprocess_version']!=registry['preprocess_version']:
        raise RuntimeError('Training/runtime crop parity has not passed')
    run('cpu-checks','-m','unittest','tests.test_release_training_data','tests.test_release_landmark_labels',
        'tests.test_release_preprocess','tests.test_local_mark_annotations','tests.test_release_readiness')
    save(marker,{'ready':True,'registry_sha256':sha(registry_path),'overlay_sha256':sha(overlay_path),
                 'source_hashes':registry['source_hashes'],'validation':validation,'smoke_valid_json_records':good,
                 'human_review_count':0,'decision':'Predeclared automatic nonempty-data checks passed; not human/clinical validation'})
    state('input-preparation','complete',marker=str(marker.relative_to(ROOT)),validation=validation)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    # The operating system releases this lock on exit, including a crashed waiter.
    with (OUT/'input-preparation.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as exc:
            state('input-preparation','failed',error_type=type(exc).__name__,message=str(exc));raise
