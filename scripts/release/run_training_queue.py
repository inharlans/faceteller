"""Serial GPU queue: comparison -> frozen evaluation -> depth -> release training.

The comparison is already running separately. This queue waits for its explicit
completion, never kills it or runs a second GPU job beside it. New label training
also requires a coordinator-written readiness marker after actual validation.
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/release';OUT.mkdir(parents=True,exist_ok=True)
STATE=OUT/'training_queue.json'


def state(stage,status,**extra):
    data={'stage':stage,'status':status,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    temporary=STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(STATE)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def run(stage,args):
    state(stage,'running')
    log=OUT/f'queue-{stage}.log'
    with log.open('ab',buffering=0) as out:
        result=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    if result.returncode:
        state(stage,'failed',returncode=result.returncode,log=str(log.relative_to(ROOT)))
        raise SystemExit(result.returncode)
    state(stage,'complete',log=str(log.relative_to(ROOT)))


def main():
    log=ROOT/'outputs/model-selection-20261014/train.log'
    state('comparison','waiting_for_existing_training')
    while True:
        paths=[p for p in (ROOT/'models/selection-20261014').glob('*/*/*.json') if not p.stem.endswith(('_progress','_test'))]
        if len(paths)==45 and 'ALL_TRAINING_COMPLETE' in log.read_text(encoding='utf-8'):
            break
        time.sleep(10)
    time.sleep(3)
    if not (ROOT/'outputs/model-selection-20261014/results.json').exists():
        run('comparison-evaluation',['scripts/model_selection_evaluate.py','all'])
    run('comparison-report',['scripts/model_selection_report.py'])
    if not (ROOT/'data/release/depth_config.json').exists():
        run('release-depth',['scripts/release/calibrate_depth.py'])
    else:
        # A previous process may have stopped after config publication but before
        # its audit completed. Recheck records/config on CPU before proceeding.
        run('release-depth-audit',['scripts/release/audit_depth.py'])
    marker=ROOT/'data/release/extended_ready.json'
    state('release-labels','waiting_for_reviewed_automatic_label_artifacts')
    while not marker.exists():time.sleep(10)
    ready=json.loads(marker.read_text(encoding='utf-8'))
    if ready.get('ready') is not True:raise RuntimeError('extended label artifacts not ready')
    run('release-registry',['scripts/release/build_training_registry.py'])
    run('release-models',['scripts/release/train_models.py','all'])
    state('training-queue','complete',next='Start release server and execute all400 actual HTTP acceptance checks')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('training-queue','failed',error_type=type(exc).__name__,message=str(exc));raise
