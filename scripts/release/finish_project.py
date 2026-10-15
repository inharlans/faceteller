"""Run the authorized local service and actual400 verification after training.

Waits for both GPU training completion and a reviewed/imported corpus marker.
Does not publish a site, send reports, replace a running server or hide failures.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import tempfile
import time

import httpx

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.release.setup_database import canonical_json
from scripts.release.build_report import TITLE as REPORT_TITLE

OUT=ROOT/'outputs/release';STATE=OUT/'completion_queue.json'
SHARE_DIR=(Path('C:/Users/user/Downloads/FaceTeller_완성프로젝트_공유자료') if os.name=='nt'
           else Path('/mnt/c/Users/user/Downloads/FaceTeller_완성프로젝트_공유자료'))


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(40):
        try:tmp.replace(path);break
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)
def state(stage,status,**extra):
    value={'stage':stage,'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}
    save(STATE,value);print(json.dumps(value,ensure_ascii=False),flush=True)
def run(stage,*args):
    state(stage,'running')
    with (OUT/f'completion-{stage}.log').open('ab',buffering=0) as log:
        subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def report_paths():
    return [directory/f'{REPORT_TITLE}.{suffix}' for directory in (OUT,SHARE_DIR)
            for suffix in ('pdf','xlsx','txt')]


def require_absent(paths):
    """Preserve old results and incomplete writes for manual inspection."""
    candidates=set()
    for path in paths:
        candidates.update((path,path.with_name(path.name+'.partial'),
                           path.with_name(path.name+'.tmp'),path.with_suffix('.tmp')))
    existing=sorted(str(path) for path in candidates if path.exists() or path.is_symlink())
    if existing:
        raise RuntimeError('Existing completion artifacts preserved; inspect/archive manually before retry: '+
                           ', '.join(existing))


def preflight_outputs():
    require_absent([OUT/name for name in ('service-smoke.json','service-verification.json',
                                        'score-influence.json','server-process.json',
                                        'frontend-actual-verification.json')]+report_paths()+ui_artifact_paths())


def ui_artifact_paths():
    # Match Windows Node os.tmpdir() precedence, then pin TEMP/TMP in its child.
    directory=Path(os.environ.get('TEMP') or os.environ.get('TMP') or tempfile.gettempdir())/'faceteller-production-ui'
    return [directory/name for name in ('verification.json','actual-result-desktop.png',
                                       'actual-result-mobile.png','actual-sample-result.pdf')]


def run_actual_ui(health):
    """Run the real Windows browser; copy only its validated scalar JSON evidence."""
    artifacts=ui_artifact_paths();source=artifacts[0]
    destination=OUT/'frontend-actual-verification.json'
    require_absent(artifacts+[destination])
    if sys.platform!='win32':raise RuntimeError('Actual release UI verification requires Windows Node/browser and project Windows .venv')
    node=shutil.which('node')
    if not node:raise RuntimeError('Windows node is not on PATH; actual UI verification was not run')
    cli=ROOT/'frontend/node_modules/playwright/cli.js'
    if not cli.is_file():raise RuntimeError('Local Playwright CLI missing; install frontend dependencies before completion')
    expected={key:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for key,name in (
        ('deployment_sha256','models/release/deployment.json'),
        ('depth_config_sha256','data/release/depth_config.json'),
        ('registry_sha256','data/release/training_registry.json'))}
    marker=load(ROOT/'data/release/literature_ready.json')
    if marker.get('ready') is not True:raise RuntimeError('Corpus readiness withdrawn before UI verification')
    expected['corpus_sha256']=marker['corpus_sha256']
    if any(health.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Actual UI local artifacts differ from the owned server health identity')
    environment=os.environ.copy()
    for name in ('PLAYWRIGHT_CHROMIUM_EXECUTABLE','LD_LIBRARY_PATH'):
        environment.pop(name,None)
    environment['FACETELLER_TEST_URL']='http://127.0.0.1:8767'
    environment['TEMP']=environment['TMP']=str(source.parent.parent)
    state('actual-ui','running')
    with (OUT/'completion-actual-ui.log').open('ab',buffering=0) as log:
        subprocess.run([node,str(cli),'test','--config','playwright.release.config.ts'],
                       cwd=ROOT/'frontend',env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not source.is_file() or source.is_symlink():raise RuntimeError('Actual UI did not create a fresh verification.json')
    raw=source.read_bytes();proof=json.loads(raw)
    if not isinstance(proof,dict) or any(proof.get(key) is not value for key,value in (
        ('pass',True),('mocked',False),('actual_models',True),('artifact_identity_verified',True),('photos_saved',False))):
        raise RuntimeError('Actual UI proof did not verify real models, identity and photo-free output')
    if any(proof.get(key)!=value for key,value in expected.items()):
        raise RuntimeError('Actual UI proof artifact hashes differ from local/health identity')
    if proof.get('source_origin')!='http://127.0.0.1:8767' or proof.get('external_requests')!=0 or proof.get('page_errors')!=[]:
        raise RuntimeError('Actual UI proof has wrong origin, external requests or browser errors')
    count=proof.get('passages')
    if type(count) is not int or not 0<=count<=5 or proof.get('pdf_passage_text_verified_count')!=count:
        raise RuntimeError('Actual UI PDF text verification count differs from returned passages')
    require_absent([destination])
    # Exclusive creation preserves an artifact appearing after the preflight.
    with destination.open('xb') as stream:stream.write(raw)
    state('actual-ui','complete',verification=str(destination),pdf_passage_text_verified_count=count)


def cleanup_owned_child(process):
    """Only the current Popen handle; never a saved PID or a process on a port."""
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            if process.poll() is None:
                process.kill()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    preflight_outputs()
    state('dependencies','waiting_for_training_and_corpus')
    marker=ROOT/'data/release/literature_ready.json'
    while True:
        training=load(OUT/'training_queue.json')
        if training['status']=='failed':raise RuntimeError('Training queue failed; inspect it before service startup')
        if training['stage']=='training-queue' and training['status']=='complete' and marker.exists():break
        if not args.wait:raise RuntimeError('Training/corpus not ready; no service started')
        time.sleep(10)
    ready=load(marker)
    corpus=load(ROOT/'data/release/literature/reference_corpus_v1.json')
    corpus_hash=hashlib.sha256(canonical_json(corpus).encode()).hexdigest()
    if ready.get('ready') is not True or ready.get('corpus_sha256')!=corpus_hash:
        raise RuntimeError('Corpus readiness differs from final source artifact')
    # A result may have appeared during the potentially long dependency wait.
    preflight_outputs()
    run('deployment-preflight','scripts/release/serve.py','--validate-only')
    # Bind proof avoids reusing or killing an unrelated server on this port.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',8767))
    preflight_outputs()
    state('service','starting')
    process=None
    try:
        with (OUT/'server.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-X','utf8','scripts/release/serve.py','--port','8767'],
                                     cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        save(OUT/'server-process.json',{'pid':process.pid,'url':'http://127.0.0.1:8767/',
                                       'started_by':'scripts/release/finish_project.py','corpus_sha256':corpus_hash})
        with httpx.Client(base_url='http://127.0.0.1:8767',timeout=5,trust_env=False) as client:
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Release server exited during startup; inspect server.log')
                try:
                    response=client.get('/api/v1/health',timeout=max(.01,min(5,deadline-time.monotonic())))
                    if response.status_code==200 and response.json().get('ready'):break
                except httpx.HTTPError:pass
                time.sleep(1)
            else:raise RuntimeError('Release server did not become ready within180s')
            if response.json().get('corpus_sha256')!=corpus_hash:raise RuntimeError('Actual DB revision differs from corpus readiness')
        smoke_path=OUT/'service-smoke.json'
        run('actual-smoke','scripts/release/verify_service.py','--limit','2','--output',str(smoke_path))
        smoke=load(smoke_path)
        if not smoke['complete'] or any(p['status']=='error' or p['confirmed']==0 for p in smoke['participants']):
            raise RuntimeError('Actual service smoke did not yield usable morphology results')
        if not any(i['kind']=='depth' and i['status']=='CONFIRMED' for p in smoke['participants'] for i in p['items']):
            raise RuntimeError('Actual smoke yielded no usable depth; inspect before400 evaluation')
        run('actual400','scripts/release/verify_service.py')
        require_absent([OUT/'score-influence.json'])
        run('score-influence','scripts/release/audit_scores.py')
        run_actual_ui(response.json())
        report_args=('--output-dir',str(OUT),'--share-dir',str(SHARE_DIR))
        require_absent(report_paths())
        run('report-preflight','scripts/release/build_report.py','--validate-only',*report_args)
        require_absent(report_paths())
        run('report','scripts/release/build_report.py',*report_args)
        report=load(OUT/'service-verification.json')
        state('actual-verification','complete',service_url='http://127.0.0.1:8767/',
              participants=len(report['participants']),errors=sum(p['status']=='error' for p in report['participants']),
              next='Review actual metrics and completed production upload UI proof; physical live-camera validation remains unperformed.')
    except BaseException as failure:
        if process is not None:
            try:
                cleanup_owned_child(process)
            except BaseException as cleanup_error:
                failure.add_note('Owned server cleanup failed; retain server-process.json and inspect manually: '+repr(cleanup_error))
        raise


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'completion-queue.lock').open('a+b') as lock:
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
            state('actual-verification','failed',error_type=type(exc).__name__,message=str(exc));raise
