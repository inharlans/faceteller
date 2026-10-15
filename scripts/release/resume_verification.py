"""Resume an incomplete HTTP cohort, then finish UI/report with the same models.

The verifier rejects input/participant/runtime identity changes. Existing completed
reports are never overwritten. A new owned server warms up before remaining rows.
"""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.release.finish_project import (OUT,SHARE_DIR,cleanup_owned_child,load,report_paths,
    require_absent,run,run_actual_ui,save,state,ui_artifact_paths)


def main():
    source=load(OUT/'service-verification.json')
    if source.get('complete') is not False:raise RuntimeError('Only an incomplete cohort can resume')
    require_absent([OUT/n for n in ('score-influence.json','frontend-actual-verification.json')]+report_paths()+ui_artifact_paths())
    run('resume-preflight','scripts/release/serve.py','--validate-only')
    with socket.socket() as probe:probe.bind(('127.0.0.1',8767))
    with (OUT/'server.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,'-X','utf8','scripts/release/serve.py','--port','8767'],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    save(OUT/'server-process.json',{'pid':process.pid,'url':'http://127.0.0.1:8767/',
         'started_by':'scripts/release/resume_verification.py','corpus_sha256':source['corpus_sha256']})
    try:
        with httpx.Client(base_url='http://127.0.0.1:8767',timeout=5,trust_env=False) as client:
            deadline=time.monotonic()+180;health=None
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Owned resume server exited')
                try:
                    response=client.get('/api/v1/health')
                    if response.status_code==200 and response.json().get('ready'):
                        health=response.json();break
                except httpx.HTTPError:pass
                time.sleep(1)
            if health is None or health!=source['service']:raise RuntimeError('Resumed server identity differs')
        run('actual400-resume','scripts/release/verify_service.py','--resume')
        run('score-influence','scripts/release/audit_scores.py')
        run_actual_ui(health)
        args=('--output-dir',str(OUT),'--share-dir',str(SHARE_DIR))
        run('report-preflight','scripts/release/build_report.py','--validate-only',*args)
        run('report','scripts/release/build_report.py',*args)
        report=load(OUT/'service-verification.json')
        state('actual-verification','complete',service_url='http://127.0.0.1:8767/',
              participants=len(report['participants']),errors=sum(p['status']=='error' for p in report['participants']),
              resumed_existing_participants=len(source['participants']),
              next='Recorded-camera/TLS and real production upload UI completed; physical webcam not tested.')
    except BaseException:
        cleanup_owned_child(process);raise


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('actual-verification','failed',error_type=type(exc).__name__,message=str(exc));raise
