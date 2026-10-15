"""Real models over local TLS + recorded-camera/identical-upload integration.

Trust is confined to this HTTP client and the certificate SPKI in this browser.
No OS trust changes, real webcam claim, photo files or public listener.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import serialization

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.release.setup_local_tls import ensure_certificate
from scripts.release.finish_project import cleanup_owned_child


def main():
    out=ROOT/'outputs/release'; destination=out/'tls-camera-verification.json'
    if destination.exists():raise RuntimeError('Existing TLS verification preserved')
    cert=ensure_certificate();context=ssl.create_default_context(cafile=cert['certfile'])
    leaf=x509.load_pem_x509_certificate(Path(cert['certfile']).read_bytes())
    spki=leaf.public_key().public_bytes(serialization.Encoding.DER,serialization.PublicFormat.SubjectPublicKeyInfo)
    pin=base64.b64encode(hashlib.sha256(spki).digest()).decode()
    with socket.socket() as probe:probe.bind(('127.0.0.1',8768))
    with (out/'tls-server.log').open('ab',buffering=0) as log:
        process=subprocess.Popen([sys.executable,'-B','-X','utf8','scripts/release/serve.py','--https'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url='https://127.0.0.1:8768',verify=context,trust_env=False,timeout=5) as client:
            deadline=time.monotonic()+180;health=None
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Owned TLS service exited')
                try:
                    response=client.get('/api/v1/health')
                    if response.status_code==200 and response.json().get('ready'):
                        health=response.json();break
                except httpx.HTTPError:pass
                time.sleep(1)
            if health is None:raise RuntimeError('TLS service did not become ready')
        default_rejected=False
        try:httpx.get('https://127.0.0.1:8768/api/v1/health',trust_env=False,timeout=5)
        except httpx.ConnectError:default_rejected=True
        if not default_rejected:raise RuntimeError('Expected self-signed development certificate to require scoped trust')
        if health.get('quality_version')!='face-roi-and-signed-poses-v3' or health.get('classifier_architectures')!=['resnet18']:
            raise RuntimeError('TLS server is not the final intended runtime')
        environment=os.environ.copy();environment['FACETELLER_TEST_URL']='https://127.0.0.1:8768'
        environment['FACETELLER_TLS_SPKI']=pin;environment['NODE_EXTRA_CA_CERTS']=cert['certfile']
        environment.pop('FACETELLER_ALLOW_SELF_SIGNED',None);environment.pop('PLAYWRIGHT_CHROMIUM_EXECUTABLE',None)
        proof_path=Path(tempfile.gettempdir())/'faceteller-recorded-camera/verification.json'
        if proof_path.exists():raise RuntimeError('Existing camera proof preserved; archive before rerun')
        with (out/'tls-camera-browser.log').open('ab',buffering=0) as log:
            subprocess.run([shutil.which('node'),str(ROOT/'frontend/node_modules/playwright/cli.js'),
                            'test','--config','playwright.camera.config.ts'],cwd=ROOT/'frontend',env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        proof=json.loads(proof_path.read_text())
        if proof.get('pass') is not True or proof.get('deployment_sha256')!=health['deployment_sha256']:
            raise RuntimeError('Camera integration proof mismatched TLS runtime')
        value={'pass':True,'actual_models':True,'url':'https://127.0.0.1:8768/',
               'certificate_verified_by_scoped_ca':True,'browser_certificate_spki_pinned':True,
               'system_default_rejected_self_signed':default_rejected,'global_trust_modified':False,
               'self_signed_development':True,'physical_camera_tested':False,'health':health,'camera':proof}
        destination.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        shutil.copy2(proof_path,out/'frontend-camera-verification.json')
        print('ACTUAL_TLS_RECORDED_CAMERA_AND_UPLOAD_PARITY_VERIFIED',flush=True)
    finally:cleanup_owned_child(process)


if __name__=='__main__':main()
