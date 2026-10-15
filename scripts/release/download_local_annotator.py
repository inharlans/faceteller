"""Fetch the official, pinned local visual annotator. No photos are uploaded."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
from huggingface_hub import HfApi,snapshot_download

ROOT=Path(__file__).resolve().parents[2]
REPO='Qwen/Qwen3-VL-2B-Instruct'
OUT=ROOT/'models/annotation/qwen3-vl-2b'


def main():
    info=HfApi().model_info(REPO,files_metadata=True)
    revision=info.sha
    snapshot_download(REPO,revision=revision,local_dir=OUT,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE*'],max_workers=2)
    hashes={}
    for path in OUT.glob('*'):
        if path.is_file():
            with path.open('rb') as stream:hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest={'repo':REPO,'revision':revision,'license':'apache-2.0','purpose':'offline automatic morphology annotation only',
              'runtime_service_uses_this_model':False,'face_uploads':False,'files':hashes,
              'primary_source':f'https://huggingface.co/{REPO}'}
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'revision':revision,'files':len(hashes),'path':str(OUT)}),flush=True)


if __name__=='__main__':main()
