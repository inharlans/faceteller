"""Recalibrate existing ResNet weights for the user's single-model decision.

Preserves the previous ensemble deployment, held-out results and all weights.
No checkpoint or boundary selection is performed using test participants.
"""
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.release.train_models import dump, load, sha
from facework.release_runtime import validate_deployment


def main():
    out=ROOT/'outputs/release'
    models=ROOT/'models/release'
    archive=out/'before-resnet-only'
    selection=models/'inference_frozen.json'
    targets=[models/'deployment.json',out/'test_metrics.json',out/'training_queue.json']
    if not selection.exists():
        if archive.exists():
            raise RuntimeError('Existing pre-transition archive requires inspection')
        validate_deployment(ROOT)
        archive.mkdir()
        identities={}
        for path in targets:
            shutil.copy2(path,archive/path.name)
            identities[path.relative_to(ROOT).as_posix()]=sha(path)
        weights={path.relative_to(ROOT).as_posix():sha(path) for path in models.glob('*/*.pt')}
        dump(archive/'identity.json',{'artifacts':identities,'weights':weights})
        config=load(models/'training_frozen.json')
        dump(selection,{'version':'release-inference-choice-v1','architectures':['resnet18'],
                        'registry_sha256':config['registry_sha256'],
                        'training_config_sha256':sha(models/'training_frozen.json'),
                        'selection_basis':'User explicitly chose one ResNet18 family. Highest selection macroF1 among the three tested single models; historical ensemble comparison preserved.',
                        'decided_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        # These exact two archived artifacts are replaced by a new calibration/test.
        for path in targets[:2]:
            if sha(path)!=identities[path.relative_to(ROOT).as_posix()]:
                raise RuntimeError('Deployment changed while archiving')
            path.unlink()
    identity=load(archive/'identity.json')
    for relative,digest in identity['artifacts'].items():
        if sha(archive/Path(relative).name)!=digest:raise RuntimeError('Archived artifact changed')
    dump(out/'training_queue.json',{'stage':'resnet-only-calibration','status':'running'})
    subprocess.run([sys.executable,'-B','-X','utf8','scripts/release/train_models.py','test'],cwd=ROOT,check=True)
    registry,manifest,depth=validate_deployment(ROOT)
    if manifest.get('inference_architectures')!=['resnet18'] or len(manifest['weights'])!=12:
        raise RuntimeError('ResNet-only deployment inventory incorrect')
    for relative,digest in identity['weights'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Existing trained weight changed')
    result=load(out/'test_metrics.json')
    if result['deployment_sha256']!=sha(models/'deployment.json'):
        raise RuntimeError('Test deployment mismatch')
    dump(out/'resnet-only-verification.json',{'pass':True,'architectures':['resnet18'],
         'active_model_count':12,'categorical_heads':len(registry['features']),
         'preserved_tensor_artifact_count':len(identity['weights']),
         'preserved_classifier_weight_count':sum('/cache/' not in path for path in identity['weights']),
         'deployment_sha256':sha(models/'deployment.json'),'test_metrics_sha256':sha(out/'test_metrics.json'),
         'inference_selection_sha256':sha(selection),'retrained':False,
         'archive':'outputs/release/before-resnet-only'})
    dump(out/'training_queue.json',{'stage':'training-queue','status':'complete',
         'resnet_only_verified':True,'ear_correction_verified':True})
    print('RESNET_ONLY_DEPLOYMENT_VERIFIED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(ROOT/'outputs/release/training_queue.json',{'stage':'resnet-only-calibration','status':'failed',
             'error_type':type(exc).__name__,'message':str(exc)})
        raise
