"""Bind the deployed classifier choice without rewriting training history."""
import hashlib
import json
from pathlib import Path


def inference_selection(root, config, manifest=None):
    root = Path(root)
    path = root / 'models/release/inference_frozen.json'
    fields = ('inference_architectures', 'inference_selection_sha256')
    if not path.exists():
        if manifest is not None and any(key in manifest for key in fields):
            raise RuntimeError('Inference selection artifact missing')
        return list(config['architectures']), {}
    raw = path.read_bytes()
    selection = json.loads(raw)
    training_path = root / 'models/release/training_frozen.json'
    if (selection.get('version') != 'release-inference-choice-v1'
            or selection.get('architectures') != ['resnet18']
            or 'resnet18' not in config['architectures']
            or selection.get('registry_sha256') != config['registry_sha256']
            or selection.get('training_config_sha256') != hashlib.sha256(training_path.read_bytes()).hexdigest()
            or json.loads(training_path.read_bytes()) != config):
        raise RuntimeError('Inference selection differs from frozen training history')
    identity = {'inference_architectures': selection['architectures'],
                'inference_selection_sha256': hashlib.sha256(raw).hexdigest()}
    if manifest is not None and any(manifest.get(key) != value for key, value in identity.items()):
        raise RuntimeError('Deployment inference selection mismatch; recalibration required')
    return selection['architectures'], identity
