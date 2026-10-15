"""Preserve the original run, replace only ear classifiers, then verify service.

This one-time correction waits for the existing serial GPU queue. It never stops
that process or starts training beside it. Baseline artifacts are moved into an
auditable archive; rerunning after a partial correction requires inspection.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/release'
CORRECTION = OUT / 'ear-input-correction'
MODELS = ROOT / 'models/release'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def state(stage, status, **extra):
    value = dict(stage=stage, status=status,
                 updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    save(OUT / 'training_queue.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(log_name, *args):
    with (OUT / log_name).open('ab', buffering=0) as stream:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', *args], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    while True:
        queue = load(OUT / 'training_queue.json')
        if queue['status'] == 'failed':
            raise RuntimeError('Original GPU queue failed; no correction attempted')
        if queue['stage'] == 'training-queue' and queue['status'] == 'complete':
            break
        time.sleep(10)
    # Original parent publishes completion only after its training child exits.
    if CORRECTION.exists():
        raise RuntimeError('Correction archive already exists; inspect before retry')
    proof = load(OUT / 'ear-v2-training-input-check.json')
    ear_manifest = ROOT / 'data/derived/release/ears-v2/manifest.json'
    registry = ROOT / 'data/release/training_registry.json'
    config_path = MODELS / 'training_frozen.json'
    config = load(config_path)
    if (proof.get('pass') is not True or proof.get('gpu_used') is not False
            or proof['cache_manifest_sha256'] != sha(ear_manifest)
            or proof['registry_sha256'] != sha(registry)):
        raise RuntimeError('Actual corrected ear inputs have not passed CPU parity')
    architectures = config['architectures']
    all_weights = sorted(MODELS.glob('*/*.pt'))
    all_weights = [p for p in all_weights if p.parent.name in architectures]
    non_ear = [p for p in all_weights if p.stem != 'ear']
    if len(all_weights) != 36 or len(non_ear) != 33:
        raise RuntimeError('Original 36 classifiers must be complete')
    unchanged = {p.relative_to(ROOT).as_posix(): sha(p) for p in non_ear}
    frozen = {p.relative_to(ROOT).as_posix(): sha(p) for p in (registry, config_path, ear_manifest)}
    paths = [MODELS / arch / name for arch in architectures
             for name in ('ear.pt', 'ear.json', 'ear_progress.json')]
    paths += [MODELS / 'cache' / f'ear_{split}.pt'
              for split in ('train', 'selection', 'calibration', 'test')]
    paths += [MODELS / 'deployment.json', OUT / 'test_metrics.json']
    if any(not p.is_file() for p in paths):
        raise RuntimeError('Baseline ear/deployment/test artifacts incomplete')
    baseline_ear_seconds = sum(load(MODELS / arch / 'ear.json')['elapsed_seconds'] for arch in architectures)
    baseline_calibration = load(MODELS / 'deployment.json')['calibration_timings']['total_seconds']
    baseline_test = load(OUT / 'test_metrics.json')['timings']['total_seconds']
    CORRECTION.mkdir()
    archive = CORRECTION / 'baseline'
    record = {'reason': 'Ear bbox normalization discarded relative ear length; fixed eye-reference scaling',
              'ear_transform_version': proof['ear_transform_version'],
              'baseline_files': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'non_ear_weights': unchanged, 'frozen_inputs': frozen,
              'baseline_ear_training_seconds': baseline_ear_seconds,
              'baseline_calibration_seconds': baseline_calibration,
              'baseline_test_seconds': baseline_test}
    save(CORRECTION / 'before.json', record)
    state('release-ear-correction', 'running', unchanged_non_ear_models=33)
    for path in paths:
        destination = archive / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        if sha(destination) != record['baseline_files'][path.relative_to(ROOT).as_posix()]:
            raise RuntimeError('Baseline archive hash mismatch')
    started = time.perf_counter()
    run('ear-correction-training.log', 'scripts/release/train_models.py', 'all')
    elapsed = time.perf_counter() - started
    save(CORRECTION / 'training-finished.json', {
        'correction_wall_seconds': elapsed,
        'timing_method': 'perf_counter around completed training/calibration/test child',
        'deployment_sha256': sha(MODELS / 'deployment.json'),
        'test_metrics_sha256': sha(OUT / 'test_metrics.json')
    })
    after = {p.relative_to(ROOT).as_posix(): sha(p) for p in non_ear}
    if after != unchanged or any(sha(ROOT / p) != digest for p, digest in frozen.items()):
        raise RuntimeError('Correction changed an unrelated weight or frozen input')
    deployment = load(MODELS / 'deployment.json')
    identity = sha(MODELS / 'ear_preprocessing_frozen.json')
    for arch in architectures:
        meta = load(MODELS / arch / 'ear.json')['meta']
        if (meta.get('ear_transform_version') != proof['ear_transform_version']
                or meta.get('ear_preprocessing_sha256') != identity):
            raise RuntimeError('Corrected ear metadata mismatch')
    if deployment.get('ear_preprocessing_sha256') != identity:
        raise RuntimeError('Deployment did not bind the corrected ear protocol')
    if any('\\' in p or sha(ROOT / p) != digest for p, digest in deployment['weights'].items()):
        raise RuntimeError('Deployment paths/hashes invalid')
    if load(OUT / 'test_metrics.json')['deployment_sha256'] != sha(MODELS / 'deployment.json'):
        raise RuntimeError('Test report does not correspond to corrected deployment')
    run('ear-correction-preflight.log', 'scripts/release/serve.py', '--validate-only')
    save(CORRECTION / 'verification.json', {
        'pass': True, 'unchanged_non_ear_models': 33, 'retrained_ear_models': 3,
        'baseline_ear_training_seconds': baseline_ear_seconds,
        'baseline_calibration_seconds': baseline_calibration,
        'baseline_test_seconds': baseline_test,
        'correction_wall_seconds': elapsed,
        'ear_transform_version': proof['ear_transform_version'],
        'before_non_ear_weights': unchanged, 'after_non_ear_weights': after,
        'deployment_sha256': sha(MODELS / 'deployment.json'),
        'ear_preprocessing_sha256': identity,
        'scope': 'Three corrected ear classifiers plus fresh full calibration/test; other33 weights unchanged'
    })
    state('training-queue', 'complete', ear_correction_verified=True)
    run('completion-coordinator.log', 'scripts/release/finish_project.py', '--wait')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        # A failed service check is separately recorded by finish_project.py.
        if not (CORRECTION / 'verification.json').exists():
            state('release-ear-correction', 'failed', error_type=type(exc).__name__, message=str(exc))
        raise
