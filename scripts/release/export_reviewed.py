"""Stage a reviewed candidate as a complete release; do not activate it here."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def registry_for_candidate(old, targets, protocol, manifest):
    from facework.release_reviewed_policy import SUPPORT_POLICY
    registry = copy.deepcopy(old)
    registry['version'] = 'reviewed-release-v1'
    registry['source_hashes']['reviewed_targets'] = protocol['targets_sha256']
    registry['reviewed_candidate_version'] = protocol['version']
    for part, inputs in protocol['inputs'].items():
        registry['parts'][part].update(inputs)
    for key, definition in targets['definitions'].items():
        f = registry['features'][key]
        f['reviewed_definition'] = definition
        f['source'] = definition['source']
        f['supported'] = definition['supported']
        f['annotation_provenance'] = targets['meaning']
        if 'cuts' in definition:
            f['thresholds'] = {
                'cuts': definition['cuts'], 'fit_split': 'train',
                'fit_cohort_people': 270,
                'train_subjects': definition.get('train_measured_subjects',f.get('thresholds',{}).get('train_subjects',270)),
                'sigma_single_capture': definition['sigma_subject'],
                'method': 'Gaussian interval soft targets; boundaries and repeat variability fitted on training participants',
            }
    registry['features']['brow_length'].update(
        ko='눈썹 털의 길이(눈 가로 대비)',
        classes_ko=['비교 자료보다 짧은 편', '보통', '비교 자료보다 긴 편'])
    for sid, entry in targets['subjects'].items():
        labels, votes = {}, {}
        for key, target in entry['targets'].items():
            t = target['subject']
            labels[key] = (targets['definitions'][key]['classes'][t['hard']]
                           if t is not None and not t['tie'] else None)
            if t is not None:
                votes[key] = t['soft']
        registry['subjects'][sid] = {'split': entry['split'], 'labels': labels, 'soft_targets': votes}
    registry['class_support'] = {
        split: {key: {'counts': row['hard'], 'unknown': row['ties'] + row['unmeasured']}
                for key, row in values.items()}
        for split, values in targets['support'].items()}
    registry['training_exposure'] = {
        'policy': SUPPORT_POLICY,
        'counts': {key: value['train_support'] for key, value in manifest['features'].items()},
        'meaning': 'Unique training people supplying a usable crop whose soft target argmax is the class; bilateral crops may contribute to different classes',
    }
    return registry


def stage(candidate, root=ROOT):
    import torch
    from facework.release_runtime import validate_checkpoint, validate_deployment
    from facework.release_thresholds import POLICY
    from scripts.release.retrain_reviewed import verify_protocol
    candidate, root = Path(candidate).resolve(), Path(root).resolve()
    if not candidate.is_relative_to(root / 'experiments'):
        raise ValueError('Candidate must be in the private experiments directory')
    protocol = verify_protocol(candidate, root)
    targets = load(candidate / 'targets.json')
    manifest = load(candidate / 'deployment_candidate.json')
    test = load(candidate / 'test_metrics.json')
    if (manifest['protocol_sha256'] != sha(candidate / 'protocol.json')
            or test['deployment_candidate_sha256'] != sha(candidate / 'deployment_candidate.json')
            or manifest['targets_sha256'] != sha(candidate / 'targets.json')):
        raise RuntimeError('Candidate/test identity mismatch')
    output = candidate / 'staged-release'
    if output.exists():
        raise RuntimeError('Staging directory already exists; preserve it and use a fresh output')
    registry = registry_for_candidate(load(root / 'data/release/training_registry.json'), targets, protocol, manifest)
    write(output / 'data/release/training_registry.json', registry)
    registry_hash = sha(output / 'data/release/training_registry.json')
    ear = load(root / 'models/release/ear_preprocessing_frozen.json')
    ear['registry_sha256'] = registry_hash
    write(output / 'models/release/ear_preprocessing_frozen.json', ear)
    ear_hash = sha(output / 'models/release/ear_preprocessing_frozen.json')
    config = {**protocol['config'], 'version': 'reviewed-release-training-v1',
              'architectures': ['resnet18'], 'registry_sha256': registry_hash,
              'preprocess_version': registry['preprocess_version'], 'split_sizes': protocol['split'],
              'candidate_protocol_sha256': sha(candidate / 'protocol.json'),
              'targets_sha256': protocol['targets_sha256'], 'human_review_count': 0}
    write(output / 'models/release/training_frozen.json', config)
    selection = {'version': 'release-inference-choice-v1', 'architectures': ['resnet18'],
                 'registry_sha256': registry_hash,
                 'training_config_sha256': sha(output / 'models/release/training_frozen.json'),
                 'selection_basis': 'User-selected ResNet18 only; reviewed targets and calibration'}
    write(output / 'models/release/inference_frozen.json', selection)
    write(output / 'models/release/threshold_policy_frozen.json', POLICY)
    deployment = {**manifest, 'status': 'staged_validated_candidate', 'config': config,
                  'registry_sha256': registry_hash, 'ear_preprocessing_sha256': ear_hash,
                  'inference_selection_sha256': sha(output / 'models/release/inference_frozen.json'),
                  'threshold_policy_sha256': sha(output / 'models/release/threshold_policy_frozen.json'),
                  'source_candidate_manifest_sha256': sha(candidate / 'deployment_candidate.json'),
                  'weights': {}}
    deployment.pop('candidate_weight_files', None)
    for part in registry['parts']:
        relative = f'models/release/resnet18/{part}.pt'
        source = candidate / f'models/resnet18/{part}.pt'
        if sha(source) != manifest['weights'][relative]:
            raise RuntimeError(f'Candidate weights changed: {part}')
        ck = torch.load(source, map_location='cpu', weights_only=True)
        if ck['meta']['targets_sha256'] != protocol['targets_sha256']:
            raise RuntimeError(f'Target metadata mismatch: {part}')
        ck['meta']['source_registry_sha256'] = ck['meta']['registry_sha256']
        ck['meta']['registry_sha256'] = registry_hash
        if part == 'ear':
            ck['meta']['ear_preprocessing_sha256'] = ear_hash
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, destination)
        deployment['weights'][relative] = sha(destination)
        validate_checkpoint(ck['meta'], 'resnet18', part, registry, deployment)
        history = load(candidate / f'models/resnet18/{part}.json')
        history['meta'] = copy.deepcopy(ck['meta'])
        history['source_candidate_weight_sha256'] = manifest['weights'][relative]
        history['deployment_weight_sha256'] = deployment['weights'][relative]
        history['status'] = 'reviewed_training_complete'
        write(output / f'models/release/resnet18/{part}.json', history)
        write(output / f'models/release/resnet18/{part}_progress.json', history)
        del ck
    write(output / 'models/release/deployment.json', deployment)
    # Read-only dependencies for complete startup validation. A hard link avoids
    # duplicating the large VGGT checkpoint; staged code never writes this file.
    depth = load(root / 'data/release/depth_config.json')
    write(output / 'data/release/depth_config.json', depth)
    depth_file = Path(depth.get('checkpoint_path', 'models/vggt/model.pt'))
    (output / depth_file).parent.mkdir(parents=True, exist_ok=True)
    os.link(root / depth_file, output / depth_file)
    shutil.copytree(root / 'third_party/vggt', output / 'third_party/vggt',
                    ignore=shutil.ignore_patterns('__pycache__', '.git'))
    validate_deployment(output)
    changed = ['data/release/training_registry.json'] + [
        p.relative_to(output).as_posix() for p in (output / 'models/release').rglob('*') if p.is_file()]
    receipt = {'status': 'staged_not_activated', 'candidate': candidate.relative_to(root).as_posix(),
               'files': {p: sha(output / p) for p in sorted(changed)},
               'test_metrics_sha256': sha(candidate / 'test_metrics.json'),
               'baseline_files': {p: sha(root / p) if (root / p).exists() else None for p in changed},
               'calibrated_heads': sum(c['calibrated'] for c in deployment['features'].values()),
               'human_review_count': 0}
    write(candidate / 'staging_receipt.json', receipt)
    return receipt


def activate(candidate, root=ROOT):
    """Activate an already validated stage while the local service is stopped.

    Every replaced artifact is archived first. Failed validation restores the
    previous bundle before returning the error. No tunnel or auth changes.
    """
    from facework.release_runtime import validate_deployment
    candidate, root = Path(candidate).resolve(), Path(root).resolve()
    if not candidate.is_relative_to(root / 'experiments'):
        raise ValueError('Candidate must be in the private experiments directory')
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', 8767)) == 0:
            raise RuntimeError('Stop the local analysis service before replacing model files')
    receipt = load(candidate / 'staging_receipt.json')
    stage_root = candidate / 'staged-release'
    if receipt['status'] != 'staged_not_activated':
        raise RuntimeError('Not an unactivated stage')
    for relative, digest in receipt['files'].items():
        if sha(stage_root / relative) != digest:
            raise RuntimeError(f'Staged artifact changed: {relative}')
        baseline = receipt['baseline_files'][relative]
        actual = sha(root / relative) if (root / relative).exists() else None
        if actual != baseline:
            raise RuntimeError(f'Active baseline changed before promotion: {relative}')
    validate_deployment(stage_root)
    backup = candidate / ('baseline-before-promotion-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    backup.mkdir()
    for relative, digest in receipt['baseline_files'].items():
        if digest is not None:
            dest = backup / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, dest)
    replaced = []
    # Receipts are part of activation too: an I/O failure after model replacement
    # must not leave an unreported new bundle active.
    report_paths=[candidate/'activation_receipt.json',
                  root/'outputs/release/reviewed-model-activation-20261015.json',
                  root/'outputs/release/reviewed-test-metrics-20261015.json']
    previous_reports={p:p.read_bytes() if p.exists() else None for p in report_paths}
    try:
        for relative in receipt['files']:
            dest = root / relative
            temp = dest.with_suffix(dest.suffix + '.reviewed-tmp')
            shutil.copy2(stage_root / relative, temp)
            temp.replace(dest)
            replaced.append(relative)
        validate_deployment(root)
        result = {**receipt, 'status': 'activated', 'baseline_archive': backup.relative_to(root).as_posix()}
        write(report_paths[0], result)
        write(report_paths[1], result)
        shutil.copy2(candidate / 'test_metrics.json', report_paths[2])
    except BaseException:
        for relative in reversed(replaced):
            if receipt['baseline_files'][relative] is None:
                (root / relative).unlink(missing_ok=True)
            else:
                shutil.copy2(backup / relative, root / relative)
        for path, content in previous_reports.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('--activate', action='store_true', help='Replace artifacts from an existing validated stage; requires stopped service')
    args = parser.parse_args()
    result = (activate if args.activate else stage)(ROOT / args.candidate)
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'calibrated_heads': result['calibrated_heads']}))
