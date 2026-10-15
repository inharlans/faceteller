"""Conservative local VLM/CV agreement; never changes original weak targets."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.release.annotate_marks_local import (
    DEFAULT_OUT, LIGHTS, PROMPT, PREPROCESS, configuration, digest, file_hash,
    load, parse_response, read_inputs, save, selected_subjects, validate_record,
)

POLICY = {
    'version': 'visible-upper-skin-vlm-cv-agreement-v1',
    'positive': 'at least two same-location VLM positives AND matching quality-passed CV stable candidate in those lights',
    'negative': 'three VLM clear_absent AND three quality-passed CV coverage>=0.65 AND no stable or unstable candidates or rejected components',
    'conflict': 'any present/absent contradiction or differing positive locations => unknown',
    'spatial_radius_eye_units': .04,
    'regions': 'forehead y<min(brow107,brow336)-0.05; cheeks y>=0.12,y<=subnasale2-0.04,abs(x)>=0.25; eye-centre coordinates',
    'excluded': 'nose/chin/other lack CV regional agreement; unknown',
    'meaning': 'algorithm-generated visible upper-skin mark target, not medical mole truth',
    'human_review_count': 0,
}
NEGATIVE_POLICIES = ('strict', 'vlm_three_absent_no_stable')


def policy_for(negative_policy):
    if negative_policy not in NEGATIVE_POLICIES:
        raise ValueError('UNKNOWN_NEGATIVE_POLICY')
    policy = {**POLICY, 'negative_policy': negative_policy}
    if negative_policy == 'vlm_three_absent_no_stable':
        policy.update(version='visible-upper-skin-vlm-cv-three-absent-v1',
            negative='three VLM clear_absent AND three CV quality-passed coverage>=0.65 AND no stable candidate; unstable/rejected texture allowed',
            limitations='predeclared weak AI agreement only; absence may be false due correlated VLM omissions; no independent human validation')
    return policy


def negative_eligible(cv, negative_policy='strict'):
    """CV-only upper bound, NEVER a label: VLM agreement is still required."""
    policy_for(negative_policy)
    try:
        frames = cv['frames']
        if (cv.get('version') != 'visible-upper-skin-dark-marks-v1' or
            len(frames) != 3 or [f.get('light') for f in frames] != list(LIGHTS) or
            not isinstance(cv['stable_marks'], list) or cv['stable_marks']):
            return False
        if not all(f.get('quality_passed') is True and isinstance(f['coverage'], (int, float)) and
                   math.isfinite(f['coverage']) and .65 <= f['coverage'] <= 1 and
                   isinstance(f['candidates'], list) and type(f['rejected_components']) is int and
                   f['rejected_components'] >= 0 for f in frames):
            return False
        return negative_policy != 'strict' or all(not f['candidates'] and f['rejected_components'] == 0 for f in frames)
    except (ValueError, TypeError, KeyError):
        return False


def eligibility_report(original):
    policies = {}
    for name in NEGATIVE_POLICIES:
        groups = {}
        for split, ids in original['groups'].items():
            eligible = [sid for sid in ids if negative_eligible(original['metrics'][sid]['pigmented_marks'], name)]
            groups[split] = {'upper_bound': len(eligible), 'eligible_subject_ids': eligible}
        policies[name] = {'policy': policy_for(name), 'policy_sha256': digest(policy_for(name)),
                          'groups': groups, 'upper_bound_total': sum(g['upper_bound'] for g in groups.values())}
    return {'version': 'local-mark-negative-eligibility-preflight-v1', 'policies': policies,
            'actual_vlm_labels': 0, 'is_label_artifact': False, 'model_loaded': False, 'gpu_used': False,
            'interpretation': 'CV-only ceilings before VLM; actual support may be much lower, including zero. No human-validated truth.'}


def candidate_location(mark, bounds):
    x, y = mark['x_eye'], mark['y_eye']
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in (x, y, *bounds.values())):
        raise ValueError('INVALID_COORDINATES')
    if y < bounds['forehead_max_y']:
        return 'forehead'
    if bounds['cheek_min_y'] <= y <= bounds['cheek_max_y']:
        if x <= -bounds['cheek_min_abs_x']:
            return 'image_left_cheek'
        if x >= bounds['cheek_min_abs_x']:
            return 'image_right_cheek'
    return None


def merge_subject(response, cv, source, negative_policy='strict'):
    """Pure decision function. Caller verifies participant/hash provenance first."""
    policy_for(negative_policy)
    result = {'label': None, 'reason': 'NO_VALID_ANNOTATION', 'matched_marks': [],
              'negative_policy': negative_policy,
              'source': 'local_single_vlm_plus_algorithm_agreement', 'confidence': None,
              'clinical_truth_validated': False, 'historical_mole_mapping': 'unmapped'}
    if response is None:
        return result
    try:
        items = parse_response(json.dumps(response))['images']
        frames = cv['frames']
        if len(frames) != 3 or [f.get('light') for f in frames] != list(LIGHTS):
            raise ValueError('INVALID_CV_LIGHTS')
        if cv.get('version') != 'visible-upper-skin-dark-marks-v1':
            raise ValueError('UNSUPPORTED_CV_VERSION')
        if len(source['frames']) != 3 or [f['light'] for f in source['frames']] != list(LIGHTS):
            raise ValueError('INVALID_SOURCE_LIGHTS')
        for key in ('capture', 'png_sha256', 'pixels_sha256'):
            if len({f[key] for f in source['frames']}) != 3:
                raise ValueError('REPEATED_SOURCE')
        stable = cv['stable_marks']
        positives = [i for i in items if i['status'] == 'clear_present']
        negatives = [i for i in items if i['status'] == 'clear_absent']
        locations = {i['location'] for i in positives}
        if (positives and negatives) or len(locations) > 1:
            result['reason'] = 'VLM_CONTRADICTION'
            return result
        clear = [f.get('quality_passed') is True and isinstance(f.get('coverage'), (int, float))
                 and math.isfinite(f['coverage']) and .65 <= f['coverage'] <= 1 for f in frames]
        if len(negatives) == 3:
            if negative_eligible(cv, negative_policy):
                result.update(label='absent', reason='THREE_CLEAR_ABSENT_AND_CLEAN_SKIN' if negative_policy == 'strict'
                              else 'THREE_VLM_ABSENT_CLEAR_SKIN_NO_STABLE_CV_WEAK')
            elif stable or (negative_policy == 'strict' and any(f['candidates'] or f['rejected_components'] != 0 for f in frames)):
                result['reason'] = 'CV_CANDIDATES_OR_TEXTURE_CONFLICT'
            else:
                result['reason'] = 'INSUFFICIENT_CLEAR_SKIN'
            return result
        if len(positives) < 2:
            result['reason'] = 'INSUFFICIENT_VLM_AGREEMENT'
            return result
        location = positives[0]['location']
        if location not in ('forehead', 'image_left_cheek', 'image_right_cheek'):
            result['reason'] = 'OUTSIDE_SHARED_OBSERVABLE_SCOPE'
            return result
        for mark in stable:
            if mark.get('repeats') not in (2, 3):
                continue
            matched = []
            for item, frame, src, usable in zip(items, frames, source['frames'], clear):
                if not usable or item['status'] != 'clear_present' or item['location'] != location:
                    continue
                for candidate in frame['candidates']:
                    if (candidate_location(candidate, src['region_bounds']) == location and
                        math.hypot(candidate['x_eye']-mark['x_eye'], candidate['y_eye']-mark['y_eye']) <= .04):
                        matched.append(frame['light'])
                        break
            if len(matched) >= 2:
                result['matched_marks'].append({**mark, 'location': location, 'lights': matched})
        if result['matched_marks']:
            result.update(label='present', reason='REPEATED_REGIONAL_VLM_CV_AGREEMENT')
        else:
            result['reason'] = 'NO_MATCHING_CV_REPEAT_IN_POSITIVE_LIGHTS'
    except (ValueError, TypeError, KeyError, OverflowError):
        result.update(label=None, reason='INVALID_EVIDENCE', matched_marks=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--negative-policy', choices=NEGATIVE_POLICIES, default='strict')
    parser.add_argument('--preflight-only', action='store_true', help='report both CV eligibility ceilings, no VLM labels/model')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    policy = policy_for(args.negative_policy)
    if args.output is None:
        if args.preflight_only:
            args.output = ROOT/'outputs/release-local-marks/negative-eligibility-preflight-v1.json'
        else:
            filename = ('merged_mark_labels.json' if args.negative_policy == 'strict' else
                        'merged_mark_labels_vlm_agreement.json')
            args.output = ROOT/'data/release/local_marks'/filename
    original_path = ROOT/'data/release/extended_labels.json'
    if args.output.resolve() == original_path.resolve():
        raise ValueError('ORIGINAL_LABELS_MUST_NOT_BE_OVERWRITTEN')
    original_hash = file_hash(original_path)
    original = load(original_path)
    protocol_path = ROOT/'models/selection-20261014/protocol.json'
    protocol_hash = file_hash(protocol_path)
    protocol = load(protocol_path)
    if original['protocol_sha256'] != protocol_hash or original['groups'] != protocol['groups']:
        raise ValueError('CV_PROTOCOL_MISMATCH')
    if args.preflight_only:
        report = eligibility_report(original)
        report['source_hashes'] = {'original_extended_labels': original_hash, 'protocol': protocol_hash,
                                   'merge_implementation': file_hash(Path(__file__))}
        save(args.output, report)
        print(json.dumps({'output': str(args.output), 'upper_bounds': {
            name: value['upper_bound_total'] for name, value in report['policies'].items()},
            'actual_vlm_labels': 0}))
        return
    if args.output.exists() and load(args.output).get('policy_sha256') != digest(policy):
        raise ValueError('EXISTING_OUTPUT_DIFFERENT_POLICY_USE_SEPARATE_ARTIFACT')
    config = load(args.annotations/'manifest.json')
    if (config != configuration(config['model'], protocol_hash, config['runtime_versions']) or
            config.get('prompt') != PROMPT or config['runtime_versions'].get('transformers') != '4.57.6'):
        raise ValueError('ANNOTATION_CONFIGURATION_MISMATCH')
    subjects, metrics, record_hashes = {}, {}, {}
    for sid, split in selected_subjects(protocol['groups'], False):
        path = args.annotations/'records'/f'{sid}.json'
        decision = merge_subject(None, {}, {}, args.negative_policy)
        if path.exists():
            record = load(path)
            validate_record(record, sid, split, config)
            record_hashes[sid] = file_hash(path)
            if record['status'] == 'ok':
                images, source = read_inputs(sid, split, protocol_hash)
                try:
                    validate_record(record, sid, split, config, source)
                    decision = merge_subject(record['response'], original['metrics'][sid]['pigmented_marks'], source, args.negative_policy)
                finally:
                    for image in images:
                        image.close()
            else:
                decision['reason'] = 'ANNOTATION_ERROR'
        subjects[sid] = {'pigmented_marks': decision['label']}
        metrics[sid] = {**decision, 'original_strict_label': original['subjects'][sid]['pigmented_marks'],
                        'cv': original['metrics'][sid]['pigmented_marks']}
    support = {}
    for split, ids in protocol['groups'].items():
        counts = Counter(subjects[s]['pigmented_marks'] for s in ids)
        support[split] = {k: counts[k] for k in ('present', 'absent')}
        support[split]['unknown'] = counts[None]
    artifact = {'version': policy['version'], 'feature': 'pigmented_marks', 'subjects': subjects,
                'metrics': metrics, 'groups': protocol['groups'], 'class_support': support,
                'policy': policy, 'policy_sha256': digest(policy), 'config_sha256': digest(config),
                'source_hashes': {'original_extended_labels': original_hash,
                                  'protocol': protocol_hash, 'records': record_hashes,
                                  'merge_implementation': file_hash(Path(__file__))},
                'preprocess_version': PREPROCESS, 'human_review_count': 0,
                'annotators': 1, 'lighting_repeats_are_independent_raters': False,
                'training_ready': False,
                'training_readiness_note': 'Main must explicitly assess class support, annotation feasibility and weak-label limitations before registry integration.'}
    if file_hash(original_path) != original_hash:
        raise ValueError('ORIGINAL_LABELS_CHANGED_DURING_MERGE')
    save(args.output, artifact)
    print(json.dumps({'output': str(args.output), 'class_support': support, 'records': len(record_hashes)}))


if __name__ == '__main__':
    main()
