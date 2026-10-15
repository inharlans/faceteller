"""Reproducible 400-person label/hold audit, without reading user uploads."""
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def audit(root=ROOT):
    paths = {'registry': 'data/release/training_registry.json',
             'deployment': 'models/release/deployment.json',
             'service': 'outputs/release/service-verification.json'}
    docs = {k: json.loads((root / p).read_text(encoding='utf-8')) for k, p in paths.items()}
    registry, deployment, service = (docs[k] for k in ('registry', 'deployment', 'service'))
    people = service['participants']
    features = {}
    for key, spec in {**registry['features'], **registry['depth_features']}.items():
        observed = [i for person in people for i in person['items'] if i['item_id'] == key]
        held = [i for i in observed if i['status'] == 'HELD']
        scores = [i['confidence'] for i in observed if i['status'] == 'CONFIRMED' and i['confidence'] is not None]
        soft_ties, soft_conflicts = [], []
        for sid, person in registry['subjects'].items():
            vector = person['soft_targets'].get(key)
            label = person['labels'].get(key)
            if vector and label in spec['classes']:
                top = max(vector)
                if vector[spec['classes'].index(label)] < top - 1e-8:
                    soft_conflicts.append(sid)
                if sum(abs(x-top)<1e-8 for x in vector) > 1:
                    soft_ties.append(sid)
        cal = deployment['features'].get(key, {})
        features[key] = {
            'definition': spec['ko'], 'classes': spec['classes'],
            'class_names': spec['classes_ko'], 'metric': spec.get('metric'),
            'label_support': {split: data[key] for split, data in registry['class_support'].items() if key in data},
            'effective_train_support': cal.get('train_support'), 'calibration_support': cal.get('support'),
            'temperature': cal.get('temperature'), 'threshold': cal.get('threshold'),
            'calibration_status': cal.get('status'), 'soft_tie_participants': soft_ties,
            'soft_hard_strict_conflicts': soft_conflicts,
            'observed_participants': len(observed), 'held_count': len(held),
            'held_reasons': dict(collections.Counter(reason for i in held for reason in i['hold_reasons'])),
            'confirmed_labels': dict(collections.Counter(i['label_id'] for i in observed if i['status']=='CONFIRMED')),
            'confirmed_confidence': {'n': len(scores), 'median': statistics.median(scores) if scores else None,
                                    'exact_one': sum(s==1 for s in scores), 'rounds_to_100_integer': sum(s>=.995 for s in scores)},
            'threshold_definition': spec.get('thresholds'),
        }
    return {'version': 'all-label-audit-20261015-v1', 'participants': len(people),
            'cnn_features': len(registry['features']), 'depth_features': len(registry['depth_features']),
            'class_labels': sum(len(f['classes']) for f in features.values()),
            'source_hashes': {paths[k]: hashlib.sha256((root/paths[k]).read_bytes()).hexdigest() for k in paths},
            'user_upload_history_available': False, 'independent_human_review': False,
            'features': features,
            'rank_one_counts': dict(collections.Counter(p['passages'][0]['passage_id'] for p in people if p['passages'])),
            'scope': 'Existing controlled K-FACE experiment records; not private user test history or proof of human-label accuracy'}


if __name__ == '__main__':
    result = audit()
    target = ROOT/'outputs/release/main-all-label-audit-20261015.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('participants','cnn_features','depth_features','class_labels','rank_one_counts')},ensure_ascii=False))
