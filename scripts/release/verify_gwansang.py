"""Check grouped retrieval against the supplied archive, then the real loopback API.

The optional cached replay checks ranking only, not fresh inference or accuracy.
Missing cached brow measurements exclude S12. Live requests check that gate with
the actual measured ratio. Saved output contains aggregates, no faces or IDs.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from facework.pilot_contract import Decision, Passage, build_result
from facework.release_gwansang import GwansangCatalog
from facework.release_repository import ReleaseRepository
from scripts.release.verify_service import image_files, local_url, request_analysis, verify_selection, overlay_summary


def oracle(document, items, measurements):
    labels = {i['label_id'] for i in items if i['status'] == 'CONFIRMED'}
    candidates = []
    for shape in document['data']['생김새']:
        key = shape['생김새코드']
        b = document['bindings']['shapes'][key]
        if not b['enabled'] or shape['대상성별'] != '공통':
            continue
        if any(label not in labels for label in b['required_labels']):
            continue
        valid = True
        for gate in b.get('measurement_gates', []):
            actual = measurements.get(gate['feature'], {})
            value = actual.get('value')
            if (actual.get('source') != gate['source'] or actual.get('unit') != gate['unit']
                    or type(value) not in (int,float) or not math.isfinite(value) or value < gate['min']):
                valid = False
        if not valid:
            continue
        source = next(s for s in shape['출처'] if s['출처id'] == b['presentation']['representative_source_id'] and s['적용성별'] == '공통')
        candidates.append(Passage('GW-'+key, frozenset(b['required_labels']), source['원문'], source['번역'], source['해설'],
                                  {'shape':shape, 'source':source, 'binding':b}, 0, int(key[1:])))
    return candidates


def verify(result, catalog, document):
    features = {f['item_id']:f for f in result.get('face_overlay', {}).get('features', [])}
    candidates = oracle(document, result['items'], features)
    summary = verify_selection(result['items'], candidates, result['passages'])
    by_id = {p.passage_id:p for p in candidates}
    for p in result['passages']:
        expected = by_id[p['passage_id']]
        shape, source, binding = (expected.source[k] for k in ('shape','source','binding'))
        assert p['original_text'] == source['원문']
        assert p['translation'] == source['번역']
        assert p['commentary'] == source['해설']
        assert p['source']['source_id'] == source['출처id']
        assert p['source']['url'] == source['github']
        assert p['source']['title'] == source['책_한글']
        assert p['source']['chapter'] == source['편']
        assert p['reading']['title'] == binding['presentation']['title']
        assert p['reading']['meaning'] == binding['presentation']['meaning']
        assert p['reading']['meaning_basis'] == 'representative'
        assert p['source']['reading_basis'] == 'reviewed_representative'
        assert p['reading']['valence'] == 'favorable'
    assert result['named_types'] == []
    assert result['ranking_policy']['grouping'] == 'one_per_shape_before_top_five'
    assert len({p['source']['shape_id'] for p in result['passages']}) == len(result['passages'])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', type=local_url, default='http://127.0.0.1:8767')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--live', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.live <= 400:
        parser.error('--live must be 1..400')
    if args.output.exists():
        parser.error('Choose a new output file to preserve previous evidence')
    catalog = GwansangCatalog.load(ROOT)
    repository = ReleaseRepository()
    with repository.snapshot() as snapshot:
        catalog.verify_snapshot(snapshot)
        stored, _ = snapshot.gwansang_catalog()
    report = {'catalog':catalog.identity, 'photos_saved':False, 'independent_human_labels':0,
              'scope':'retrieval, representative sources, AND/G/ranking and integration; not model accuracy'}
    rows = []
    if args.replay:
        rows = json.loads(args.replay.read_bytes())['participants']
        histogram = Counter(); counts = Counter()
        with repository.snapshot() as snapshot:
            repo = catalog.repository(snapshot)
            for row in rows:
                decisions = [Decision(i['item_id'], i['region'], i['status'], i['label_id'], i['label_id'],
                                      i['confidence'], None, i['measurement_score'], 'cached replay',
                                      tuple(i['hold_reasons']), i['kind']) for i in row['items']]
                result = catalog.apply(build_result(decisions, repo))
                verify(result, catalog, stored)
                histogram[len(result['passages'])] += 1
                counts.update(p['passage_id'] for p in result['passages'])
        report['cached_replay'] = {'participants':len(rows), 'passed':True, 'card_count_histogram':dict(histogram),
                                   'selected_shapes':dict(counts), 'measurement_limitation':'cached report has no brow ratio; S12 excluded, not fabricated'}
    # Include confirmed long brows and both high-depth features when available.
    priority = []
    for labels in ({'brow_length:long'}, {'nose_projection:high','nose_bridge:high'}, {'brow_shape:straight'}):
        matches = [r['subject_id'] for r in rows if labels <= {i['label_id'] for i in r['items'] if i['status']=='CONFIRMED'}]
        priority.extend(matches[:2])
    registry = json.loads((ROOT/'data/release/training_registry.json').read_bytes())
    subjects = list(dict.fromkeys(priority+['19062431']+sorted(registry['subjects'])))[:args.live]
    from facework.kface import Store
    store = Store(2)
    live = []
    try:
        with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
            health = client.get('/api/v1/health'); health.raise_for_status()
            for k,v in catalog.identity.items():
                assert health.json()[k] == v, k
            for subject in subjects:
                code, terminal, seconds, _ = request_analysis(client, image_files(store, subject, None))
                assert code == 200 and terminal['type'] == 'result', terminal.get('code')
                result = terminal['data']
                for k,v in catalog.identity.items():
                    assert result['runtime'][k] == v
                verify(result, catalog, stored)
                overlay = overlay_summary(result['face_overlay'])
                live.append({'passed':True,'seconds':seconds,'cards':len(result['passages']),
                             'shape_ids':[p['source']['shape_id'] for p in result['passages']],
                             'overlay_features':overlay['overlay_feature_count']})
                print(f'live {len(live)}/{len(subjects)} passed ({seconds:.2f}s)', flush=True)
            report['service'] = {k:health.json()[k] for k in ('deployment_sha256','depth_config_sha256','registry_sha256','repository')}
    finally:
        store.close()
    report['live'] = {'participants':len(live),'passed':True,'requests':live}
    report['complete'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'complete':True,'live':len(live),'cached_replay':len(rows)},ensure_ascii=False))


if __name__ == '__main__':
    main()
