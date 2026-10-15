"""Audit CNN/depth influence from actual frozen-service responses, CPU only."""
import hashlib
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.pilot_contract import Decision,Passage,PassageSelector
from scripts.release.setup_database import canonical_json


def summary(values):
    if not values:return {'n':0}
    a=np.asarray(values,float)
    if not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid score')
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),
            'ge_099':float((a>=.99).mean()),'zero_count':int((a==0).sum()),
            'mean_negative_log_clipped_1e15':float(-np.log(np.maximum(a,1e-15)).mean())}


def audit(rows,passages):
    scores=defaultdict(list);by_feature=defaultdict(list);counts=Counter({k:0 for k in ('participants','request_errors','selected_mixed','selected_depth_only','selected_cnn_only','order_changed_by_depth','selected_set_changed_by_depth','people_with_passages')});gaps=[];wins=[]
    for row in rows:
        counts['participants']+=1
        if row['status']=='error':counts['request_errors']+=1;continue
        decisions=[];accepted=defaultdict(list)
        for raw in row['items']:
            kind=raw['kind'];s=raw['confidence'] if kind=='classification' else raw['measurement_score']
            if s is not None:scores[f'{kind}_all'].append(s);by_feature[raw['item_id']].append(s)
            if raw['status']=='CONFIRMED':
                if s is None:raise ValueError('Confirmed score missing')
                accepted[kind].append(s);scores[f'{kind}_confirmed'].append(s)
            decisions.append(Decision(item_id=raw['item_id'],region=raw['region'],status=raw['status'],
                label_id=raw['label_id'],label_name=None,confidence=raw['confidence'],measurement=None,
                measurement_score=raw['measurement_score'],basis='saved response',
                hold_reasons=tuple(raw['hold_reasons']),kind=kind))
        cnn,depth=accepted['classification'],accepted['depth']
        if cnn and depth:
            gaps.append(float(np.mean(depth)-np.mean(cnn)))
            wins.append(float((np.asarray(depth)[:,None]>np.asarray(cnn)[None,:]).mean()))
        mixed=PassageSelector('include_measurement_scores').select(decisions,passages)
        old=PassageSelector('classification_only').select(decisions,passages)
        actual=row['passages']
        if [p['passage_id'] for p in actual]!=[p['passage_id'] for p in mixed]:raise ValueError('Actual ordering differs from frozen corpus scoring')
        for a,b in zip(actual,mixed):
            if not math.isclose(a['G'],b['G'],abs_tol=1e-12,rel_tol=1e-12):raise ValueError('Actual G mismatch')
            kinds={c['kind'] for c in b['score_components']}
            counts['selected_mixed' if len(kinds)==2 else 'selected_depth_only' if 'depth' in kinds else 'selected_cnn_only']+=1
        old_ids=[p['passage_id'] for p in old];mixed_ids=[p['passage_id'] for p in mixed]
        counts['order_changed_by_depth']+=old_ids!=mixed_ids
        counts['selected_set_changed_by_depth']+=set(old_ids)!=set(mixed_ids)
        counts['people_with_passages']+=bool(mixed_ids)
    return {'counts':dict(counts),'score_distributions':{k:summary(v) for k,v in scores.items()},
            'features':{k:summary(v) for k,v in by_feature.items()},
            'paired_people':len(gaps),'mean_depth_minus_cnn_confirmed':float(np.mean(gaps)) if gaps else None,
            'mean_pairwise_depth_higher_rate':float(np.mean(wins)) if wins else None}


def main():
    out=ROOT/'outputs/release';response_path=out/'service-verification.json'
    response=json.loads(response_path.read_text(encoding='utf-8'))
    corpus=json.loads((ROOT/'data/release/literature/reference_corpus_v1.json').read_text(encoding='utf-8'))
    corpus_hash=hashlib.sha256(canonical_json(corpus).encode()).hexdigest()
    if not response.get('complete') or not response.get('all400') or len(response['participants'])!=400:raise ValueError('Actual complete400 verification required')
    if response['corpus_sha256']!=corpus_hash:raise ValueError('Corpus changed after verification')
    passages=[Passage(p['passage_id'],frozenset(p['required_labels']),p['original_text'],p['translation'],
                      p['commentary'],p['source'],p['literature_priority'],p['passage_order'])
              for p in corpus['passages'] if p['service_eligible']]
    groups={split:[r for r in response['participants'] if r['split']==split]
            for split in ('train','selection','calibration','test')}
    report={'version':'actual-release-score-influence-v1','service_report_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),
            'corpus_sha256':corpus_hash,'all400':audit(response['participants'],passages),
            'splits':{k:audit(v,passages) for k,v in groups.items()},
            'interpretation':[
                'Each required feature has equal weight in log G. Feature count is the geometric-mean root, not an extra reward.',
                'Equal mathematical weight does not prove equal score distributions or calibrated correctness.',
                'CNN confidence is validation-temperature adjusted. Depth score is interval membership from repeat variation.',
                'Saturation and rank-change statistics describe observed influence, not fairness or ground-truth validity.',
                'Comparison excluding depth replays an earlier scoring convention; it is a diagnostic, not the deployed mode.',
                'No weights, thresholds or prompts are selected from this test or all400 diagnostic.'],
            'personal_identifiers_in_report':False,'human_ground_truth_count':0}
    (out/'score-influence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report['all400']['counts'],ensure_ascii=False))


if __name__=='__main__':main()
