"""Validation-only reject-option selection, with an explicit operational cost.

The costs express a service preference, not a measured probability or clinical
claim: one wrong confirmation costs as much as four withheld judgments.
"""
import numpy as np

POLICY = {
    'version': 'validation-reject-cost-v1', 'split': 'calibration',
    'wrong_confirmation_cost': 1.0, 'withheld_cost': 0.25,
    'initial_threshold': 0.6,
    'binary_floor': 0.6, 'multiclass_floor': 0.5,
    'candidates': [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0],
    'tie_break': 'lowest loss, most accepted, closest to initial0.6, lowest threshold',
    'scope': 'same calibration34 as temperature fitting; development criterion, not independent accuracy',
    'test_used': False,
}


def select_threshold(probabilities, truth, *, calibrated, train_support, validation_ids, test_ids):
    if set(validation_ids) & set(test_ids):
        raise ValueError('Threshold fitting cannot use test participants')
    p=np.asarray(probabilities,dtype=float); y=np.asarray(truth,dtype=int)
    if p.ndim!=2 or len(p)!=len(y) or len(p)!=len(validation_ids) or p.shape[1]<2:
        raise ValueError('Invalid threshold calibration dimensions')
    if (not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1)
            or len(train_support)!=p.shape[1]):
        raise ValueError('Invalid threshold probabilities or support')
    valid=y>=0
    if (y[valid]>=p.shape[1]).any():raise ValueError('Invalid threshold target class')
    base={'threshold':POLICY['initial_threshold'],'policy_version':POLICY['version'],
          'source_split':'calibration','test_used':False,'n_valid':int(valid.sum())}
    if not calibrated:
        return {**base,'status':'disabled_uncalibrated','candidates':[]}
    if not valid.any():
        return {**base,'status':'no_valid_calibration_labels','candidates':[]}
    p=p[valid]; y=y[valid]; predicted=p.argmax(1); confidence=p.max(1)
    trained=np.asarray(train_support)[predicted]>0
    floor=POLICY['binary_floor'] if p.shape[1]==2 else POLICY['multiclass_floor']
    candidates=[]
    for threshold in POLICY['candidates']:
        if threshold<floor:continue
        accepted=(confidence>=threshold)&trained
        errors=int(((predicted!=y)&accepted).sum()); count=int(accepted.sum()); held=len(y)-count
        candidates.append({'threshold':threshold,'accepted':count,'withheld':held,'wrong_confirmed':errors,
                           'coverage':count/len(y),'accepted_error_rate':errors/count if count else None,
                           'loss':(errors*POLICY['wrong_confirmation_cost']+held*POLICY['withheld_cost'])/len(y)})
    selected=min(candidates,key=lambda c:(c['loss'],-c['accepted'],abs(c['threshold']-.6),c['threshold']))
    return {**base,'threshold':selected['threshold'],'status':'selected_on_calibration',
            'selected':selected,'candidates':candidates}
