"""Transparent morphology measurements; no physical scale or fortune inference."""
import math, statistics
from collections import defaultdict

def finite_number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def observations(metrics,spec,quality_flags=()):
    rows=[]
    for feature in spec['features']:
        key=next(iter(feature['metric_keys']),None)
        value=metrics.get(key) if key else None
        status=('not_implemented' if feature['measurement_status']=='not_implemented'
                else 'missing_measurement' if not finite_number(value)
                else 'capture_review_required' if quality_flags else 'uncalibrated')
        rows.append({'id':feature['id'],'label_ko':feature['label_ko'],
                     'value':value if finite_number(value) else None,'unit':feature['unit'],
                     'status':status,'size_label':None,'formula':feature['formula'],
                     'limitation':feature['limitation']})
    return rows

def quantile(values,p):
    values=sorted(values)
    if not values or not 0<=p<=1:raise ValueError('invalid quantile input')
    pos=(len(values)-1)*p;lo=math.floor(pos);hi=math.ceil(pos)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

def pilot_reference(rows,metric_keys,cohort):
    """Training-only descriptive cutpoints, one median per subject, not deployed labels."""
    if not rows:raise ValueError('no rows')
    if any(r['split']!='train' for r in rows):raise ValueError('reference must contain train only')
    if any(r['condition']!='neutral_front' for r in rows):raise ValueError('reference must be neutral_front')
    seen=set()
    for r in rows:
        if not r['subject_id']:raise ValueError('subject ID required')
        if r['image_id'] in seen:raise ValueError('duplicate image ID')
        seen.add(r['image_id'])
    result={}
    for key in metric_keys:
        groups=defaultdict(list)
        for r in rows:
            v=r['metrics'].get(key)
            if finite_number(v):groups[r['subject_id']].append(v)
        values=[statistics.median(v) for v in groups.values()]
        if len(values)<4:raise ValueError(f'insufficient subjects for {key}')
        low,high=quantile(values,.25),quantile(values,.75)
        result[key]={'subjects':len(values),'p25':low,'median':statistics.median(values),'p75':high,
                     'range':[min(values),max(values)],'degenerate':low==high,
                     'boundary_error_margin':None,'threshold_ready':False}
    return {'version':'descriptive-pilot-v1','cohort':cohort,'input_images':len(rows),
            'subjects':len({r['subject_id'] for r in rows}),'subject_ids':sorted({r['subject_id'] for r in rows}),
            'quantile_method':'linear interpolation; one median per subject',
            'deployment_ready':False,'use':'distribution_review_only','features':result}
