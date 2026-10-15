"""Verify all 400 cached frontal coordinate sets; save counts, never face paths."""
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.release_overlay import face_overlay
from scripts.release.prepare_landmark_labels import measurements_from_points


def main():
    registry=json.loads((ROOT/'data/release/training_registry.json').read_text(encoding='utf-8'))
    counts=Counter();issues=[];maximum_error=0.;splits=Counter()
    for sid,subject in registry['subjects'].items():
        meta=json.loads((ROOT/'data/derived/release/ears'/f'{sid}.json').read_text(encoding='utf-8'))
        frame=next(f for f in meta['parsing_frames'] if f['capture']=='S001|L1|E01|C7')
        w,h=frame['mask']['size']
        with np.load(ROOT/'data/derived/kface/subjects'/f'{sid}.npz') as a:
            index=list(a['keys']).index('S001|L1|E01|C7|core');p=a['landmarks'][index]
        overlay=face_overlay(p,w,h);metrics=measurements_from_points(p)
        splits[subject['split']]+=1
        if len(overlay['features'])!=19:issues.append({'subject_id':sid,'reason':'missing_features'})
        for f in overlay['features']:
            counts[f['item_id']]+=1
            value=metrics['brow_outer_inner_tilt' if f['item_id']=='brow_tail' else f['item_id']]
            maximum_error=max(maximum_error,abs(value-f['value']))
            if any(x<0 or x>w or y<0 or y>h for path in f['paths'] for x,y in path):
                issues.append({'subject_id':sid,'feature':f['item_id'],'reason':'outside_photo'})
        json.dumps(overlay,allow_nan=False)
    report={'pass':not issues and maximum_error<1e-10,'participants':len(registry['subjects']),
            'splits':dict(splits),'feature_counts':dict(counts),'maximum_value_difference':maximum_error,
            'issues':issues,'photos_or_coordinates_saved':False,
            'scope':'cached actual frontal geometry and numeric parity, not CNN accuracy or a new landmark detection'}
    (ROOT/'outputs/release/overlay-verification-20261015.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
