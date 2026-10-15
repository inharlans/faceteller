"""Candidate visible-hair measurements; audit only, never overwrite old labels."""
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.release_preprocess import eye_affine
from facework.morphometrics import EYE,BROW


def measure(mask,points):
    q,affine=eye_affine(points)
    y,x=np.where(np.isin(mask,[2,3]))
    if len(x)<20:return {}
    hair=np.c_[x,y,np.ones(len(x))]@affine.T
    result={}
    for side,e in EYE.items():
        guide=q[BROW[side]['upper']+BROW[side]['lower']]
        width=float(np.linalg.norm(q[e['outer']]-q[e['inner']]))
        low,high=guide.min(0)-width*.3,guide.max(0)+width*.3
        selected=hair[((hair>=low)&(hair<=high)).all(1)]
        if len(selected)<20:continue
        left,right=np.quantile(selected[:,0],[.02,.98])
        if right-left<width*.3:continue
        # Central hair line, robust to isolated segmentation pixels at the edge.
        xx=np.linspace(left,right,25);step=(right-left)/24;yy=[]
        for value in xx:
            ys=selected[np.abs(selected[:,0]-value)<=step,1]
            yy.append(float(np.median(ys)) if len(ys)>=2 else np.nan)
        ok=np.isfinite(yy)
        if sum(ok)<20:continue
        yy=np.interp(xx,xx[ok],np.asarray(yy)[ok]);yy=gaussian_filter1d(yy,1)
        length=float(np.hypot(np.diff(xx),np.diff(yy)).sum()/width)
        # Decide inner end from the actual eye landmark, not a mask class index.
        inner_left=abs(left-q[e['inner'],0])<abs(right-q[e['inner'],0])
        inner_y,outer_y=(yy[0],yy[-1]) if inner_left else (yy[-1],yy[0])
        result[side]={'length_eye_ratio':length,'horizontal_eye_ratio':float((right-left)/width),
                      'tail_angle_deg':float(np.degrees(np.arctan2(inner_y-outer_y,right-left))),
                      'hair_pixels':len(selected),'columns_valid':int(sum(ok))}
    return result


def main():
    registry=json.loads((ROOT/'data/release/training_registry.json').read_text(encoding='utf-8'))
    output={'version':'visible-brow-mask-candidate-v1','status':'audit_candidate_not_deployed','subjects':{}}
    for count,(sid,subject) in enumerate(sorted(registry['subjects'].items()),1):
        folder=ROOT/'data/derived/release/ears';meta=json.loads((folder/f'{sid}.json').read_text(encoding='utf-8'))
        frames=[]
        with np.load(ROOT/'data/derived/kface/subjects'/f'{sid}.npz') as archive, (folder/f'{sid}.bin').open('rb') as binary:
            index={str(k):i for i,k in enumerate(archive['keys'])}
            for frame in meta['parsing_frames']:
                if frame['camera']!='C7' or frame['status']!='parsed':continue
                i=index.get(frame['capture']+'|core')
                if i is None:continue
                raw=frame['mask'];binary.seek(raw['offset'])
                with Image.open(io.BytesIO(binary.read(raw['length']))) as im:mask=np.asarray(im).copy()
                frames.append({'light':frame['light'],'sides':measure(mask,archive['landmarks'][i])})
        values={}
        for metric in ['length_eye_ratio','horizontal_eye_ratio','tail_angle_deg']:
            all_values=[np.mean([f['sides'][s][metric] for s in ('left','right')]) for f in frames if len(f['sides'])==2]
            values[metric]={'n':len(all_values),'median':float(np.median(all_values)) if all_values else None,
                            'sd':float(np.std(all_values,ddof=1)) if len(all_values)>1 else None}
        output['subjects'][sid]={'split':subject['split'],'frames':frames,'measurements':values}
        if count%100==0:print('BROW_MASK_AUDIT',count,flush=True)
    target=ROOT/'data/relabel-20261015/brow_mask_audit.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for metric in ['length_eye_ratio','horizontal_eye_ratio','tail_angle_deg']:
        v=[d['measurements'][metric]['median'] for d in output['subjects'].values() if d['split']=='train' and d['measurements'][metric]['n']>=2]
        print(metric,len(v),np.quantile(v,[0,.1,.25,.5,.75,.9,1]).tolist())


if __name__=='__main__':main()
