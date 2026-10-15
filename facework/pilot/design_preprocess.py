"""Design 5.2 eye-centre normalization pilot; not the legacy CNN training transform."""
import cv2,math
import numpy as np
from PIL import Image
from facework.morphometrics import EYE,BROW,MOUTH,OVAL,WINDOWS

def eye_normalization(points,target_eye_distance=128):
    p=np.asarray(points)[:,:2]
    right=(p[33]+p[133])/2;left=(p[263]+p[362])/2
    delta=left-right;distance=np.linalg.norm(delta)
    if distance<1e-6:raise ValueError('eye centres coincide')
    roll=math.degrees(math.atan2(delta[1],delta[0]));scale=target_eye_distance/distance
    M=cv2.getRotationMatrix2D(tuple((right+left)/2),roll,scale)
    q=np.c_[p,np.ones(len(p))]@M.T
    return q,M,{'target_eye_distance':target_eye_distance,'input_eye_distance':float(distance),'scale':float(scale),'roll':roll}

def design_crops(image,points):
    q,M,meta=eye_normalization(points);arr=np.asarray(image.convert('RGB'));fw=np.linalg.norm(q[234]-q[454]);out={}
    for part in ['eye_right','eye_left','brow_right','brow_left','nose','mouth','face']:
        kind=part.split('_')[0]
        if kind=='eye':center=q[EYE[part.split('_')[1]]['ring']].mean(0)
        elif kind=='brow':
            b=BROW[part.split('_')[1]];center=np.r_[q[b['upper']],q[b['lower']]].mean(0)+[0,.035*fw]
        elif kind=='nose':center=.4*q[168]+.6*q[2]
        elif kind=='mouth':center=q[MOUTH].mean(0)-[0,.04*fw]
        else:center=(q[OVAL].min(0)+q[OVAL].max(0))/2
        h,w=WINDOWS[kind];s=.5 if kind=='face' else 1
        S=np.array([[s,0,w/2-s*center[0]],[0,s,h/2-s*center[1]],[0,0,1]])
        T=(S@np.vstack([M,[0,0,1]]))[:2]
        crop=cv2.warpAffine(arr,T,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(127,127,127))
        if part.endswith('_left'):crop=crop[:,::-1]
        out[part]=Image.fromarray(np.ascontiguousarray(crop))
    return out,meta
