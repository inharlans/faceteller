"""Measured 2D paths in the EXIF-oriented front photo, for optional visual QA.

These are measurement locations, not CNN attention maps or 3D projections.
Nothing is written to disk. A path is omitted when its points are invalid.
"""
import numpy as np
from facework.morphometrics import EYE, BROW, landmark_metrics
from facework.release_preprocess import eye_affine


def face_overlay(points, width, height, *, visible_brows=None):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]<2 or len(p)<468 or not np.isfinite(p).all():
        return {'width':width,'height':height,'features':[]}
    p=p[:,:2]
    try:
        q,affine=eye_affine(p)
        # Diagnostics must not break the result when a measurement is degenerate.
        denominators=[(234,454),(10,152),(61,291),(168,2),(33,133),(263,362)]
        if any(np.linalg.norm(q[a]-q[b])<1e-6 for a,b in denominators):
            return {'width':int(width),'height':int(height),'features':[]}
        metrics=landmark_metrics(q)
    except ValueError:
        return {'width':int(width),'height':int(height),'features':[]}
    slopes=[]
    for b in BROW.values():
        outer=(q[b['upper'][0]]+q[b['lower'][0]])/2
        inner=(q[b['upper'][-1]]+q[b['lower'][-1]])/2
        slopes.append(np.degrees(np.arctan2(inner[1]-outer[1],abs(outer[0]-inner[0]))))
    metrics['brow_tail']=float(np.mean(slopes))
    segment=lambda a,b:np.array([p[a],p[b]])
    face_width=segment(234,454);face_height=segment(10,152)
    eye_widths=[segment(e['inner'],e['outer']) for e in EYE.values()]
    eye_heights=[segment(e['top'],e['bottom']) for e in EYE.values()]
    paths={
        'eye_size':('눈 가로 길이',eye_widths+[face_width]),
        'eye_roundness':('눈의 가로·세로',eye_widths+eye_heights),
        'eye_tilt':('눈꼬리 기울기',eye_widths),
        'eye_asymmetry':('좌우 눈의 가로·세로',eye_widths+eye_heights),
        'brow_length':('눈썹 기준선',[(p[b['upper']]+p[b['lower']])/2 for b in BROW.values()]+eye_widths),
        'brow_tail':('눈썹 앞머리와 꼬리',[np.array([(p[b['upper'][i]]+p[b['lower'][i]])/2 for i in (0,-1)]) for b in BROW.values()]),
        'brow_eye_gap':('눈과 눈썹 사이',list(eye_widths)),
        'brow_gap':('눈썹 사이 간격',[np.array([(p[107]+p[55])/2,(p[336]+p[285])/2]),face_width]),
        'nose_width':('코 너비',[segment(98,327),face_width]),
        'nose_length':('코 길이',[segment(168,2),face_height]),
        'mouth_width':('입 너비',[segment(61,291),face_width]),
        'lip_thickness':('윗입술·아랫입술 두께',[segment(0,13),segment(14,17),segment(61,291)]),
        'lip_balance':('윗입술과 아랫입술 비율',[segment(0,13),segment(14,17)]),
        'mouth_corner':('입꼬리 위치',[segment(61,291)]),
        'face_length':('얼굴 세로·가로',[face_height,face_width]),
        'jaw_width':('턱 너비',[segment(172,397),face_width]),
        'eye_spacing':('눈 사이 간격',[segment(133,362)]+eye_widths),
        'philtrum_length':('인중 길이',[segment(2,0),face_height]),
        'lower_face_ratio':('얼굴 가운데와 아래 길이',[segment(168,2),segment(2,152)]),
    }
    for side,e in EYE.items():
        brow=BROW[side]['lower'];index=brow[int(np.argmax(q[brow,1]))]
        # The formula uses vertical separation after roll alignment, not the
        # diagonal distance between the two landmarks.
        guide=np.array([q[e['top']], [q[e['top'],0],q[index,1]]])
        inverse=np.linalg.inv(np.vstack([affine,[0,0,1]]))
        paths['brow_eye_gap'][1].append((np.c_[guide,np.ones(2)]@inverse.T)[:,:2])
    centre_y=(q[13,1]+q[14,1])/2
    for corner in (61,291):
        guide=np.array([q[corner],[q[corner,0],centre_y]])
        paths['mouth_corner'][1].append((np.c_[guide,np.ones(2)]@inverse.T)[:,:2])
    output=[]
    for key,(label,segments) in paths.items():
        source='landmark'
        if visible_brows is not None and key in ('brow_length','brow_tail'):
            selected=[]
            for brow in visible_brows.values():
                curve=np.asarray(brow.get('path',[]),dtype=float)
                if curve.ndim==2 and curve.shape[1]==2 and len(curve)>=2:
                    selected.append(curve if key=='brow_length' else curve[[0,-1]])
            if selected:
                segments=selected+(eye_widths if key=='brow_length' else [])
                label='눈썹 털의 길이' if key=='brow_length' else '눈썹 털의 앞머리와 꼬리'
                source='segmentation'
            else:
                continue
        if not segments or any(not np.isfinite(s).all() for s in segments):continue
        entry={'item_id':key,'label':label,'source':source,
               'paths':[np.round(s,3).tolist() for s in segments]}
        # A visible-hair path is not the old Face Mesh guide measurement.
        if key in metrics and np.isfinite(metrics[key]) and not (source=='segmentation' and key.startswith('brow_')):
            unit='degrees' if key in ('eye_tilt','brow_tail') else 'log_ratio' if key=='eye_asymmetry' else 'ratio'
            entry.update(value=metrics[key],unit=unit)
        if source=='segmentation' and len(visible_brows)==2:
            metric='length_eye_ratio' if key=='brow_length' else 'tail_angle_deg'
            values=[b.get(metric) for b in visible_brows.values()]
            if all(isinstance(v,(int,float)) and np.isfinite(v) for v in values):
                entry.update(value=float(np.mean(values)),unit='ratio' if key=='brow_length' else 'degrees')
        output.append(entry)
    return {'width':int(width),'height':int(height),'features':output}
