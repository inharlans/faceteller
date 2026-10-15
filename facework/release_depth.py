"""Identical face framing and geometry measurements for cohort and web requests."""
import numpy as np
from facework.morphometrics import OVAL
from facework.pilot.depth import measure_geometry, FEATURES
from facework.pilot.scoring import boundary_scores
from facework.pilot_contract import Observation

VERSION='landmark-guide-1.5-vggt518-v1'


def guide_images(images,detections):
    result=[]
    for image,detection in zip(images,detections):
        points=np.asarray(detection['landmarks'])[OVAL,:2]
        if not np.isfinite(points).all():raise ValueError('invalid face framing')
        low,high=points.min(0),points.max(0);centre=(low+high)/2
        edge=float(max(high-low))*1.5
        if edge<32:raise ValueError('face framing too small')
        box=(*np.round(centre-edge/2).astype(int),*np.round(centre+edge/2).astype(int))
        result.append(image.crop(box))
    return result


def estimate(engine,images,detections,detect):
    guides=guide_images(images,detections);geometry=None
    try:
        geometry=engine.infer(guides)
        mapped=[detect(image)[0] for image in geometry['prepared_images']]
        result=measure_geometry(geometry,mapped)
        result['framing_version']=VERSION
        result['network_seconds']=geometry['network_seconds']
        result['inference_seconds']=geometry['inference_seconds']
        return result
    finally:
        for image in guides:image.close()
        if geometry:
            for image in geometry['prepared_images']:image.close()
        geometry=None


def observations(result,detections,config,schema):
    if config['framing_version']!=VERSION:raise ValueError('depth calibration framing mismatch')
    poses=[d['yaw'] for d in detections]
    directions_ok=(abs(poses[0])<=12 and -60<=poses[1]<=-20 and 20<=poses[2]<=60
                   and all(abs(d.get('pitch',-5)+5)<=16 and abs(d.get('roll',0))<=12 for d in detections))
    reasons=[]
    if not directions_ok:reasons.append('DEPTH_VIEW_DIRECTIONS_INVALID')
    if any(d['smile']>.3 or d['blink']>.5 for d in detections):reasons.append('DEPTH_EXPRESSION_CHANGED')
    if result['status']!='measured':reasons.append('DEPTH_MEASUREMENT_FAILED')
    quality=config['quality']
    confidence=result.get('depth_confidence_min')
    cross=result.get('cross_view_midline_error_ratio')
    if confidence is None or not np.isfinite(confidence) or confidence<quality['min_confidence']:reasons.append('DEPTH_CONFIDENCE_LOW')
    if cross is None or not np.isfinite(cross) or cross>quality['max_cross_view_error']:reasons.append('DEPTH_VIEWS_INCONSISTENT')
    out=[]
    for key in FEATURES:
        f=schema[key];c=config['features'][key];value=result.get('values',{}).get(key)
        valid=value is not None and np.isfinite(value)
        score=None;label=None;label_name=None
        if valid:
            scored=boundary_scores(value,c['lower'],c['upper'],c['sigma_repeat_train'])
            index=['low','medium','high'].index(scored['label'])
            score=scored['score'];label=f'{key}:{f["classes"][index]}';label_name=f['classes_ko'][index]
        out.append(Observation(key,f['part'],'depth',label,label_name,
            measurement={'indicator':key,'value':float(value),'unit':'ratio'} if valid else None,
            measurement_score=score,quality_passed=not reasons and valid,boundaries_validated=True,
            basis='세 사진에서 복원한 상대 비율을 400명 기준 분포와 비교합니다. 경계점수에는 반복 촬영에 따른 변동을 반영합니다.',
            hold_reasons=tuple(reasons)))
    return out
