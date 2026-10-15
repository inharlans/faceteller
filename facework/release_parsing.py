"""Local CPU face parsing and explicitly weak observable measurements.

Parsing probabilities are never classifier confidence. Returned masks/boxes use
original image pixels. No network calls, GPU providers, file writes or model
downloads occur in this module. The caller owns private offline artifacts.
"""
from pathlib import Path
import hashlib
import math
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'bisenet-visible-geometry-v1'
EAR_TRANSFORM_VERSION = 'visible-ear-eye-scale-v2'
MODEL_SHA256 = '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
SKIN, HAIR, HAT, EARS = 1, 17, 18, (7, 8)
FEATURE_CLASSES = {
    'forehead_height': ['low', 'medium', 'high'],
    'forehead_width': ['narrow', 'medium', 'wide'],
    'ear_length': ['short', 'medium', 'long'],
    'ear_width': ['narrow', 'medium', 'wide'],
    'pigmented_marks': ['absent', 'present'],
}


def eye_geometry(landmarks):
    p = np.asarray(landmarks, dtype=float)[:, :2]
    if len(p) < 468 or not np.isfinite(p).all():
        raise ValueError('finite original-pixel face landmarks required')
    right, left = (p[33]+p[133])/2, (p[263]+p[362])/2
    d = left-right
    e = float(np.linalg.norm(d))
    if e < 4:
        raise ValueError('eye baseline too small')
    return p, (right+left)/2, e, math.degrees(math.atan2(d[1], d[0]))


def head_box(image_size, landmarks):
    """Broad head ROI retains hair/ears; never invents an ear box."""
    w, h = image_size
    if landmarks is None:
        return [0, 0, w, h]
    p, _, _, _ = eye_geometry(landmarks)
    lo, hi = p.min(0), p.max(0)
    fw, fh = hi-lo
    return [max(0, int(lo[0]-.45*fw)), max(0, int(lo[1]-.70*fh)),
            min(w, int(math.ceil(hi[0]+.45*fw))), min(h, int(math.ceil(hi[1]+.18*fh)))]


class ReleaseParser:
    """parse(PIL image, original pixel landmarks=None) -> uint8 HxW mask."""
    def __init__(self, model_path=None, threads=2):
        vendor = ROOT/'models/face_parsing/python'
        if vendor.exists() and str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        import onnxruntime as ort
        path = Path(model_path or ROOT/'models/face_parsing/resnet18.onnx')
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.model_sha256 != MODEL_SHA256:
            raise ValueError('unrecognized face parsing model checksum')
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise RuntimeError('CPU-only provider required')
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def parse(self, image, landmarks=None):
        if not isinstance(image, Image.Image):
            raise TypeError('PIL.Image required')
        box = head_box(image.size, landmarks)
        x0, y0, x1, y1 = box
        # Isotropic square padding keeps inverse mapping exact and face shapes intact.
        rgb = np.asarray(image.convert('RGB'))[y0:y1, x0:x1]
        h, w = rgb.shape[:2]
        scale = min(512/w, 512/h)
        nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
        dx, dy = (512-nw)//2, (512-nh)//2
        canvas = np.full((512, 512, 3), 127, np.uint8)
        canvas[dy:dy+nh, dx:dx+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.astype(np.float32)/255
        x = (x-np.array([.485,.456,.406], np.float32))/np.array([.229,.224,.225], np.float32)
        output = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(x.transpose(2,0,1)[None])})[0]
        mask = output[0].argmax(0).astype(np.uint8)
        if mask.shape != (512,512):
            mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_NEAREST)
        result = np.zeros((image.height, image.width), np.uint8)
        result[y0:y1, x0:x1] = cv2.resize(mask[dy:dy+nh,dx:dx+nw], (w,h), interpolation=cv2.INTER_NEAREST)
        return result


def visible_ear_box(mask, landmarks=None):
    """Largest plausible visible ear component, xyxy exclusive, else None.

    No inferred hidden ear, ear-ring (class9), or guessed landmark location.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError('2D class mask required')
    e = eye_geometry(landmarks)[2] if landmarks is not None else min(mask.shape)/5
    n, _, stats, _ = cv2.connectedComponentsWithStats(np.isin(mask, EARS).astype(np.uint8), 8)
    choices = []
    for x,y,w,h,area in stats[1:]:
        if area < max(25,.01*e*e) or h < max(12,.20*e) or w < max(5,.055*e):
            continue
        if not 1.05 <= h/w <= 5 or area/(w*h) < .35:
            continue
        if x <= 1 or y <= 1 or x+w >= mask.shape[1]-1 or y+h >= mask.shape[0]-1:
            continue
        choices.append((int(area), [int(x),int(y),int(x+w),int(y+h)]))
    return max(choices, key=lambda v:v[0])[1] if choices else None


def forehead_metrics(mask, landmarks):
    """Visible skin/hair-boundary height and skin width / eye-centre distance.

    This conservative mask gate cannot establish anatomical hairline truth; labels
    remain weak. No visible skin/hair transition (including bald/hidden) => unknown.
    """
    p, centre, e, roll = eye_geometry(landmarks)
    m = cv2.getRotationMatrix2D(tuple(centre), roll, 128/e)
    m[:,2] += np.array([256,280])-centre
    q = np.c_[p,np.ones(len(p))] @ m.T
    a = cv2.warpAffine(np.asarray(mask,np.uint8),m,(512,512),flags=cv2.INTER_NEAREST)
    by = float(np.mean(q[[107,336],1]))
    bx = float(np.mean(q[[107,336],0]))
    start = int(by-.12*128)
    bounds=[]
    for x in range(int(bx-48),int(bx+49),3):
        if not 3 <= x < 509 or not 8 <= start < 510:
            continue
        col = a[:,x]
        if np.mean(col[start-12:start+1] == SKIN) < .85:
            continue
        for y in range(start-12,max(8,int(by-1.65*128)),-1):
            if np.mean(col[y-7:y] == HAIR) >= .85 and np.mean(col[y:y+12] == SKIN) >= .85:
                bounds.append(y)
                break
    reasons=[]
    if len(bounds) < 25:
        reasons.append('HAIRLINE_NOT_SUFFICIENTLY_VISIBLE')
    hair_y = float(np.median(bounds)) if bounds else None
    if bounds and np.ptp(bounds) > .35*128:
        reasons.append('HAIRLINE_IRREGULAR_OR_OCCLUDED')
    if hair_y is not None and hair_y > q[10,1]+.05*128:
        reasons.append('LOW_HAIR_EDGE_OR_BANGS_UNCERTAIN')
    if np.any(a[max(0,int(by-160)):max(0,int(by)),int(bx-64):int(bx+65)] == HAT):
        reasons.append('HAT_OCCLUSION')
    height = None if hair_y is None else (by-hair_y)/128
    width = None
    if height is not None:
        if not .30 <= height <= 1.65:
            reasons.append('FOREHEAD_HEIGHT_IMPLAUSIBLE')
        row = int((hair_y+by)/2)
        x = int(bx)
        if 0 <= row < 512 and a[row,x] == SKIN:
            l=r=x
            while l>0 and a[row,l-1] == SKIN:l-=1
            while r<511 and a[row,r+1] == SKIN:r+=1
            width=(r-l+1)/128
            if not 1 <= width <= 2.8 or l<2 or r>509:
                reasons.append('FOREHEAD_WIDTH_NOT_VISIBLE')
        else:reasons.append('FOREHEAD_SKIN_NOT_VISIBLE')
    return {'values': {'forehead_height':height if not reasons else None,
                       'forehead_width':width if not reasons else None},
            'reasons':reasons,'boundary_columns':len(bounds),'weak_automated':True}


def ear_metrics(mask, landmarks, yaw_degrees):
    p, centre, e, roll = eye_geometry(landmarks)
    box=visible_ear_box(mask,landmarks)
    values={'ear_length':None,'ear_width':None}
    reasons=[]
    if box is None:reasons.append('EAR_NOT_VISIBLE')
    if not math.isfinite(yaw_degrees) or not 20 <= abs(yaw_degrees) <= 60:
        reasons.append('EAR_VIEW_OUTSIDE_OBLIQUE_RANGE')
    if box is not None and not reasons:
        x0,y0,x1,y1=box
        yy,xx=np.where(np.isin(mask[y0:y1,x0:x1],EARS))
        xy=np.c_[xx+x0,yy+y0].astype(float)
        theta=math.radians(roll)
        rot=np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
        aligned=(xy-centre)@rot.T
        w,h=np.ptp(aligned,axis=0)+1
        # Weak-perspective correction, not a recovered physical length.
        reference=e/math.cos(math.radians(yaw_degrees))
        values={'ear_length':float(h/reference),'ear_width':float(w/h)}
        if not .20 <= values['ear_length'] <= 1.8 or not .15 <= values['ear_width'] <= .95:
            reasons.append('EAR_GEOMETRY_IMPLAUSIBLE');values={k:None for k in values}
    return {'values':values,'reasons':reasons,'box':box,'weak_automated':True}


def stable_measurement(values, minimum=2, relative_spread=.15):
    """At least two visible repeated lights, max relative range <= predeclared gate."""
    a=np.array([v for v in values if v is not None and math.isfinite(v)],float)
    if len(a)<minimum:return {'value':None,'reason':'INSUFFICIENT_VISIBLE_REPEATS','n':len(a)}
    mid=float(np.median(a));spread=float(np.ptp(a)/max(abs(mid),1e-8))
    return {'value':mid if spread<=relative_spread else None,
            'reason':None if spread<=relative_spread else 'LIGHTING_UNSTABLE',
            'n':len(a),'relative_range':spread}


def fit_thresholds(subject_metrics, train_ids, feature, minimum=20):
    ids=sorted(set(train_ids))
    selected=[(sid,subject_metrics.get(sid,{}).get(feature)) for sid in ids]
    selected=[(sid,float(v)) for sid,v in selected if v is not None and math.isfinite(v)]
    provenance={'fit_split':'train','train_id_sha256':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                'eligible_train_n':len(selected),'quantiles':[1/3,2/3]}
    if len(selected)<minimum:return {**provenance,'status':'unavailable','reason':'INSUFFICIENT_TRAIN_SUPPORT'}
    lower,upper=map(float,np.quantile([v for _,v in selected],[1/3,2/3]))
    if upper-lower<1e-6:return {**provenance,'status':'unavailable','reason':'DEGENERATE_TRAIN_DISTRIBUTION'}
    return {**provenance,'status':'fitted','lower':lower,'upper':upper}


def metric_label(value, thresholds, classes):
    if value is None or not math.isfinite(value) or thresholds.get('status')!='fitted':return None
    return classes[0 if value<thresholds['lower'] else 1 if value<thresholds['upper'] else 2]


MARK_VERSION = 'visible-upper-skin-dark-marks-v1'
MARK_SCOPE = ('small dark visible marks in exposed upper-face skin inside the face oval, '
              'above subnasale; eyes/brows/nose/lips/hair and lower beard-prone face excluded; '
              'not whole-face absence, not clinical mole diagnosis')


def visible_mark_candidates(image, mask, landmarks):
    """Bounded, uncalibrated candidate detector; absence needs three clean repeats.

    Fixed geometric/contrast gates, not tuned on held-out subjects. Scalar candidate
    positions use the eye-centre coordinate system; no image leaves this function.
    """
    from facework.morphometrics import OVAL
    p,centre,e,roll=eye_geometry(landmarks)
    m=cv2.getRotationMatrix2D(tuple(centre),roll,128/e)
    m[:,2]+=np.array([256,280])-centre
    q=np.c_[p,np.ones(len(p))]@m.T
    a=cv2.warpAffine(np.asarray(mask,np.uint8),m,(512,512),flags=cv2.INTER_NEAREST)
    rgb=cv2.warpAffine(np.asarray(image.convert('RGB')),m,(512,512),flags=cv2.INTER_LINEAR,borderValue=(127,127,127))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    area=np.zeros((512,512),np.uint8)
    cv2.fillConvexPoly(area,cv2.convexHull(np.round(q[OVAL]).astype(np.int32)),1)
    area[max(0,int(q[2,1]-5)):]=0  # no upper lip/chin/beard inference
    # Anatomical exclusions remain excluded even if the parser mistakes a brow for skin.
    for ids in ([33,133,159,145],[263,362,386,374],[70,63,105,66,107],
                [336,296,334,293,300],[168,98,327,2]):
        box=q[list(ids)];lo=np.floor(box.min(0)-7).astype(int);hi=np.ceil(box.max(0)+7).astype(int)
        area[max(0,lo[1]):min(512,hi[1]+1),max(0,lo[0]):min(512,hi[0]+1)]=0
    potential=cv2.erode(area,np.ones((5,5),np.uint8))>0
    skin=cv2.erode((a==SKIN).astype(np.uint8),np.ones((7,7),np.uint8))>0
    allowed=potential & skin
    denominator=int(potential.sum());coverage=float(allowed.sum()/max(denominator,1))
    reasons=[]
    if denominator<1000 or coverage<.65:reasons.append('INSUFFICIENT_CLEAR_SKIN_COVERAGE')
    pixels=gray[allowed]
    if len(pixels)<600 or (len(pixels) and (np.median(pixels)<45 or np.median(pixels)>225 or np.percentile(pixels,95)>250)):
        reasons.append('LIGHTING_OR_RESOLUTION_UNSUITABLE')
    background=cv2.GaussianBlur(gray.astype(np.float32),(0,0),3)
    residual=background-gray.astype(np.float32)
    blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    noise=float(np.median(np.abs(residual[allowed]-np.median(residual[allowed])))) if len(pixels) else 0
    threshold=max(12.,4*1.4826*noise)
    candidate_mask=((residual>=threshold)&(blackhat>=15)&allowed).astype(np.uint8)
    n,components,stats,_=cv2.connectedComponentsWithStats(candidate_mask,8)
    candidates=[];rejected=0
    for number,(x,y,w,h,count) in enumerate(stats[1:],1):
        if not 3<=count<=55 or not 2<=w<=11 or not 2<=h<=11 or max(w,h)/min(w,h)>2:
            rejected+=1;continue
        component=(components[y:y+h,x:x+w]==number).astype(np.uint8)
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        perimeter=sum(cv2.arcLength(c,True) for c in contours)
        circularity=min(1.,4*math.pi*count/max(perimeter**2,1))
        if circularity<.45:rejected+=1;continue
        yy,xx=np.where(component)
        cx,cy=float(np.mean(xx)+x),float(np.mean(yy)+y)
        # Every pixel of a margin must remain inside the eroded observable skin.
        if not allowed[max(0,y-2):y+h+2,max(0,x-2):x+w+2].all():
            rejected+=1;continue
        candidates.append({'x_eye':(cx-256)/128,'y_eye':(cy-280)/128,
                           'area_eye2':float(count/128**2),'contrast':float(np.mean(residual[y+yy,x+xx])),
                           'circularity':float(circularity)})
    if len(candidates)>12 or rejected>25:reasons.append('TEXTURE_OR_SHADOW_UNCERTAIN')
    # Discarding dark non-circular components cannot establish clean-skin absence.
    return {'version':MARK_VERSION,'scope':MARK_SCOPE,'quality_passed':not reasons,
            'coverage':coverage,'candidates':candidates,'rejected_components':rejected,
            'reasons':reasons,'contrast_threshold':threshold,'confidence':None}


def repeated_mark_label(frames, maximum_distance=.04):
    """One candidate per light per match; two-of-three repeat evidence, else null."""
    lights=[f.get('light') for f in frames if f.get('light') is not None]
    if len(lights)!=len(set(lights)):
        return {'label':None,'reason':'DUPLICATE_LIGHT_NOT_INDEPENDENT_REPEAT','stable_marks':[]}
    usable=[f for f in frames if f.get('quality_passed')]
    if len(usable)<2:return {'label':None,'reason':'INSUFFICIENT_CLEAR_REPEATS','stable_marks':[]}
    clusters=[]
    for light_index,frame in enumerate(usable):
        for mark in frame['candidates']:
            xy=np.array([mark['x_eye'],mark['y_eye']])
            choices=[(float(np.linalg.norm(xy-c['position'])),i) for i,c in enumerate(clusters) if light_index not in c['lights']]
            nearest=min(choices,default=(float('inf'),-1))
            if nearest[0]<=maximum_distance:
                c=clusters[nearest[1]];c['points'].append(xy);c['lights'].append(light_index);c['position']=np.mean(c['points'],axis=0)
            else:clusters.append({'position':xy,'points':[xy],'lights':[light_index]})
    stable=[{'x_eye':float(c['position'][0]),'y_eye':float(c['position'][1]),'repeats':len(c['lights'])} for c in clusters if len(c['lights'])>=2]
    if any(len(c['lights'])<2 for c in clusters):return {'label':None,'reason':'UNSTABLE_DARK_CANDIDATES','stable_marks':stable}
    if stable:return {'label':'present','reason':None,'stable_marks':stable}
    if len(usable)==3 and all(f.get('rejected_components',0)==0 for f in usable):
        return {'label':'absent','reason':None,'stable_marks':[]}
    return {'label':None,'reason':'ABSENCE_NOT_ESTABLISHED','stable_marks':[]}
