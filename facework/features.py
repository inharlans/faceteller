from pathlib import Path
import math
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/face_landmarker.task'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

# Indices are MediaPipe Face Mesh topology. Left/right refer to the subject.
PARTS = {
 'eye_right': [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
 'eye_left': [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466],
 'brow_right': [70,63,105,66,107,55,65,52,53,46],
 'brow_left': [300,293,334,296,336,285,295,282,283,276],
 'nose': [168,6,197,195,5,4,1,19,94,2,98,97,326,327,49,279,129,358],
 'mouth': [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185],
 'face': [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],
}

def detector():
    return mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))

def analyze(image, landmarker):
    arr=np.asarray(image.convert('RGB'))
    h,w=arr.shape[:2]
    result=landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr)))
    if len(result.face_landmarks)!=1:
        return {'status':'reject', 'reason':'no_face' if not result.face_landmarks else 'multiple_faces'}
    p=np.array([[l.x*w,l.y*h] for l in result.face_landmarks[0]])
    # Rotate image and landmarks using the same affine matrix. No nonrigid warping.
    v=p[263]-p[33]
    roll=math.degrees(math.atan2(v[1],v[0]))
    M=cv2.getRotationMatrix2D(tuple((p[33]+p[263])/2), roll, 1)
    aligned=cv2.warpAffine(arr,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    q=np.c_[p,np.ones(len(p))]@M.T
    dist=lambda a,b:float(np.linalg.norm(q[a]-q[b]))
    face_width=dist(234,454)
    metrics={
      'roll_deg':roll, 'face_width_px':face_width,
      'face_visible_height_width':dist(10,152)/face_width,
      'nose_width_face':dist(98,327)/face_width,
      'mouth_width_face':dist(61,291)/face_width,
      'lip_vermilion_mouth':(dist(0,13)+dist(14,17))/dist(61,291),
      'upper_lower_lip_ratio':dist(0,13)/max(dist(14,17),1),
      'mouth_open_ratio':dist(13,14)/dist(61,291),
      'jaw_width_face':dist(172,397)/face_width,
    }
    for side,outer,inner,top,bottom,brow in [('right',33,133,159,145,[70,63,105,66,107]),('left',263,362,386,374,[300,293,334,296,336])]:
        ew=dist(outer,inner)
        metrics[f'eye_{side}_aperture_ratio']=ew/max(dist(top,bottom),1)
        metrics[f'eye_{side}_width_face']=ew/face_width
        metrics[f'eye_{side}_outer_tilt_deg']=math.degrees(math.atan2(q[inner,1]-q[outer,1], abs(q[outer,0]-q[inner,0])))
        metrics[f'brow_{side}_length_eye']=sum(np.linalg.norm(q[a]-q[b]) for a,b in zip(brow[:-1],brow[1:]))/ew
    scores={b.category_name:float(b.score) for b in result.face_blendshapes[0]}
    metrics['smile_score']=max(scores.get('mouthSmileLeft',0),scores.get('mouthSmileRight',0))
    # Signed orientation of the fitted canonical mesh, used only as a capture-quality heuristic.
    if result.facial_transformation_matrixes:
        from scipy.spatial.transform import Rotation
        rot=np.asarray(result.facial_transformation_matrixes[0])[:3,:3]
        angles=Rotation.from_matrix(rot).as_euler('xyz',degrees=True)
        metrics['mesh_pitch_deg']=float(angles[0]);metrics['mesh_yaw_deg']=float(angles[1])
    crops={}; boxes={}
    for part,inds in PARTS.items():
        ps=q[inds]; lo=ps.min(0); hi=ps.max(0); extent=hi-lo
        # Brows/eyes require surrounding texture, not just the landmark hull.
        padx=max(12,extent[0]*(.16 if 'eye_' in part else .12))
        pady=max(18,extent[1]*(.65 if 'eye_' in part else .40 if 'brow_' in part else .15))
        if part=='nose': padx=max(padx,extent[0]*.25)
        box=[max(0,int(lo[0]-padx)),max(0,int(lo[1]-pady)),min(w,int(hi[0]+padx+1)),min(h,int(hi[1]+pady+1))]
        crops[part]=Image.fromarray(aligned).crop(box)
        boxes[part]=box
    facegray=cv2.cvtColor(np.asarray(crops['face']),cv2.COLOR_RGB2GRAY)
    metrics['face_laplacian_variance']=float(cv2.Laplacian(facegray,cv2.CV_64F).var())
    return {'status':'ok','metrics':{k:round(float(v),6) for k,v in metrics.items()},'landmarks_original':p.tolist(),'landmarks_aligned':q.tolist(),'affine_original_to_aligned':M.tolist(),'boxes_aligned':boxes,'crops':crops,'aligned':Image.fromarray(aligned)}
