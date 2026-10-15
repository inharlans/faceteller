"""VGGT RGB inference and *separate* landmark-guided shape measurements.

No face detector or landmarks are passed to the VGGT network. Depth/camera heads
are frozen; their output is used to lift 2D anatomical locations into 3D.
"""
from pathlib import Path
import sys, time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
VGGT_SOURCE = ROOT / 'third_party/vggt'
FEATURES = ['nose_projection','nose_bridge','lip_projection','chin_projection']
LANDMARK_IDS = [1,98,327,168,133,362,0,17,61,291,175,2,33,263,10,6]

class VGGTEngine:
    def __init__(self, size=518, device='cuda', weights_path=None):
        import torch
        sys.path.insert(0, str(VGGT_SOURCE))
        from vggt.models.vggt import VGGT
        self.torch, self.size, self.device = torch, size, device
        torch.set_num_threads(2)
        start=time.perf_counter()
        model=VGGT(enable_point=False, enable_track=False)
        self.weights_path=Path(weights_path) if weights_path is not None else ROOT/'models/vggt/model.pt'
        state=torch.load(self.weights_path,map_location='cpu',weights_only=True,mmap=True)
        incompatible=model.load_state_dict(state,strict=False,assign=True)
        assert not incompatible.missing_keys, incompatible.missing_keys
        assert all(k.startswith(('point_head.','track_head.')) for k in incompatible.unexpected_keys)
        del state
        model.eval().requires_grad_(False)
        # 8GB deployment: store transformer weights in BF16, numerical heads in FP32.
        model.aggregator.to(device=device,dtype=torch.bfloat16)
        model.camera_head.to(device=device,dtype=torch.float32)
        model.depth_head.to(device=device,dtype=torch.float32)
        self.model=model
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-start

    def prepare(self, images, size=None):
        """Aspect-preserving padding; mapping is explicit and identical for landmarks."""
        import torch
        size=size or self.size
        if size%14:raise ValueError('size must be a multiple of 14')
        tensors=[]; prepared=[];maps=[]
        for image in images:
            w,h=image.size;scale=size/max(w,h)
            nw,nh=round(w*scale),round(h*scale)
            dx,dy=(size-nw)//2,(size-nh)//2
            im=Image.new('RGB',(size,size),(255,255,255))
            im.paste(image.convert('RGB').resize((nw,nh),Image.Resampling.BICUBIC),(dx,dy))
            prepared.append(im);maps.append({'scale_xy':[nw/w,nh/h],'offset':[dx,dy],'input_size':[w,h]})
            tensors.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255)
        return torch.stack(tensors),prepared,maps

    def infer(self, images, size=None):
        torch=self.torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        t=time.perf_counter();x,prepared,maps=self.prepare(images,size)
        x=x.unsqueeze(0).to(self.device)
        torch.cuda.synchronize();t0=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                tokens,ps=self.model.aggregator(x)
            tokens=[v.float() if v is not None else None for v in tokens]
            pose=self.model.camera_head(tokens)[-1]
            depth,confidence=self.model.depth_head(tokens,images=x,patch_start_idx=ps,frames_chunk_size=1)
            extr,intr=pose_encoding_to_extri_intri(pose,x.shape[-2:])
        torch.cuda.synchronize();net_seconds=time.perf_counter()-t0
        d=depth[0].cpu().numpy();conf=confidence[0].cpu().numpy()
        e=extr[0].cpu().numpy();k=intr[0].cpu().numpy()
        points=unproject_depth_map_to_point_map(d,e,k)
        peak=torch.cuda.max_memory_allocated()/1024**3
        del tokens,pose,depth,confidence,extr,intr,x
        return {'points':points,'depth':d,'depth_confidence':conf,'extrinsics':e,'intrinsics':k,
                'prepared_images':prepared,'transforms':maps,'network_seconds':net_seconds,
                'inference_seconds':time.perf_counter()-t,'peak_allocated_gib':peak}


def sample_points(points, landmarks):
    """Median 3x3 3D neighborhood; landmarks have the same image coordinates."""
    h,w=points.shape[:2];out={}
    for i in LANDMARK_IDS:
        u,v=np.round(landmarks[i,:2]).astype(int)
        if not (1<=u<w-1 and 1<=v<h-1):raise ValueError('landmark outside image')
        out[i]=np.median(points[v-1:v+2,u-1:u+2].reshape(-1,3),axis=0)
        if not np.isfinite(out[i]).all():raise ValueError('invalid point')
    return out


def measurements_from_points(p, camera_extrinsic):
    """Signed frontal-axis projection / 3D outer-eye-corner distance.

    Positive direction is determined by the camera, never by requiring a positive nose.
    Chin uses subnasale(2) as reference, so negative values are expected.
    """
    x=p[263]-p[33];width=float(np.linalg.norm(x))
    if width<1e-8:raise ValueError('degenerate eye baseline')
    x=x/width;down=p[175]-p[10];down-=x*np.dot(x,down)
    if np.linalg.norm(down)<1e-8:raise ValueError('degenerate vertical axis')
    down=down/np.linalg.norm(down);normal=np.cross(x,down)
    toward_camera=-camera_extrinsic[:,:3][2]
    if np.dot(normal,toward_camera)<0:normal=-normal
    z=lambda i:float(np.dot(p[i],normal)/width)
    values={'nose_projection':z(1)-(z(98)+z(327))/2,
            'nose_bridge':z(168)-(z(133)+z(362))/2,
            'lip_projection':(z(0)+z(17))/2-(z(61)+z(291))/2,
            'chin_projection':z(175)-z(2)}
    return values,width,normal


def measure_geometry(geometry, detections):
    """Join VGGT geometry with independently produced MediaPipe 2D landmarks.

    Side-view measurements are diagnostic. Frontal samples define the cohort value;
    multi-view fusion is performed by VGGT's attention, not by averaging occluded points.
    """
    if detections[0] is None:return {'status':'held','reason':'front_landmarks_missing'}
    try:
        p=sample_points(geometry['points'][0],detections[0]['landmarks'])
        values,width,normal=measurements_from_points(p,geometry['extrinsics'][0])
    except ValueError as e:return {'status':'held','reason':str(e)}
    confidences=[]
    for i in LANDMARK_IDS:
        u,v=np.round(detections[0]['landmarks'][i,:2]).astype(int)
        confidences.append(float(geometry['depth_confidence'][0,v,u]))
    per_view=[];cross=[]
    for v in range(1,len(detections)):
        if detections[v] is None:continue
        try:
            other=sample_points(geometry['points'][v],detections[v]['landmarks'])
            vals,_,_=measurements_from_points(other,geometry['extrinsics'][v]);per_view.append(vals)
            # Midline positions tend to be observable at +/-40 deg; contours excluded.
            cross.extend(float(np.linalg.norm(other[i]-p[i])/width) for i in [1,168,0,17,2])
        except ValueError:pass
    return {'status':'measured','values':values,'eye_distance_model_units':width,
            'depth_confidence_min':min(confidences),'depth_confidence_median':float(np.median(confidences)),
            'cross_view_midline_error_ratio':float(np.median(cross)) if cross else None,
            'side_measurements':per_view,'unit':'ratio',
            'quality_policy':'exploratory_finite_geometry_only; confidence/consistency thresholds not calibrated'}
