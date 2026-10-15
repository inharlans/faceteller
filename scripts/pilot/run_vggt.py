"""Real K-FACE VGGT cohort pilot. Saves scalar research records, never face images."""
import argparse,csv,io,json,sys,time,zipfile,hashlib,platform,traceback
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from facework.pilot.depth import VGGTEngine,measure_geometry
OUT=ROOT/'data/derived/pilot-20261014';OUT.mkdir(exist_ok=True,parents=True)

def triplet(sid,light='L1',cameras=('C7','C4','C10'),mode='guide',margin=1.5):
    ims=[]
    with zipfile.ZipFile(ROOT/'data/raw/kface/zips'/f'{sid}.zip') as z:
        for cam in cameras:
            base=f'S001/{light}/E01/{cam}'
            im=Image.open(io.BytesIO(z.read(base+'.jpg'))).convert('RGB')
            if mode=='guide':
                annotation_key=base+'.txt'
                # L2 has RGB frames but no provider TXT boxes. Reuse the same
                # camera's L1 guide box, rather than deriving it from VGGT/MediaPipe.
                if annotation_key not in z.namelist():annotation_key=f'S001/L1/E01/{cam}.txt'
                rows=[[int(v) for v in line.split()] for line in z.read(annotation_key).decode().splitlines() if line.strip()]
                x,y,w,h=next(row for row in rows if len(row)==4)
                cx,cy=x+w/2,y+h/2;edge=max(w,h)*margin
                im=im.crop((round(cx-edge/2),round(cy-edge/2),round(cx+edge/2),round(cy+edge/2)))
            ims.append(im)
    return ims

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','cohort','repeat','controls'],default='smoke');ap.add_argument('--size',type=int,default=518);ap.add_argument('--limit',type=int,default=400)
    args=ap.parse_args();target=OUT/f'{args.phase}_{args.size}.jsonl'
    refs=list(csv.DictReader((ROOT/'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')))
    split={r['subject_id']:r['split'] for r in refs};ids=sorted(split)
    # Repeat cohort selected uniformly through sorted IDs, not after seeing quality.
    repeat_ids=ids[::10]
    eng=VGGTEngine(args.size)
    meta={'phase':args.phase,'input_size':args.size,'gpu':eng.torch.cuda.get_device_name(0),'torch':eng.torch.__version__,'python':platform.python_version(),'weights':'facebook/VGGT-1B original research checkpoint','code_commit':'a288dd0f14786c93483e45524328726ab7b1b4ce','precision':'BF16 transformer weights/autocast; FP32 camera/depth heads','model_load_seconds':eng.load_seconds,'geometry':'depth + predicted cameras -> common-world point map','model_training':False,'landmarks_are_network_input':False,'crop':'provider bounding box expanded 1.5x (simulates capture guide); unaligned; aspect-preserving padding','stored_images':False}
    (OUT/f'{args.phase}_{args.size}_meta.json').write_text(json.dumps(meta,indent=2))
    print('LOADED',json.dumps(meta),flush=True)
    jobs=[]
    if args.phase=='smoke':
        # This forward is executed BEFORE importing any MediaPipe code.
        t=time.perf_counter();raw=eng.infer(triplet(ids[0],mode='raw'))
        independent={'mediapipe_imported_before_forward':'mediapipe' in sys.modules,'network_input_shape':[3,3,args.size,args.size],'depth_shape':list(raw['depth'].shape),'depth_finite_fraction':float(np.isfinite(raw['depth']).mean()),'seconds':raw['inference_seconds'],'peak_gib':raw['peak_allocated_gib']}
        (OUT/'vggt_without_mediapipe.json').write_text(json.dumps(independent,indent=2));print('INDEPENDENT_FORWARD',independent,flush=True);del raw
        jobs=[(s,'L1','guide',('C7','C4','C10'),1.5) for s in ids[:args.limit]]
    elif args.phase=='cohort':jobs=[(s,'L1','guide',('C7','C4','C10'),1.5) for s in ids[:args.limit]]
    elif args.phase=='repeat':jobs=[(s,l,'guide',('C7','C4','C10'),1.5) for s in repeat_ids[:args.limit] for l in ['L2','L3']]
    else:
        for s in repeat_ids[:args.limit]:
            jobs.extend([(s,'L1','raw',('C7','C4','C10'),1.5),(s,'L1','guide',('C7',),1.5),(s,'L1','guide',('C7','C7','C7'),1.5),(s,'L1','guide',('C7','C5','C9'),1.5),(s,'L1','guide',('C7','C4','C10'),1.3)])
    from facework.analysis_v2 import detect
    prior={}
    if target.exists():
        for ln in target.read_text().splitlines():
            row=json.loads(ln);prior[row['job']]=row
    for j,(sid,light,mode,cams,margin) in enumerate(jobs):
        key=f'{sid}|{light}|{mode}|{",".join(cams)}|{margin}'
        if key in prior and prior[key]['status']!='error':continue
        t=time.perf_counter();row={'job':key,'subject_id':sid,'split':split[sid],'light':light,'mode':mode,'cameras':cams,'margin':margin,'crop_annotation_policy':'requested light; L1 same-camera fallback if annotation absent'}
        try:
            ims=triplet(sid,light,cams,mode,margin)
            g=eng.infer(ims);t1=time.perf_counter();detections=[detect(im)[0] for im in g['prepared_images']]
            row.update(measure_geometry(g,detections));row.update(network_seconds=g['network_seconds'],inference_seconds=g['inference_seconds'],landmark_and_measure_seconds=time.perf_counter()-t1,peak_allocated_gib=g['peak_allocated_gib'],poses=[{'yaw':d['yaw'],'pitch':d['pitch']} if d else None for d in detections],depth_positive_fraction=float((g['depth']>0).mean()))
            if args.phase=='smoke' and j==0:
                np.savez_compressed(OUT/'private_depth_example.npz',depth=g['depth'],confidence=g['depth_confidence'],points=g['points'],extrinsics=g['extrinsics'],intrinsics=g['intrinsics'])
                import matplotlib;matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(1,3,figsize=(12,4))
                for ax,dd,title in zip(axes,g['depth'],['front','left oblique','right oblique']):
                    v=dd[...,0];im=ax.imshow(v,cmap='viridis',vmin=np.quantile(v,.05),vmax=np.quantile(v,.95));ax.set_title(title);ax.axis('off');fig.colorbar(im,ax=ax,fraction=.04)
                fig.tight_layout();fig.savefig(OUT/'private_depth_example.png',dpi=110);plt.close(fig)
            del g,ims,detections
        except Exception as e:
            row.update(status='error',error=repr(e));traceback.print_exc()
            if 'out of memory' in str(e).lower():eng.torch.cuda.empty_cache()
        row['wall_seconds_including_zip']=time.perf_counter()-t
        with target.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(args.phase,j+1,len(jobs),sid,row['status'],round(row.get('inference_seconds',0),3),row.get('values',row.get('error')),flush=True)
    print('DONE',target,flush=True)
if __name__=='__main__':main()
