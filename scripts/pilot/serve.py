"""Local-only pilot server. Warms both networks before accepting any requests."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PIL import Image
from facework.pilot.runtime import RuntimeResources,make_app,INFERENCE_WORKER

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--exploratory-depth',action='store_true',help='Show cohort-relative depth labels without claiming validated 3D accuracy')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args();t=time.perf_counter();r=RuntimeResources()
    def warmup():
        from facework.morphometrics import WINDOWS
        crops={}
        for part in ['eye_right','eye_left','brow_right','brow_left','nose','mouth','face']:
            h,w=WINDOWS[part.split('_')[0]];crops[part]=Image.new('RGB',(w,h),(127,127,127))
        r.cnn(None,None,crops)
        r.engine.infer([Image.new('RGB',(640,640),(127,127,127)) for _ in range(3)])
    INFERENCE_WORKER.submit(warmup).result()
    app=make_app(r,exploratory_depth=args.exploratory_depth)
    print(f'Pilot ready after model load and warmup: {time.perf_counter()-t:.2f}s',flush=True)
    if args.self_check:
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            checks={p:c.get(p).status_code for p in ['/pilot','/pilot.js','/pilot.css']}
            assert all(v==200 for v in checks.values())
        report={'status':'passed','startup_seconds_including_warmup':time.perf_counter()-t,'static_routes':checks,'scope':'startup + synthetic warmup + same-app UI assets; no physical camera test'}
        (ROOT/'outputs/pilot-20261014/startup_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        INFERENCE_WORKER.shutdown(wait=True)
        return
    import uvicorn
    print(f'http://127.0.0.1:{args.port}/pilot | exploratory_depth={args.exploratory_depth}',flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
if __name__=='__main__':main()
