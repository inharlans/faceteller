"""Trial server for the web front end: the reference structure behind the endpoint the front end already calls.

  POST /api/v1/analysis   multipart: front, left_oblique, right_oblique (images), input_mode
  -> application/x-ndjson: {"type":"progress","stage":...}* then {"type":"result","data":{...}} or {"type":"error",...}
  data = {status:'COMPLETE', items:[], passages:[], notice, research_report}; research_report is run_person.Runner.analyze
  (schema faceteller-research-report-v1) and is what the front end's research view shows.
  GET /healthz            {"status":"ok", ...} once the models are loaded
Photos are decoded in memory and never written to disk; the report holds no photo and no landmark coordinates.
The CNN ensembles and VGGT are loaded at start-up and stay on the GPU; one analysis at a time.
If the front end build exists (--frontend-dist, default <repo>/frontend/dist) it is served at /.

  python deploy/v1/serve_person.py --models-dir ~/deploy_v1/experiments/feature17-20261109 \\
      --public-models-dir ~/faceteller/models --vggt-dtype fp16 --host 0.0.0.0 --port 8000
  environment instead of arguments: FACETELLER_HOST, FACETELLER_PORT and the variables listed in settings.py"""
import argparse, asyncio, json, os, sys, threading, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import settings as ST
import run_person as RP
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
VIEWS = ('front', 'left_oblique', 'right_oblique'); MAX_BYTES = 12 * 2**20
NOTICE = '전통 문헌 조회 결과이며 개인의 성격·지능·건강·운명을 판정하거나 예측하지 않습니다.'
line = lambda e: (json.dumps(e, ensure_ascii=False) + '\n').encode('utf8')


def create_app(runner: RP.Runner, frontend_dist=None):
    app = FastAPI(title='FaceTeller research trial', docs_url=None, redoc_url=None); gpu = threading.Lock()

    @app.get('/healthz')
    def healthz(): return JSONResponse({'status': 'ok', 'vggt_dtype': runner.dtype, 'check_cnn': bool(runner.s.check), 'gpu_allocated_gib': round(torch.cuda.memory_allocated() / 2**30, 2)})

    @app.post('/api/v1/analysis')
    async def analysis(front: UploadFile = File(...), left_oblique: UploadFile = File(...), right_oblique: UploadFile = File(...), input_mode: str = Form('upload')):
        raw = [await f.read(MAX_BYTES + 1) for f in (front, left_oblique, right_oblique)]; loop = asyncio.get_running_loop(); q: asyncio.Queue = asyncio.Queue()
        put = lambda e: loop.call_soon_threadsafe(q.put_nowait, e)

        def work():
            try:
                ims = []
                for i, b in enumerate(raw):
                    if len(b) > MAX_BYTES: return put({'type': 'error', 'message': '사진 파일이 너무 큽니다.', 'direction': VIEWS[i], 'code': 'IMAGE_TOO_LARGE'})
                    try: ims.append(RP.open_image(b))
                    except Exception: return put({'type': 'error', 'message': '사진을 읽을 수 없습니다.', 'direction': VIEWS[i], 'code': 'IMAGE_UNREADABLE'})
                with gpu: rep = runner.analyze(ims, progress=lambda s: put({'type': 'progress', 'stage': s}), resident=True)
                put({'type': 'result', 'data': {'status': 'COMPLETE', 'items': [], 'passages': [], 'notice': NOTICE, 'research_report': rep}})
            except RP.PhotoError as e: put({'type': 'error', 'message': '얼굴을 하나만 찾을 수 있어야 합니다.' if e.reason == 'multiple_faces' else '얼굴을 찾지 못했습니다.', 'direction': VIEWS[e.index], 'code': 'FACE_NOT_FOUND'})
            except Exception as e: put({'type': 'error', 'message': f'분석 중 오류가 발생했습니다. ({type(e).__name__})', 'code': 'ANALYSIS_FAILED'}); print('ERROR', repr(e), flush=True)
            finally: put(None)

        async def stream():
            threading.Thread(target=work, daemon=True).start()
            while (e := await q.get()) is not None: yield line(e)
        return StreamingResponse(stream(), media_type='application/x-ndjson', headers={'Cache-Control': 'no-store'})

    if frontend_dist and (Path(frontend_dist) / 'index.html').exists(): app.mount('/', StaticFiles(directory=str(frontend_dist), html=True), name='frontend')
    return app


def main():
    ap = ST.add_arguments(argparse.ArgumentParser(description='FaceTeller research trial server')); ap.add_argument('--host', default=os.environ.get('FACETELLER_HOST', '127.0.0.1'))
    ap.add_argument('--port', type=int, default=int(os.environ.get('FACETELLER_PORT', '8777'))); ap.add_argument('--frontend-dist', default=os.environ.get('FACETELLER_FRONTEND_DIST', str(ROOT / 'frontend/dist')))
    a = ap.parse_args(); t0 = time.perf_counter(); runner = RP.Runner(ST.resolve(a)); runner.load(); print('closing remark:', (f'model {runner.summary.model} ready' if runner.summary.warm() else 'model not available - template remarks'), flush=True)
    print(f'models loaded in {time.perf_counter() - t0:.1f} s: VGGT {runner.dtype}, check CNN {"on" if runner.s.check else "off"}, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated; http://{a.host}:{a.port}/', flush=True)
    uvicorn.run(create_app(runner, os.path.expanduser(a.frontend_dist)), host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
