"""Run the reference structure (v10: plan C, confidence floor 0.6) on one person's three photos: frontal + two obliques.
Stand-alone: needs this folder, the repository's facework/ and third_party/vggt, the model files and config/*.json -
no dataset, no label files.

  CLI      python deploy/v1/run_person.py <front> <oblique> <oblique> [--out report.json] [settings, see settings.py]
  library  Runner(settings).analyze([front, oblique, oblique] PIL RGB images, progress=None, resident=False) -> report
           (schema faceteller-research-report-v1: judgements, plan_C_selected, all_matching, blocked_by_hold,
           held_judgements, callouts (one sentence per region), cnn_check; no photo, no landmark coordinates)
resident=True keeps the CNN ensembles and VGGT on the GPU between calls (the server). The check-only CNNs run only with
--check / FACETELLER_CHECK_CNN=1 and are loaded and released per call."""
import argparse, io, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
import torch
from PIL import Image, ImageOps
import settings as ST
import structure as S
import cnn as C
import measures as M
import vggt_depth as VD
import facework.features as FF
from facework.analysis_v2 import detect
from facework.release_parsing import ReleaseParser
CHECK_PARTS = ('glabella', 'face', 'philtrum')  # ensembles the structure does not use; loaded only for the optional check
STAGES = ('얼굴 정렬', '부위 추출', '형태 분류', '깊이 분석', '문헌 선정')


class PhotoError(Exception):
    def __init__(self, index, reason): super().__init__(reason); self.index, self.reason = index, reason


class Runner:
    def __init__(self, settings: ST.Settings):
        self.s = settings; self.cfg = S.Config(settings.config_dir); self.parts = list(self.cfg.cnn_parts)
        missing = [f'{what}: {p}' for p, what in ST.required_files(settings, self.parts) if not p.exists()]
        if missing: raise SystemExit('필요한 파일이 없습니다:\n  ' + '\n  '.join(missing))
        FF.MODEL = settings.public_models_dir / 'face_landmarker.task'  # the landmark model path of facework.features.detector
        torch.hub.set_dir(str(settings.public_models_dir / 'torch_cache/hub')); self.dtype = VD.resolve_dtype(settings.vggt_dtype)
        self.template = np.load(settings.models_dir / 'posenorm_template.npy'); self.depth = VD.DepthFrames(np.load(settings.models_dir / 'vggt3/vggt_template.npy'))
        self.nets, self.engine, self.parser = {}, None, None

    # ---- models
    def _nets(self, folder, part, keep):
        key = (str(folder), part); nets = self.nets.get(key) or C.load_part(folder, part)
        if keep: self.nets[key] = nets
        return nets

    def load(self):
        """Put the structure's CNN ensembles and VGGT on the GPU (the server calls this at start-up)."""
        for part in self.parts: self._nets(self.s.models_dir / 'candidate-v2-posenorm', part, True)
        if self.engine is None: self.engine = VD.make_engine(self.s.public_models_dir / 'vggt/model.pt', self.dtype)
        if self.parser is None: self.parser = ReleaseParser(model_path=self.s.public_models_dir / 'face_parsing/resnet18.onnx', threads=2)

    # ---- one person
    def analyze(self, ims, progress=None, resident=False):
        cfg = self.cfg; say = progress or (lambda s: None); say(STAGES[0]); dets = []; t0 = time.perf_counter()
        for i, im in enumerate(ims):
            d, err = detect(im)
            if d is None: raise PhotoError(i, err)
            dets.append(d)
        res = {'schema': 'faceteller-research-report-v1', 'structure': '구조 v10 · C안 · 하한 0.6',
               'photos': [{'view': v, 'size': list(im.size), 'yaw': round(d['yaw'], 1), 'pitch': round(d['pitch'], 1), 'roll': round(d['roll'], 1), 'blink': round(float(d['blink']), 2), 'smile': round(float(d['smile']), 2)} for v, im, d in zip(('front', 'oblique_1', 'oblique_2'), ims, dets)]}
        img, det = ims[0], dets[0]; P3 = det['landmarks'].astype(float); res['face_width_px'] = round(float(np.linalg.norm(P3[234, :2] - P3[454, :2])), 1); raw, value, note, detail = {}, {}, {}, {}
        # CNN
        say(STAGES[1]); check = self.s.check and cfg.check is not None; X = C.tiles(img, P3, self.template, self.parts + ([p for p in CHECK_PARTS if p not in self.parts] if check else [])); say(STAGES[2]); cnn_all = {}
        for part in self.parts: cnn_all.update(C.ensemble(self._nets(self.s.models_dir / 'candidate-v2-posenorm', part, resident), X[part]))
        for k in cfg.cnn_features:
            c = S.calibrated(*cnn_all[k], cfg.cnn['features'][k]['classes'], cfg.cnn['features'][k]['temperature']); value[k] = None; detail[k] = {'probabilities': dict(zip(c['classes'], c['probabilities']))}
            if c['unjudgeable'] is not None and c['unjudgeable'] > .5: note[k] = f"판정 불가({c['unjudgeable']:.2f})"; continue
            raw[k] = (c['index'], c['confidence'])
        # numeric F07, F12
        if self.parser is None or not resident: parser = self.parser or ReleaseParser(model_path=self.s.public_models_dir / 'face_parsing/resnet18.onnx', threads=2)
        else: parser = self.parser
        if resident: self.parser = parser
        mask = parser.parse(img, P3[:, :2]); f07, f12 = M.brow_gap_and_mouth_width(P3, mask); refs = M.reference_measures(P3, mask, self.template)
        for k, r in refs.items():
            if k in detail: detail[k]['reference'] = r  # geometric measure shown next to the CNN result; not used for the judgement
        for k, v in (('F07', f07), ('F12', f12)):
            value[k] = v; f = cfg.numeric['features'][k]; detail[k] = {'measure': S.cut_measure(*M.INFO[k], v, f['cuts'], cfg.classes[k])}
            if v is None: note[k] = '측정 불가'; continue
            raw[k] = S.judge_value(f['sign'] * v, f['cuts'], f['sigma'] * f['sigma_multiplier'])

        def measured(k, v, components=None):
            value[k] = v; f = cfg.measure_spec[k]; detail[k] = {'measure': {**S.cut_measure(*M.INFO[k], v, f['cuts'], cfg.classes[k]), **({'components': components} if components else {})}}
            if v is None: note[k] = '측정 불가'; return
            raw[k] = S.judge_value(v, f['cuts'], f['sigma'] * cfg.k_sigma)
        blink = det['blink'] > .5
        ev, ecomp = (None, []) if blink else M.eye_size_detail(img, P3, cfg.eye['standardisation_train270'])
        measured('F09', None if blink else M.nose_length(P3)); measured('F01', ev, ecomp); measured('F17', M.chin_angle(P3))
        measured('F06', None if blink else refs.get('F06', {}).get('value')); measured('F13', refs['F13']['value']); measured('F15', refs['F15']['value'])  # team rank cuts of 2026-11-17
        # check-only CNNs (optional)
        chk = self._check(X, cnn_all) if check else {}
        # VGGT
        say(STAGES[3]); engine = self.engine or VD.make_engine(self.s.public_models_dir / 'vggt/model.pt', self.dtype)
        if resident: self.engine = engine
        Pv, g = VD.landmark_points(engine, ims, dets, detect)
        if Pv is None or not np.isfinite(Pv[list(range(468))]).all(): note.update({k: 'VGGT로 얼굴 점을 얻지 못함' for k in ('F10', 'F20', 'F21')}); value.update({k: None for k in ('F10', 'F20', 'F21')})
        else:
            measured('F10', self.depth.nose_height(Pv)); c = self.depth.frames(Pv)
            for k in cfg.formula:
                m = cfg.depth['features'][k]; f = cfg.formula_spec[k]; value[k] = self.depth.score(m, c); i, sc = S.judge_value(value[k], f['cuts'], f['sigma'] * cfg.k_sigma); raw[k] = ((2 if i else 1), sc); detail[k] = {'measure': S.cut_measure(*M.INFO[k], value[k], f['cuts'], cfg.classes[k][1:])}
        res['vggt'] = {'dtype': self.dtype, 'network_seconds': round(float(g.get('network_seconds', 0)), 2)}
        for im in g.get('prepared_images', []): im.close()
        del engine, g
        if not resident: torch.cuda.empty_cache()
        # hold rules, passages
        say(STAGES[4]); held = {k: (cfg.threshold(k) is not None and raw[k][1] < cfg.threshold(k)) for k in raw}; J = {k: v for k, v in raw.items() if not held[k]}
        res['judgements'] = {k: {'name': cfg.name[k], 'region': cfg.region[k], 'method': cfg.method[k], 'value': value.get(k), 'class': cfg.classes[k][raw[k][0]] if k in raw else None, 'confidence': raw[k][1] if k in raw else None,
                                 'threshold': cfg.threshold(k), 'status': note.get(k) or ('보류' if held[k] else '확정'), **detail.get(k, {})} for k in cfg.order}
        res['cnn_check'] = chk; res['callouts'] = [{'region': r['region'], 'tone': r['tone'], 'text': r['text']} for r in S.region_phrases(J, cfg)];  # the report sheet of the frontend reads research_report.callouts
         cand = S.matches(J, cfg); sel = S.select_c(cand, cfg); cand_raw = S.matches(raw, cfg); ok_ids = {c[1] for c in cand}
        res['plan_C_selected'] = [S.passage_row(c, J, held, cfg) for c in sel]; res['all_matching'] = [S.passage_row(c, J, held, cfg) for c in cand]
        res['blocked_by_hold'] = [S.passage_row(c, raw, held, cfg) for c in cand_raw if c[1] not in ok_ids]; blocked = {r['representative'] for r in res['blocked_by_hold']}
        res['held_judgements'] = [{'feature': k, 'name': cfg.name[k], 'class': cfg.classes[k][raw[k][0]], 'confidence': raw[k][1], 'threshold': cfg.threshold(k), 'reason': S.held_reason(k, cfg.classes[k][raw[k][0]], blocked, cfg)} for k in cfg.order if held.get(k)]
        res['seconds'] = round(time.perf_counter() - t0, 2); return res

    def _check(self, X, cnn_all):
        """Check-only CNNs: old heads of the structure model set and the trial models (config/check_models.json). Missing model files are skipped."""
        cfg = self.cfg; chk = {}; base = self.s.models_dir / 'candidate-v2-posenorm'; have = lambda folder, part: all((folder / 'models' / f'{part}_fold{k}.pt').exists() for k in range(C.FOLDS))
        for part in [p for p in CHECK_PARTS if p not in self.parts]:
            if have(base, part): cnn_all.update(C.ensemble(self._nets(base, part, False), X[part]))
        for k, o in cfg.check['old_heads'].items():
            if k in cnn_all: chk.setdefault(k, {})[o['tag']] = S.calibrated(*cnn_all[k], o['classes'], o['temperature'])
        for m in cfg.check['models']:
            folder = self.s.check_models_dir / m['folder']
            if have(folder, m['part']): chk.setdefault(m['feature'], {})[m['tag']] = S.calibrated(*C.ensemble(self._nets(folder, m['part'], False), X[m['part']])[m['head']], m['classes'], m['temperature'])
        torch.cuda.empty_cache(); return chk


def open_image(source):
    """Path or bytes -> RGB image with the EXIF orientation applied (decoded in memory)."""
    return ImageOps.exif_transpose(Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)).convert('RGB')


def main():
    ap = ST.add_arguments(argparse.ArgumentParser(description='Run the FaceTeller reference structure on three photos.')); ap.add_argument('front'); ap.add_argument('oblique_1'); ap.add_argument('oblique_2'); ap.add_argument('--out')
    a = ap.parse_args(); paths = [Path(a.front), Path(a.oblique_1), Path(a.oblique_2)]; runner = Runner(ST.resolve(a)); torch.cuda.reset_peak_memory_stats()
    try: res = runner.analyze([open_image(p) for p in paths])
    except PhotoError as e: raise SystemExit(f'얼굴을 찾지 못함: {paths[e.index].name} ({e.reason})')
    res['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if a.out: Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf8')
    out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8'); f2 = lambda x: ' -  ' if x is None else f'{x:.2f}'
    print('VGGT', res['vggt'], 'seconds', res['seconds'], 'GPU peak GiB', res['gpu_peak_gib'], file=out)
    for k, r in res['judgements'].items(): print(f"{k} {r['name']:9s} {r['method']:9s} value {('-' if r['value'] is None else round(r['value'], 4))!s:>9} -> {str(r['class']):8s} conf {f2(r['confidence'])} (>= {f2(r['threshold'])}) {r['status']}", file=out)
    for k, d in res['cnn_check'].items():
        for tag, c in d.items(): print('check', k, tag, c['class'], f2(c['confidence']), file=out)
    for title, key in (('PLAN C', 'plan_C_selected'), ('ALL MATCHING', 'all_matching'), ('BLOCKED BY HOLD', 'blocked_by_hold')):
        print('---', title, len(res[key]), file=out)
        for r in res[key]: print(f"  G {r['G']:.2f} {r['valence']} {' / '.join(r['requirement'])} | {r['source']}: {r['translation']}", file=out)
    out.flush()


if __name__ == '__main__':
    main()
