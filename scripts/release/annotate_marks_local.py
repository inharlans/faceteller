"""Offline, one-annotator / three-light weak annotation. Import is CPU-only."""
import argparse
import hashlib
import io
import importlib.metadata
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'local-visible-marks-v2'
PREPROCESS = 'eye-centre-isotropic-v2'
LIGHTS = ('L1', 'L2', 'L3')
STATUSES = ('clear_present', 'clear_absent', 'uncertain')
LOCATIONS = ('forehead', 'image_left_cheek', 'image_right_cheek', 'nose', 'chin', 'other')
DEFAULT_OUT = ROOT / 'data/release/local_marks/qwen3-vl-2b-v2'
PROMPT = '''The three images are ordered L1, L2, L3: lighting repeats of one participant.
Inspect each image separately for small dark discrete visible skin marks ONLY on exposed forehead and upper cheek skin above the base of the nose.
Exclude hair, eyebrows, eyes, nostrils, lips, lower face/beard, shadows, illumination edges, and compression artifacts. This is NOT medical mole identification. Do not infer identity, age, personality, health, or fortune.
clear_present means an unmistakable small dark discrete mark in this scope. Choose its location; when several exist, choose the clearest repeat-visible mark consistently across images. Left/right mean IMAGE left/right, never anatomical left/right.
clear_absent requires sufficiently clear exposed skin throughout this scope and no visible mark. Occlusion, beard intrusion, blur, shadow, ambiguity, or insufficient coverage means uncertain. Never guess.
Return ONLY one JSON object with exactly this structure, three ordered entries, and no explanation or markdown:
{"images":[{"light":"L1","status":"uncertain","location":null},{"light":"L2","status":"uncertain","location":null},{"light":"L3","status":"uncertain","location":null}]}
status must be clear_present, clear_absent, or uncertain. For clear_present, location must be forehead, image_left_cheek, image_right_cheek, nose, chin, or other. For clear_absent or uncertain, location must be null. No other fields.'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('DUPLICATE_JSON_KEY')
        result[k] = v
    return result


def _json_transport(text):
    """Accept only a whole-response json/plain fence; never salvage prose."""
    match = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*', text)
    if match is not None and '```' not in match.group(1):
        return match.group(1), True
    return text, False


def parse_response(text):
    """Normalize one transport fence; keep JSON and all schema checks strict."""
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError('INVALID_RESPONSE')
    text, _ = _json_transport(text)
    def invalid_constant(_):
        raise ValueError('NONFINITE_JSON')
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError('INVALID_JSON') from exc
    if not isinstance(value, dict) or set(value) != {'images'} or not isinstance(value['images'], list) or len(value['images']) != 3:
        raise ValueError('INVALID_SCHEMA')
    for light, item in zip(LIGHTS, value['images']):
        if not isinstance(item, dict) or set(item) != {'light', 'status', 'location'}:
            raise ValueError('INVALID_SCHEMA')
        if item['light'] != light or item['status'] not in STATUSES:
            raise ValueError('INVALID_LIGHT_OR_STATUS')
        if item['status'] == 'clear_present':
            if item['location'] not in LOCATIONS:
                raise ValueError('INVALID_LOCATION')
        elif item['location'] is not None:
            raise ValueError('INVALID_LOCATION')
    return value


def selected_subjects(groups, smoke):
    sizes = {'train': 270, 'selection': 34, 'calibration': 34, 'test': 62}
    if set(groups) != set(sizes) or any(len(groups[k]) != n for k, n in sizes.items()):
        raise ValueError('UNEXPECTED_FROZEN_SPLIT')
    subjects = {str(s): split for split, ids in groups.items() for s in ids}
    if len(subjects) != 400:
        raise ValueError('DUPLICATE_SUBJECT')
    if smoke:
        train = sorted(map(str, groups['train']))
        # Midpoint of each of twenty equal-width strata, independent of labels.
        ids = [train[((2*i + 1)*len(train)) // 40] for i in range(20)]
    else:
        ids = sorted(subjects)
    return [(sid, subjects[sid]) for sid in ids]


def verify_model(model_dir):
    model_dir = Path(model_dir)
    provenance = load(model_dir / 'provenance.json')
    if provenance.get('repo') != 'Qwen/Qwen3-VL-2B-Instruct' or not provenance.get('revision'):
        raise ValueError('UNEXPECTED_MODEL')
    files = provenance.get('files', {})
    if not {'model.safetensors', 'config.json', 'preprocessor_config.json', 'chat_template.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json'} <= set(files):
        raise ValueError('INCOMPLETE_MODEL_HASHES')
    for name, expected in files.items():
        path = model_dir / name
        if Path(name).name != name or file_hash(path) != expected:
            raise ValueError('MODEL_HASH_MISMATCH')
    return provenance


def configuration(provenance, protocol_hash, runtime_versions=None):
    return {'version': VERSION, 'prompt': PROMPT, 'prompt_sha256': digest(PROMPT),
            'response_transport': 'strict-json-or-single-whole-json-plain-fence-v2',
            'model': provenance, 'protocol_sha256': protocol_hash, 'preprocess_version': PREPROCESS,
            'generation': {'max_new_tokens': 180, 'do_sample': False, 'num_beams': 1,
                           'dtype': 'bfloat16', 'attention': 'sdpa', 'batch_size': 1, 'seed': 0},
            'annotators': 1, 'lighting_repeats': 3, 'human_review_count': 0,
            'runtime_versions': runtime_versions if runtime_versions is not None else
                {name: importlib.metadata.version(name) for name in ('transformers', 'torch', 'accelerate')},
            'implementation_sha256': file_hash(Path(__file__))}


def read_inputs(sid, split, protocol_hash, cache=None, landmarks_dir=None):
    """Verify private local bytes, return PIL images and scalar provenance only."""
    from PIL import Image
    import numpy as np
    cache = Path(cache or ROOT / 'data/derived/release/subjects')
    landmarks_dir = Path(landmarks_dir or ROOT / 'data/derived/kface/subjects')
    meta_path, binary = cache / f'{sid}.json', cache / f'{sid}.bin'
    meta = load(meta_path)
    if (meta.get('subject_id') != sid or meta.get('split') != split or
            meta.get('preprocess_version') != PREPROCESS or meta.get('protocol_sha256') != protocol_hash):
        raise ValueError('SOURCE_IDENTITY_MISMATCH')
    if file_hash(binary) != meta.get('sha256'):
        raise ValueError('CACHE_HASH_MISMATCH')
    points_path = landmarks_dir / f'{sid}.npz'
    with np.load(points_path, allow_pickle=False) as archive:
        keys = list(map(str, archive['keys']))
        points = archive['landmarks'].copy()
        statuses = archive['status'].copy()
    images, sources = [], []
    try:
        with binary.open('rb') as stream:
            for light in LIGHTS:
                frames = [f for f in meta['frames'] if f.get('light') == light]
                capture = f'S001|{light}|E01|C7|core'
                if len(frames) != 1 or frames[0].get('status') != 'ready' or frames[0].get('capture') != capture or keys.count(capture) != 1:
                    raise ValueError('MISSING_OR_REPEATED_CAPTURE')
                crop = frames[0]['crops']['mole']
                offset, length = crop['offset'], crop['length']
                if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or offset + length > binary.stat().st_size:
                    raise ValueError('INVALID_CROP_RANGE')
                if crop.get('size') != [352, 416] or not .95 <= crop.get('valid_fraction', 0) <= 1:
                    raise ValueError('INSUFFICIENT_CROP_COVERAGE')
                stream.seek(offset)
                payload = stream.read(length)
                with Image.open(io.BytesIO(payload)) as original:
                    if original.format != 'PNG' or original.size != (352, 416):
                        raise ValueError('INVALID_PNG')
                    image = original.convert('RGB')
                images.append(image)
                index = keys.index(capture)
                p = np.asarray(points[index], dtype=float)[:, :2]
                if statuses[index] != 1 or len(p) < 468 or not np.isfinite(p).all():
                    raise ValueError('INVALID_LANDMARKS')
                left, right = (p[33]+p[133])/2, (p[263]+p[362])/2
                direction = right-left
                distance = np.linalg.norm(direction)
                if distance < 4:
                    raise ValueError('INVALID_EYE_SCALE')
                unit = direction/distance
                q = (p-(left+right)/2) @ np.array([unit, [-unit[1], unit[0]]]).T / distance
                sources.append({'light': light, 'capture': capture,
                    'png_sha256': hashlib.sha256(payload).hexdigest(),
                    'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                    'crop': crop, 'region_bounds': {'forehead_max_y': float(min(q[107,1],q[336,1])-.05),
                    'cheek_min_y': .12, 'cheek_max_y': float(q[2,1]-.04), 'cheek_min_abs_x': .25}})
        source = {'subject_id': sid, 'split': split, 'protocol_sha256': protocol_hash,
                  'preprocess_version': PREPROCESS, 'metadata_sha256': file_hash(meta_path),
                  'binary_sha256': meta['sha256'], 'landmarks_sha256': file_hash(points_path), 'frames': sources}
        validate_source(source, sid, split, protocol_hash)
        return images, source
    except Exception:
        for image in images:
            image.close()
        raise


def validate_source(source, sid, split, protocol_hash):
    if any(source.get(k) != v for k, v in {'subject_id': sid, 'split': split,
            'protocol_sha256': protocol_hash, 'preprocess_version': PREPROCESS}.items()):
        raise ValueError('SOURCE_IDENTITY_MISMATCH')
    frames = source.get('frames', [])
    if len(frames) != 3 or [f.get('light') for f in frames] != list(LIGHTS):
        raise ValueError('INVALID_SOURCE_LIGHTS')
    for field in ('capture', 'png_sha256', 'pixels_sha256'):
        values = [f.get(field) for f in frames]
        if not all(isinstance(v, str) and v for v in values) or len(set(values)) != 3:
            raise ValueError('REPEATED_IMAGE_IDENTITY')
    for frame in frames:
        if frame['capture'] != f"S001|{frame['light']}|E01|C7|core":
            raise ValueError('INVALID_CAPTURE')


def validate_record(record, sid, split, config, source=None):
    if record.get('subject_id') != sid or record.get('split') != split or record.get('config_sha256') != digest(config):
        raise ValueError('RESUME_IDENTITY_MISMATCH')
    stored = record.get('source')
    if stored is not None:
        validate_source(stored, sid, split, config['protocol_sha256'])
        if record.get('source_sha256') != digest(stored):
            raise ValueError('SOURCE_HASH_MISMATCH')
    if source is not None and stored != source:
        raise ValueError('RESUME_SOURCE_CHANGED')
    if record.get('status') == 'ok':
        if stored is None:
            raise ValueError('MISSING_SOURCE')
        parse_response(json.dumps(record.get('response')))
    elif record.get('status') != 'unknown' or record.get('response') is not None or not record.get('error'):
        raise ValueError('INVALID_RECORD')


def annotate(generate, images, source, config):
    record = {'subject_id': source['subject_id'], 'split': source['split'],
              'config_sha256': digest(config), 'source': source, 'source_sha256': digest(source),
              'status': 'unknown', 'response': None, 'error': None}
    try:
        raw = generate(images)
        record['response_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        record['format_diagnostics'] = {'characters': len(raw),
            'single_complete_json_fence': _json_transport(raw)[1]}
        generation = getattr(generate, 'format_diagnostics', {})
        for key in ('generated_tokens', 'max_new_tokens', 'reached_token_limit', 'last_token_is_eos'):
            if key in generation:
                record['format_diagnostics'][key] = generation[key]
        record['response'] = parse_response(raw)
        record['status'] = 'ok'
    except Exception as exc:
        # Never persist freeform model output, diagnoses, images, or exception payloads.
        record['error'] = {'stage': 'generation_or_validation', 'type': type(exc).__name__}
        if isinstance(exc, ValueError) and str(exc) in {
                'INVALID_RESPONSE', 'INVALID_JSON', 'INVALID_SCHEMA',
                'INVALID_LIGHT_OR_STATUS', 'INVALID_LOCATION'}:
            record['error']['code'] = str(exc)
        if isinstance(exc.__cause__, json.JSONDecodeError):
            record['error']['json_error_position'] = exc.__cause__.pos
    return record


def initialize_or_record_failure(model_dir, images, source, config, path):
    try:
        return load_generator(model_dir)
    except Exception as exc:
        record = {'subject_id': source['subject_id'], 'split': source['split'],
                  'config_sha256': digest(config), 'source': source, 'source_sha256': digest(source),
                  'status': 'unknown', 'response': None,
                  'error': {'stage': 'model_initialization', 'type': type(exc).__name__}}
        save(path, record)
        # Stop the run instead of attempting a broken/OOM initialization 400 times.
        raise RuntimeError('MODEL_INITIALIZATION_FAILED_UNKNOWN_RECORDED') from None


def load_generator(model_dir):
    """The only model/GPU entry point; never called by preflight or CPU tests."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    import transformers
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    if transformers.__version__ != '4.57.6':
        raise RuntimeError('PINNED_TRANSFORMERS_4_57_6_REQUIRED')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA_BF16_REQUIRED')
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 'cuda:0'})
    model.eval()
    def generate(images):
        content = []
        for light, image in zip(LIGHTS, images):
            content.extend([{'type': 'text', 'text': light}, {'type': 'image', 'image': image}])
        content.append({'type': 'text', 'text': PROMPT})
        inputs = processor.apply_chat_template([{'role': 'user', 'content': content}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda:0')
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=180, do_sample=False, num_beams=1,
                                    use_cache=True)
        tokens = output[0, inputs['input_ids'].shape[-1]:]
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else (eos or [])
        generate.format_diagnostics = {'generated_tokens': len(tokens), 'max_new_tokens': 180,
            'reached_token_limit': len(tokens) >= 180,
            'last_token_is_eos': bool(len(tokens) and tokens[-1].item() in eos)}
        return processor.decode(tokens, skip_special_tokens=True)
    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--all', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='hash/input checks only; no model or GPU import')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'models/annotation/qwen3-vl-2b')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    protocol_path = ROOT/'models/selection-20261014/protocol.json'
    protocol = load(protocol_path)
    selected = selected_subjects(protocol['groups'], args.smoke)
    config = configuration(verify_model(args.model_dir), file_hash(protocol_path))
    manifest = args.output_dir/'manifest.json'
    if manifest.exists() and load(manifest) != config:
        raise ValueError('MANIFEST_CHANGED_USE_NEW_OUTPUT_DIRECTORY')
    save(manifest, config)
    report = {'mode': 'smoke' if args.smoke else 'all', 'preflight_only': args.preflight_only,
              'config_sha256': digest(config), 'participants': [], 'model_loaded': False}
    generate = None
    for sid, split in selected:
        path = args.output_dir/'records'/f'{sid}.json'
        images = []
        try:
            images, source = read_inputs(sid, split, config['protocol_sha256'])
        except Exception as exc:
            record = {'subject_id': sid, 'split': split, 'config_sha256': digest(config),
                      'source': None, 'source_sha256': None, 'response': None, 'status': 'unknown',
                      'error': {'stage': 'input', 'type': type(exc).__name__}}
            # Input failures are always rechecked; never overwrite a completed annotation.
            if path.exists():
                previous = load(path)
                validate_record(previous, sid, split, config)
                if previous.get('source') is not None or previous.get('error', {}).get('stage') != 'input':
                    raise ValueError('EXISTING_RECORD_SOURCE_UNVERIFIABLE') from exc
            if not args.preflight_only:
                save(path, record)
            report['participants'].append({'subject_id': sid, 'split': split, 'status': 'input_unknown'})
            continue
        try:
            if args.preflight_only:
                status = 'input_verified'
            elif path.exists():
                record = load(path)
                if record.get('source') is None and record.get('error', {}).get('stage') == 'input':
                    # Recovered input: now annotate; prior input error was not a model result.
                    validate_record(record, sid, split, config)
                    if generate is None:
                        generate = initialize_or_record_failure(args.model_dir, images, source, config, path)
                        report['model_loaded'] = True
                    record = annotate(generate, images, source, config)
                    save(path, record)
                else:
                    validate_record(record, sid, split, config, source)
                status = record['status']
            else:
                if generate is None:
                    generate = initialize_or_record_failure(args.model_dir, images, source, config, path)
                    report['model_loaded'] = True
                record = annotate(generate, images, source, config)
                save(path, record)
                status = record['status']
            report['participants'].append({'subject_id': sid, 'split': split, 'status': status,
                                          'source_sha256': digest(source)})
            print(f'{len(report["participants"])}/{len(selected)} {sid} {status}', flush=True)
        finally:
            for image in images:
                image.close()
    name = report['mode'] + ('-preflight' if args.preflight_only else '-run') + '.json'
    save(args.output_dir/name, report)


if __name__ == '__main__':
    main()
