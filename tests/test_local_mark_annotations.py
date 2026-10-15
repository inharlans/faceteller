"""CPU fixtures only: no transformers, model, network, or GPU initialization."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.release import annotate_marks_local as a
from scripts.release import merge_mark_labels as merger
from scripts.release.merge_mark_labels import merge_subject


def response(statuses=('clear_present', 'clear_present', 'uncertain'), location='image_left_cheek'):
    return {'images': [{'light': light, 'status': status,
                        'location': location if status == 'clear_present' else None}
                       for light, status in zip(a.LIGHTS, statuses)]}


def source():
    return {'subject_id': 'sample', 'split': 'train', 'protocol_sha256': 'protocol',
            'preprocess_version': a.PREPROCESS,
            'frames': [{'light': light, 'capture': f'S001|{light}|E01|C7|core',
                        'png_sha256': 'png'+light, 'pixels_sha256': 'pixels'+light,
                        'region_bounds': {'forehead_max_y': -.3, 'cheek_min_y': .12,
                                          'cheek_max_y': .8, 'cheek_min_abs_x': .25}}
                       for light in a.LIGHTS]}


def cv_evidence():
    mark = {'x_eye': -.65, 'y_eye': .4}
    return {'version': 'visible-upper-skin-dark-marks-v1', 'label': None,
            'reason': 'UNSTABLE_DARK_CANDIDATES', 'stable_marks': [{**mark, 'repeats': 2}],
            'frames': [{'light': light, 'quality_passed': True, 'coverage': .9,
                        'candidates': [dict(mark)] if light != 'L3' else [],
                        'rejected_components': 0} for light in a.LIGHTS]}


class AnnotationTests(unittest.TestCase):
    def test_strict_json_accepts_exact_shape(self):
        value = response()
        self.assertEqual(a.parse_response(json.dumps(value)), value)

    def test_rejects_bad_json_partial_duplicate_extra_and_invalid_enums(self):
        good = response()
        mutations = []
        for change in ('partial', 'light', 'enum', 'extra', 'location', 'nonpresent_location'):
            v = copy.deepcopy(good)
            if change == 'partial': v['images'].pop()
            if change == 'light': v['images'][1]['light'] = 'L1'
            if change == 'enum': v['images'][0]['status'] = 'mole'
            if change == 'extra': v['diagnosis'] = 'not allowed'
            if change == 'location': v['images'][0]['location'] = 'left_cheek'
            if change == 'nonpresent_location': v['images'][2]['location'] = 'forehead'
            mutations.append(json.dumps(v))
        mutations += ['{', 'null',
                      '{"images":[],"images":[]}', '{"images":NaN}', json.dumps(good)+' explanation']
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.parse_response(value)

    def test_single_transport_fence_preserves_exact_schema(self):
        good = response()
        for opening, newline in [('```json', '\n'), ('```', '\r\n')]:
            raw = ' \n'+opening+newline+json.dumps(good)+newline+'```\n '
            self.assertEqual(a.parse_response(raw), good)
            record = a.annotate(lambda _: raw, [], source(), {'version': 'fixture'})
            self.assertEqual(record['status'], 'ok')
            self.assertTrue(record['format_diagnostics']['single_complete_json_fence'])
            self.assertEqual(record['response_sha256'], hashlib.sha256(raw.encode()).hexdigest())
            self.assertNotIn('```', json.dumps(record))

    def test_fence_never_salvages_prose_multiple_blocks_or_invalid_schema(self):
        good = json.dumps(response())
        fence = '```json\n'+good+'\n```'
        bad = ['explanation\n'+fence, fence+'\nexplanation', fence+'\n'+fence,
               '```python\n'+good+'\n```', '```json\n'+good,
               '```json\n{"images":[],"images":[]}\n```',
               '```json\n{"images":NaN}\n```',
               '```json\n'+json.dumps({'images':response()['images'][:2]})+'\n```']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                a.parse_response(raw)

    def test_format_failure_records_only_allowlisted_diagnostics(self):
        raw = 'PRIVATE FREEFORM TEXT'
        record = a.annotate(lambda _: raw, [], source(), {'version': 'fixture'})
        self.assertEqual(record['error']['code'], 'INVALID_JSON')
        self.assertEqual(record['error']['json_error_position'], 0)
        self.assertEqual(record['format_diagnostics']['characters'], len(raw))
        self.assertNotIn(raw, json.dumps(record))
        def generate(_):
            raise ValueError('PRIVATE EXCEPTION TEXT')
        record = a.annotate(generate, [], source(), {'version': 'fixture'})
        self.assertNotIn('code', record['error'])
        self.assertNotIn('PRIVATE EXCEPTION TEXT', json.dumps(record))

    def test_smoke_is_systematic_train_only_and_split_disjoint(self):
        groups = {'train': [f'{i:04}' for i in range(270)],
                  'selection': [f'{i:04}' for i in range(270,304)],
                  'calibration': [f'{i:04}' for i in range(304,338)],
                  'test': [f'{i:04}' for i in range(338,400)]}
        chosen = a.selected_subjects(groups, True)
        self.assertEqual(len(set(chosen)), 20)
        self.assertTrue(all(split == 'train' and sid in groups['train'] for sid,split in chosen))
        self.assertEqual(chosen, a.selected_subjects({k:list(reversed(v)) for k,v in groups.items()}, True))
        groups['test'][0] = groups['train'][0]
        with self.assertRaises(ValueError): a.selected_subjects(groups, True)

    def test_model_or_parse_error_is_unknown_without_freeform(self):
        for generate in (lambda _: 'diagnosis forbidden', lambda _: (_ for _ in ()).throw(RuntimeError('private text'))):
            result = a.annotate(generate, [], source(), {'version': 'fixture'})
            self.assertEqual(result['status'], 'unknown')
            self.assertIsNone(result['response'])
            self.assertNotIn('private text', json.dumps(result))
            self.assertNotIn('diagnosis forbidden', json.dumps(result))

    def test_initialization_failure_saved_then_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            with patch.object(a, 'load_generator', side_effect=RuntimeError('sensitive payload')):
                with self.assertRaisesRegex(RuntimeError, 'UNKNOWN_RECORDED'):
                    a.initialize_or_record_failure(directory, [], source(), {}, path)
            record = a.load(path)
            self.assertEqual(record['status'], 'unknown')
            self.assertEqual(record['error']['stage'], 'model_initialization')
            self.assertNotIn('sensitive payload', json.dumps(record))

    def test_model_file_hashes_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ('model.safetensors', 'config.json', 'preprocessor_config.json', 'chat_template.json',
                     'tokenizer_config.json', 'tokenizer.json', 'generation_config.json')
            files = {}
            for name in names:
                (root/name).write_bytes(b'fixture only')
                files[name] = a.file_hash(root/name)
            a.save(root/'provenance.json', {'repo': 'Qwen/Qwen3-VL-2B-Instruct', 'revision': 'fixture', 'files': files})
            a.verify_model(root)
            (root/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'MODEL_HASH_MISMATCH'): a.verify_model(root)

    def test_resume_detects_participant_config_and_source_changes(self):
        config = {'protocol_sha256': 'protocol'}
        record = a.annotate(lambda _: json.dumps(response()), [], source(), config)
        a.validate_record(record, 'sample', 'train', config, source())
        for field in ('subject_id', 'config_sha256', 'source_sha256'):
            changed = copy.deepcopy(record); changed[field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.validate_record(changed, 'sample', 'train', config, source())
        changed_source = source(); changed_source['frames'][0]['png_sha256'] = 'new'
        with self.assertRaises(ValueError): a.validate_record(record, 'sample', 'train', config, changed_source)

    def test_repeat_light_capture_and_pixel_identity_rejected(self):
        for field in ('light', 'capture', 'png_sha256', 'pixels_sha256'):
            value = source(); value['frames'][1][field] = value['frames'][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.validate_source(value, 'sample', 'train', 'protocol')

    def test_input_cache_hash_decode_and_repeat_pixels(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = bytearray(); frames = []
            for i, light in enumerate(a.LIGHTS):
                buffer = io.BytesIO()
                with Image.new('RGB', (352,416), (80+i*10,90,100)) as image:
                    image.save(buffer, format='PNG')
                data = buffer.getvalue()
                frames.append({'light': light, 'capture': f'S001|{light}|E01|C7|core', 'status': 'ready',
                               'crops': {'mole': {'offset': len(payload), 'length': len(data),
                                                  'size': [352,416], 'valid_fraction': 1.}}})
                payload.extend(data)
            meta = {'subject_id': 'sample', 'split': 'train', 'preprocess_version': a.PREPROCESS,
                    'protocol_sha256': 'protocol', 'sha256': hashlib.sha256(payload).hexdigest(), 'frames': frames}
            (root/'sample.bin').write_bytes(payload)
            a.save(root/'sample.json', meta)
            p = np.zeros((3,468,3)); p[:,[33,133],:2] = [100,100]; p[:,[263,362],:2] = [200,100]
            p[:,[107,336],:2] = [150,70]; p[:,2,:2] = [150,170]
            np.savez(root/'sample.npz', keys=np.array([f['capture'] for f in frames]), landmarks=p, status=np.ones(3))
            images, src = a.read_inputs('sample', 'train', 'protocol', root, root)
            self.assertEqual(len(images), 3)
            self.assertAlmostEqual(src['frames'][0]['region_bounds']['forehead_max_y'], -.35)
            for image in images: image.close()
            # Different light names cannot disguise reusing the exact same crop.
            meta['frames'][1]['crops']['mole'] = dict(meta['frames'][0]['crops']['mole'])
            a.save(root/'sample.json', meta)
            with self.assertRaisesRegex(ValueError, 'REPEATED_IMAGE_IDENTITY'):
                a.read_inputs('sample', 'train', 'protocol', root, root)
            (root/'sample.bin').write_bytes(payload+b'changed')
            with self.assertRaisesRegex(ValueError, 'CACHE_HASH_MISMATCH'):
                a.read_inputs('sample', 'train', 'protocol', root, root)

    def test_preflight_never_loads_model(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = {'train': [str(i) for i in range(270)], 'selection': [str(i) for i in range(270,304)],
                      'calibration': [str(i) for i in range(304,338)], 'test': [str(i) for i in range(338,400)]}
            with patch('sys.argv', ['annotate', '--smoke', '--preflight-only', '--output-dir', directory]), \
                 patch.object(a, 'load', return_value={'groups': groups}), \
                 patch.object(a, 'file_hash', return_value='hash'), \
                 patch.object(a, 'verify_model', return_value={'fixture': True}), \
                 patch.object(a.importlib.metadata, 'version', return_value='fixture'), \
                 patch.object(a, 'read_inputs', return_value=([], source())), \
                 patch.object(a, 'load_generator') as loader, patch('builtins.print'):
                a.main()
                loader.assert_not_called()
            report = json.loads((Path(directory)/'smoke-preflight.json').read_text())
            self.assertEqual(len(report['participants']),20)
            self.assertFalse(report['model_loaded'])
            self.assertFalse((Path(directory)/'records').exists())


class MergeTests(unittest.TestCase):
    def test_optional_weak_negative_predeclared_gates_and_positive_invariance(self):
        cv = cv_evidence(); cv['stable_marks'] = []
        cv['frames'][0]['rejected_components'] = 2
        absent = response(('clear_absent',)*3)
        weak = 'vlm_three_absent_no_stable'
        self.assertIsNone(merge_subject(absent, cv, source())['label'])
        self.assertEqual(merge_subject(absent, cv, source(), weak)['label'], 'absent')
        self.assertEqual(merge_subject(response(), cv_evidence(), source(), weak)['label'], 'present')
        for field, value in (('quality_passed',False), ('coverage',.64), ('light','L2')):
            changed = copy.deepcopy(cv); changed['frames'][0][field] = value
            self.assertIsNone(merge_subject(absent, changed, source(), weak)['label'])
        self.assertIsNone(merge_subject(absent, cv_evidence(), source(), weak)['label'])
        self.assertIsNone(merge_subject(response(('clear_absent','clear_absent','uncertain')), cv, source(), weak)['label'])
        self.assertIsNone(merge_subject(response(('clear_absent','clear_absent','clear_present')), cv, source(), weak)['label'])
        with self.assertRaises(ValueError): merge_subject(absent, cv, source(), 'invented')

    def test_eligibility_is_only_an_upper_bound_and_policy_hashes_differ(self):
        cv = cv_evidence(); cv['stable_marks'] = []
        original = {'groups': {'train': ['sample']}, 'metrics': {'sample': {'pigmented_marks': cv}}}
        report = merger.eligibility_report(original)
        self.assertEqual(report['policies']['strict']['upper_bound_total'], 0)
        self.assertEqual(report['policies']['vlm_three_absent_no_stable']['upper_bound_total'], 1)
        self.assertEqual(report['actual_vlm_labels'],0)
        self.assertFalse(report['is_label_artifact'])
        self.assertNotEqual(report['policies']['strict']['policy_sha256'], report['policies']['vlm_three_absent_no_stable']['policy_sha256'])

    def test_merge_cli_overlay_preserves_original_and_rejects_wrong_participant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            groups = {'train': [str(i) for i in range(270)], 'selection': [str(i) for i in range(270,304)],
                      'calibration': [str(i) for i in range(304,338)], 'test': [str(i) for i in range(338,400)]}
            protocol = root/'models/selection-20261014/protocol.json'
            a.save(protocol, {'groups': groups})
            protocol_hash = a.file_hash(protocol)
            original = {'groups': groups, 'protocol_sha256': protocol_hash,
                        'subjects': {str(i): {'pigmented_marks':'present'} for i in range(400)},
                        'metrics': {str(i): {'pigmented_marks':cv_evidence()} for i in range(400)}}
            original_path = root/'data/release/extended_labels.json'
            a.save(original_path, original)
            original_bytes = original_path.read_bytes()
            annotations = root/'annotations'
            config = a.configuration({'fixture':True}, protocol_hash, {'transformers':'4.57.6'})
            a.save(annotations/'manifest.json', config)
            src = source(); src.update(subject_id='0', protocol_sha256=protocol_hash)
            record = a.annotate(lambda _: json.dumps(response()), [], src, config)
            record_path = annotations/'records/0.json'
            a.save(record_path, record)
            output = root/'merged_mark_labels.json'
            with patch.object(merger, 'ROOT', root), \
                 patch.object(merger, 'read_inputs', return_value=([], src)), \
                 patch('sys.argv', ['merge', '--annotations',str(annotations),'--output',str(output)]), \
                 patch('builtins.print'):
                merger.main()
                result = a.load(output)
                self.assertEqual(result['subjects']['0']['pigmented_marks'], 'present')
                self.assertIsNone(result['subjects']['1']['pigmented_marks'])
                self.assertEqual(result['class_support']['train'], {'present':1,'absent':0,'unknown':269})
                self.assertFalse(result['training_ready'])
                self.assertEqual(original_path.read_bytes(), original_bytes)
                record['subject_id'] = '1'; a.save(record_path, record)
                with self.assertRaisesRegex(ValueError, 'IDENTITY_MISMATCH'): merger.main()

    def test_stable_candidates_can_corroborate_strict_unknown(self):
        cv = cv_evidence()
        cv['frames'][2]['candidates'] = [{'x_eye': .7, 'y_eye': .6}]
        result = merge_subject(response(), cv, source())
        self.assertEqual(result['label'], 'present')
        self.assertEqual(result['matched_marks'][0]['lights'], ['L1','L2'])

    def test_strict_positive_requires_corroboration(self):
        cv = cv_evidence(); cv['label'] = 'present'
        for value in (None, response(('uncertain',)*3), response(('clear_present','uncertain','uncertain'))):
            self.assertIsNone(merge_subject(value, cv, source())['label'])

    def test_contradiction_wrong_region_or_wrong_lights_unknown(self):
        variants = [response(('clear_present','clear_present','clear_absent')),
                    response(location='image_right_cheek'), response(location='nose'),
                    response(('uncertain','clear_present','clear_present'))]
        mismatch = response(); mismatch['images'][1]['location'] = 'forehead'; variants.append(mismatch)
        for value in variants:
            with self.subTest(response=value):
                self.assertIsNone(merge_subject(value, cv_evidence(), source())['label'])

    def test_negative_requires_three_clear_and_no_candidates_or_texture(self):
        cv = cv_evidence(); cv['stable_marks'] = []
        for frame in cv['frames']: frame['candidates'] = []
        absent = response(('clear_absent',)*3)
        self.assertEqual(merge_subject(absent, cv, source())['label'], 'absent')
        for field, value in (('coverage',.64), ('quality_passed',False), ('rejected_components',1),
                             ('candidates',[{'x_eye':.5,'y_eye':.4}])):
            changed = copy.deepcopy(cv); changed['frames'][0][field] = value
            with self.subTest(field=field): self.assertIsNone(merge_subject(absent, changed, source())['label'])
        self.assertIsNone(merge_subject(response(('clear_absent','clear_absent','uncertain')),cv,source())['label'])

    def test_corrupt_partial_repeated_cv_unknown(self):
        for field, value in (('frames',cv_evidence()['frames'][:2]), ('version','unknown'),
                             ('stable_marks',[{'x_eye':float('nan'),'y_eye':.4,'repeats':2}])):
            cv = cv_evidence(); cv[field] = value
            self.assertIsNone(merge_subject(response(), cv, source())['label'])
        cv = cv_evidence(); cv['frames'][1]['light'] = 'L1'
        self.assertIsNone(merge_subject(response(), cv, source())['label'])
        src = source(); src['frames'][1]['pixels_sha256'] = src['frames'][0]['pixels_sha256']
        self.assertIsNone(merge_subject(response(), cv_evidence(), src)['label'])
        self.assertIsNone(merge_subject({'images':[]}, cv_evidence(), source())['label'])


if __name__ == '__main__':
    unittest.main()
