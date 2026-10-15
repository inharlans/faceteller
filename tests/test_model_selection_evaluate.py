"""CPU-only provenance regressions using tiny synthetic artifacts in temp dirs.

No real training files, checkpoints, images, evaluation stages, or GPU are used.
"""
import copy
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import torch

torch.set_num_threads(2)
from scripts import model_selection_evaluate as ev
from scripts.model_selection_metrics import aggregate_metrics, fit_temperature, paired_bootstrap


class EvaluationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / 'models'; self.out.mkdir()
        self.report = Path(self.temp.name) / 'report'; self.report.mkdir()
        self.parts = ['eye', 'brow', 'nose', 'mouth', 'face']
        self.arches = ['resnet18', 'efficientnet_b0', 'convnext_tiny']
        self.seeds = [20261014, 20261015, 20261016]
        self.heads = {part: [part + str(i) for i in range(3)] for part in self.parts}
        self.schema = {k: {'part': part, 'source': 'visual', 'classes': ['a', 'b']}
                       for part, keys in self.heads.items() for k in keys}
        self.groups = {split: [split + '0', split + '1'] for split in ('train', 'selection', 'calibration', 'test')}
        self.refs = {sid: {k: ['a', 'b'][i] for k in self.schema}
                     for ids in self.groups.values() for i, sid in enumerate(ids)}
        self.protocol = {'groups': self.groups, 'heads': self.heads, 'version': 'synthetic-test-only'}
        self.context = self.refs, self.groups, self.heads, self.protocol
        self.families = {'+'.join(c): list(c) for size in (1, 2, 3)
                         for c in itertools.combinations(self.arches, size)}
        replacements = dict(OUT=self.out, REPORT=self.report, SCHEMA=self.schema, PARTS=self.parts,
                            ARCHES=self.arches, SEEDS=self.seeds, FAMILIES=self.families,
                            setup=Mock(return_value=self.context))
        binding = patch.multiple(ev, **replacements); binding.start(); self.addCleanup(binding.stop)
        guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('GPU prohibited in CPU tests'))
        guard.start(); self.addCleanup(guard.stop)
        output = redirect_stdout(io.StringIO()); output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        ev.dump(self.out / 'protocol.json', self.protocol)
        self.prob = [[.8, .2], [.2, .8]]
        for arch in self.arches:
            for seed in self.seeds:
                folder = self.out / arch / str(seed); folder.mkdir(parents=True)
                for part in self.parts:
                    meta = {'arch': arch, 'part': part, 'seed': seed,
                            'protocol_sha256': ev.sha(self.out / 'protocol.json'),
                            'classification_features': self.heads[part], 'regression_features': [],
                            'classes': {k: ['a', 'b'] for k in self.heads[part]},
                            'train_people': 2, 'selection_people': 2, 'calibration_people': 2}
                    predictions = {k: {'truth': [0, 1], 'prob': self.prob} for k in self.heads[part]}
                    ev.dump(folder / (part + '.json'), {'meta': meta, 'selection': predictions, 'calibration': predictions})
                    (folder / (part + '.pt')).write_bytes(f'synthetic weights {arch} {seed} {part}'.encode())
        y = {k: np.array([0, 1]) for k in self.schema}
        p = {k: np.array(self.prob) for k in self.schema}
        metric = aggregate_metrics(y, p)
        fitted = fit_temperature(self.prob, np.array([0, 1]))
        families = {name: {'architectures': arches, 'runs': {
            str(seed): {'temperatures': {k: copy.deepcopy(fitted) for k in self.schema},
                        'raw': metric, 'calibrated': metric} for seed in self.seeds}}
                    for name, arches in self.families.items()}
        ev.dump(self.report / 'development.json', {'protocol_sha256': ev.sha(self.out / 'protocol.json'),
                'training_manifest': ev.training_manifest(), 'families': families})
        ev.dump(self.report / 'latency.json', {'protocol_sha256': ev.sha(self.out / 'protocol.json'),
                'development_sha256': ev.sha(self.report / 'development.json'),
                'families': {name: {'mean_ms': float(i + 1)} for i, name in enumerate(self.families)}})

    def freeze(self):
        ev.freeze()
        return ev.load(self.report / 'choice_frozen.json')

    def write_tests(self, choice):
        for arch in self.arches:
            for seed in self.seeds:
                for part in self.parts:
                    base = f'{arch}/{seed}/{part}'
                    value = {'predictions': {k: {'truth': [0, 1], 'prob': self.prob} for k in self.heads[part]},
                             'subject_ids': self.groups['test'], 'classes': {k: ['a', 'b'] for k in self.heads[part]},
                             'choice_sha256': ev.sha(self.report / 'choice_frozen.json'),
                             'checkpoint_sha256': choice['training_manifest'][base + '.pt'],
                             'metadata_sha256': choice['training_manifest'][base + '.json']}
                    ev.dump(self.out / (base + '_test.json'), value)

    def first_test(self):
        return self.out / self.arches[0] / str(self.seeds[0]) / 'eye_test.json'

    def test_freeze_snapshots_all_45_checkpoints_and_metadata(self):
        choice = self.freeze()
        self.assertEqual(len(choice['training_manifest']), 90)
        self.assertEqual(sum(k.endswith('.pt') for k in choice['training_manifest']), 45)
        self.assertFalse(choice['test_predictions_present_at_freeze'])
        self.assertEqual(choice['selected_family'], self.arches[0])
        self.assertEqual(ev.validate_frozen()[0], choice)

    def test_modified_checkpoint_or_metadata_rejected(self):
        self.freeze()
        for suffix in ('.pt', '.json'):
            path = self.out / self.arches[0] / str(self.seeds[0]) / ('eye' + suffix)
            original = path.read_bytes()
            with self.subTest(suffix=suffix):
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(RuntimeError, 'changed training artifact'):
                    ev.validate_frozen()
            path.write_bytes(original)

    def test_reuse_rejects_wrong_choice_checkpoint_and_metadata_hash(self):
        self.write_tests(self.freeze()); path = self.first_test(); original = ev.load(path)
        for field in ('choice_sha256', 'checkpoint_sha256', 'metadata_sha256'):
            value = copy.deepcopy(original); value[field] = 'wrong'; ev.dump(path, value)
            with self.subTest(field=field), patch.object(ev, 'prepare') as prepare:
                with self.assertRaisesRegex(RuntimeError, 'stale test'):
                    ev.test_inference()
                prepare.assert_not_called()
        ev.dump(path, original)

    def test_reuse_rejects_participant_and_class_order(self):
        self.write_tests(self.freeze()); path = self.first_test(); original = ev.load(path)
        for field in ('people', 'classes'):
            value = copy.deepcopy(original)
            if field == 'people': value['subject_ids'].reverse()
            else: value['classes']['eye0'].reverse()
            ev.dump(path, value)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'order mismatch'):
                ev.summary()

    def test_reuse_rejects_truth_head_and_shape(self):
        self.write_tests(self.freeze()); path = self.first_test(); original = ev.load(path)
        for mode in ('truth', 'head', 'shape', 'probability'):
            value = copy.deepcopy(original)
            if mode == 'truth': value['predictions']['eye0']['truth'].reverse()
            elif mode == 'head': del value['predictions']['eye0']
            elif mode == 'shape': value['predictions']['eye0']['prob'] = [[.5, .5]]
            else: value['predictions']['eye0']['prob'][0] = [.2, .2]
            ev.dump(path, value)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError): ev.summary()

    def test_changed_latency_rejected_before_summary_work(self):
        self.freeze()
        path = self.report / 'latency.json'; value = ev.load(path)
        value['families'][self.arches[0]]['mean_ms'] += 10; ev.dump(path, value)
        with patch.object(ev, 'paired_bootstrap') as boot, patch.object(ev, 'test_records') as tests:
            with self.assertRaisesRegex(RuntimeError, 'latency_sha256 mismatch'): ev.summary()
            boot.assert_not_called(); tests.assert_not_called()

    def test_changed_protocol_or_development_rejected_early(self):
        self.freeze()
        for path in (self.out / 'protocol.json', self.report / 'development.json'):
            original = path.read_bytes(); path.write_bytes(original + b' ')
            with self.subTest(path=path), patch.object(ev, 'training_manifest') as manifest:
                with self.assertRaisesRegex(RuntimeError, 'frozen .* mismatch'): ev.summary()
                manifest.assert_not_called()
            path.write_bytes(original)

    def test_existing_test_file_prevents_any_choice_write(self):
        ev.dump(self.first_test(), {'stale': True})
        with self.assertRaisesRegex(RuntimeError, 'test predictions already exist'): ev.freeze()
        self.assertFalse((self.report / 'choice_frozen.json').exists())

    def test_existing_choice_is_validated_and_not_rewritten(self):
        self.freeze(); path = self.report / 'choice_frozen.json'; original = path.read_bytes()
        ev.freeze(); self.assertEqual(path.read_bytes(), original)
        weight = self.out / self.arches[0] / str(self.seeds[0]) / 'eye.pt'
        weight.write_bytes(b'replaced')
        with self.assertRaisesRegex(RuntimeError, 'changed training artifact'): ev.freeze()
        self.assertEqual(path.read_bytes(), original)

    def test_changed_frozen_selection_is_rejected(self):
        choice = self.freeze(); choice['selected_family'] = self.arches[1]
        ev.dump(self.report / 'choice_frozen.json', choice)
        with self.assertRaisesRegex(RuntimeError, 'frozen decision mismatch'): ev.validate_frozen()

    def test_valid_resume_skips_all_image_and_model_work(self):
        self.write_tests(self.freeze())
        with patch.object(ev, 'prepare') as prepare, patch.object(ev, 'load_model') as model, patch.object(ev, 'predict') as predict:
            ev.test_inference()
            prepare.assert_not_called(); model.assert_not_called(); predict.assert_not_called()

    def test_valid_summary_end_to_end_without_gpu(self):
        self.write_tests(self.freeze())
        def small_boot(*args, **kwargs):
            return paired_bootstrap(*args, n_boot=20, **kwargs)
        with patch.object(ev, 'paired_bootstrap', side_effect=small_boot): ev.summary()
        result = ev.load(self.report / 'results.json')
        self.assertEqual(len(result['families']), 7)
        self.assertEqual(len(result['training_runs']), 45)
        self.assertEqual(result['families'][self.arches[0]]['test']['macro_f1']['mean'], 1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_resume_preserves_skip_development_and_latency(self):
        self.freeze()
        with patch.object(sys, 'argv', ['evaluate', 'all']), \
             patch.object(torch, 'set_num_threads'), patch.object(torch, 'set_num_interop_threads'), \
             patch.object(ev, 'development') as dev, patch.object(ev, 'latency') as latency, \
             patch.object(ev, 'freeze') as freeze, patch.object(ev, 'test_inference') as test, patch.object(ev, 'summary') as summary:
            ev.main()
            dev.assert_not_called(); latency.assert_not_called(); freeze.assert_not_called()
            test.assert_called_once(); summary.assert_called_once()

    def test_training_records_validate_reference_truth_and_class_order(self):
        path = self.out / self.arches[0] / str(self.seeds[0]) / 'eye.json'; original = ev.load(path)
        for mode in ('truth', 'classes', 'heads'):
            value = copy.deepcopy(original)
            if mode == 'truth': value['selection']['eye0']['truth'].reverse()
            elif mode == 'classes': value['meta']['classes']['eye0'].reverse()
            else: value['meta']['classification_features'].reverse()
            ev.dump(path, value)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError): ev.all_training()

    def test_temperature_failure_cannot_silently_become_identity(self):
        for item in ({'temperature': None, 'n_val': 0, 'calibrated': False, 'optimizer_success': False},
                     {'temperature': 1., 'n_val': 2, 'calibrated': True, 'optimizer_success': False},
                     {'temperature': float('nan'), 'n_val': 2, 'calibrated': True, 'optimizer_success': True}):
            with self.subTest(item=item), self.assertRaisesRegex(RuntimeError, 'temperature fit missing/failed'):
                ev.calibrated({'h': np.array(self.prob)}, {'h': item})

    def test_development_failure_does_not_publish_calibrated_report(self):
        path = self.report / 'development.json'; original = path.read_bytes()
        failed = {'temperature': None, 'n_val': 0, 'calibrated': False, 'optimizer_success': False}
        with patch.object(ev, 'fit_temperature', return_value=failed):
            with self.assertRaisesRegex(RuntimeError, 'temperature fit missing/failed'): ev.development()
        self.assertEqual(path.read_bytes(), original)

    def test_development_and_latency_cannot_overwrite_after_freeze(self):
        self.freeze()
        for function in (ev.development, ev.latency):
            with self.subTest(stage=function.__name__), self.assertRaisesRegex(RuntimeError, 'already frozen'): function()

    def test_crop_cache_requires_reference_order_and_truth(self):
        cache = {'subject_ids': self.groups['test'], 'images': np.zeros((2, 1)),
                 'owners': np.array([0, 1]), 'truth': np.array([[0, 0, 0], [1, 1, 1]])}
        ev.validate_capture_cache(cache, 'eye', 'test', self.context)
        cache['subject_ids'] = list(reversed(self.groups['test']))
        with self.assertRaisesRegex(RuntimeError, 'crop cache participant order'): ev.validate_capture_cache(cache, 'eye', 'test', self.context)

    def test_checkpoint_metadata_checked_before_gpu_model_creation(self):
        path = self.out / self.arches[0] / str(self.seeds[0]) / 'eye.json'
        meta = ev.load(path)['meta']; meta['classes']['eye0'].reverse()
        with patch.object(torch, 'load', return_value={'meta': meta}), patch.object(ev, 'PartNet') as constructor:
            with self.assertRaisesRegex(RuntimeError, 'wrong class order'):
                ev.load_model(self.arches[0], 'eye', self.seeds[0], self.context)
            constructor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
