"""Independent CPU readiness, provenance and holdout-isolation regressions."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts.release import finish_training_inputs as ready
from scripts.release import prepare_landmark_labels as landmarks
from scripts.release.annotate_marks_local import digest, selected_subjects

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    groups = {'train': [str(i) for i in range(270)], 'selection': [str(i) for i in range(270,304)],
              'calibration': [str(i) for i in range(304,338)], 'test': [str(i) for i in range(338,400)]}
    registry = {'subjects': {}, 'parts': {p: {'features': [p+'_fixture']} for p in ready.PART_CROPS}}
    overlay = {'subjects': {}, 'groups': groups, 'class_support': {},
               'source_hashes': {'records': {str(i): 'fixture' for i in range(400)}}}
    for split, ids in groups.items():
        for index, sid in enumerate(ids):
            value = ('present' if index < 2 else 'absent' if index < 4 else None) if split != 'test' else None
            registry['subjects'][sid] = {'split': split, 'labels': {'pigmented_marks': value}}
            overlay['subjects'][sid] = {'pigmented_marks': value}
    recount(overlay)
    return registry, overlay


def recount(overlay):
    for split, ids in overlay['groups'].items():
        values = [overlay['subjects'][sid]['pigmented_marks'] for sid in ids]
        overlay['class_support'][split] = {'present': values.count('present'), 'absent': values.count('absent'),
                                          'unknown': values.count(None)}


class ReadinessTests(unittest.TestCase):
    def test_minimum_support_and_test_labels_independent(self):
        registry, overlay = fixture()
        result = ready.check_ready(registry, overlay)
        self.assertFalse(result['test_used_for_readiness'])
        self.assertFalse(result['independent_accuracy_verified'])
        for sid in overlay['groups']['test']:
            overlay['subjects'][sid]['pigmented_marks'] = 'present'
            registry['subjects'][sid]['labels']['pigmented_marks'] = 'present'
        recount(overlay)
        changed = ready.check_ready(registry, overlay)
        self.assertEqual(result['train_per_class_minimum'], changed['train_per_class_minimum'])

    def test_insufficient_train_or_selection_calibration_rejected(self):
        for split, keep in (('train','present'), ('train','absent'), ('selection',None), ('calibration',None)):
            registry, overlay = fixture()
            for sid in overlay['groups'][split]:
                value = keep if split == 'train' else None
                registry['subjects'][sid]['labels']['pigmented_marks'] = value
                overlay['subjects'][sid]['pigmented_marks'] = value
            recount(overlay)
            with self.subTest(split=split, keep=keep), self.assertRaises(RuntimeError):
                ready.check_ready(registry, overlay)

    def test_missing_part_empty_part_or_missing_subject_rejected(self):
        for defect in ('missing_part','extra_part','empty_part','missing_subject'):
            registry, overlay = fixture()
            if defect == 'missing_part': registry['parts'].pop('mole')
            if defect == 'extra_part': registry['parts']['invented'] = {'features':['fake']}
            if defect == 'empty_part': registry['parts']['mole']['features'] = []
            if defect == 'missing_subject': registry['subjects'].pop('0')
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                ready.check_ready(registry, overlay)

    def test_record_keys_labels_split_and_counts_cross_checked(self):
        for defect in ('record_key','overlay_key','split','label','count'):
            registry, overlay = fixture()
            if defect == 'record_key': overlay['source_hashes']['records']['wrong'] = overlay['source_hashes']['records'].pop('0')
            if defect == 'overlay_key': overlay['subjects']['wrong'] = overlay['subjects'].pop('0')
            if defect == 'split': registry['subjects']['0']['split'] = 'test'
            if defect == 'label': registry['subjects']['0']['labels']['pigmented_marks'] = None
            if defect == 'count': overlay['class_support']['train']['absent'] += 1
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                ready.check_ready(registry, overlay)

    def test_smoke_fixed_ids_digest_and_eighteen_threshold(self):
        _, overlay = fixture(); manifest = {'fixture': True}
        smoke = {'config_sha256': digest(manifest), 'participants': [
            {'subject_id': sid, 'split': split, 'status': 'ok' if i<18 else 'unknown'}
            for i,(sid,split) in enumerate(selected_subjects(overlay['groups'], True))]}
        self.assertEqual(ready.check_smoke(smoke,overlay['groups'],manifest),18)
        for defect in ('seventeen','duplicate','different_train','heldout','config','short'):
            value = copy.deepcopy(smoke)
            if defect == 'seventeen': value['participants'][0]['status'] = 'unknown'
            if defect == 'duplicate': value['participants'][0] = dict(value['participants'][1])
            if defect == 'different_train': value['participants'][0]['subject_id'] = '0'
            if defect == 'heldout': value['participants'][0]['split'] = 'test'
            if defect == 'config': value['config_sha256'] = 'changed'
            if defect == 'short': value['participants'].pop()
            with self.subTest(defect=defect), self.assertRaises(RuntimeError):
                ready.check_smoke(value, overlay['groups'], manifest)

    def test_queue_stages_cannot_launch_annotation_early(self):
        for stage,status in [('comparison','waiting_for_existing_training'), ('release-depth','running'),
                             ('release-labels','running'), ('release-labels','failed')]:
            with tempfile.TemporaryDirectory() as directory, patch.object(ready,'ROOT',Path(directory)), \
                 patch.object(ready,'state'), patch.object(ready,'load',return_value={'stage':stage,'status':status}), \
                 patch.object(ready,'run') as run, patch('sys.argv',['finish']):
                with self.subTest(stage=stage,status=status), self.assertRaises(RuntimeError): ready.main()
                run.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.object(ready,'ROOT',Path(directory)), \
             patch.object(ready,'state'), patch.object(ready,'load',return_value={
                 'stage':'release-labels','status':'waiting_for_reviewed_automatic_label_artifacts'}), \
             patch.object(ready,'run',side_effect=RuntimeError('stop-before-any-real-subprocess')) as run, \
             patch('sys.argv',['finish']):
            with self.assertRaisesRegex(RuntimeError,'stop-before-any-real-subprocess'): ready.main()
            run.assert_called_once_with('smoke','scripts/release/annotate_marks_local.py','--smoke',label_python=True)

    def test_run_uses_isolated_python_for_annotation_and_serial_checked_process(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ready,'OUT',Path(directory)), \
             patch.object(ready,'state'), patch.object(ready.subprocess,'run') as run:
            ready.run('smoke','scripts/release/annotate_marks_local.py','--smoke',label_python=True)
            args, kwargs = run.call_args
            self.assertEqual(args[0][0],str(ready.LABEL_PYTHON))
            self.assertTrue(kwargs['check'])
            self.assertEqual(kwargs['cwd'],ready.ROOT)
            ready.run('merge','scripts/release/merge_mark_labels.py','--negative-policy','strict')
            self.assertEqual(run.call_args.args[0][0],sys.executable)

    def test_actual_os_lock_excludes_second_process_and_releases_on_exit(self):
        # Execute ONLY the actual main guard's lock wrapper with a harmless main.
        # No orchestrator main(), model, live state, or GPU entry point is called.
        module = ast.parse(Path(ready.__file__).read_text(encoding='utf-8'))
        guard = [n for n in module.body if isinstance(n,ast.If)][-1]
        wrapper = compile(ast.Module(body=guard.body,type_ignores=[]),str(ready.__file__),'exec')
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'input-preparation.lock')
            probe = '''import os,sys
f=open(sys.argv[1],'a+b'); f.seek(0)
try:
 if os.name=='nt':
  import msvcrt; msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
 else:
  import fcntl; fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
except OSError: sys.exit(23)
sys.exit(0)
'''
            def harmless_main():
                result = subprocess.run([sys.executable,'-B','-c',probe,path],capture_output=True,timeout=10)
                self.assertEqual(result.returncode,23,result.stderr.decode(errors='replace'))
            exec(wrapper,{'OUT':Path(directory),'os':os,'main':harmless_main,'state':lambda *a,**k:None})
            result = subprocess.run([sys.executable,'-B','-c',probe,path],capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))


class LandmarkIsolationTests(unittest.TestCase):
    def test_all_nineteen_features_ignore_heldout_values_and_repeat_noise(self):
        schema = json.loads((ROOT/'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))['features']
        features = {k:v for k,v in schema.items() if v['source']=='landmark'}
        self.assertEqual(len(features),19)
        train_ids = ['a','b','c','d']
        for key,feature in features.items():
            samples = {sid:[base-.03,base,base+.03,base+.01] for sid,base in zip(train_ids,[.5,1.,1.5,2.])}
            samples.update(selection=[1,2,3],calibration=[2,3,4],test=[3,4,5])
            original = landmarks.fit_feature(samples,train_ids,feature,key)
            train_labels = [landmarks.label(samples[s],original) for s in train_ids]
            for sid in ('selection','calibration','test'): samples[sid] = [-1e12,0,1e12]*100
            changed = landmarks.fit_feature(samples,train_ids,feature,key)
            with self.subTest(feature=key):
                self.assertEqual(original,changed)
                self.assertEqual(train_labels,[landmarks.label(samples[s],changed) for s in train_ids])
                # A training change must still affect cutpoints/guard: not an inert helper.
                for sid in train_ids: samples[sid] = [v*3+1 for v in samples[sid]]
                self.assertNotEqual(original,landmarks.fit_feature(samples,train_ids,feature,key))

    def test_physical_centres_and_class_order_for_all_four_special_features(self):
        schema = json.loads((ROOT/'data/label_schema_kface_v3.json').read_text(encoding='utf-8'))['features']
        for key,centre in landmarks.CENTRES.items():
            samples = {str(i):[centre+i-.001,centre+i,centre+i+.001] for i in range(3)}
            definition = landmarks.fit_feature(samples,list(samples),schema[key],key)
            with self.subTest(feature=key):
                self.assertAlmostEqual(sum(definition['cuts'])/2,centre)
                self.assertEqual(landmarks.label([centre]*3,definition),schema[key]['classes'][1])
                self.assertEqual(landmarks.label([centre-.1]*3,definition),schema[key]['classes'][0])
                self.assertEqual(landmarks.label([centre+.1]*3,definition),schema[key]['classes'][2])
                self.assertIsNone(landmarks.label([definition['cuts'][0]]*3,definition))

    def test_brow_endpoint_sign_arch_independence_and_similarity_invariance(self):
        points = np.zeros((468,3),float)
        points[[33,133],:2] = [100,100]; points[[263,362],:2] = [200,100]
        for sign in (-1,0,1):
            p = points.copy()
            for side, brow in landmarks.BROW.items():
                x = np.linspace(60,120,5) if side == 'right' else np.linspace(240,180,5)
                y = np.array([70-sign*10,60,30,60,70])
                for index in range(5):
                    p[brow['upper'][index],:2] = [x[index],y[index]-2]
                    p[brow['lower'][index],:2] = [x[index],y[index]+2]
            # Isolate endpoint implementation from unrelated degenerate fake-face ratios.
            with patch.object(landmarks,'landmark_metrics',return_value={}):
                value = landmarks.measurements_from_points(p)['brow_outer_inner_tilt']
                self.assertAlmostEqual(value,np.degrees(np.arctan2(sign*10,60)))
                for brow in landmarks.BROW.values():
                    p[brow['upper'][2],1] -= 30; p[brow['lower'][2],1] -= 30
                self.assertAlmostEqual(value,landmarks.measurements_from_points(p)['brow_outer_inner_tilt'])
                angle=.3; rotate=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                p[:,:2] = p[:,:2] @ rotate.T * 2.5 + [40,90]
                self.assertAlmostEqual(value,landmarks.measurements_from_points(p)['brow_outer_inner_tilt'],places=9)


class FrozenProvenanceTests(unittest.TestCase):
    def test_train_protocol_requires_exact_source_map_and_current_hash(self):
        from scripts.release import train_models as train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.json'; source.write_text('original')
            registry_path = root/'registry.json'
            registry = {'parts':{'mole':{'features':['pigmented_marks']}},
                        'source_hashes':{'source.json':train.sha(source)},'split_sizes':{'train':270}}
            ready.save(registry_path,registry)
            choice = root/'outputs/model-selection-20261014/choice_frozen.json'
            ready.save(choice,{'selected_architectures':['resnet18']})
            marker = root/'data/release/extended_ready.json'
            valid = {'ready':True,'registry_sha256':train.sha(registry_path),'source_hashes':registry['source_hashes']}
            ready.save(root/'cache/crop_manifest.json',{'complete':True})
            with patch.object(train,'ROOT',root),patch.object(train,'REGISTRY',registry_path), \
                 patch.object(train,'DATA',root/'cache'),patch.object(train,'OUT',root/'models'):
                ready.save(marker,{**valid,'source_hashes':{}})
                with self.assertRaisesRegex(RuntimeError,'source list'): train.protocol()
                ready.save(marker,valid); source.write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'source changed'): train.protocol()
                source.write_text('original')
                _,config = train.protocol()
                self.assertEqual(config['registry_sha256'],train.sha(registry_path))
                self.assertTrue((root/'models/training_frozen.json').exists())


if __name__ == '__main__':
    unittest.main()
