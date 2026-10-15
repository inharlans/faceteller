import unittest
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from scripts.release import audit_depth, calibrate_depth, run_training_queue
from scripts.release.audit_depth import repeat_summary, analyze, validate_records, FEATURES


def synthetic_cohort():
    ids=[f'{i:04d}' for i in range(400)]
    groups=dict(train=ids[:270],selection=ids[270:304],calibration=ids[304:338],test=ids[338:])
    owners={sid:split for split,people in groups.items() for sid in people}
    jobs=[(sid,'L1') for sid in ids]+[(sid,light) for sid in ids[::10] for light in ('L2','L3')]
    rows=[dict(job=f'{sid}:{light}',subject_id=sid,light=light,split=owners[sid],
               framing_version='synthetic',status='measured',
               values={key:int(sid)/400+{'L1':0,'L2':.01,'L3':-.01}[light] for key in FEATURES})
          for sid,light in jobs]
    lower,upper=np.quantile(np.arange(400)/400,[.25,.75])
    config={'framing_version':'synthetic','features':{
        key:dict(lower=float(lower),upper=float(upper),sigma_repeat_train=.01,repeat_train_people=27)
        for key in FEATURES}}
    return groups,rows,config


class RepeatAuditTests(unittest.TestCase):
    def test_identical_captures_are_perfect_repeatability_not_accuracy(self):
        report=repeat_summary([[0,0,0],[1,1,1],[2,2,2]],{'lower':.5,'upper':1.5})
        self.assertEqual(report['icc_1_1'],1)
        self.assertEqual(report['mean_absolute_change_from_L1'],0)
        self.assertEqual(report['all_three_labels_agree'],1)
        self.assertNotIn('accuracy',report)
    def test_lighting_shift_and_boundary_crossings_remain_visible(self):
        report=repeat_summary([[0,1,2],[1,2,3],[2,3,4]],{'lower':.5,'upper':2.5})
        self.assertEqual(report['pooled_repeat_sd'],1)
        self.assertEqual(report['mean_shift_L2'],1)
        self.assertEqual(report['mean_shift_L3'],2)
        self.assertAlmostEqual(report['label_agreement_L1_L2'],1/3)
        self.assertEqual(report['all_three_labels_agree'],0)
        self.assertAlmostEqual(report['icc_1_1'],.4)
        with self.assertRaises(ValueError):repeat_summary([[0,np.nan,1]],{'lower':.5,'upper':2.5})


class RecordValidationTests(unittest.TestCase):
    def setUp(self):
        self.groups,self.rows,self.config=synthetic_cohort()

    def test_full_cohort_reproduces_training_sigma_and_boundaries(self):
        result=analyze(self.rows,self.config,self.groups)
        self.assertEqual(result['base_participants'],400)
        self.assertEqual(result['repeat_jobs_attempted'],80)
        self.assertEqual(result['repeat_jobs_measured'],80)
        for feature in result['features'].values():
            self.assertEqual(feature['repeat']['all']['people'],40)
            self.assertEqual(feature['repeat']['train']['people'],27)
            self.assertAlmostEqual(feature['repeat']['train']['pooled_repeat_sd'],.01)

    def test_split_names_counts_and_disjointness(self):
        wrong=[]
        g=deepcopy(self.groups);g['extra']=[];wrong.append(g)
        g=deepcopy(self.groups);g['selection'].append(g['train'].pop());wrong.append(g)
        g=deepcopy(self.groups);g['train'][0]=g['test'][0];wrong.append(g)
        g=deepcopy(self.groups);g['train'][0]=g['train'][1];wrong.append(g)
        g=deepcopy(self.groups);g['train'][0]=123;wrong.append(g)
        for groups in wrong:
            with self.subTest(groups=groups.keys()),self.assertRaises(ValueError):
                validate_records([],groups,'synthetic')

    def test_invalid_attempt_cannot_be_hidden_by_later_valid_retry(self):
        variants=[{'subject_id':'0001'},{'light':'L3'},{'light':'L4'},
                  {'subject_id':'unknown'},{'split':'test'},{'framing_version':'old'},
                  {'job':'different-job'}]
        for fields in variants:
            earlier={**self.rows[0],**fields}
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                analyze([earlier]+self.rows,self.config,self.groups)

    def test_unscheduled_repeat_is_rejected_even_if_failed(self):
        extra={**self.rows[1],'job':'0001:L2','light':'L2','status':'error'}
        with self.assertRaisesRegex(ValueError,'scheduled'):
            analyze(self.rows+[extra],self.config,self.groups)

    def test_valid_retry_uses_last_attempt_and_held_repeat_is_not_measured(self):
        failed={**self.rows[0],'status':'error','values':{}}
        owners,jobs=validate_records([failed]+self.rows,self.groups,'synthetic')
        self.assertEqual(len(owners),400)
        self.assertEqual(len(jobs),480)
        self.assertEqual(jobs['0000:L1']['status'],'measured')
        held={**next(r for r in self.rows if r['job']=='0390:L3'),'status':'held','values':{}}
        report=analyze([failed]+self.rows+[held],self.config,self.groups)
        self.assertEqual(report['repeat_jobs_attempted'],80)
        self.assertEqual(report['repeat_jobs_measured'],79)
        self.assertEqual(report['features'][FEATURES[0]]['repeat']['all']['people'],39)

    def test_changed_boundary_or_sigma_is_rejected(self):
        for field in ('lower','sigma_repeat_train','repeat_train_people'):
            config=deepcopy(self.config);config['features'][FEATURES[0]][field]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                analyze(self.rows,config,self.groups)

    def test_calibration_rejects_bad_resume_before_engine_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'records';out.mkdir()
            protocol=root/'models/selection-20261014/protocol.json';protocol.parent.mkdir(parents=True)
            protocol.write_text(json.dumps({'groups':self.groups}),encoding='utf-8')
            valid={**self.rows[0],'framing_version':calibrate_depth.VERSION}
            bad={**valid,'subject_id':'0001'}
            (out/'depth_records.jsonl').write_text(json.dumps(bad)+'\n'+json.dumps(valid)+'\n',encoding='utf-8')
            with patch.object(calibrate_depth,'ROOT',root),patch.object(calibrate_depth,'OUT',out), \
                 patch.object(calibrate_depth,'VGGTEngine') as engine,patch('sys.argv',['calibrate_depth.py']):
                with self.assertRaisesRegex(ValueError,'job'):
                    calibrate_depth.main()
                engine.assert_not_called()


class PublicationAndResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.target=self.root/'data/release/depth_config.json'
        self.target.parent.mkdir(parents=True)

    def test_atomic_publication_audits_published_file_once(self):
        self.target.write_text('{"old":true}',encoding='utf-8')
        config={'new':True}
        def audit():
            self.assertEqual(json.loads(self.target.read_text(encoding='utf-8')),config)
            self.assertFalse(self.target.with_suffix('.json.tmp').exists())
        with patch.object(calibrate_depth,'ROOT',self.root),patch.object(audit_depth,'main',side_effect=audit) as check:
            calibrate_depth.publish_config(config)
        check.assert_called_once_with()

    def test_serialization_or_replace_failure_preserves_old_config(self):
        original=b'{"old":true}'
        for phase in ('serialize','replace'):
            with self.subTest(phase=phase):
                self.target.write_bytes(original)
                with patch.object(calibrate_depth,'ROOT',self.root),patch.object(audit_depth,'main') as check:
                    if phase=='serialize':
                        with self.assertRaises(ValueError):calibrate_depth.publish_config({'bad':float('nan')})
                    else:
                        with patch.object(Path,'replace',side_effect=OSError('simulated interruption')):
                            with self.assertRaises(OSError):calibrate_depth.publish_config({'new':True})
                    check.assert_not_called()
                self.assertEqual(self.target.read_bytes(),original)
                self.assertFalse(self.target.with_suffix('.json.tmp').exists())

    def test_audit_failure_leaves_complete_config_but_never_reports_completion(self):
        with patch.object(calibrate_depth,'ROOT',self.root), \
             patch.object(audit_depth,'main',side_effect=ValueError('invalid records')),patch('builtins.print') as output:
            with self.assertRaisesRegex(ValueError,'invalid records'):
                calibrate_depth.publish_config({'complete_json':True})
        self.assertEqual(json.loads(self.target.read_bytes()),{'complete_json':True})
        output.assert_not_called()

    def prepare_queue(self):
        log=self.root/'outputs/model-selection-20261014/train.log';log.parent.mkdir(parents=True)
        log.write_text('ALL_TRAINING_COMPLETE',encoding='utf-8')
        (log.parent/'results.json').write_text('{}',encoding='utf-8')
        checkpoints=self.root/'models/selection-20261014/arch/part';checkpoints.mkdir(parents=True)
        for i in range(45):(checkpoints/f'{i}.json').write_text('{}',encoding='utf-8')
        (self.target.parent/'extended_ready.json').write_text('{"ready":true}',encoding='utf-8')

    def test_queue_fresh_calibration_or_existing_cpu_audit_exactly_once(self):
        self.prepare_queue()
        for existing in (False,True):
            with self.subTest(existing=existing):
                if existing:self.target.write_text('{}',encoding='utf-8')
                with patch.object(run_training_queue,'ROOT',self.root),patch.object(run_training_queue,'state') as state, \
                     patch.object(run_training_queue.time,'sleep'),patch.object(run_training_queue,'run') as run:
                    run_training_queue.main()
                depth_calls=[call.args for call in run.call_args_list if call.args[0].startswith('release-depth')]
                expected=('release-depth-audit',['scripts/release/audit_depth.py']) if existing else \
                         ('release-depth',['scripts/release/calibrate_depth.py'])
                self.assertEqual(depth_calls,[expected])
                self.assertTrue(any(call.args[:2]==('training-queue','complete') for call in state.call_args_list))

    def test_failed_resumed_audit_blocks_downstream_and_completion(self):
        self.prepare_queue();self.target.write_text('{}',encoding='utf-8')
        def run(stage,args):
            if stage=='release-depth-audit':raise SystemExit(1)
        with patch.object(run_training_queue,'ROOT',self.root),patch.object(run_training_queue,'state') as state, \
             patch.object(run_training_queue.time,'sleep'),patch.object(run_training_queue,'run',side_effect=run) as execute:
            with self.assertRaises(SystemExit):run_training_queue.main()
        self.assertEqual([call.args[0] for call in execute.call_args_list],['comparison-report','release-depth-audit'])
        self.assertFalse(any(call.args[:2]==('training-queue','complete') for call in state.call_args_list))

    def test_actual_cpu_audit_binds_config_and_rejects_changed_records(self):
        groups,rows,config=synthetic_cohort()
        records=self.root/'data/derived/release/depth_records.jsonl';records.parent.mkdir(parents=True)
        raw=('\n'.join(json.dumps(row) for row in rows)+'\n').encode()
        records.write_bytes(raw);config['records_sha256']=hashlib.sha256(raw).hexdigest()
        self.target.write_text(json.dumps(config),encoding='utf-8')
        protocol=self.root/'models/selection-20261014/protocol.json';protocol.parent.mkdir(parents=True)
        protocol.write_text(json.dumps({'groups':groups}),encoding='utf-8')
        report_path=self.root/'outputs/release/depth-stability.json'
        with patch.object(audit_depth,'ROOT',self.root):
            audit_depth.main()
            report_bytes=report_path.read_bytes();report=json.loads(report_bytes)
            self.assertEqual(report['depth_config_sha256'],hashlib.sha256(self.target.read_bytes()).hexdigest())
            self.assertEqual(report['records_sha256'],config['records_sha256'])
            self.assertEqual(report['metric_3d_ground_truth_count'],0)
            records.write_bytes(raw+b'\n')
            with self.assertRaisesRegex(ValueError,'changed after calibration'):audit_depth.main()
            self.assertEqual(report_path.read_bytes(),report_bytes)
