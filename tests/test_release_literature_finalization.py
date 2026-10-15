"""CPU finalizer tests: temporary files, mocked jobs, isolated PostgreSQL only."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from facework.release_repository import local_config_path
from scripts.release import finalize_literature as final
from tests.test_release_repository import corpus,source_section
from tests import test_release_repository as repository_tests


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1),encoding='utf-8')


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
        root=Path(directory);out=root/'outputs/release';out.mkdir(parents=True)
        stack.enter_context(patch.object(final,'ROOT',root));stack.enter_context(patch.object(final,'OUT',out))
        stack.enter_context(patch('builtins.print'))
        ids=[f'{i:04d}' for i in range(400)]
        groups=dict(train=ids[:270],selection=ids[270:304],calibration=ids[304:338],test=ids[338:])
        put(root/'models/selection-20261014/protocol.json',{'groups':groups})
        put(root/'data/release/extended_labels.json',{'synthetic':True})
        put(root/'scripts/release/merge_mark_labels.py',{'synthetic':True})
        registry={'features':{'pigmented_marks':{'classes':['absent','present']}},'subjects':{},
                  'preprocess_version':'fixture','human_review_count':0,'class_support':{}}
        overlay={'subjects':{},'groups':groups,'class_support':{},'preprocess_version':'fixture','human_review_count':0,
                 'source_hashes':{'records':{}}}
        for split,people in groups.items():
            for index,sid in enumerate(people):
                label='present' if index<2 else 'absent' if index<4 else None
                registry['subjects'][sid]={'split':split,'labels':{'pigmented_marks':label}}
                overlay['subjects'][sid]={'pigmented_marks':label}
                record=root/f'{final.RECORDS}/{sid}.json';put(record,{'synthetic':True,'subject_id':sid})
                overlay['source_hashes']['records'][sid]=final.sha(record)
        for key,path in [('protocol','models/selection-20261014/protocol.json'),
                         ('original_extended_labels','data/release/extended_labels.json'),
                         ('merge_implementation','scripts/release/merge_mark_labels.py')]:
            overlay['source_hashes'][key]=final.sha(root/path)
        def bind():
            for split,people in groups.items():
                labels=[registry['subjects'][sid]['labels']['pigmented_marks'] for sid in people]
                support={'absent':labels.count('absent'),'present':labels.count('present'),'unknown':labels.count(None)}
                overlay['class_support'][split]=support
                registry['class_support'][split]={'pigmented_marks':{'counts':[support['absent'],support['present']],'unknown':support['unknown']}}
            put(root/final.OVERLAY,overlay)
            registry['source_hashes']={name:final.sha(root/name) for name in
                ('models/selection-20261014/protocol.json','data/release/extended_labels.json',final.OVERLAY)}
            put(root/'data/release/training_registry.json',registry)
            put(root/'data/release/extended_ready.json',{'ready':True,'registry_sha256':final.sha(root/'data/release/training_registry.json'),
                'overlay_sha256':final.sha(root/final.OVERLAY),'source_hashes':registry['source_hashes'],'human_review_count':0,
                'validation':{'class_support':overlay['class_support'],'train_per_class_minimum':2,'selection_known_minimum':2,
                              'calibration_known_minimum':2,'test_used_for_readiness':False,'independent_accuracy_verified':False}})
        bind()
        pending=root/'data/release/literature/pending_labels.json'
        put(pending,{'features':{'pigmented_marks':['absent','present'],'other_pending':['x']},'status':'pending','source':'historical dispatch'})
        for name in ('data/release/literature/source_registry.json','data/release/literature/snapshots/snapshot_manifest.json',
                     'data/release/literature/annotations/frozen.json','scripts/release/build_reference_corpus.py','tests/test_release_literature.py'):
            put(root/name,{'synthetic':True})
        yield root,out,registry,overlay,bind


@contextmanager
def mocked_pipeline(root,document,fail=None,after=None):
    stages=[];digest=final.canonical_hash(document)
    result={'sha256':digest,'sources':len(document['sources']),'passages':len(document['passages']),
            'source_sections':len(document.get('source_sections',[])),'eligible':sum(p['service_eligible'] for p in document['passages'])}
    def stage(name,*args,**kwargs):
        stages.append(name)
        if name==fail:raise RuntimeError('synthetic stage failure')
        if name=='build':put(root/'data/release/literature/reference_corpus_v1.json',document)
        if after:after(name)
    with ExitStack() as stack:
        stack.enter_context(patch.object(final,'run_stage',side_effect=stage))
        stack.enter_context(patch.object(final,'run_isolated_repository_tests',side_effect=lambda admin:stage('repository-tests')))
        stack.enter_context(patch.object(final,'database_dsn',side_effect=lambda **kw:'admin' if kw else 'reader'))
        stack.enter_context(patch.object(final.db,'migrate'))
        importer=stack.enter_context(patch.object(final.db,'import_corpus',return_value=result))
        stack.enter_context(patch.object(final.db,'status',return_value={'imports':1,'corpus_sha256':digest}))
        verifier=stack.enter_context(patch.object(final,'verify_database',return_value={'corpus_sha256':digest}))
        yield stages,importer,verifier


class FinalizationTests(unittest.TestCase):
    def test_windows_source_keys_validate_without_rewriting_frozen_artifacts(self):
        with fixture() as (root,out,registry,overlay,bind):
            registry_path=root/'data/release/training_registry.json'
            ready_path=root/'data/release/extended_ready.json'
            registry['source_hashes']={name.replace('/','\\'):digest for name,digest in registry['source_hashes'].items()}
            put(registry_path,registry)
            ready=final.load(ready_path)
            ready.update(source_hashes=registry['source_hashes'],registry_sha256=final.sha(registry_path))
            put(ready_path,ready)
            before={path:path.read_bytes() for path in (registry_path,ready_path,root/final.OVERLAY)}
            proof=final.validate_readiness()
            self.assertEqual(proof['registry_sha256'],final.sha(registry_path))
            self.assertEqual(proof['validation']['class_support']['train']['present'],2)
            for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)

    def test_source_path_aliases_rejected_even_with_equal_hashes(self):
        name=final.OVERLAY
        for alias in (name.replace('/','\\'),'./'+name,name.replace('/','//'),name.upper()):
            for other_hash in ('a'*64,'b'*64):
                with self.subTest(alias=alias,hash=other_hash),self.assertRaisesRegex(RuntimeError,'Duplicate'):
                    final.normalized_sources({name:'a'*64,alias:other_hash})
        self.assertEqual(final.normalized_sources({name.replace('/','\\'):'a'*64}),{name:'a'*64})
        with fixture() as (root,out,registry,overlay,bind):
            registry_path=root/'data/release/training_registry.json';ready_path=root/'data/release/extended_ready.json'
            registry['source_hashes'][name.replace('/','\\')]=registry['source_hashes'][name]
            put(registry_path,registry);ready=final.load(ready_path)
            ready.update(source_hashes=registry['source_hashes'],registry_sha256=final.sha(registry_path));put(ready_path,ready)
            with self.assertRaisesRegex(RuntimeError,'Duplicate'):final.validate_readiness()

    def test_raw_marker_map_must_still_equal_registry_before_normalization(self):
        with fixture() as (root,out,registry,overlay,bind):
            ready_path=root/'data/release/extended_ready.json';ready=final.load(ready_path)
            ready['source_hashes']={name.replace('/','\\'):digest for name,digest in ready['source_hashes'].items()}
            put(ready_path,ready)
            with self.assertRaisesRegex(RuntimeError,'source map mismatch'):final.validate_readiness()

    def test_portable_source_key_rejects_absolute_and_parent_paths(self):
        for name in ('../outside','data\\..\\outside','C:\\outside','C:outside','/outside','\\\\server\\share\\file',''):
            with self.subTest(name=name),self.assertRaises(RuntimeError):final.source_path_key(name)
        self.assertEqual(final.source_path_key('.\\data\\release//file.json'),'data/release/file.json')

    def test_readiness_source_hash_and_recomputed_minimum_support(self):
        with fixture() as (root,out,registry,overlay,bind):
            proof=final.validate_readiness()
            self.assertEqual(proof['validation']['class_support']['train']['present'],2)
            record=root/f'{final.RECORDS}/0000.json';original=record.read_bytes();record.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'source hash'):final.validate_readiness()
            record.write_bytes(original)
            registry['subjects']['0000']['labels']['pigmented_marks']=None
            overlay['subjects']['0000']['pigmented_marks']=None
            bind()
            with self.assertRaisesRegex(RuntimeError,'at least 2'):final.validate_readiness()

    def test_registry_hash_split_and_source_map_mismatches_rejected(self):
        for defect in ('registry','map','split','calibration'):
            with self.subTest(defect=defect),fixture() as (root,out,registry,overlay,bind):
                if defect=='registry':
                    put(root/'data/release/training_registry.json',{'changed':True})
                elif defect=='map':
                    path=root/'data/release/extended_ready.json';data=final.load(path);data['source_hashes']={};put(path,data)
                elif defect=='split':
                    registry['subjects']['0000']['split']='test';bind()
                else:
                    for sid in overlay['groups']['calibration']:
                        registry['subjects'][sid]['labels']['pigmented_marks']=None;overlay['subjects'][sid]['pigmented_marks']=None
                    bind()
                with self.assertRaises(RuntimeError):final.validate_readiness()

    def test_promotion_backup_is_original_and_restart_changes_no_annotations(self):
        with fixture() as (root,out,registry,overlay,bind):
            path=root/'data/release/literature/pending_labels.json';original=path.read_bytes()
            annotations=final.literature_inputs();proof=final.validate_readiness()
            first=final.promote_pending(proof)
            self.assertEqual((out/'literature-pending-labels.original.json').read_bytes(),original)
            self.assertEqual(final.load(path)['features'],{'other_pending':['x']})
            self.assertEqual(final.load(path)['ready_proof'],proof)
            self.assertEqual(final.promote_pending(proof),first)
            self.assertEqual(final.literature_inputs(),annotations)
            data=final.load(path);data['source']='unrelated new edit';put(path,data)
            with self.assertRaisesRegex(RuntimeError,'outside this finalization'):final.promote_pending(proof)
            self.assertEqual((out/'literature-pending-labels.original.json').read_bytes(),original)

    def test_no_readiness_or_existing_marker_never_mutates_pending_or_imports(self):
        with fixture() as (root,out,*_),mocked_pipeline(root,corpus()) as (stages,importer,verifier):
            path=root/'data/release/literature/pending_labels.json';original=path.read_bytes()
            ready=root/'data/release/extended_ready.json';ready.unlink()
            with self.assertRaisesRegex(RuntimeError,'absent'):final.finalize()
            self.assertEqual(path.read_bytes(),original);importer.assert_not_called()
            marker=root/'data/release/literature_ready.json';marker.write_bytes(b'existing')
            with self.assertRaisesRegex(RuntimeError,'preserved'):final.finalize(wait=True)
            self.assertEqual(marker.read_bytes(),b'existing');self.assertEqual(stages,[])

    def test_pipeline_publishes_compact_actual_import_hash_only_after_all_checks(self):
        with fixture() as (root,out,*_),mocked_pipeline(root,corpus()) as (stages,importer,verifier):
            final.finalize()
            self.assertEqual(stages,['build','literature-tests','repository-tests'])
            self.assertEqual(importer.call_count,2);verifier.assert_called_once()
            report=final.load(out/'literature-finalization.json')
            marker=final.load(root/'data/release/literature_ready.json')
            self.assertTrue(marker['ready']);self.assertTrue(report['idempotent_reimport'])
            self.assertEqual(marker['corpus_sha256'],final.canonical_hash(corpus()))
            self.assertEqual(marker['import_sha256'],marker['corpus_sha256'])
            self.assertEqual(marker['finalization_report_sha256'],final.sha(out/'literature-finalization.json'))
            self.assertNotEqual(marker['corpus_sha256'],final.sha(root/'data/release/literature/reference_corpus_v1.json'))
            with self.assertRaisesRegex(RuntimeError,'preserved'):final.finalize()

    def test_failed_preimport_job_resumes_from_original_pending_backup(self):
        with fixture() as (root,out,*_):
            for stage in ('build','literature-tests','repository-tests'):
                with mocked_pipeline(root,corpus(),fail=stage) as (_,importer,_):
                    with self.assertRaisesRegex(RuntimeError,'synthetic'):final.finalize()
                    importer.assert_not_called()
                    self.assertFalse((root/'data/release/literature_ready.json').exists())
            with mocked_pipeline(root,corpus()):final.finalize()
            self.assertTrue((root/'data/release/literature_ready.json').is_file())

    def test_hash_import_or_database_failure_never_publishes_marker(self):
        for defect in ('import_hash','idempotence','db','inputs'):
            with self.subTest(defect=defect),fixture() as (root,out,*_):
                def after(stage):
                    if defect=='inputs' and stage=='literature-tests':put(root/'data/release/literature/annotations/frozen.json',{'changed':True})
                with mocked_pipeline(root,corpus(),after=after) as (_,importer,verifier):
                    if defect=='import_hash':importer.return_value={**importer.return_value,'sha256':'0'*64}
                    if defect=='idempotence':importer.side_effect=[importer.return_value,{**importer.return_value,'eligible':99}]
                    if defect=='db':verifier.side_effect=RuntimeError('DB proof failed')
                    with self.assertRaises(RuntimeError):final.finalize()
                    self.assertFalse((root/'data/release/literature_ready.json').exists())
                    if defect=='inputs':importer.assert_not_called()

    def test_wait_rechecks_marker_and_failed_input_preparation(self):
        with fixture() as (root,out,*_),mocked_pipeline(root,corpus()) as (_,importer,_):
            (root/'data/release/extended_ready.json').unlink()
            put(out/'input_preparation.json',{'status':'failed'})
            with self.assertRaisesRegex(RuntimeError,'Input preparation failed'):final.finalize(wait=True)
            importer.assert_not_called()

    def test_atomic_exclusive_publication_preserves_existing_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ready.json';path.write_bytes(b'prior')
            with self.assertRaises(FileExistsError):final.write_atomic(path,{'ready':True},exclusive=True)
            self.assertEqual(path.read_bytes(),b'prior')

    def test_os_lock_prevents_another_process_and_releases_after_exit(self):
        with fixture() as (root,out,*_):
            code="from pathlib import Path; from scripts.release import finalize_literature as f; import sys; f.OUT=Path(sys.argv[1]);\nwith f.operation_lock(): pass"
            with final.operation_lock():
                result=subprocess.run([sys.executable,'-c',code,str(out)],capture_output=True,timeout=15)
                self.assertNotEqual(result.returncode,0)
            result=subprocess.run([sys.executable,'-c',code,str(out)],capture_output=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipUnless(os.environ.get('FACETELLER_ADMIN_DATABASE_URL') or local_config_path().exists(),'Isolated PostgreSQL required')
class FinalizerPostgreSQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the already reviewed fixture contract, with its OWN class state.
        cls.fixture=type('FinalizerIsolatedFixture',(repository_tests.PostgreSQLRepositoryTests,),{})
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.fixture.setUpClass()

    def setUp(self):
        self.instance=self.fixture('test_and_matching_retains_all_conditions');self.instance.setUp()

    def test_actual_readonly_all_and_archive_proof_in_disposable_database(self):
        document=corpus();document['source_sections']=[source_section()]
        digest=final.db.import_corpus(self.fixture.dsn,document)['sha256']
        report=final.verify_database(document,digest,self.fixture.dsn,self.fixture.reader)
        self.assertEqual(report['eligible'],5);self.assertEqual(report['source_sections'],1)
        self.assertEqual(report['and_checks']['eligible_condition_sets'],5)
        self.assertEqual(report['and_checks']['missing_condition_checks'],6)
        self.assertTrue(report['readonly_role_verified']);self.assertTrue(report['private_archive_hidden_from_reader'])
        self.assertTrue(self.fixture.db.startswith('faceteller_test_'))

    def test_actual_archive_or_runtime_provenance_corruption_rejected(self):
        import psycopg
        document=corpus();document['source_sections']=[source_section()]
        digest=final.db.import_corpus(self.fixture.dsn,document)['sha256']
        with psycopg.connect(self.fixture.dsn) as conn:
            conn.execute("UPDATE face_release.passage SET annotation_status='human_reviewed' WHERE passage_id='2'")
        with self.assertRaisesRegex(RuntimeError,'provenance'):final.verify_database(document,digest,self.fixture.dsn,self.fixture.reader)
        final.db.import_corpus(self.fixture.dsn,document)
        with psycopg.connect(self.fixture.dsn) as conn:conn.execute('DELETE FROM face_release.source_section')
        with self.assertRaisesRegex(RuntimeError,'source archive'):final.verify_database(document,digest,self.fixture.dsn,self.fixture.reader)


if __name__=='__main__':unittest.main()
