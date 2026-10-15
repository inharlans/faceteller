"""CPU orchestration tests. All network/process launches are mocked."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.release import finish_project as finish


@contextmanager
def sandbox(fail_stage=None, after_stage=None):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root=Path(directory);out=root/'outputs/release';share=root/'share'
        out.mkdir(parents=True);share.mkdir()
        for name,value in [('ROOT',root),('OUT',out),('STATE',out/'completion_queue.json'),('SHARE_DIR',share)]:
            stack.enter_context(patch.object(finish,name,value))
        temp=root/'temp';temp.mkdir()
        stack.enter_context(patch.dict(os.environ,{'TEMP':str(temp),'TMP':str(temp)}))
        stack.enter_context(patch.object(finish.shutil,'which',return_value='C:/fixture/node.exe'))
        cli=root/'frontend/node_modules/playwright/cli.js';cli.parent.mkdir(parents=True);cli.write_text('// fixture only')
        stack.enter_context(patch('sys.argv',['finish_project']))
        stack.enter_context(patch('builtins.print'))
        stack.enter_context(patch.object(finish.time,'sleep'))
        corpus={'fixture':'no real data or model'}
        corpus_hash=hashlib.sha256(finish.canonical_json(corpus).encode()).hexdigest()
        finish.save(root/'data/release/literature/reference_corpus_v1.json',corpus)
        finish.save(root/'data/release/literature_ready.json',{'ready':True,'corpus_sha256':corpus_hash})
        finish.save(out/'training_queue.json',{'stage':'training-queue','status':'complete'})
        child=MagicMock();child.pid=123456;child.poll.return_value=None;child.wait.return_value=0
        popen=stack.enter_context(patch.object(finish.subprocess,'Popen',return_value=child))
        socket_factory=stack.enter_context(patch.object(finish.socket,'socket'))
        client_factory=stack.enter_context(patch.object(finish.httpx,'Client'))
        client=client_factory.return_value.__enter__.return_value
        response=client.get.return_value;response.status_code=200
        identity={'corpus_sha256':corpus_hash}
        for key,name in [('deployment_sha256','models/release/deployment.json'),
                         ('depth_config_sha256','data/release/depth_config.json'),
                         ('registry_sha256','data/release/training_registry.json')]:
            path=root/name;finish.save(path,{'fixture':name})
            identity[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        response.json.return_value={'ready':True,**identity}
        ui_proof={'pass':True,'mocked':False,'actual_models':True,'artifact_identity_verified':True,
                  **identity,'photos_saved':False,'source_origin':'http://127.0.0.1:8767',
                  'external_requests':0,'page_errors':[],'passages':1,'pdf_passage_text_verified_count':1,
                  'participants':1,'split':'train','confirmed':4,'held':3,'pdf_pages':1,
                  'one_page_button':True,'received_to_result_seconds':.7}
        smoke={'complete':True,'participants':[
            {'status':'PARTIAL','confirmed':1,'items':[{'kind':'depth','status':'CONFIRMED'}]} for _ in range(2)]}
        stages=[]
        def fake_run(command,**kwargs):
            script=command[3];args=command[4:]
            if command[0]=='C:/fixture/node.exe':stage='actual-ui'
            elif script.endswith('serve.py'):stage='deployment-preflight'
            elif script.endswith('verify_service.py'):stage='actual-smoke' if '--limit' in args else 'actual400'
            elif script.endswith('audit_scores.py'):stage='score-influence'
            else:stage='report-preflight' if '--validate-only' in args else 'report'
            stages.append(stage)
            if stage==fail_stage:
                # Model a verifier that wrote useful partial evidence before failing.
                if stage=='actual400':finish.save(out/'service-verification.json',{'complete':False,'participants':[{'fixture':'partial'}]})
                if stage=='actual-ui':finish.save(finish.ui_artifact_paths()[0],ui_proof)
                raise subprocess.CalledProcessError(7,command)
            if stage=='actual-smoke':finish.save(out/'service-smoke.json',smoke)
            if stage=='actual400':finish.save(out/'service-verification.json',{'complete':True,'participants':[{'status':'PARTIAL'}]*400})
            if stage=='score-influence':finish.save(out/'score-influence.json',{'fixture':True})
            if stage=='actual-ui':
                finish.save(finish.ui_artifact_paths()[0],ui_proof)
                for path in finish.ui_artifact_paths()[1:]:path.write_bytes(b'photo-free fixture result')
            if stage=='report':
                for path in finish.report_paths():path.write_bytes(b'fixture report')
            if after_stage:after_stage(stage,out,share)
            return subprocess.CompletedProcess(command,0)
        run=stack.enter_context(patch.object(finish.subprocess,'run',side_effect=fake_run))
        yield SimpleNamespace(root=root,out=out,share=share,child=child,popen=popen,socket=socket_factory,
                              client=client,run=run,stages=stages,smoke=smoke,ui_proof=ui_proof)


class CompletionTests(unittest.TestCase):
    def test_success_retains_child_and_uses_checked_report_destinations(self):
        with sandbox() as env:
            finish.main()
            self.assertEqual(env.stages,['deployment-preflight','actual-smoke','actual400',
                                        'score-influence','actual-ui','report-preflight','report'])
            env.child.terminate.assert_not_called();env.child.kill.assert_not_called();env.child.wait.assert_not_called()
            env.popen.assert_called_once()
            self.assertEqual(finish.load(env.out/'completion_queue.json')['status'],'complete')
            args=env.run.call_args.args[0]
            self.assertEqual(args[-4:],['--output-dir',str(env.out),'--share-dir',str(env.share)])
            self.assertTrue(all(path.exists() for path in finish.report_paths()))
            self.assertEqual(finish.load(env.out/'frontend-actual-verification.json'),env.ui_proof)
            self.assertEqual((env.out/'frontend-actual-verification.json').read_bytes(),finish.ui_artifact_paths()[0].read_bytes())
            self.assertTrue((env.out/'completion-actual-ui.log').exists())
            self.assertFalse((env.out/'actual-result-desktop.png').exists())
            self.assertFalse((env.out/'actual-sample-result.pdf').exists())

    def test_stale_local_and_shared_outputs_block_before_any_subprocess(self):
        cases=[('local','service-smoke.json'),('local','service-smoke.json.tmp'),
               ('local','service-verification.json'),('local','service-verification.json.partial'),
               ('local','score-influence.json'),('local','server-process.json'),('local','server-process.tmp'),
               ('local','frontend-actual-verification.json'),('local','frontend-actual-verification.json.partial')]
        for target in ('local','shared'):
            for suffix in ('pdf','xlsx','txt'):
                cases.extend([(target,f'{finish.REPORT_TITLE}.{suffix}'),
                              (target,f'{finish.REPORT_TITLE}.{suffix}.partial')])
        for target,name in cases:
            with self.subTest(target=target,name=name),sandbox() as env:
                path=(env.out if target=='local' else env.share)/name
                path.write_bytes(b'existing evidence')
                with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
                env.popen.assert_not_called();env.run.assert_not_called();env.socket.assert_not_called()
                self.assertEqual(path.read_bytes(),b'existing evidence')

    def test_artifact_appearing_during_preflight_blocks_popen(self):
        def appear(stage,out,share):
            if stage=='deployment-preflight':(share/f'{finish.REPORT_TITLE}.pdf').write_bytes(b'other worker report')
        with sandbox(after_stage=appear) as env:
            with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
            self.assertEqual(env.stages,['deployment-preflight']);env.popen.assert_not_called()
            self.assertEqual((env.share/f'{finish.REPORT_TITLE}.pdf').read_bytes(),b'other worker report')

    def test_each_post_launch_subprocess_failure_cleans_only_owned_child(self):
        for stage in ('actual-smoke','actual400','score-influence','actual-ui','report-preflight','report'):
            with self.subTest(stage=stage),sandbox(fail_stage=stage) as env:
                with self.assertRaises(subprocess.CalledProcessError):finish.main()
                env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)
                env.child.kill.assert_not_called();env.popen.assert_called_once()
                self.assertTrue((env.out/'server-process.json').exists())
                self.assertTrue((env.out/'server.log').exists())
                self.assertNotEqual(finish.load(env.out/'completion_queue.json')['status'],'complete')
                if stage=='actual-ui':
                    self.assertTrue(finish.ui_artifact_paths()[0].exists())
                    self.assertFalse((env.out/'frontend-actual-verification.json').exists())

    def test_partial_verification_retained_and_retry_refuses_launch(self):
        with sandbox(fail_stage='actual400') as env:
            with self.assertRaises(subprocess.CalledProcessError):finish.main()
            path=env.out/'service-verification.json';before=path.read_bytes()
            env.popen.reset_mock();env.run.reset_mock()
            with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
            env.popen.assert_not_called();env.run.assert_not_called()
            self.assertEqual(path.read_bytes(),before)

    def test_unusable_smoke_stops_before_all400_and_preserves_evidence(self):
        with sandbox() as env:
            env.smoke['participants'][0]['confirmed']=0
            with self.assertRaisesRegex(RuntimeError,'usable morphology'):finish.main()
            self.assertNotIn('actual400',env.stages)
            self.assertEqual(finish.load(env.out/'service-smoke.json')['participants'][0]['confirmed'],0)
            env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)

    def test_startup_timeout_and_corpus_mismatch_cleanup(self):
        with sandbox() as env,patch.object(finish.time,'monotonic',side_effect=[0,181]):
            with self.assertRaisesRegex(RuntimeError,'within180s'):finish.main()
            env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)
            self.assertEqual(env.stages,['deployment-preflight']);env.client.get.assert_not_called()
        with sandbox() as env:
            env.client.get.return_value.json.return_value={'ready':True,'corpus_sha256':'wrong'}
            with self.assertRaisesRegex(RuntimeError,'DB revision'):finish.main()
            env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)

    def test_process_record_write_failure_and_keyboard_interrupt_cleanup(self):
        with sandbox() as env:
            original=finish.save
            def save(path,value):
                if path.name=='server-process.json':raise OSError('fixture write failure')
                return original(path,value)
            with patch.object(finish,'save',side_effect=save),self.assertRaises(OSError):finish.main()
            env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)
        with sandbox() as env:
            env.client.get.side_effect=KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):finish.main()
            env.child.terminate.assert_called_once();env.child.wait.assert_called_once_with(timeout=15)

    def test_dead_child_reaped_without_termination_and_spawn_failure_touches_nobody(self):
        with sandbox() as env:
            env.child.poll.return_value=9
            with self.assertRaisesRegex(RuntimeError,'exited during startup'):finish.main()
            env.child.terminate.assert_not_called();env.child.kill.assert_not_called()
            env.child.wait.assert_called_once_with(timeout=15)
        with sandbox() as env:
            env.popen.side_effect=OSError('spawn failed')
            with self.assertRaises(OSError):finish.main()
            env.child.terminate.assert_not_called();env.child.kill.assert_not_called();env.child.wait.assert_not_called()

    def test_occupied_port_or_incomplete_training_never_touches_existing_server(self):
        with sandbox() as env:
            env.socket.return_value.__enter__.return_value.bind.side_effect=OSError('address already in use')
            with self.assertRaises(OSError):finish.main()
            env.popen.assert_not_called();env.child.terminate.assert_not_called()
        with sandbox() as env:
            finish.save(env.out/'training_queue.json',{'stage':'release-models','status':'running'})
            with self.assertRaisesRegex(RuntimeError,'not ready'):finish.main()
            env.popen.assert_not_called();env.run.assert_not_called()

    def test_late_report_conflict_preserved_and_child_cleaned(self):
        def appear(stage,out,share):
            if stage=='report-preflight':(share/f'{finish.REPORT_TITLE}.xlsx.partial').write_bytes(b'partial evidence')
        with sandbox(after_stage=appear) as env:
            with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
            self.assertNotIn('report',env.stages)
            self.assertEqual((env.share/f'{finish.REPORT_TITLE}.xlsx.partial').read_bytes(),b'partial evidence')
            env.child.terminate.assert_called_once()

    def test_cleanup_timeout_escalates_to_kill_then_reaps(self):
        child=MagicMock();child.poll.return_value=None
        child.wait.side_effect=[subprocess.TimeoutExpired('owned',15),0]
        finish.cleanup_owned_child(child)
        child.terminate.assert_called_once();child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count,2)

    def test_cleanup_failure_preserves_original_failure_and_ownership_record(self):
        with sandbox(fail_stage='actual400') as env,patch.object(finish,'cleanup_owned_child',side_effect=OSError('cleanup denied')):
            with self.assertRaises(subprocess.CalledProcessError) as caught:finish.main()
            self.assertIn('cleanup denied',caught.exception.__notes__[0])
            self.assertTrue((env.out/'server-process.json').exists())
            self.assertFalse(finish.load(env.out/'service-verification.json')['complete'])

    def test_actual_ui_uses_windows_node_local_cli_cwd_and_clean_environment(self):
        with sandbox() as env,patch.dict(os.environ,{
            'PLAYWRIGHT_CHROMIUM_EXECUTABLE':'/home/user/chromium',
            'LD_LIBRARY_PATH':'/tmp/wsl-libs','FACETELLER_TEST_URL':'http://unrelated.invalid'}):
            finish.main()
            call=next(call for call in env.run.call_args_list if call.args[0][0]=='C:/fixture/node.exe')
            self.assertEqual(call.args[0],['C:/fixture/node.exe',str(env.root/'frontend/node_modules/playwright/cli.js'),
                                         'test','--config','playwright.release.config.ts'])
            self.assertEqual(call.kwargs['cwd'],env.root/'frontend')
            self.assertIs(call.kwargs['check'],True)
            child_env=call.kwargs['env']
            self.assertEqual(child_env['FACETELLER_TEST_URL'],'http://127.0.0.1:8767')
            self.assertEqual(child_env['TEMP'],str(env.root/'temp'))
            self.assertNotIn('PLAYWRIGHT_CHROMIUM_EXECUTABLE',child_env)
            self.assertNotIn('LD_LIBRARY_PATH',child_env)
            self.assertEqual(os.environ['PLAYWRIGHT_CHROMIUM_EXECUTABLE'],'/home/user/chromium')

    def test_existing_temp_ui_evidence_is_preserved_before_any_work(self):
        for index in range(4):
            with self.subTest(index=index),sandbox() as env:
                path=finish.ui_artifact_paths()[index];path.parent.mkdir(parents=True)
                path.write_bytes(b'prior result')
                with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
                env.popen.assert_not_called();env.run.assert_not_called()
                self.assertEqual(path.read_bytes(),b'prior result')

    def test_late_temp_ui_evidence_blocks_ui_and_cleans_owned_child(self):
        def appear(stage,out,share):
            if stage=='score-influence':finish.save(finish.ui_artifact_paths()[0],{'prior':True})
        with sandbox(after_stage=appear) as env:
            with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
            self.assertNotIn('actual-ui',env.stages)
            self.assertNotIn('report-preflight',env.stages)
            self.assertEqual(finish.load(finish.ui_artifact_paths()[0]),{'prior':True})
            env.child.terminate.assert_called_once()

    def test_ui_nonzero_missing_or_invalid_proof_never_reaches_report(self):
        variants=[{'pass':False},{'pass':1},{'mocked':True},{'actual_models':False},
                  {'artifact_identity_verified':False},{'photos_saved':True},
                  {'source_origin':'http://localhost:8767'},{'external_requests':1},
                  {'page_errors':['browser failed']},{'pdf_passage_text_verified_count':0}]
        variants.extend({key:'f'*64} for key in ('deployment_sha256','depth_config_sha256','registry_sha256','corpus_sha256'))
        for fields in variants:
            with self.subTest(fields=fields),sandbox() as env:
                env.ui_proof.update(fields)
                with self.assertRaisesRegex(RuntimeError,'Actual UI'):finish.main()
                self.assertNotIn('report-preflight',env.stages)
                self.assertFalse((env.out/'frontend-actual-verification.json').exists())
                self.assertTrue(finish.ui_artifact_paths()[0].exists())
                env.child.terminate.assert_called_once()
        for mode in ('missing','malformed'):
            def damage(stage,out,share):
                if stage=='actual-ui':
                    path=finish.ui_artifact_paths()[0]
                    if mode=='missing':path.unlink()
                    else:path.write_bytes(b'{broken')
            with self.subTest(mode=mode),sandbox(after_stage=damage) as env:
                with self.assertRaises((RuntimeError,json.JSONDecodeError)):finish.main()
                self.assertNotIn('report-preflight',env.stages)
                self.assertFalse((env.out/'frontend-actual-verification.json').exists())
                env.child.terminate.assert_called_once()

    def test_zero_passages_is_valid_but_records_zero_pdf_text_checks(self):
        with sandbox() as env:
            env.ui_proof.update(passages=0,pdf_passage_text_verified_count=0)
            finish.main()
            self.assertEqual(finish.load(env.out/'frontend-actual-verification.json')['pdf_passage_text_verified_count'],0)

    def test_changed_local_identity_or_missing_node_blocks_before_browser(self):
        for mode in ('local','health','node','cli'):
            def change(stage,out,share):
                if stage=='score-influence' and mode=='local':finish.save(finish.ROOT/'models/release/deployment.json',{'changed':True})
            with self.subTest(mode=mode),sandbox(after_stage=change) as env:
                if mode=='health':env.client.get.return_value.json.return_value['depth_config_sha256']='bad'
                if mode=='node':finish.shutil.which.return_value=None
                if mode=='cli':(env.root/'frontend/node_modules/playwright/cli.js').unlink()
                with self.assertRaises(RuntimeError):finish.main()
                self.assertNotIn('actual-ui',env.stages)
                self.assertNotIn('report-preflight',env.stages)
                env.child.terminate.assert_called_once()

    def test_destination_appearing_during_ui_is_not_overwritten(self):
        def appear(stage,out,share):
            if stage=='actual-ui':(out/'frontend-actual-verification.json').write_bytes(b'other evidence')
        with sandbox(after_stage=appear) as env:
            with self.assertRaisesRegex(RuntimeError,'preserved'):finish.main()
            self.assertEqual((env.out/'frontend-actual-verification.json').read_bytes(),b'other evidence')
            self.assertNotIn('report-preflight',env.stages)
            env.child.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
