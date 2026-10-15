"""CPU-only synthetic adapter tests. Never load real models or real faces."""
import asyncio
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from facework.pilot_contract import DIRECTIONS,Decision,InputSet,Observation,Passage,STAGES,build_result
from facework.release_preprocess import Crop,INPUT_SIZES,PART_CROPS,VERSION
from facework.release_runtime import (REGIONS,AdmissionMiddleware,InferenceWorker,ReleaseService,
    RuntimeResources,make_app,quality_reasons,sha256,temperature_probability,validate_checkpoint,validate_deployment)
from facework.release_depth import VERSION as DEPTH_VERSION,FEATURES
from facework.release_parsing import EAR_TRANSFORM_VERSION
from scripts.release.verify_service import local_url,summarize_result,upload_bytes,error_checks,verify_identity,aggregate


def registry_fixture():
    features={part+'_test':{'part':part,'kind':'classification','classes':['a','b'],
                           'classes_ko':['A','B'],'annotation_provenance':'synthetic'} for part in REGIONS}
    return {'preprocess_version':VERSION,'features':features,
            'parts':{part:{'crops':list(PART_CROPS[part]),'input_size':list(INPUT_SIZES[part]),
                           'features':[part+'_test']} for part in REGIONS},
            'depth_features':{key:{'part':'nose','classes':['low','medium','high'],'classes_ko':['low','medium','high']} for key in FEATURES},
            'class_support':{'train':{key:{'counts':[10,10]} for key in features}}}


def calibration_fixture(registry):
    return {'config':{'architectures':['resnet18'],'preprocess_version':VERSION,'registry_sha256':'fixture'},
            'ear_transform_version':EAR_TRANSFORM_VERSION,'ear_preprocessing_sha256':'f'*64,
            'features':{key:{'temperature':1.,'calibrated':True,'threshold':.6,'train_support':[10,10],
                             'n_val':34,'optimizer_success':True,'test_used_for_fit':False} for key in registry['features']}}


def fake_landmarks():
    p=np.full((478,3),200.,dtype=np.float32); p[:,2]=0
    p[234,:2]=[90,200];p[454,:2]=[310,200]
    p[33,:2]=[130,160];p[133,:2]=[170,160];p[263,:2]=[270,160];p[362,:2]=[230,160]
    p[61,:2]=[160,260];p[291,:2]=[240,260];p[13,:2]=[200,259];p[14,:2]=[200,261]
    return p


def blobs():
    with Image.new('RGB',(400,400),(170,170,170)) as im:
        stream=io.BytesIO();im.save(stream,format='PNG');raw=stream.getvalue()
    return [(d,(d+'.png',raw,'image/png')) for d in DIRECTIONS]


def inputs():
    files=blobs()
    return InputSet({d:bytearray(f[1]) for d,f in files},'upload',{d:'image/png' for d in DIRECTIONS})


class FakeRepository:
    def __init__(self): self.calls=[];self.fail=False;self.revision='a'*64
    def corpus_revision(self):return self.revision
    @contextmanager
    def snapshot(self):yield self
    def load_candidates(self,labels):
        self.calls.append(labels.copy())
        if self.fail: raise RuntimeError('PRIVATE_CONTENT_MUST_NOT_APPEAR')
        return [Passage('stable:fixture',frozenset({'eye_test:a'}),'original','translation','commentary',
                        {'title':'Synthetic','url':'https://example.invalid','chapter':'one','locator':'two'})] if 'eye_test:a' in labels else []


class FakeResources:
    def __init__(self):
        self.registry=registry_fixture();self.manifest=calibration_fixture(self.registry)
        self.depth_config={};self.repository=FakeRepository();self.ready=True
        self.identity={'repository':'postgresql_readonly','deployment_sha256':'synthetic','depth_config_sha256':'synthetic',
                       'corpus_sha256':self.repository.corpus_revision()}
        self.detected_images=[];self.detect_failure=None;self.failed_parts=set();self.depth_fail=False
        self.parser=SimpleNamespace(parse=lambda image,landmarks:np.zeros((image.height,image.width),np.uint8))
        self.visible_ear_box=lambda mask,landmarks:[260,130,300,230]
        self.forehead_metrics=lambda mask,landmarks:{'values':{},'reasons':[]}
        self.ear_metrics=lambda *args:{'values':{},'reasons':[],'box':[260,130,300,230]}
        self.visible_mark_candidates=lambda *args:{'quality_passed':True,'reasons':[]}
        self.engine=object()
    def detect(self,image):
        self.detected_images.append(image)
        if self.detect_failure: return None,self.detect_failure
        return {'landmarks':fake_landmarks(),'yaw':[0,-40,40][(len(self.detected_images)-1)%3],
                'pitch':-5,'smile':0.,'blink':0.,'brow_down':0.},None
    def predict_part(self,part,crops):
        if part in self.failed_parts:raise RuntimeError('part failure')
        return {key:np.array([.9,.1]) for key in self.registry['parts'][part]['features']}
    def depth_estimate(self,*args):
        if self.depth_fail:raise RuntimeError('depth failure')
        return {}
    def depth_observations(self,*args):
        return [Observation(key,'nose','depth',key+':medium','medium',
                            measurement={'indicator':key,'unit':'ratio','value':.4},measurement_score=.8,
                            quality_passed=True,boundaries_validated=True,basis='synthetic') for key in FEATURES]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.registry=registry_fixture();self.manifest=calibration_fixture(self.registry)
        self.depth={'framing_version':DEPTH_VERSION,'cohort_n':400,
            'features':{key:{'lower':.2,'upper':.8,'sigma_repeat_train':.1,'n':400,'repeat_train_people':10} for key in FEATURES},
            'quality':{'min_confidence':.1,'max_cross_view_error':.2}}
        self.write('data/release/training_registry.json',self.registry)
        self.manifest['config']['registry_sha256']=sha256(self.root/'data/release/training_registry.json')
        self.write('models/release/ear_preprocessing_frozen.json',
                   {'ear_transform_version':EAR_TRANSFORM_VERSION,
                    'registry_sha256':self.manifest['config']['registry_sha256']})
        self.manifest['ear_preprocessing_sha256']=sha256(self.root/'models/release/ear_preprocessing_frozen.json')
        self.manifest['weights']={}
        for part in REGIONS:
            relative=f'models/release/resnet18/{part}.pt';path=self.root/relative
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic-not-a-torch-file')
            self.manifest['weights'][relative]=sha256(path)
        self.write('models/release/deployment.json',self.manifest)
        checkpoint=self.root/'models/vggt/model.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic-depth')
        self.depth.update(checkpoint='synthetic research checkpoint',checkpoint_sha256=sha256(checkpoint),code_commit='synthetic-commit')
        source=self.root/'third_party/vggt/vggt';source.mkdir(parents=True);(source/'synthetic.py').write_text('# synthetic')
        self.write('third_party/vggt/provenance.json',{'commit':'synthetic-commit','files':{'vggt/synthetic.py':sha256(source/'synthetic.py')}})
        self.write('data/release/depth_config.json',self.depth)
    def write(self,path,value):
        target=self.root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value),encoding='utf8')
    def test_manifest_checks_without_loading_models(self):
        r,m,d=validate_deployment(self.root)
        self.assertEqual(len(r['parts']),12);self.assertEqual(d['cohort_n'],400)
    def test_optional_observation_region_independent_of_input_part(self):
        for region in ('eye','face',None,'not-a-region',[]):
            with self.subTest(region=region):
                self.registry['features']['face_test']['region']=region
                self.write('data/release/training_registry.json',self.registry)
                self.manifest['config']['registry_sha256']=sha256(self.root/'data/release/training_registry.json')
                self.write('models/release/ear_preprocessing_frozen.json',
                           {'ear_transform_version':EAR_TRANSFORM_VERSION,
                            'registry_sha256':self.manifest['config']['registry_sha256']})
                self.manifest['ear_preprocessing_sha256']=sha256(self.root/'models/release/ear_preprocessing_frozen.json')
                self.write('models/release/deployment.json',self.manifest)
                if region in ('eye','face'):
                    registry,_,_=validate_deployment(self.root)
                    self.assertEqual(registry['features']['face_test']['part'],'face')
                    self.assertEqual(registry['features']['face_test']['region'],region)
                else:
                    with self.assertRaisesRegex(RuntimeError,'Invalid observation region'):
                        validate_deployment(self.root)
    def test_missing_deployment_fails_before_torch_import(self):
        (self.root/'models/release/deployment.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'artifact missing'):RuntimeResources.load(self.root)
    def test_changed_weight_and_registry_rejected(self):
        (self.root/'models/release/resnet18/eye.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'weight hash'):validate_deployment(self.root)
        self.write('data/release/training_registry.json',{'bad':'registry'})
        with self.assertRaises((RuntimeError,KeyError)):validate_deployment(self.root)
    def test_old_depth_boundaries_and_failed_temperature_rejected(self):
        self.depth['framing_version']='old-provider-box';self.write('data/release/depth_config.json',self.depth)
        with self.assertRaisesRegex(RuntimeError,'release depth'):validate_deployment(self.root)
        self.depth['framing_version']=DEPTH_VERSION;self.write('data/release/depth_config.json',self.depth)
        self.manifest['features']['eye_test']['calibrated']=False;self.write('models/release/deployment.json',self.manifest)
        with self.assertRaisesRegex(RuntimeError,'calibration'):validate_deployment(self.root)
    def test_legitimate_no_calibration_labels_disables_only_head(self):
        self.manifest['features']['eye_test'].update(calibrated=False,temperature=None,n_val=0,
                                                   status='no_valid_validation_labels',optimizer_success=False)
        self.write('models/release/deployment.json',self.manifest)
        validate_deployment(self.root)
        self.manifest['features']['nose_test']['temperature']=float('nan')
        self.write('models/release/deployment.json',self.manifest)
        with self.assertRaisesRegex(RuntimeError,'calibration'):validate_deployment(self.root)
    def test_single_class_calibration_is_valid_only_as_disabled_with_matching_support(self):
        c=self.manifest['features']['eye_test']
        c.update(calibrated=False,temperature=None,n_val=34,support=[34,0],
                 status='single_class_calibration_support',
                 coverage_policy='at_least_two_observed_classes_v1',test_used_for_fit=False)
        self.write('models/release/deployment.json',self.manifest)
        validate_deployment(self.root)
        for bad_support in ([17,17],[33,0],[True,33]):
            c['support']=bad_support
            self.write('models/release/deployment.json',self.manifest)
            with self.subTest(support=bad_support),self.assertRaisesRegex(RuntimeError,'disabled calibration'):
                validate_deployment(self.root)
        c.update(support=[34,0],calibrated=True,temperature=1.,optimizer_success=True)
        self.write('models/release/deployment.json',self.manifest)
        with self.assertRaisesRegex(RuntimeError,'class coverage'):validate_deployment(self.root)
    def test_head_and_class_order_checked(self):
        part='eye';spec=self.registry['parts'][part]
        meta={'arch':'resnet18','part':part,'classification_features':spec['features'],
              'classes':{'eye_test':['a','b']},'crops':spec['crops'],'input_size':spec['input_size'],
              'registry_sha256':self.manifest['config']['registry_sha256'],'preprocess_version':VERSION,
              'train_support':{'eye_test':[10,10]}}
        validate_checkpoint(meta,'resnet18',part,self.registry,self.manifest)
        meta['classes']['eye_test'].reverse()
        with self.assertRaisesRegex(RuntimeError,'class order'):validate_checkpoint(meta,'resnet18',part,self.registry,self.manifest)
    def test_actual_crop_support_can_be_below_registry_but_not_above(self):
        self.manifest['features']['eye_test']['train_support']=[0,7]
        self.write('models/release/deployment.json',self.manifest)
        _,manifest,_=validate_deployment(self.root)
        self.assertEqual(manifest['features']['eye_test']['train_support'],[0,7])
        for support in ([11,7],[0,-1],[True,7],[0,7.0],[7]):
            with self.subTest(support=support):
                self.manifest['features']['eye_test']['train_support']=support
                self.write('models/release/deployment.json',self.manifest)
                with self.assertRaisesRegex(RuntimeError,'support'):validate_deployment(self.root)
    def test_checkpoint_support_matches_actual_deployment_for_each_arch(self):
        spec=self.registry['parts']['eye']
        self.manifest['features']['eye_test']['train_support']=[0,7]
        meta={'arch':'resnet18','part':'eye','classification_features':spec['features'],
              'classes':{'eye_test':['a','b']},'crops':spec['crops'],'input_size':spec['input_size'],
              'registry_sha256':self.manifest['config']['registry_sha256'],'preprocess_version':VERSION,
              'train_support':{'eye_test':[0,7]}}
        validate_checkpoint(meta,'resnet18','eye',self.registry,self.manifest)
        for support in (None,{}, {'eye_test':[10,10]}, {'eye_test':[False,7]},
                        {'eye_test':[0,7.0]}, {'eye_test':[0,7],'extra':[1]}):
            with self.subTest(support=support):
                meta['train_support']=support
                with self.assertRaisesRegex(RuntimeError,'metadata|support'):
                    validate_checkpoint(meta,'resnet18','eye',self.registry,self.manifest)
    def test_old_ear_checkpoint_and_modified_ear_protocol_are_rejected(self):
        spec=self.registry['parts']['ear']
        meta={'arch':'resnet18','part':'ear','classification_features':spec['features'],
              'classes':{'ear_test':['a','b']},'crops':spec['crops'],'input_size':spec['input_size'],
              'registry_sha256':self.manifest['config']['registry_sha256'],'preprocess_version':VERSION,
              'train_support':{'ear_test':[10,10]}}
        with self.assertRaisesRegex(RuntimeError,'metadata'):
            validate_checkpoint(meta,'resnet18','ear',self.registry,self.manifest)
        meta.update(ear_transform_version=EAR_TRANSFORM_VERSION,
                    ear_preprocessing_sha256=self.manifest['ear_preprocessing_sha256'])
        validate_checkpoint(meta,'resnet18','ear',self.registry,self.manifest)
        self.write('models/release/ear_preprocessing_frozen.json',{'ear_transform_version':'old'})
        with self.assertRaisesRegex(RuntimeError,'Ear preprocessing'):
            validate_deployment(self.root)
    def test_depth_checkpoint_code_commit_and_source_hash_enforced(self):
        self.depth['code_commit']='wrong';self.write('data/release/depth_config.json',self.depth)
        with self.assertRaisesRegex(RuntimeError,'code commit'):validate_deployment(self.root)
        self.depth['code_commit']='synthetic-commit';self.write('data/release/depth_config.json',self.depth)
        (self.root/'third_party/vggt/vggt/synthetic.py').write_text('# changed')
        with self.assertRaisesRegex(RuntimeError,'source checksum'):validate_deployment(self.root)
        (self.root/'models/vggt/model.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'checkpoint hash'):validate_deployment(self.root)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.r=FakeResources();self.worker=InferenceWorker();self.addCleanup(self.worker.close)
    def app(self):return make_app(self.r,worker=self.worker)
    def post(self,client):return client.post('/api/v1/analysis',data={'input_mode':'upload'},files=blobs())
    def test_actual_progress_partial_depth_failure_and_cleanup(self):
        self.r.depth_fail=True
        with TestClient(self.app()) as client:
            response=self.post(client)
        events=[json.loads(line) for line in response.content.splitlines()]
        self.assertEqual([e['stage'] for e in events[:-1]],list(STAGES))
        result=events[-1]['data'];items=result['items']
        self.assertEqual(len(items),16)
        self.assertTrue(all(i['status']=='HELD' for i in items if i['kind']=='depth'))
        self.assertTrue(any(i['status']=='CONFIRMED' for i in items if i['kind']=='classification'))
        self.assertEqual(result['status'],'PARTIAL');self.assertTrue(result['passages'])
        self.assertEqual(set(i['region'] for i in items),set(REGIONS))
        for image in self.r.detected_images:
            with self.assertRaises(ValueError): image.getpixel((0,0))
    def test_one_part_failure_and_zero_support_are_independent(self):
        self.r.failed_parts={'nose'};self.r.manifest['features']['eye_test']['train_support']=[0,10]
        with TestClient(self.app()) as client:result=json.loads(self.post(client).content.splitlines()[-1])['data']
        items={i['item_id']:i for i in result['items']}
        self.assertIn('CNN_PART_FAILED',items['nose_test']['hold_reasons'])
        self.assertIn('PREDICTED_CLASS_NOT_TRAINED',items['eye_test']['hold_reasons'])
        self.assertEqual(items['eye_test']['status'],'HELD')
        self.assertNotIn('eye_test:a',self.r.repository.calls[-1])
        self.assertEqual(items['mouth_test']['status'],'CONFIRMED')
    def test_missing_crop_holds_only_that_part(self):
        from facework.release_preprocess import frontal_crops
        def missing(image,points):
            result=frontal_crops(image,points);result.pop('nose').image.close();return result
        with patch('facework.release_preprocess.frontal_crops',side_effect=missing),TestClient(self.app()) as client:
            result=json.loads(self.post(client).content.splitlines()[-1])['data']
        items={i['item_id']:i for i in result['items']}
        self.assertIn('CROP_UNAVAILABLE',items['nose_test']['hold_reasons'])
        self.assertEqual(items['mouth_test']['status'],'CONFIRMED')
    def test_no_face_decode_cleanup_and_private_errors(self):
        self.r.detect_failure='no_face'
        with TestClient(self.app()) as client:
            response=self.post(client)
        self.assertEqual(response.status_code,422)
        self.assertEqual(response.json()['code'],'FACE_NOT_FOUND')
        with self.assertRaises(ValueError):self.r.detected_images[0].getpixel((0,0))
        self.r.detect_failure=None;self.r.repository.fail=True;self.r.detected_images.clear()
        with TestClient(self.app()) as client:response=self.post(client)
        self.assertEqual(json.loads(response.content.splitlines()[-1])['code'],'SERVER_ERROR')
        self.assertNotIn(b'PRIVATE_CONTENT',response.content)
    def test_strict_three_inputs_and_health_and_frontend(self):
        with TestClient(self.app()) as client:
            response=client.post('/api/v1/analysis',data={'input_mode':'upload'},files=blobs()[:2])
            self.assertEqual(response.status_code,400)
            self.assertTrue(client.get('/api/v1/health').json()['ready'])
            self.assertEqual(client.get('/').status_code,200)
            self.assertIn('file_bytes',client.get('/api/v1/service-info').json()['limits'])
    def test_quality_scopes_reach_new_regions(self):
        images=[Image.new('RGB',(400,400),'gray') for _ in range(3)]
        try:
            detections=[self.r.detect(im)[0] for im in images]
            detections[0].update(smile=.8,blink=.8,brow_down=.8)
            flags=quality_reasons(images,detections)
            for part in ('under_eye','philtrum','cheekbone','forehead','glabella'):
                self.assertTrue(flags[part],part)
            self.assertFalse(flags['mole'])
        finally:
            for image in images:image.close()
    def test_temperature_exact_and_no_smoothing(self):
        p=np.array([.81,.19,0]);expected=np.sqrt(p);expected/=expected.sum()
        np.testing.assert_allclose(temperature_probability(p,2),expected)
        self.assertEqual(temperature_probability(p,2)[-1],0.)
    def test_imagenet_and_equal_model_bilateral_mean_before_temperature(self):
        import torch
        torch.set_num_threads(2)
        captured=[]
        class Model:
            def __init__(self,p):self.p=p
            def __call__(self,x):
                captured.append(x.clone())
                return None,[torch.log(torch.tensor(self.p,dtype=torch.float32))]
        r=RuntimeResources();r.device='cpu';r.registry=self.r.registry
        r.models={'eye':[Model([[.99,.01],[.7,.3]]),Model([[.6,.4],[.2,.8]])]}
        arrays=[np.full((4,6,3),255,np.uint8),np.zeros((4,6,3),np.uint8)]
        with patch('facework.release_preprocess.network_array',side_effect=arrays):
            p=r.predict_part('eye',[object(),object()])['eye_test']
        np.testing.assert_allclose(p,[.6225,.3775],rtol=1e-6)
        self.assertAlmostEqual(float(captured[0][0,0,0,0]),(1-.485)/.229,places=5)
        self.assertAlmostEqual(float(captured[0][1,2,0,0]),-.406/.225,places=5)
        q=temperature_probability(p,2)
        expected=np.sqrt([.6225,.3775]);expected/=expected.sum()
        np.testing.assert_allclose(q,expected,rtol=1e-6)
    def test_ear_orientation_and_crop_images_are_closed(self):
        from facework.release_preprocess import ear_crop,frontal_crops
        mirrors=[];created=[]
        def ears(image,box,*,landmarks,yaw_degrees,mirror=False):
            mirrors.append(mirror)
            np.testing.assert_array_equal(landmarks,fake_landmarks())
            self.assertEqual(yaw_degrees,40 if mirror else -40)
            crop=ear_crop(image,box,landmarks=landmarks,yaw_degrees=yaw_degrees,mirror=mirror)
            created.append(crop.image);return crop
        def front(image,points):
            crops=frontal_crops(image,points);created.extend(c.image for c in crops.values());return crops
        with patch('facework.release_preprocess.ear_crop',side_effect=ears),patch('facework.release_preprocess.frontal_crops',side_effect=front),TestClient(self.app()) as client:
            self.post(client)
        self.assertEqual(mirrors,[False,True])
        self.assertTrue(created)
        for image in created:
            with self.assertRaises(ValueError):image.getpixel((0,0))
    def test_skin_visibility_holds_only_mole(self):
        self.r.visible_mark_candidates=lambda *args:{'quality_passed':False,'reasons':['SKIN_UNAVAILABLE']}
        with TestClient(self.app()) as client:result=json.loads(self.post(client).content.splitlines()[-1])['data']
        items={i['item_id']:i for i in result['items']}
        self.assertIn('SKIN_UNAVAILABLE',items['mole_test']['hold_reasons'])
        self.assertEqual(items['nose_test']['status'],'CONFIRMED')
    def test_pigmented_marks_absence_basis_scopes_only_this_head(self):
        key='pigmented_marks'
        self.r.registry['features'][key]={**self.r.registry['features']['mole_test'],
                                         'classes':['absent','present'],'classes_ko':['없음','있음']}
        self.r.registry['parts']['mole']['features'].append(key)
        self.r.manifest['features'][key]=copy.deepcopy(self.r.manifest['features']['mole_test'])
        with TestClient(self.app()) as client:result=json.loads(self.post(client).content.splitlines()[-1])['data']
        items={i['item_id']:i for i in result['items']}
        scope='머리카락·눈썹·눈·코·입과 아랫얼굴을 제외한 이마·윗볼의 작은 어두운 표시만 판정합니다. 없음은 이 관찰 범위에 대한 결과입니다.'
        self.assertEqual(items[key]['status'],'CONFIRMED')
        self.assertEqual(items[key]['label_id'],'pigmented_marks:absent')
        self.assertEqual(items[key]['basis'],items['mole_test']['basis']+' '+scope)
        self.assertTrue(all(scope not in item['basis'] for name,item in items.items() if name!=key))
    def test_uncalibrated_rare_head_holds_without_temperature_or_disabling_part(self):
        self.r.manifest['features']['eye_test'].update(calibrated=False,temperature=None,n_val=0,status='no_valid_validation_labels')
        with TestClient(self.app()) as client:result=json.loads(self.post(client).content.splitlines()[-1])['data']
        items={i['item_id']:i for i in result['items']}
        self.assertEqual(items['eye_test']['status'],'HELD')
        self.assertIsNone(items['eye_test']['confidence'])
        self.assertIn('CONFIDENCE_NOT_CALIBRATED',items['eye_test']['hold_reasons'])
        self.assertEqual(items['nose_test']['status'],'CONFIRMED')
    def test_http_error_probes_and_report_schema(self):
        detect=self.r.detect
        self.r.detect=lambda image: (None,'no_face') if image.getpixel((0,0))==(0,0,0) else detect(image)
        with TestClient(self.app()) as client:
            checks=error_checks(client)
            self.assertTrue(all(c['passed'] for c in checks),checks)
            # Reset synthetic direction counter after the deliberate no-face probe.
            self.r.detected_images.clear()
            event=json.loads(self.post(client).content.splitlines()[-1])
        summary=summarize_result(event,self.r.repository)
        self.assertEqual(summary['passage_count'],len(summary['passages']))
        self.assertTrue(summary['db_source_verified'])
        self.assertEqual(summary['corpus_sha256'],self.r.repository.corpus_revision())
        self.assertEqual(set(summary['items'][0]),{'item_id','region','kind','status','confidence',
                                                 'measurement_score','label_id','hold_reasons'})
        self.assertNotIn('measurement',summary['items'][0])
        self.assertNotIn('original_text',summary['passages'][0])
        verify_identity(summary,self.r.identity)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            verify_identity(summary,{'deployment_sha256':'other','depth_config_sha256':'synthetic'})

    def test_current_revision_exposed_and_changed_corpus_verification_rejected(self):
        with TestClient(self.app()) as client:
            self.r.repository.revision='b'*64  # identity startup value is intentionally stale
            self.assertEqual(client.get('/api/v1/service-info').json()['corpus_sha256'],'b'*64)
            event=json.loads(self.post(client).content.splitlines()[-1])
        self.assertEqual(event['data']['runtime']['corpus_sha256'],'b'*64)
        summary=summarize_result(event,self.r.repository)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            verify_identity(summary,self.r.identity)
        self.r.repository.revision='c'*64
        with self.assertRaisesRegex(RuntimeError,'corpus revision'):
            summarize_result(event,self.r.repository)

    def test_unknown_active_revision_is_not_ready(self):
        self.r.repository.revision=None
        with TestClient(self.app()) as client:
            response=client.get('/api/v1/health')
        self.assertEqual(response.status_code,503)
        self.assertFalse(response.json()['ready'])


class SelectionProofTests(unittest.TestCase):
    @staticmethod
    def decision(key,region,kind,value):
        return Decision(key,region,'CONFIRMED',key+':a','A',value if kind=='classification' else None,
                        None,value if kind=='depth' else None,'synthetic',(),kind,
                        'geometry_membership_not_calibrated_accuracy' if kind=='depth' else None)
    @staticmethod
    def passage(pid,labels,priority=0,order=0):
        return Passage(pid,frozenset(labels),'Original historical quote','Literal translation','Historical note',
                       {'title':'Synthetic source','annotation_status':'ai_annotated_unreviewed'},priority,order)
    def fixture(self,decisions=None,passages=None):
        decisions=decisions or [self.decision('eye','eye','classification',.81),self.decision('depth','nose','depth',.64)]
        passages=passages if passages is not None else [self.passage('mixed',{'eye:a','depth:a'})]
        repo=FakeRepository()
        repo.load_candidates=lambda labels:[p for p in passages if p.required_labels and p.required_labels<=labels]
        data=build_result(decisions,repo);data['runtime']={'corpus_sha256':repo.revision}
        return {'type':'result','data':data},repo
    def test_mixed_score_is_independently_point72_and_scalar_proof(self):
        event,repo=self.fixture();summary=summarize_result(event,repo)
        self.assertAlmostEqual(summary['passages'][0]['G'],.72)
        self.assertTrue(summary['score_formula_verified']);self.assertTrue(summary['selection_verified'])
        self.assertEqual(summary['score_formula_verified_count'],1)
        self.assertEqual(summary['score_components_verified_count'],2)
        self.assertNotIn('original_text',summary['passages'][0])
    def test_wrong_g_components_counts_and_evidence_rejected(self):
        event,repo=self.fixture()
        changes=[('G',.725),('G',float('nan')),('score_feature_count',1),('rank',2),
                 ('evidence_item_ids',['eye']),('evidence_item_ids',['eye','depth','depth'])]
        for field,value in changes:
            with self.subTest(field=field,value=value):
                bad=copy.deepcopy(event);bad['data']['passages'][0][field]=value
                with self.assertRaisesRegex(RuntimeError,'Selection verification'):summarize_result(bad,repo)
        for change in ('wrong_score','wrong_kind','duplicate_feature','omit_depth'):
            with self.subTest(change=change):
                bad=copy.deepcopy(event);components=bad['data']['passages'][0]['score_components']
                if change=='wrong_score':components[0]['score']=.7
                elif change=='wrong_kind':components[0]['score_kind']='calibrated_classification_confidence'
                elif change=='duplicate_feature':components[0]=copy.deepcopy(components[1])
                else:components.pop(0)
                with self.assertRaisesRegex(RuntimeError,'components'):summarize_result(bad,repo)
    def test_missing_confirmed_condition_and_omitted_selection_rejected(self):
        event,repo=self.fixture();event['data']['items'][1].update(status='HELD',label_id=None)
        with self.assertRaisesRegex(RuntimeError,'selected IDs'):summarize_result(event,repo)
        event,repo=self.fixture();event['data']['passages']=[]
        with self.assertRaisesRegex(RuntimeError,'selected IDs'):summarize_result(event,repo)
    def test_priority_order_numeric_id_and_both_caps(self):
        decisions=[self.decision(k,k,'classification',.8) for k in ('eye','nose','face')]
        passages=[self.passage(pid,{'eye:a'},priority,order) for pid,priority,order in
                  [('10',0,0),('2',0,0),('alpha',0,0),('zorder',0,-1),('zpriority',-1,0)]]
        passages += [self.passage(pid,{region+':a'}) for pid,region in
                     [('n3','nose'),('n2','nose'),('n1','nose'),('f1','face')]]
        event,repo=self.fixture(decisions,passages)
        self.assertEqual([p['passage_id'] for p in event['data']['passages']],['zpriority','zorder','2','f1','n1'])
        summarize_result(event,repo)
        swapped=copy.deepcopy(event);swapped['data']['passages'][0:2]=reversed(swapped['data']['passages'][0:2])
        for rank,p in enumerate(swapped['data']['passages'],1):p['rank']=rank
        with self.assertRaisesRegex(RuntimeError,'rank/tie/caps'):summarize_result(swapped,repo)
        bad=copy.deepcopy(event);bad['data']['passages'][2]['passage_id']='10'
        with self.assertRaisesRegex(RuntimeError,'rank/tie/caps'):summarize_result(bad,repo)
    def test_multiregion_passage_consumes_each_regions_cap(self):
        decisions=[self.decision(k,k,'classification',.8) for k in ('eye','nose')]
        passages=[self.passage('combo',{'eye:a','nose:a'},-1)]
        passages += [self.passage(region+str(i),{region+':a'}) for region in ('eye','nose') for i in (1,2,3)]
        event,repo=self.fixture(decisions,passages)
        self.assertEqual([p['passage_id'] for p in event['data']['passages']],['combo','eye1','eye2','nose1','nose2'])
        summarize_result(event,repo)
        bad=copy.deepcopy(event);bad['data']['passages'][3]['passage_id']='eye3'
        with self.assertRaisesRegex(RuntimeError,'rank/tie/caps'):summarize_result(bad,repo)
    def test_zero_and_tiny_positive_log_domain(self):
        for values,expected in [((0.,.64),0.),((1e-250,1e-250),1e-250)]:
            with self.subTest(values=values):
                decisions=[self.decision(k,'nose','depth',v) for k,v in zip(('eye','depth'),values)]
                event,repo=self.fixture(decisions)
                result=summarize_result(event,repo)
                self.assertAlmostEqual(result['passages'][0]['G']/expected if expected else result['passages'][0]['G'],1 if expected else 0)
                event['data']['passages'][0]['G']=1e-12 if expected==0 else 0
                with self.assertRaisesRegex(RuntimeError,'geometric G'):summarize_result(event,repo)
    def test_production_selector_wrong_math_detected_independently(self):
        event,repo=self.fixture();wrong=copy.deepcopy(event['data']['passages']);wrong[0]['G']=.8
        with patch('scripts.release.verify_service.PassageSelector.select',return_value=wrong):
            with self.assertRaisesRegex(RuntimeError,'independent G'):summarize_result(event,repo)
    def test_empty_results_errors_and_aggregate_counts_are_honest(self):
        event,repo=self.fixture(passages=[]);empty=summarize_result(event,repo)
        self.assertTrue(empty['selection_verified']);self.assertEqual(empty['score_formula_verified_count'],0)
        error=summarize_result({'type':'error','code':'SERVER_ERROR'},repo)
        self.assertFalse(error['score_formula_verified'])
        event,repo=self.fixture();ok=summarize_result(event,repo)
        summary=aggregate([{**row,'split':'test','seconds':1.} for row in (ok,empty,error)])['test']
        self.assertFalse(summary['score_formula_verified']);self.assertEqual(summary['score_formula_verified_count'],1)
        self.assertEqual(summary['score_components_verified_count'],2)
        self.assertEqual(summary['selection_verified_participants'],2)


class ReceiveToResultTimingTests(unittest.TestCase):
    def test_continuous_clock_includes_decode_validation_and_stages_before_serialization(self):
        from facework import pilot_contract as contract
        marks={};clock_samples=[]
        receive=contract.receive_inputs;validate=contract.validate_images;serialize=contract.ndjson
        async def delayed_receive(request,limits):
            await asyncio.sleep(.04)
            result=await receive(request,limits)
            marks['receive_complete']=time.perf_counter()
            return result
        def delayed_decode(inputs,limits):
            marks['decode_start']=time.perf_counter();time.sleep(.02);validate(inputs,limits)
        def clock():
            sample=time.perf_counter();clock_samples.append(sample);return sample
        def delayed_serialize(event):
            if event['type']=='result':
                marks['serialize_start']=time.perf_counter();time.sleep(.04)
            return serialize(event)
        class Service:
            async def validate(self,inputs):
                await asyncio.sleep(.02)
            async def analyze(self,inputs):
                yield {'type':'progress','stage':STAGES[0]}
                await asyncio.sleep(.02)
                marks['result_ready']=time.perf_counter()
                yield {'type':'result','data':{'runtime':{'timings':{'existing_stage_seconds':.001}}}}
        with patch.object(contract,'receive_inputs',side_effect=delayed_receive), \
             patch.object(contract,'validate_images',side_effect=delayed_decode), \
             patch.object(contract,'perf_counter',side_effect=clock), \
             patch.object(contract,'ndjson',side_effect=delayed_serialize), \
             TestClient(contract.create_app(Service)) as client:
            start=time.perf_counter()
            response=client.post('/api/v1/analysis',data={'input_mode':'upload'},files=blobs())
            http_seconds=time.perf_counter()-start
        timings=json.loads(response.content.splitlines()[-1])['data']['runtime']['timings']
        measured=timings['received_to_result_seconds']
        self.assertEqual(len(clock_samples),2)
        self.assertGreaterEqual(clock_samples[0],marks['receive_complete'])
        self.assertLessEqual(clock_samples[0],marks['decode_start'])
        self.assertGreaterEqual(clock_samples[1],marks['result_ready'])
        self.assertLessEqual(clock_samples[1],marks['serialize_start'])
        self.assertEqual(measured,clock_samples[1]-clock_samples[0])
        self.assertGreaterEqual(measured,.05)  # decode + face validation + stage delays
        self.assertGreaterEqual(http_seconds-measured,.07)  # receive and terminal encoding excluded
        self.assertEqual(timings['existing_stage_seconds'],.001)  # not an invented stage sum
    def test_pilot_without_runtime_and_terminal_errors_remain_unchanged(self):
        from facework.pilot_contract import create_app
        for terminal in ({'type':'result','data':{'status':'COMPLETE'}},
                         {'type':'error','code':'SYNTHETIC_ERROR'}):
            with self.subTest(terminal=terminal):
                class Service:
                    async def validate(self,inputs):pass
                    async def analyze(self,inputs):yield copy.deepcopy(terminal)
                with TestClient(create_app(Service)) as client:
                    response=client.post('/api/v1/analysis',data={'input_mode':'upload'},files=blobs())
                self.assertEqual(json.loads(response.content),terminal)


class AsyncCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_waits_for_worker_then_closes_images(self):
        worker=InferenceWorker();r=FakeResources();service=ReleaseService(r,worker);raw=inputs()
        entered=threading.Event();release=threading.Event();old=r.detect
        def blocked(image):entered.set();release.wait(3);return old(image)
        r.detect=blocked
        task=asyncio.create_task(service.validate(raw))
        try:
            while not entered.is_set():await asyncio.sleep(.005)
            task.cancel();await asyncio.sleep(.02)
            self.assertFalse(task.done());self.assertEqual(service.images[0].getpixel((0,0)),(170,170,170))
            task.cancel();release.set()
            with self.assertRaises(asyncio.CancelledError):await task
            self.assertEqual(service.images,[])
            for image in r.detected_images:
                with self.assertRaises(ValueError):image.getpixel((0,0))
        finally:release.set();raw.clear();worker.close()
    async def test_admission_rejects_without_reading_body_and_releases(self):
        entered=asyncio.Event();release=asyncio.Event();sent=[]
        async def inner(scope,receive,send):entered.set();await release.wait()
        async def receive():raise AssertionError('busy request body was read')
        async def send(event):sent.append(event)
        middleware=AdmissionMiddleware(inner)
        scope={'type':'http','path':'/api/v1/analysis'}
        first=asyncio.create_task(middleware(scope,receive,send));await entered.wait()
        await middleware(scope,receive,send)
        self.assertEqual(sent[0]['status'],503);self.assertEqual(middleware.active,1)
        release.set();await first;self.assertEqual(middleware.active,0)


class VerificationClientTests(unittest.TestCase):
    def test_remote_urls_rejected(self):
        for url in ('https://example.com','http://example.com','http://localhost.evil','http://user:pw@localhost','http://localhost/x'):
            with self.assertRaises(ValueError):local_url(url)
        self.assertEqual(local_url('http://127.0.0.1:8767'),'http://127.0.0.1:8767')
    def test_whole_image_resize_preserves_aspect_and_original_bytes(self):
        with Image.new('RGB',(1600,800),'gray') as image:
            buffer=io.BytesIO();image.save(buffer,format='JPEG');raw=buffer.getvalue()
        self.assertEqual(upload_bytes(raw)[0],raw)
        resized,mime=upload_bytes(raw,1280)
        with Image.open(io.BytesIO(resized)) as image:self.assertEqual(image.size,(1280,640))
        self.assertEqual(mime,'image/png')
    def test_source_proof_rejects_invented_reference(self):
        result={'type':'result','data':{'items':[],'passages':[{'passage_id':'invented'}]}}
        with self.assertRaisesRegex(RuntimeError,'PostgreSQL'):summarize_result(result,FakeRepository())


if __name__=='__main__':unittest.main()
