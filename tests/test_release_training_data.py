import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

from scripts.release import train_models as train


class ReleaseDataTest(unittest.TestCase):
    def test_single_class_calibration_cannot_authorize_service_confidence(self):
        probabilities=np.asarray([[.7,.3],[.8,.2],[.9,.1]])
        result=train.release_temperature(probabilities,np.asarray([0,0,-1]),
            validation_ids=['cal-a','cal-b','cal-c'],test_ids=['test-a'])
        self.assertEqual(result['support'],[2,0])
        self.assertFalse(result['calibrated'])
        self.assertIsNone(result['temperature'])
        self.assertEqual(result['status'],'single_class_calibration_support')
        result=train.release_temperature(probabilities,np.asarray([0,1,-1]),
            validation_ids=['cal-a','cal-b','cal-c'],test_ids=['test-a'])
        self.assertTrue(result['calibrated'])
        with self.assertRaisesRegex(ValueError,'overlap'):
            train.release_temperature(probabilities,np.asarray([0,1,-1]),
                validation_ids=['cal-a','cal-b','cal-c'],test_ids=['cal-c'])

    def test_asymmetry_uses_joint_face_pixels_in_training_and_runtime(self):
        from facework.release_preprocess import Crop, network_array
        from facework.release_runtime import ReleaseService

        registry=deepcopy(train.load(train.REGISTRY))
        self.assertEqual((len(registry['parts']),len(registry['features'])),(12,41))
        self.assertEqual(len(registry['parts']['eye']['features']),6)
        self.assertEqual(len(registry['parts']['face']['features']),7)
        self.assertEqual(registry['features']['eye_asymmetry']['part'],'face')
        self.assertEqual(registry['features']['eye_asymmetry']['region'],'eye')
        self.assertNotIn('eye_asymmetry',registry['parts']['eye']['features'])
        self.assertEqual(registry['parts']['face']['crops'],['face'])
        heads=registry['parts']['face']['features']
        self.assertIn('eye_asymmetry',heads)
        labels={k:None for k in registry['features']}
        labels['eye_asymmetry']='asymmetric'
        registry['subjects']={'fixture':{'split':'train','labels':labels,'soft_targets':{}}}
        h,w=registry['parts']['face']['input_size']
        joint=np.zeros((h,w,3),dtype=np.uint8)
        joint[:,:w//2]=[210,30,40]
        joint[:,w//2:]=[20,50,220]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'data'/'subjects';folder.mkdir(parents=True)
            blob=io.BytesIO();crops={}
            for name,array in [('face',joint),('eye_right',np.full((160,256,3),61,np.uint8)),
                               ('eye_left',np.full((160,256,3),173,np.uint8))]:
                offset=blob.tell()
                with Image.fromarray(array) as image:image.save(blob,format='PNG')
                crops[name]={'offset':offset,'length':blob.tell()-offset,'valid_fraction':1}
            (folder/'fixture.bin').write_bytes(blob.getvalue())
            (folder/'fixture.json').write_text(json.dumps({'preprocess_version':train.VERSION,
                'frames':[{'light':light,'status':'ready','crops':crops,'blend':blend}
                          for light,blend in [('L1',[0,0,0,0,0]),
                                              ('L2',[0,.6,0,0,0]),
                                              ('L3',[0,0,.6,0,0]),
                                              ('L4',[.4,0,0,0,0])]]}))
            with patch.object(train,'DATA',root/'data'),patch.object(train,'OUT',root/'models'):
                cached=train.cache('face','train',registry,{'registry_sha256':'fixture'})
            self.assertEqual([row['crop'] for row in cached['rows']],['face'])
            self.assertEqual([row['capture'] for row in cached['rows']],['L1'])
            actual=cached['images'][0].permute(1,2,0).numpy()
            np.testing.assert_array_equal(actual,joint)
            target=registry['features']['eye_asymmetry']['classes'].index('asymmetric')
            self.assertEqual(cached['truth'][0,heads.index('eye_asymmetry')].item(),target)

            calls={}
            def predict(part,available):
                calls[part]=available
                if part=='face':
                    self.assertEqual(len(available),1)
                    np.testing.assert_array_equal(network_array(available[0],part),actual)
                return {k:np.full(len(registry['features'][k]['classes']),
                                  1/len(registry['features'][k]['classes']))
                        for k in registry['parts'][part]['features']}
            manifest={'features':{k:{'calibrated':True,'temperature':1.,
                                    'train_support':[1]*len(f['classes'])}
                                   for k,f in registry['features'].items()}}
            service=ReleaseService(SimpleNamespace(registry=registry,manifest=manifest,
                                                   predict_part=predict),None)
            service.crops={'face':Crop(Image.fromarray(joint),np.eye(3),1.),
                           'eye_right':Crop(Image.new('RGB',(256,160),(61,61,61)),np.eye(3),1.),
                           'eye_left':Crop(Image.new('RGB',(256,160),(173,173,173)),np.eye(3),1.)}
            try:
                observations=service.classify()
                item=next(o for o in observations if o.item_id=='eye_asymmetry')
                self.assertEqual(item.region,'eye')
                self.assertFalse(item.hold_reasons)
                self.assertIs(calls['face'][0],service.crops['face'])
                self.assertEqual(len(calls['eye']),2)
                for reason in ('EXPRESSION_BLINK','EXPRESSION_SMILE'):
                    with self.subTest(reason=reason):
                        service.flags['eye']=[reason]
                        items={o.item_id:o for o in service.classify()}
                        self.assertEqual(items['eye_asymmetry'].hold_reasons,(reason,))
                        self.assertIsNone(items['eye_asymmetry'].confidence)
                        self.assertIsNone(items['eye_asymmetry'].label_id)
                        for key in heads:
                            if key!='eye_asymmetry':
                                self.assertEqual(items[key].region,'face')
                                self.assertFalse(items[key].hold_reasons)
                                self.assertIsNotNone(items[key].label_id)
            finally:service.close()

    def test_runtime_rejects_checkpoint_with_old_asymmetry_head_assignment(self):
        from facework.release_runtime import validate_checkpoint
        registry=train.load(train.REGISTRY)
        manifest={'config':{'registry_sha256':'fixture'},'features':{
            k:{'train_support':[1]*len(f['classes'])} for k,f in registry['features'].items()}}
        for part in ('eye','face'):
            spec=registry['parts'][part]
            meta={'arch':'fixture','part':part,'classification_features':list(spec['features']),
                  'classes':{k:registry['features'][k]['classes'] for k in spec['features']},
                  'crops':spec['crops'],'input_size':spec['input_size'],
                  'registry_sha256':'fixture','preprocess_version':registry['preprocess_version'],
                  'train_support':{k:manifest['features'][k]['train_support'] for k in spec['features']}}
            validate_checkpoint(meta,'fixture',part,registry,manifest)
            if part=='face':meta['classification_features'].remove('eye_asymmetry')
            else:meta['classification_features'].append('eye_asymmetry')
            with self.subTest(part=part),self.assertRaisesRegex(RuntimeError,'metadata/head/class order mismatch'):
                validate_checkpoint(meta,'fixture',part,registry,manifest)

    def test_support_counts_seen_people_not_repeated_crops_or_absent_people(self):
        data={'owners':torch.tensor([0,0,2,2,2]),'truth':torch.tensor([[1,-1],[1,-1],[0,2],[0,2],[0,2]])}
        features={'a':{'classes':['no','yes']},'b':{'classes':['low','medium','high']}}
        self.assertEqual(train.training_support(data,['a','b'],features),{'a':[1,1],'b':[0,0,1]})
        data['truth'][1,0]=0
        with self.assertRaisesRegex(RuntimeError,'Inconsistent participant'):
            train.training_support(data,['a','b'],features)

    def test_participant_split_repeat_selection_and_unknown_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';folder=data/'subjects';folder.mkdir(parents=True)
            features={'a':{'classes':['no','yes']},'b':{'classes':['low','medium','high']}}
            registry={'parts':{'nose':{'features':['a','b'],'crops':['nose'],'input_size':[16,16]}},
                      'features':features,'subjects':{}}
            for sid,split in [('one','train'),('two','selection'),('three','test')]:
                registry['subjects'][sid]={'split':split,'labels':{'a':'yes','b':None},'soft_targets':{}}
                blob=io.BytesIO();frames=[]
                for light in ['L1','L2']:
                    offset=blob.tell();Image.new('RGB',(16,16),(20,30,40)).save(blob,format='PNG')
                    frames.append({'light':light,'capture':light,'status':'ready','crops':{'nose':{
                        'offset':offset,'length':blob.tell()-offset,'valid_fraction':1}}})
                (folder/f'{sid}.bin').write_bytes(blob.getvalue())
                (folder/f'{sid}.json').write_text(json.dumps({'frames':frames,'preprocess_version':train.VERSION}))
            with patch.object(train,'DATA',data),patch.object(train,'OUT',root/'models'):
                config={'registry_sha256':'frozen'}
                a=train.cache('nose','train',registry,config)
                b=train.cache('nose','selection',registry,config)
                c=train.cache('nose','test',registry,config)
                self.assertEqual([len(x['images']) for x in [a,b,c]],[2,1,1])
                self.assertEqual([x['ids'] for x in [a,b,c]],[['one'],['two'],['three']])
                self.assertEqual(a['truth'].tolist(),[[1,-1],[1,-1]])
                np.testing.assert_array_equal(a['votes'][:,1].numpy(),0)
                with self.assertRaisesRegex(RuntimeError,'Stale crop cache'):
                    train.cache('nose','train',registry,{'registry_sha256':'changed'})

    def test_actual_400_registry_support_and_disjoint_groups(self):
        registry=train.load(train.REGISTRY)
        self.assertEqual(len(registry['subjects']),400)
        self.assertEqual(registry['split_sizes'],{'train':270,'selection':34,'calibration':34,'test':62})
        self.assertEqual(registry['human_review_count'],0)
        for split,n in registry['split_sizes'].items():
            people=[p for p in registry['subjects'].values() if p['split']==split]
            self.assertEqual(len(people),n)
            for key,feature in registry['features'].items():
                support=registry['class_support'][split][key]
                self.assertEqual(sum(support['counts'])+support['unknown'],n)
                for person in people:
                    self.assertTrue(person['labels'][key] is None or person['labels'][key] in feature['classes'])
                    if key in person['soft_targets']:
                        self.assertAlmostEqual(sum(person['soft_targets'][key]),1)


if __name__=='__main__':unittest.main()
