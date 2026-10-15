"""CPU geometry, visibility, split isolation and actual ONNX smoke checks."""
import unittest
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from facework.release_parsing import (ReleaseParser, forehead_metrics, visible_ear_box,
    ear_metrics, fit_thresholds, metric_label, stable_measurement, head_box, MODEL_SHA256,
    visible_mark_candidates, repeated_mark_label)


def landmarks():
    p=np.tile([256.,280.,0.],(478,1))
    p[33,:2]=[182,280];p[133,:2]=[202,280]
    p[263,:2]=[330,280];p[362,:2]=[310,280]
    p[107,:2]=[230,240];p[336,:2]=[282,240];p[10,:2]=[256,170]
    p[234,:2]=[140,320];p[454,:2]=[372,320];p[152,:2]=[256,420]
    return p


class VisibleGeometryTests(unittest.TestCase):
    def test_forehead_requires_skin_and_visible_hair_boundary(self):
        mask=np.zeros((512,512),np.uint8)
        mask[40:175,130:382]=17;mask[175:430,140:372]=1
        result=forehead_metrics(mask,landmarks())
        self.assertEqual(result['reasons'],[])
        # Boundary windows tolerate one non-hair pixel; localization is pixel-quantized.
        self.assertAlmostEqual(result['values']['forehead_height'],65/128,delta=1/128)
        self.assertAlmostEqual(result['values']['forehead_width'],232/128)
        mask[mask==17]=0
        self.assertIsNone(forehead_metrics(mask,landmarks())['values']['forehead_height'])

    def test_bangs_and_hat_remain_unknown(self):
        mask=np.zeros((512,512),np.uint8);mask[40:205,130:382]=17;mask[205:430,140:372]=1
        r=forehead_metrics(mask,landmarks())
        self.assertIn('LOW_HAIR_EDGE_OR_BANGS_UNCERTAIN',r['reasons'])
        self.assertIsNone(r['values']['forehead_width'])
        mask[180:200,230:280]=18
        self.assertIn('HAT_OCCLUSION',forehead_metrics(mask,landmarks())['reasons'])

    def test_actual_component_box_excludes_jewelry_and_clipped_ear(self):
        mask=np.zeros((512,512),np.uint8);mask[220:310,390:430]=7
        mask[90:200,10:80]=9
        self.assertEqual(visible_ear_box(mask,landmarks()),[390,220,430,310])
        r=ear_metrics(mask,landmarks(),40)
        self.assertEqual(r['reasons'],[])
        self.assertAlmostEqual(r['values']['ear_length'],90/(128/np.cos(np.deg2rad(40))))
        self.assertAlmostEqual(r['values']['ear_width'],40/90)
        self.assertIsNone(ear_metrics(mask,landmarks(),0)['values']['ear_length'])
        mask[:]=0;mask[0:100,390:430]=8
        self.assertIsNone(visible_ear_box(mask,landmarks()))

    def test_repeat_instability_and_missing_observations_do_not_become_labels(self):
        self.assertIsNone(stable_measurement([None,1])['value'])
        self.assertEqual(stable_measurement([1,1.05,None])['n'],2)
        self.assertIsNone(stable_measurement([1,1.5,1.1])['value'])

    def test_threshold_fit_cannot_read_nontrain_values(self):
        train=[str(i) for i in range(30)]
        metrics={k:{'x':i/30} for i,k in enumerate(train)}
        metrics['test']={'x':10000};first=fit_thresholds(metrics,train,'x')
        metrics['test']['x']=-10000;second=fit_thresholds(metrics,train,'x')
        self.assertEqual(first,second)
        self.assertEqual(first['eligible_train_n'],30)
        self.assertEqual(metric_label(first['lower'],first,['low','medium','high']),'medium')
        self.assertEqual(metric_label(first['upper'],first,['low','medium','high']),'high')
        self.assertIsNone(metric_label(None,first,['low','medium','high']))
        self.assertEqual(fit_thresholds(metrics,train[:2],'x')['status'],'unavailable')

    def test_head_roi_and_landmark_failures(self):
        b=head_box((512,512),landmarks())
        self.assertTrue(b[0]<140 and b[2]>372 and b[1]<170)
        bad=landmarks();bad[33,0]=np.nan
        with self.assertRaises(ValueError):head_box((512,512),bad)


class RealCPUSmokeTests(unittest.TestCase):
    @unittest.skipUnless(Path('models/face_parsing/resnet18.onnx').exists(),'official local model asset required')
    def test_actual_model_provider_shape_determinism_without_gpu(self):
        parser=ReleaseParser(threads=1)
        self.assertEqual(parser.session.get_providers(),['CPUExecutionProvider'])
        self.assertEqual(parser.model_sha256,MODEL_SHA256)
        with Image.new('RGB',(864,576),(127,127,127)) as image:
            a=parser.parse(image);b=parser.parse(image)
        self.assertEqual(a.shape,(576,864));self.assertEqual(a.dtype,np.uint8)
        np.testing.assert_array_equal(a,b)
        self.assertLessEqual(int(a.max()),18)


class VisibleMarkTests(unittest.TestCase):
    def setUp(self):
        from facework.morphometrics import OVAL
        self.points=landmarks()
        for i,t in zip(OVAL,np.linspace(-np.pi/2,3*np.pi/2,len(OVAL),endpoint=False)):
            self.points[i,:2]=[256+116*np.cos(t),280+140*np.sin(t)]
        for i,xy in {2:(256,340),168:(256,285),98:(242,330),327:(270,330)}.items():self.points[i,:2]=xy
        self.mask=np.zeros((512,512),np.uint8)
        cv2.fillConvexPoly(self.mask,cv2.convexHull(np.round(self.points[OVAL,:2]).astype(np.int32)),1)
        self.rgb=np.full((512,512,3),170,np.uint8)

    def detect(self,rgb=None,mask=None):
        with Image.fromarray(self.rgb if rgb is None else rgb) as image:
            return visible_mark_candidates(image,self.mask if mask is None else mask,self.points)

    def test_injected_spot_repeats_and_clear_skin_absence(self):
        clear=self.detect();self.assertEqual(repeated_mark_label([clear,clear,clear])['label'],'absent')
        rows=[]
        for shift,light in zip([0,1,-1],['L1','L2','L3']):
            rgb=self.rgb.copy();cv2.circle(rgb,(220+shift,205),3,(60,60,60),-1)
            row=self.detect(rgb);row['light']=light;rows.append(row)
        result=repeated_mark_label(rows)
        self.assertEqual(result['label'],'present');self.assertEqual(result['stable_marks'][0]['repeats'],3)
        self.assertIsNone(rows[0]['confidence'])
        self.assertIsNone(repeated_mark_label([rows[0],rows[0]])['label'])

    def test_hair_and_lower_beard_zone_not_candidates(self):
        rgb=self.rgb.copy();cv2.circle(rgb,(220,205),3,(40,40,40),-1);cv2.circle(rgb,(256,380),3,(40,40,40),-1)
        mask=self.mask.copy();mask[190:220,205:235]=17
        self.assertEqual(self.detect(rgb,mask)['candidates'],[])

    def test_smooth_illumination_and_long_shadow_edge_not_stable_spots(self):
        gradient=np.linspace(110,210,512,dtype=np.uint8)[None,:,None]
        rgb=np.broadcast_to(gradient,(512,512,3)).copy()
        self.assertEqual(self.detect(rgb)['candidates'],[])
        rgb[:,250:]=np.maximum(rgb[:,250:].astype(int)-50,0).astype(np.uint8)
        self.assertEqual(self.detect(rgb)['candidates'],[])

    def test_single_light_spot_and_poor_coverage_stay_unknown(self):
        clear=self.detect();rgb=self.rgb.copy();cv2.circle(rgb,(220,205),3,(60,60,60),-1)
        self.assertIsNone(repeated_mark_label([self.detect(rgb),clear,clear])['label'])
        mask=self.mask.copy();mask[:270]=17
        hidden=self.detect(mask=mask)
        self.assertFalse(hidden['quality_passed'])
        self.assertIsNone(repeated_mark_label([hidden,hidden,clear])['label'])


if __name__=='__main__':unittest.main()
