import unittest
from PIL import Image, ImageDraw
import numpy as np
from facework.release_runtime import quality_reasons
from tests.test_release_runtime import fake_landmarks
from facework.release_depth import observations, FEATURES, VERSION


class FaceQualityTests(unittest.TestCase):
    def flags(self, background, face):
        with Image.new('RGB',(1000,1000),(background,)*3) as image:
            ImageDraw.Draw(image).rectangle((90,160,310,261),fill=(face,)*3)
            detections=[{'landmarks':fake_landmarks(),'yaw':yaw,'pitch':-5,'smile':0,'blink':0} for yaw in (0,-40,40)]
            return quality_reasons([image]*3,detections)
    def test_bright_face_on_dark_background_is_accepted(self):
        self.assertTrue(all('ILLUMINATION_TOO_DARK' not in reasons for reasons in self.flags(0,180).values()))
    def test_dark_face_on_bright_background_is_held(self):
        self.assertTrue(all('ILLUMINATION_TOO_DARK' in reasons for reasons in self.flags(180,20).values()))
    def test_overexposed_face_matches_acquisition_gate(self):
        self.assertTrue(all('ILLUMINATION_TOO_BRIGHT' in reasons for reasons in self.flags(30,250).values()))
    def test_invalid_landmarks_are_held_without_pixel_lookup(self):
        points=fake_landmarks();points[0,0]=np.nan
        with Image.new('RGB',(400,400)) as image:
            flags=quality_reasons([image], [{'landmarks':points}])
        self.assertTrue(all(reasons==['LANDMARKS_INVALID'] for reasons in flags.values()))
    def test_swapped_oblique_slots_hold_ears_without_discarding_frontal_parts(self):
        detections=[{'landmarks':fake_landmarks(),'yaw':yaw,'pitch':-5,'roll':0,'smile':0,'blink':0} for yaw in (0,40,-40)]
        with Image.new('RGB',(400,400),(170,)*3) as image:
            flags=quality_reasons([image]*3,detections)
        self.assertIn('EAR_VIEW_DIRECTIONS_INVALID',flags['ear'])
        self.assertNotIn('EAR_VIEW_DIRECTIONS_INVALID',flags['nose'])
    def test_roll_matches_acquisition_limit(self):
        detections=[{'landmarks':fake_landmarks(),'yaw':yaw,'pitch':-5,'roll':13 if yaw==0 else 0,'smile':0,'blink':0} for yaw in (0,-40,40)]
        with Image.new('RGB',(400,400),(170,)*3) as image:
            flags=quality_reasons([image]*3,detections)
        self.assertTrue(all('FRONTAL_POSE_INVALID' in reasons for reasons in flags.values()))
    def test_depth_checks_slot_sign_not_only_opposing_angles(self):
        config={'framing_version':VERSION,'quality':{'min_confidence':0,'max_cross_view_error':1},
                'features':{k:{'lower':.2,'upper':.8,'sigma_repeat_train':.1} for k in FEATURES}}
        schema={k:{'part':'nose','classes':['low','medium','high'],'classes_ko':['낮음','보통','높음']} for k in FEATURES}
        measured={'status':'measured','depth_confidence_min':1,'cross_view_midline_error_ratio':0,
                  'values':{k:.5 for k in FEATURES}}
        for yaws,valid in [((0,-40,40),True),((0,40,-40),False),((0,0,0),False)]:
            d=[{'yaw':y,'pitch':-5,'roll':0,'smile':0,'blink':0} for y in yaws]
            items=observations(measured,d,config,schema)
            self.assertTrue(all(i.quality_passed==valid for i in items))


if __name__=='__main__':unittest.main()
