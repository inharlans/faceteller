"""CPU ear-scale regressions: target signal, similarity, yaw, roll and rejection."""
import math
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image, ImageDraw
from facework.release_preprocess import ear_crop, network_array
from scripts.release.prepare_ear_crops_v2 import independent_transform, preserve_manifest


def points(distance=100.,centre=(220.,220.)):
    p=np.tile(centre,(478,1)).astype(float)
    p[[33,133]]=np.array(centre)-[distance/2,0]
    p[[263,362]]=np.array(centre)+[distance/2,0]
    return p


def fixture(box=(320,200,360,300),size=(600,600)):
    image=Image.new('RGB',size,(0,0,0))
    ImageDraw.Draw(image).rectangle((box[0],box[1],box[2]-1,box[3]-1),fill=(255,0,0))
    return image


def red_height(crop):
    a=network_array(crop,'ear');ys,_=np.where((a[:,:,0]>200)&(a[:,:,1]<20))
    return int(np.ptp(ys)+1)


class EarScaleTests(unittest.TestCase):
    def test_manifest_rerun_preserves_bytes_even_when_elapsed_changes(self):
        value={'complete':True,'records':{'s':{'metadata_sha256':'fixture'}},'elapsed_seconds':1.}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';preserve_manifest(path,value);before=path.read_bytes()
            result=preserve_manifest(path,{**value,'elapsed_seconds':9.})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(result,value)

    def test_manifest_rejects_real_changes_beyond_record_hashes(self):
        value={'ear_transform_version':'v2','preprocess_version':'global','participants':400,'complete':True,
               'records':{'s':{'metadata_sha256':'fixture'}},'parity':{'pass':True},
               'support':{'train':270},'rejection_counts':{},'labels_sha256':'labels','elapsed_seconds':1.}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';preserve_manifest(path,value);before=path.read_bytes()
            for key in value:
                if key=='elapsed_seconds':continue
                changed=copy.deepcopy(value);changed[key]=None
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Existing manifest changed'):
                    preserve_manifest(path,changed)
                self.assertEqual(path.read_bytes(),before)

    def test_different_relative_lengths_remain_different_pixels(self):
        heights=[]
        for box in ((320,200,360,260),(320,200,360,320)):
            with fixture(box) as image:
                crop=ear_crop(image,box,landmarks=points(),yaw_degrees=40)
                try:heights.append(red_height(crop));self.assertEqual(crop.image.size,(192,256))
                finally:crop.image.close()
        self.assertGreater(heights[1],heights[0]*1.9)

    def test_proportional_face_and_ear_scaling_preserves_geometry_and_height(self):
        box=np.array([320,200,360,300]);crops=[]
        try:
            for factor in (1,2):
                with fixture(tuple(box*factor),(600*factor,600*factor)) as image:
                    crops.append(ear_crop(image,box*factor,landmarks=points()*factor,yaw_degrees=40))
            np.testing.assert_allclose(crops[0].transform[:,:2],crops[1].transform[:,:2]*2,atol=1e-10)
            np.testing.assert_allclose(crops[0].transform[:,2],crops[1].transform[:,2],atol=1e-10)
            self.assertLessEqual(abs(red_height(crops[0])-red_height(crops[1])),1)
        finally:
            for c in crops:c.image.close()

    def test_projected_eye_baseline_and_yaw_use_same_measurement_reference(self):
        box=(320,200,360,300);transforms=[]
        for yaw in (20,40,60):
            with fixture(box) as image:
                crop=ear_crop(image,box,landmarks=points(100*math.cos(math.radians(yaw))),yaw_degrees=yaw)
                try:transforms.append(crop.transform.copy())
                finally:crop.image.close()
        for t in transforms[1:]:np.testing.assert_allclose(t,transforms[0],atol=1e-10)

    def test_roll_alignment_and_mirror_keep_isotropic_measurement_scale(self):
        angle=.3;rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        p=(points()-220)@rot.T+220;box=(320,200,360,300)
        with fixture(box) as image:
            a=ear_crop(image,box,landmarks=p,yaw_degrees=-40)
            b=ear_crop(image,box,landmarks=p,yaw_degrees=-40,mirror=True)
            try:
                np.testing.assert_array_equal(np.asarray(a.image)[:,::-1],np.asarray(b.image))
                np.testing.assert_allclose(b.transform,independent_transform(box,p,-40,True),atol=1e-10)
                aligned=np.c_[p,np.ones(len(p))]@a.transform.T
                self.assertAlmostEqual(aligned[33,1],aligned[263,1])
                singular=np.linalg.svd(a.transform[:,:2],compute_uv=False)
                np.testing.assert_allclose(singular,[128*math.cos(math.radians(40))/100]*2)
            finally:a.image.close();b.image.close()

    def test_missing_reference_never_falls_back_to_bbox_fit(self):
        with fixture() as image:
            with self.assertRaises(TypeError):ear_crop(image,(320,200,360,300))

    def test_truncated_ear_is_rejected_not_shrunk(self):
        with fixture() as image:
            with self.assertRaisesRegex(ValueError,'EAR_WINDOW_TRUNCATED'):
                ear_crop(image,(320,100,380,500),landmarks=points(),yaw_degrees=40)
            with self.assertRaisesRegex(ValueError,'EAR_SOURCE_TRUNCATED'):
                ear_crop(image,(0,200,40,300),landmarks=points(),yaw_degrees=40)
            with self.assertRaisesRegex(ValueError,'EAR_VIEW_OUTSIDE_OBLIQUE_RANGE'):
                ear_crop(image,(320,200,360,300),landmarks=points(),yaw_degrees=float('nan'))


if __name__=='__main__':unittest.main()
