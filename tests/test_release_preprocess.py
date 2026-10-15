import unittest
import numpy as np
from PIL import Image
from facework.release_preprocess import eye_affine, frontal_crops, letterbox, ear_crop, PART_CROPS


def points():
    rng = np.random.default_rng(7)
    p = rng.uniform([100, 80], [400, 450], (478, 2))
    p[33], p[133], p[362], p[263] = [150,200], [210,200], [290,200], [350,200]
    return p


class PreprocessTest(unittest.TestCase):
    def test_eye_scale_and_rotation(self):
        p = points()
        a = .32
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        transformed = 1.4 * p @ R.T + [50,70]
        q,_ = eye_affine(transformed)
        r,l = (q[33]+q[133])/2, (q[263]+q[362])/2
        self.assertAlmostEqual(l[0]-r[0],128,places=6)
        self.assertAlmostEqual(l[1],r[1],places=6)

    def test_crop_transform_invariant_under_source_similarity(self):
        p = points()
        im = Image.new('RGB',(512,512))
        a = frontal_crops(im,p)
        b = frontal_crops(Image.new('RGB',(1024,1024)),p*2)
        self.assertEqual(set(a),{n for part,names in PART_CROPS.items() if part!='ear' for n in names})
        homogeneous = np.c_[p,np.ones(len(p))]
        doubled = np.c_[p*2,np.ones(len(p))]
        for name in a:
            np.testing.assert_allclose(homogeneous@a[name].transform.T,doubled@b[name].transform.T,atol=1e-6)

    def test_letterbox_preserves_ratio_with_padding(self):
        im=Image.new('RGB',(80,40),(255,0,0))
        out=np.asarray(letterbox(im,(100,100)))
        red=(out[:,:,0]==255)&(out[:,:,1]==0)
        ys,xs=np.where(red)
        self.assertEqual((xs.max()-xs.min()+1,ys.max()-ys.min()+1),(100,50))
        self.assertTrue((out[0]==127).all())

    def test_invalid_landmarks_and_ears_rejected(self):
        p=points();p[33]=p[263];p[133]=p[362]
        with self.assertRaises(ValueError):eye_affine(p)
        with self.assertRaises(ValueError):ear_crop(Image.new('RGB',(100,100)),(5,5,6,50),landmarks=points(),yaw_degrees=35)
        p=points();p[0,0]=np.nan
        with self.assertRaises(ValueError):eye_affine(p)

    def test_out_of_frame_padding_is_reported(self):
        crops=frontal_crops(Image.new('RGB',(80,80)),points())
        self.assertLess(crops['face'].valid_fraction,.2)


if __name__=='__main__':unittest.main()
