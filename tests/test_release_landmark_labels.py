import unittest
from scripts.release.prepare_landmark_labels import fit_feature,label


class LandmarkTargetsTest(unittest.TestCase):
    def test_holdout_extremes_cannot_change_cutpoints_or_repeat_guard(self):
        samples={'a':[1,1.1,.9],'b':[2,2.1,1.9],'c':[3,3.1,2.9],'held':[1e8,2e8,3e8]}
        feature={'metric':'ratio','classes':['low','medium','high']}
        original=fit_feature(samples,['a','b','c'],feature,'ratio')
        samples['held']=[-1e8,1e10,2e10]
        self.assertEqual(original,fit_feature(samples,['a','b','c'],feature,'ratio'))
        self.assertIsNone(label([1],original))

    def test_positive_tilt_is_never_down_due_to_population_percentile(self):
        samples={'a':[3,3.1,2.9],'b':[4,4.1,3.9],'c':[5,5.1,4.9]}
        f={'metric':'eye_tilt','classes':['down','level','up']}
        d=fit_feature(samples,list(samples),f,'eye_tilt')
        self.assertLess(d['cuts'][0],0);self.assertGreater(d['cuts'][1],0)
        self.assertEqual(label(samples['a'],d),'up')
        self.assertEqual(label([-.7,-.8,-.9],d),'down')
        self.assertIsNone(label([d['cuts'][1]]*3,d))


if __name__=='__main__':unittest.main()
