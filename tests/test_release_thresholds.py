import unittest
from facework.release_thresholds import select_threshold


class ThresholdTests(unittest.TestCase):
    def fit(self,p,y,**kwargs):
        options={'calibrated':True,'train_support':[10]*len(p[0]),
                 'validation_ids':[str(i) for i in range(len(p))],'test_ids':['outside']}
        options.update(kwargs)
        return select_threshold(p,y,**options)
    def test_holds_low_confidence_error_and_keeps_correct(self):
        r=self.fit([[.9,.1],[.6,.4]],[0,1])
        self.assertEqual(r['threshold'],.65)
        self.assertEqual(r['selected']['accepted'],1)
        self.assertEqual(r['selected']['wrong_confirmed'],0)
    def test_equal_decisions_keep_initial_threshold(self):
        self.assertEqual(self.fit([[.95,.05]],[0])['threshold'],.6)
    def test_multiclass_can_reduce_threshold_with_validation_support(self):
        self.assertEqual(self.fit([[.55,.25,.2]],[0])['threshold'],.55)
    def test_binary_floor_and_untrained_class(self):
        r=self.fit([[.55,.45]],[0]);self.assertGreaterEqual(r['threshold'],.6)
        r=self.fit([[.01,.99]],[1],train_support=[10,0]);self.assertEqual(r['selected']['accepted'],0)
    def test_unknown_labels_do_not_change_selection(self):
        r=self.fit([[.9,.1],[.6,.4],[.99,.01]],[0,1,-1])
        self.assertEqual(r['threshold'],.65);self.assertEqual(r['n_valid'],2)
    def test_disabled_calibration_stays_disabled(self):
        r=self.fit([[.9,.1]],[0],calibrated=False)
        self.assertEqual(r['status'],'disabled_uncalibrated');self.assertEqual(r['candidates'],[])
    def test_test_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError,'test participants'):
            self.fit([[.9,.1]],[0],test_ids=['0'])


if __name__=='__main__':unittest.main()
