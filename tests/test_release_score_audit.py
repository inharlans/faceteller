import unittest
from facework.pilot_contract import Passage
from scripts.release.audit_scores import audit


class ActualScoreAuditTest(unittest.TestCase):
    def test_replays_mixed_rank_and_rejects_corrupted_actual_score(self):
        def item(key,kind,score):
            return {'item_id':key,'region':'nose','kind':kind,'status':'CONFIRMED',
                    'confidence':score if kind=='classification' else None,
                    'measurement_score':score if kind=='depth' else None,
                    'label_id':key+':high','hold_reasons':[]}
        passages=[Passage('a',frozenset({'width:high'}),'原文','직역','문헌 설명',{},1,1),
                  Passage('b',frozenset({'bridge:high'}),'原文二','직역2','문헌 설명2',{},1,2)]
        row={'status':'COMPLETE','items':[item('width','classification',.6),item('bridge','depth',.9)],
             'passages':[{'passage_id':'b','G':.9},{'passage_id':'a','G':.6}]}
        result=audit([row],passages)
        self.assertEqual(result['counts']['order_changed_by_depth'],1)
        self.assertEqual(result['counts']['selected_mixed'],0)
        self.assertAlmostEqual(result['mean_depth_minus_cnn_confirmed'],.3)
        row['passages'][0]['G']=.99
        with self.assertRaisesRegex(ValueError,'Actual G mismatch'):audit([row],passages)

    def test_one_clause_requiring_both_scores_uses_the_joint_geometric_mean(self):
        items=[{'item_id':'width','region':'nose','kind':'classification','status':'CONFIRMED',
                'confidence':.6,'measurement_score':None,'label_id':'width:wide','hold_reasons':[]},
               {'item_id':'bridge','region':'nose','kind':'depth','status':'CONFIRMED',
                'confidence':None,'measurement_score':.9,'label_id':'bridge:high','hold_reasons':[]}]
        passage=Passage('fixture',frozenset({'width:wide','bridge:high'}),'合成','합성 조합검사','실제 문헌 아님',{},1,1)
        row={'status':'COMPLETE','items':items,'passages':[{'passage_id':'fixture','G':(.6*.9)**.5}]}
        result=audit([row],[passage])
        self.assertEqual(result['counts']['selected_mixed'],1)
        self.assertEqual(result['counts']['selected_depth_only'],0)
        self.assertEqual(result['counts']['selected_cnn_only'],0)

    def test_absent_measurement_has_no_invented_score(self):
        result=audit([{'status':'PARTIAL','items':[{'item_id':'bridge','region':'nose','kind':'depth',
                      'status':'HELD','label_id':None,'confidence':None,'measurement_score':None,
                      'hold_reasons':['DEPTH_MEASUREMENT_FAILED']}],'passages':[]}],[])
        self.assertEqual(result['score_distributions'],{})
        self.assertIsNone(result['mean_pairwise_depth_higher_rate'])


if __name__=='__main__':unittest.main()
