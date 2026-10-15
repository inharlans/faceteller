import json,unittest
from pathlib import Path
from facework.measurement_rules import observations,pilot_reference
ROOT=Path(__file__).resolve().parents[1]

class KnowledgeTests(unittest.TestCase):
 def test_catalog_complete_and_evidence_matches(self):
  catalog=json.loads((ROOT/'data/interpretation_catalog_v1.json').read_text(encoding='utf8'))
  rows=catalog['records'];self.assertEqual(len(rows),103)
  self.assertEqual(len({r['record_id'] for r in rows}),103)
  for r in rows:
   self.assertTrue(r['interpretation_ko']);self.assertFalse(r['independent_review']);self.assertFalse(r['personal_prediction_allowed'])
   self.assertTrue(all(e in r['original_section'] for e in r['interpretation_evidence']))
  req=json.loads((ROOT/'data/type_requirements.json').read_text(encoding='utf8'))['records']
  self.assertEqual({r['record_id'] for r in rows},{r['interpretation_record_id'] for r in req})
  self.assertTrue(all(not r['type_gold_label_ready'] for r in req))
 def test_no_size_labels_without_calibration_or_scale(self):
  spec=json.loads((ROOT/'data/measurement_definitions.json').read_text(encoding='utf8'))
  for flags in [[],['bad capture']]:
   rows=observations({'nose_width_face':.3,'mouth_width_face':float('nan'),'jaw_width_face':float('inf')},spec,flags)
   nose=next(r for r in rows if r['id']=='nose_relative_width')
   self.assertEqual(nose['status'],'capture_review_required' if flags else 'uncalibrated')
   self.assertTrue(all(r['size_label'] is None for r in rows))
   self.assertIsNone(next(r for r in rows if r['id']=='face_absolute_size')['value'])
   json.dumps(rows,allow_nan=False)
 def rows(self):
  return [dict(image_id=str(i),subject_id=str(i),split='train',condition='neutral_front',metrics={'ratio':v}) for i,v in enumerate([1.,2.,3.,100.])]
 def test_subject_balancing(self):
  rows=self.rows();base=pilot_reference(rows,['ratio'],'test')
  rows.extend(dict(image_id=f'dup{i}',subject_id='3',split='train',condition='neutral_front',metrics={'ratio':100.}) for i in range(20))
  dup=pilot_reference(rows,['ratio'],'test')
  self.assertEqual(base['features'],dup['features']);self.assertEqual(dup['subjects'],4);self.assertFalse(dup['deployment_ready'])
 def test_reject_test_split_and_condition_leakage(self):
  for field,value in [('split','test'),('split','validation'),('condition','smiling_front')]:
   rows=self.rows();rows[0][field]=value
   with self.assertRaises(ValueError):pilot_reference(rows,['ratio'],'test')
 def test_reject_duplicate_or_inadequate_data(self):
  rows=self.rows()
  with self.assertRaises(ValueError):pilot_reference(rows+[rows[0]],['ratio'],'test')
  with self.assertRaises(ValueError):pilot_reference(rows[:3],['ratio'],'test')
 def test_collapsed_thresholds_are_not_usable(self):
  rows=self.rows()
  for r in rows:r['metrics']['ratio']=1.
  value=pilot_reference(rows,['ratio'],'test')['features']['ratio']
  self.assertTrue(value['degenerate']);self.assertFalse(value['threshold_ready'])
if __name__=='__main__':unittest.main()
