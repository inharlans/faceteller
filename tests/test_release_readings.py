import hashlib
from copy import deepcopy
import unittest

from facework.pilot_contract import Passage, MemoryRepository
from facework.release_readings import ReadingCatalog


def fixtures():
    text='目尾朝天，福祿綿綿。'
    corpus={'passages':[{'passage_id':'eye','original_text':text,'required_labels':['eye_tilt:up'],
                         'service_eligible':True}]}
    reading={'title':'올라간 눈꼬리','translation':'눈꼬리가 하늘을 향하면 복이 이어진다.',
             'meaning':'옛 글은 이런 눈매를 길하게 보았다.','valence':'favorable',
             'valence_label':'길한 풀이','topics':['복']}
    document={'version':'test','authoring':'ai_edited_unreviewed','readings':{'eye':reading},'named_types':[]}
    policy={'version':'test','excluded_passages':[{'passage_id':'eye',
            'original_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'reason':'A positive angle alone does not establish the stronger literary condition.'}]}
    return document,corpus,policy


class ReadingTests(unittest.TestCase):
    def test_semantic_gate_filters_before_selection_not_just_display(self):
        d,c,p=fixtures();catalog=ReadingCatalog(d,c,p)
        repo=MemoryRepository([Passage('eye',frozenset({'eye_tilt:up'}),'original','translation','note',{})])
        self.assertEqual(catalog.repository(repo).load_candidates({'eye_tilt:up'}),[])

    def test_review_cannot_silently_apply_to_changed_chinese(self):
        d,c,p=fixtures();c['passages'][0]['original_text']='changed'
        with self.assertRaises(ValueError):ReadingCatalog(d,c,p)

    def test_missing_editorial_record_is_rejected(self):
        d,c,p=fixtures();d['readings']={}
        with self.assertRaises(ValueError):ReadingCatalog(d,c,p)

    def test_edit_preserves_original_and_score(self):
        d,c,p=fixtures();p['excluded_passages']=[];catalog=ReadingCatalog(d,c,p)
        result={'items':[],'passages':[{'passage_id':'eye','original_text':c['passages'][0]['original_text'],
                'translation':'old','commentary':'old','source':{'url':'source'},'G':.83}], 'ranking_policy':{}}
        out=catalog.apply(deepcopy(result))
        self.assertEqual(out['passages'][0]['G'],.83)
        self.assertEqual(out['passages'][0]['original_text'],result['passages'][0]['original_text'])
        self.assertEqual(out['passages'][0]['translation'],d['readings']['eye']['translation'])
        self.assertEqual(out['passages'][0]['source']['url'],'source')
        self.assertEqual(out['named_types'],[])


if __name__=='__main__':unittest.main()
