"""CPU rejection checks for actual-browser evidence consumed by the final report."""
import unittest
from copy import deepcopy
from scripts.release.build_report import validate_ui_evidence, EvidenceError

class BrowserEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.hashes={'deployment':'a'*64,'depth':'b'*64,'registry':'c'*64}
        self.corpus='d'*64
        self.ui={'pass':True,'actual_models':True,'mocked':False,'artifact_identity_verified':True,
                 'deployment_sha256':'a'*64,'depth_config_sha256':'b'*64,'registry_sha256':'c'*64,
                 'corpus_sha256':self.corpus,'participants':1,'split':'train','photos_saved':False,
                 'external_requests':0,'page_errors':[],'passages':2,'pdf_passage_text_verified_count':2,
                 'pdf_pages':1,'one_page_button':True,'confirmed':20,'held':25}
    def test_rejects_mismatched_artifact_or_mocked_results(self):
        for key in ('deployment_sha256','depth_config_sha256','registry_sha256','corpus_sha256'):
            with self.subTest(key=key):
                row=deepcopy(self.ui);row[key]='e'*64
                with self.assertRaises(EvidenceError):validate_ui_evidence(row,self.hashes,self.corpus)
        for key,value in [('pass',False),('actual_models',False),('mocked',True),('artifact_identity_verified',False),('external_requests',1),('page_errors',['error']),('photos_saved',True)]:
            with self.subTest(key=key):
                row=deepcopy(self.ui);row[key]=value
                with self.assertRaises(EvidenceError):validate_ui_evidence(row,self.hashes,self.corpus)
    def test_text_count_and_page_claim_must_match(self):
        validate_ui_evidence(self.ui,self.hashes,self.corpus)
        for key,value in [('pdf_passage_text_verified_count',1),('pdf_pages',2),('participants',400),('split','test'),('confirmed',None),('one_page_button',None)]:
            row=deepcopy(self.ui);row[key]=value
            with self.assertRaises(EvidenceError):validate_ui_evidence(row,self.hashes,self.corpus)
        empty=deepcopy(self.ui);empty.update(passages=0,pdf_passage_text_verified_count=0)
        validate_ui_evidence(empty,self.hashes,self.corpus)

if __name__=='__main__':unittest.main()
