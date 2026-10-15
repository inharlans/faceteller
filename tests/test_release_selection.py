"""Model selection must not reuse ensemble calibration or alter train history."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from facework.release_selection import inference_selection


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.folder=self.root/'models/release'; self.folder.mkdir(parents=True)
        self.config={'architectures':['resnet18','efficientnet_b0','convnext_tiny'],'registry_sha256':'a'*64}
        self.training=self.folder/'training_frozen.json'; self.training.write_text(json.dumps(self.config))
        self.choice={'version':'release-inference-choice-v1','architectures':['resnet18'],
                     'registry_sha256':'a'*64,'training_config_sha256':hashlib.sha256(self.training.read_bytes()).hexdigest()}
    def write_choice(self):
        (self.folder/'inference_frozen.json').write_text(json.dumps(self.choice))
    def test_original_history_without_override(self):
        self.assertEqual(inference_selection(self.root,self.config)[0],self.config['architectures'])
    def test_single_model_and_exact_identity(self):
        self.write_choice(); active,identity=inference_selection(self.root,self.config)
        self.assertEqual(active,['resnet18'])
        self.assertEqual(inference_selection(self.root,self.config,identity)[0],active)
    def test_old_ensemble_calibration_rejected(self):
        self.write_choice()
        with self.assertRaisesRegex(RuntimeError,'recalibration'):
            inference_selection(self.root,self.config,{'config':self.config})
    def test_modified_training_or_reintroduced_ensemble_rejected(self):
        self.choice['architectures'].append('efficientnet_b0');self.write_choice()
        with self.assertRaises(RuntimeError):inference_selection(self.root,self.config)
        self.choice['architectures']=['resnet18'];self.write_choice()
        self.training.write_text('{}')
        with self.assertRaises(RuntimeError):inference_selection(self.root,self.config)
    def test_missing_choice_cannot_fall_back_to_ensemble(self):
        with self.assertRaisesRegex(RuntimeError,'missing'):
            inference_selection(self.root,self.config,{'inference_architectures':['resnet18']})


if __name__=='__main__':unittest.main()
