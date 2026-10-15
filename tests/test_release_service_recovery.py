import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.release.verify_service import save_report


class AtomicReportTests(unittest.TestCase):
    def test_temporary_windows_lock_retries_without_losing_old_report(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.json';target.write_text('{"old": true}')
            replace=Path.replace;calls=[]
            def locked(source,destination):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(target.read_text()),{'old':True})
                    raise PermissionError('simulated reader lock')
                return replace(source,destination)
            with patch.object(Path,'replace',locked),patch('scripts.release.verify_service.time.sleep'):
                save_report(target,{'received':397})
            self.assertEqual(json.loads(target.read_text()),{'received':397})
            self.assertEqual(len(calls),3)
    def test_persistent_lock_preserves_both_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'report.json';target.write_text('{"received": 396}')
            with patch.object(Path,'replace',side_effect=PermissionError('locked')),patch('scripts.release.verify_service.time.sleep'):
                with self.assertRaises(PermissionError):save_report(target,{'received':397})
            self.assertEqual(json.loads(target.read_text()),{'received':396})
            self.assertEqual(json.loads(target.with_suffix('.json.tmp').read_text()),{'received':397})


if __name__=='__main__':unittest.main()
