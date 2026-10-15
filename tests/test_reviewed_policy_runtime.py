"""Candidate calibration must never confirm an unsupported class."""
import copy
import unittest

import numpy as np

from facework.release_reviewed_policy import (
    validate_reviewed_calibration, confirmation_class_allowed,
)
from scripts.release.retrain_reviewed import calibrate_head


def calibrated_fixture(supported=True):
    y = np.array([0] * 14 + [1] * 17 + [2] * 3)
    p = np.full((34, 3), .1)
    p[np.arange(34), y] = .8
    return calibrate_head(p, y, {
        'classes': ['a', 'b', 'c'], 'supported': supported,
        'unsupported_reason': None if supported else 'No reliable reference labels',
    }, [60, 150, 60], validation_ids=list(range(34)), test_ids=list(range(100, 162)))


class ReviewedCalibrationTests(unittest.TestCase):
    def test_minority_class_cannot_confirm_even_at_high_score(self):
        c = calibrated_fixture()
        validate_reviewed_calibration(c, ['a', 'b', 'c'], [60, 150, 60])
        self.assertTrue(c['calibrated'])
        self.assertTrue(confirmation_class_allowed(c, 'a'))
        self.assertFalse(confirmation_class_allowed(c, 'c'))

    def test_unsupported_head_never_confirms(self):
        c = calibrated_fixture(False)
        validate_reviewed_calibration(c, ['a', 'b', 'c'], [60, 150, 60])
        self.assertFalse(confirmation_class_allowed(c, 'a'))

    def test_bad_metadata_rejected(self):
        c = calibrated_fixture()
        changes = [
            {'allowed_confirmation_classes': ['a', 'b', 'c']},
            {'supported': False}, {'support': [14, 17, 5]},
            {'train_support': [60, 149, 60]}, {'test_used_for_fit': True},
            {'temperature': float('nan')}, {'optimizer_success': False},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_reviewed_calibration({**copy.deepcopy(c), **change}, ['a', 'b', 'c'], [60, 150, 60])

    def test_legacy_class_gate_unchanged(self):
        self.assertTrue(confirmation_class_allowed({'calibrated': True}, 'c'))


if __name__ == '__main__':
    unittest.main()
