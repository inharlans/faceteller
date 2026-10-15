"""CPU-only numeric and statistical regression checks; no model imports."""
import json
import unittest

import numpy as np

from scripts.model_selection_metrics import (
    aggregate_metrics, apply_temperature, fit_temperature, metrics, paired_bootstrap,
)


class MetricsTests(unittest.TestCase):
    def test_missing_rows_are_excluded_even_with_invalid_probabilities(self):
        result = metrics([[.8, .2], [np.nan, np.nan], [.1, .9]], np.array([0, -1, 1]))
        self.assertEqual(result['valid_count'], 2)
        self.assertEqual(result['missing_count'], 1)
        self.assertEqual(result['accuracy'], 1)
        self.assertEqual(result['support'], [1, 1])
        self.assertAlmostEqual(result['nll'], -np.log(.8 * .9) / 2)
        self.assertAlmostEqual(result['brier'], .05)
        self.assertAlmostEqual(result['ece'], .15)

    def test_absent_truth_class_false_prediction_is_not_ignored(self):
        result = metrics([[.9, .1, 0], [.1, .9, 0]], np.array([0, 0]))
        self.assertEqual(result['macro_f1_classes'], [0, 1])
        self.assertEqual(result['support'], [2, 0, 0])
        self.assertAlmostEqual(result['macro_f1'], 1 / 3)
        self.assertEqual(result['balanced_accuracy'], .5)

    def test_absent_from_truth_and_prediction_does_not_penalize(self):
        result = metrics([[1, 0, 0]], np.array([0]))
        self.assertEqual(result['macro_f1'], 1)
        self.assertEqual(result['macro_f1_classes'], [0])

    def test_balanced_accuracy_uses_true_class_recall(self):
        p = np.eye(3)[[0, 0, 0, 2]]
        result = metrics(p, np.array([0, 0, 0, 1]))
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['macro_f1_classes'], [0, 1, 2])

    def test_threshold_and_ece_bin_edges(self):
        result = metrics([[.6, .4], [.5, .5], [0, 1]], np.array([0, 1, 1]))
        self.assertEqual(result['threshold']['accepted_count'], 2)
        self.assertAlmostEqual(result['threshold']['coverage'], 2 / 3)
        self.assertEqual(result['threshold']['accuracy'], 1)
        self.assertEqual(result['ece_bins'][6]['count'], 1)
        self.assertEqual(result['ece_bins'][5]['count'], 1)
        self.assertEqual(result['ece_bins'][9]['count'], 1)
        self.assertAlmostEqual(result['ece'], .3)

    def test_empty_labels_and_no_acceptances_have_defined_nulls(self):
        empty = metrics(np.full((2, 3), np.nan), np.array([-1, -1]))
        self.assertIsNone(empty['macro_f1'])
        self.assertIsNone(empty['threshold']['coverage'])
        self.assertEqual(empty['support'], [0, 0, 0])
        json.dumps(empty, allow_nan=False)
        result = metrics([[.5, .5]], np.array([0]))
        self.assertEqual(result['threshold']['coverage'], 0)
        self.assertIsNone(result['threshold']['accuracy'])

    def test_bad_valid_inputs_are_rejected(self):
        for p, y in [([[2, -1]], [0]), ([[.1, .2]], [0]), ([[np.nan, .1]], [0]),
                     ([[.5, .5]], [-2]), ([[.5, .5]], [2]), ([[.5, .5]], [0.0])]:
            with self.subTest(p=p, y=y), self.assertRaises(ValueError):
                metrics(p, np.array(y))

    def test_mean_head_weighting_and_incomplete_run(self):
        y = {'large': np.array([0, 1, 0, 1]), 'small': np.array([0, -1, -1, -1])}
        p = {'large': np.eye(2)[[0, 1, 0, 1]], 'small': np.eye(2)[[1, 1, 1, 1]]}
        result = aggregate_metrics(y, p)
        self.assertEqual(result['primary_score'], .5)  # Not pooled accuracy 4/5.
        self.assertEqual(result['mean_per_head']['accuracy'], .5)
        y['small'][:] = -1
        result = aggregate_metrics(y, p)
        self.assertIsNone(result['primary_score'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['excluded_features'], ['small'])


class TemperatureTests(unittest.TestCase):
    def test_exact_power_and_identity_preserve_zeros_and_argmax(self):
        p = np.array([[.64, .36, 0], [.1, .2, .7]])
        expected = np.sqrt(p); expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(apply_temperature(p, 2), expected, rtol=1e-14)
        np.testing.assert_allclose(apply_temperature(p, 1), p, atol=1e-15)
        np.testing.assert_array_equal(apply_temperature(p, 2).argmax(1), p.argmax(1))
        self.assertEqual(apply_temperature(p[0], 2)[2], 0)

    def test_validation_fit_has_analytically_known_optimum(self):
        # 75% correct, every confidence .9: optimal calibrated confidence is .75.
        p = np.tile([.9, .1], (4, 1)); y = np.array([0, 0, 0, 1])
        p_before = p.copy(); y_before = y.copy()
        result = fit_temperature(p, y, validation_ids=[1, 2, 3, 4], test_ids=[5, 6])
        self.assertAlmostEqual(result['temperature'], np.log(9) / np.log(3), places=5)
        self.assertLess(result['nll_after'], result['nll_before'])
        self.assertTrue(result['calibrated'])
        np.testing.assert_array_equal(p, p_before)
        np.testing.assert_array_equal(y, y_before)

    def test_validation_only_and_disjoint_people(self):
        p, y = [[.8, .2]], np.array([0])
        with self.assertRaises(ValueError):
            fit_temperature(p, y, split='test')
        with self.assertRaises(ValueError):
            fit_temperature(p, y, validation_ids=[7], test_ids=[7])
        with self.assertRaises(ValueError):
            fit_temperature(p, y, validation_ids=[7])

    def test_missing_validation_labels_and_empty_fit(self):
        result = fit_temperature([[.9, .1], [np.nan, np.nan]], np.array([0, -1]))
        self.assertEqual(result['n_val'], 1)
        self.assertGreaterEqual(result['log_temperature'], -2)
        self.assertLessEqual(result['log_temperature'], 2)
        empty = fit_temperature([[np.nan, np.nan]], np.array([-1]))
        self.assertFalse(empty['calibrated'])
        self.assertIsNone(empty['temperature'])

    def test_flat_validation_objective_keeps_identity(self):
        result = fit_temperature([[.5, .5], [.5, .5]], np.array([0, 1]))
        self.assertEqual(result['temperature'], 1)
        self.assertEqual(result['status'], 'identity_no_improvement')


class BootstrapTests(unittest.TestCase):
    def test_identical_models_have_exact_zero_paired_interval(self):
        y = {'x': np.array([0, 1, 0, 1]), 'z': np.array([1, 0, 1, 0])}
        p = {'x': np.array([[.7, .3], [.2, .8], [.4, .6], [.9, .1]]),
             'z': np.array([[.4, .6], [.7, .3], [.8, .2], [.4, .6]])}
        result = paired_bootstrap(y, p, p, n_boot=100)
        self.assertEqual(result['mean_macro_f1_difference'], 0)
        self.assertEqual(result['ci95'], [0, 0])

    def test_shared_participant_resample_preserves_cross_head_cancellation(self):
        y = {'x': np.array([0, 1, 0, 1]), 'z': np.array([0, 1, 0, 1])}
        good = np.eye(2)[[0, 1, 0, 1]]; bad = np.eye(2)[[1, 0, 1, 0]]
        a, b = {'x': good, 'z': bad}, {'x': bad, 'z': good}
        result = paired_bootstrap(y, a, b, n_boot=100)
        self.assertEqual(result['ci95'], [0, 0])
        self.assertEqual(result, paired_bootstrap(y, a, b, n_boot=100))

    def test_sign_and_per_head_not_pooled(self):
        y = {'x': np.array([0, 1, 0, 1]), 'z': np.array([0, -1, -1, -1])}
        good = np.eye(2)[[0, 1, 0, 1]]; bad = np.eye(2)[[1, 0, 1, 0]]
        a, b = {'x': good, 'z': good}, {'x': good, 'z': bad}
        result = paired_bootstrap(y, a, b, n_boot=200)
        self.assertEqual(result['mean_macro_f1_difference'], .5)
        self.assertEqual(result['ci95'], [.5, .5])
        self.assertGreater(result['discarded_bootstrap_samples'], 0)
        reverse = paired_bootstrap(y, b, a, n_boot=200)
        self.assertEqual(reverse['ci95'], [-.5, -.5])

    def test_invalid_alignment_and_duplicate_participants(self):
        y = {'x': np.array([0, 1])}; p = {'x': np.eye(2)}
        with self.assertRaises(ValueError):
            paired_bootstrap(y, p, {'other': np.eye(2)})
        with self.assertRaises(ValueError):
            paired_bootstrap(y, p, {'x': np.ones((2, 3)) / 3})
        with self.assertRaises(ValueError):
            paired_bootstrap(y, p, p, participant_ids=['same', 'same'])
        with self.assertRaises(ValueError):
            paired_bootstrap({'x': np.array([-1, -1])}, p, p)


if __name__ == '__main__':
    unittest.main()
