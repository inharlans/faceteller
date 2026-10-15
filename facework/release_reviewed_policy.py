"""Serving checks for the reviewed, class-supported calibration protocol."""
from __future__ import annotations

import math

COVERAGE_POLICY = 'min5_per_class_two_classes_v1'
SUPPORT_POLICY = 'actual_unique_people_with_training_crop_target_argmax_v1'


def validate_reviewed_calibration(entry, classes, exposure):
    """Reject inconsistent candidate metadata before loading any model."""
    def require(condition):
        if not condition:
            raise RuntimeError('Reviewed calibration provenance/support mismatch')

    n = len(classes)
    support = entry.get('support')
    train = entry.get('train_support')
    for counts in (support, train, exposure):
        require(isinstance(counts, list) and len(counts) == n
                and all(type(v) is int and v >= 0 for v in counts))
    require(entry.get('coverage_policy') == COVERAGE_POLICY
            and entry.get('train_support_policy') == SUPPORT_POLICY
            and train == exposure and entry.get('classes') == classes
            and type(entry.get('n_val')) is int and sum(support) == entry['n_val']
            and entry.get('raw_calibration_support') == support
            and entry.get('test_used_for_fit') is False
            and entry.get('participant_separation_checked') is True
            and type(entry.get('supported')) is bool)
    adequate = [i for i, v in enumerate(support) if v >= 5]
    allowed = [i for i in adequate if train[i] > 0]
    rule = entry.get('rule', {})
    require(rule == {'fit_temperature': len(adequate) >= 2,
                     'adequate_classes': adequate, 'allowed_classes': allowed,
                     'minimum_per_class': 5, 'coverage_policy': COVERAGE_POLICY})
    calibrated = entry.get('calibrated')
    if calibrated is True:
        temperature = entry.get('temperature')
        require(entry['supported'] and len(adequate) >= 2
                and isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
                and math.isfinite(temperature) and temperature > 0
                and entry.get('optimizer_success') is True)
        fit = entry.get('fit', {})
        require(fit.get('calibrated') is True and fit.get('optimizer_success') is True
                and fit.get('temperature') == temperature)
        require(entry.get('allowed_confirmation_classes') == [classes[i] for i in allowed])
    else:
        require(calibrated is False and entry.get('temperature') is None
                and entry.get('optimizer_success') is False
                and entry.get('allowed_confirmation_classes') == [])
        expected = ('unsupported_head' if not entry['supported'] else
                    'insufficient_calibration_support' if len(adequate) < 2 else
                    'temperature_fit_failed')
        require(entry.get('status') == expected)
        if expected == 'unsupported_head':
            require(bool(entry.get('unsupported_reason')))
        if expected == 'temperature_fit_failed':
            fit = entry.get('fit', {})
            require(not (fit.get('calibrated') is True and fit.get('optimizer_success') is True))


def confirmation_class_allowed(entry, class_name):
    """Legacy releases have no class gate; reviewed releases must declare it."""
    if entry.get('coverage_policy') != COVERAGE_POLICY:
        return True
    return entry.get('supported') is True and class_name in entry['allowed_confirmation_classes']
