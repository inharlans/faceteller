"""Independent CPU metrics for participant-aligned, multi-head classification.

Only NumPy/SciPy are used; importing this module does not change thread settings,
network access, device state, or files. See docs/model_selection_metrics.md.
Rows are people, columns are a fixed class order, and truth=-1 means missing.
"""
from collections.abc import Mapping

import numpy as np
from scipy.optimize import minimize_scalar

NLL_EPS = 1e-15


def _probabilities(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError('probabilities must have shape (N, C), C >= 1')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('probabilities must be finite and within [0, 1]')
    totals = p.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1.0, atol=1e-6, rtol=0):
        raise ValueError('probability rows must sum to one (absolute tolerance 1e-6)')
    return p / totals  # Remove only tolerated floating-point summation drift.


def _inputs(probabilities, truth):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(truth)
    if p.ndim != 2 or p.shape[1] < 1 or y.ndim != 1 or len(y) != len(p):
        raise ValueError('expected probabilities (N, C) and truth (N,)')
    if y.dtype.kind not in 'iu':
        raise ValueError('truth must be an integer array with missing labels encoded as -1')
    if (y < -1).any() or (y >= p.shape[1]).any():
        raise ValueError('truth values must be -1 or a valid class index')
    valid = y != -1
    # Invalid probabilities on missing-label rows are deliberately ignored.
    return _probabilities(p[valid]), y[valid].astype(np.int64), valid, p.shape[1]


def _confusion_scores(y, predicted, classes):
    truth_support = np.bincount(y, minlength=classes)
    pred_support = np.bincount(predicted, minlength=classes)
    true_positive = np.bincount(y[y == predicted], minlength=classes)
    union = (truth_support + pred_support) > 0
    present_truth = truth_support > 0
    f1 = np.zeros(classes, dtype=float)
    np.divide(2 * true_positive, truth_support + pred_support, out=f1, where=union)
    recall = np.zeros(classes, dtype=float)
    np.divide(true_positive, truth_support, out=recall, where=present_truth)
    return {
        'accuracy': float((y == predicted).mean()) if len(y) else None,
        'macro_f1': float(f1[union].mean()) if union.any() else None,
        'balanced_accuracy': float(recall[present_truth].mean()) if present_truth.any() else None,
        'support': truth_support.tolist(), 'predicted_support': pred_support.tolist(),
        'macro_f1_classes': np.flatnonzero(union).tolist(),
        'balanced_accuracy_classes': np.flatnonzero(present_truth).tolist(),
    }


def _nll(p, y):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], NLL_EPS)).mean())


def metrics(probabilities, truth, *, threshold=0.6):
    """Return JSON-safe metrics for one head; missing-label rows are excluded.

    macro_f1 uses the truth/prediction union, so predicting an absent true class
    contributes a zero F1. Balanced accuracy averages recall over true classes.
    Brier is the multiclass squared-error SUM, averaged over valid people.
    ECE uses ten fixed bins [0,.1), ... [.9,1], weighted by valid sample count.
    Threshold acceptance is max(p) >= threshold; coverage denominator is valid_count.
    Empty denominators return None, including selective accuracy with no acceptances.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and within [0, 1]')
    p, y, valid, classes = _inputs(probabilities, truth)
    n = len(y)
    predicted = p.argmax(axis=1)
    result = {'total_count': len(valid), 'valid_count': n, 'missing_count': int((~valid).sum()),
              'class_count': classes, **_confusion_scores(y, predicted, classes)}
    confidence = p.max(axis=1)
    correct = predicted == y
    bins = []
    ece = 0.0
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        keep = (confidence >= low) & ((confidence < high) if i < 9 else (confidence <= 1))
        count = int(keep.sum())
        accuracy = float(correct[keep].mean()) if count else None
        mean_confidence = float(confidence[keep].mean()) if count else None
        if count:
            ece += count / n * abs(accuracy - mean_confidence)
        bins.append({'lower': low, 'upper': high, 'count': count,
                     'accuracy': accuracy, 'mean_confidence': mean_confidence})
    accepted = confidence >= threshold
    # Avoid creating an N x C identity target array.
    squared_error = (p * p).sum(axis=1) - 2 * p[np.arange(n), y] + 1
    result.update(nll=_nll(p, y) if n else None,
                  brier=float(squared_error.mean()) if n else None,
                  ece=float(ece) if n else None, ece_bins=bins, nll_epsilon=NLL_EPS,
                  threshold={'value': float(threshold), 'accepted_count': int(accepted.sum()),
                             'coverage': float(accepted.mean()) if n else None,
                             'accuracy': float(correct[accepted].mean()) if accepted.any() else None})
    return result


def apply_temperature(probabilities, temperature):
    """Exact normalized p**(1/T), computed in log space. Zeros stay zero.

    Accepts a class vector or an N x C matrix. No probability flooring is used
    here; NLL's numerical floor is separate from this transformation.
    """
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be a finite positive scalar')
    values = np.asarray(probabilities, dtype=np.float64)
    single = values.ndim == 1
    p = _probabilities(values[None, :] if single else values)
    logits = np.full_like(p, -np.inf)
    np.log(p, out=logits, where=p > 0)
    logits /= float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(axis=1, keepdims=True)
    return q[0] if single else q


def fit_temperature(val_probabilities, val_truth, *, split='validation',
                    validation_ids=None, test_ids=None):
    """Fit logT in [-2,2] by validation NLL only; never accepts a test split.

    Optional validation_ids/test_ids check participant disjointness. Both must
    be provided together; validation_ids must contain one unique ID per row.
    This interface cannot independently establish the provenance of unlabeled
    arrays. The caller must pass genuine, non-training validation predictions.
    """
    if split != 'validation':
        raise ValueError('temperature fitting accepts split="validation" only')
    p, y, valid, classes = _inputs(val_probabilities, val_truth)
    checked_ids = validation_ids is not None or test_ids is not None
    if checked_ids:
        if validation_ids is None or test_ids is None:
            raise ValueError('provide both validation_ids and test_ids to check separation')
        ids = list(validation_ids); forbidden = set(test_ids)
        if len(ids) != len(valid) or len(set(ids)) != len(ids):
            raise ValueError('validation_ids must be unique and match validation rows')
        if set(ids) & forbidden:
            raise ValueError('validation/test participant overlap')
    result = {'temperature': None, 'log_temperature': None, 'n_val': len(y),
              'class_count': classes, 'support': np.bincount(y, minlength=classes).tolist(),
              'calibrated': False, 'split': split, 'test_used_for_fit': False,
              'participant_separation_checked': checked_ids,
              'log_temperature_bounds': [-2.0, 2.0], 'nll_epsilon': NLL_EPS,
              'nll_before': None, 'nll_after': None, 'optimizer_success': False,
              'at_bound': False}
    if not len(y):
        return {**result, 'status': 'no_valid_validation_labels'}
    baseline = _nll(p, y)
    objective = lambda log_t: _nll(apply_temperature(p, np.exp(log_t)), y)
    fit = minimize_scalar(objective, bounds=(-2.0, 2.0), method='bounded', options={'xatol': 1e-7})
    success = bool(fit.success and np.isfinite(fit.fun))
    # Identity is feasible. Keep it for flat objectives or numerical non-improvement.
    improved = success and float(fit.fun) < baseline - 1e-12
    log_t = float(fit.x) if improved else 0.0
    result.update(temperature=float(np.exp(log_t)) if success else None,
                  log_temperature=log_t if success else None, calibrated=success,
                  nll_before=baseline, nll_after=objective(log_t) if success else None,
                  optimizer_success=success, optimizer_evaluations=int(fit.nfev),
                  at_bound=success and abs(log_t) > 2 - 1e-5,
                  status='fitted' if improved else 'identity_no_improvement' if success else 'optimizer_failed')
    return result


def _aligned_features(y_by_feature, *probability_maps):
    if not isinstance(y_by_feature, Mapping) or not y_by_feature:
        raise ValueError('y_by_feature must be a nonempty mapping')
    features = sorted(y_by_feature)
    if any(not isinstance(p, Mapping) or set(p) != set(features) for p in probability_maps):
        raise ValueError('all mappings must contain exactly the same feature keys')
    lengths = set()
    for k in features:
        shape = None
        for predictions in probability_maps:
            _, _, valid, classes = _inputs(predictions[k], y_by_feature[k])
            lengths.add(len(valid))
            if shape is not None and shape != classes:
                raise ValueError(f'class counts differ between models for feature {k}')
            shape = classes
    if len(lengths) != 1:
        raise ValueError('all features must have the same participant row count/order')
    return features, lengths.pop()


def aggregate_metrics(y_by_feature, probabilities_by_feature, *, threshold=0.6):
    """Equal weight per head, never pool observations across differently sized heads.

    primary_score is mean head macro-F1, but is None if a requested head has no
    reference labels. mean_per_head still exposes descriptive available-head
    means, with denominator counts; it must not silently rank incomplete runs.
    """
    features, participants = _aligned_features(y_by_feature, probabilities_by_feature)
    heads = {k: metrics(probabilities_by_feature[k], y_by_feature[k], threshold=threshold) for k in features}
    means, counts = {}, {}
    for name in ('accuracy', 'macro_f1', 'balanced_accuracy', 'nll', 'brier', 'ece'):
        vals = [v[name] for v in heads.values() if v[name] is not None]
        means[name] = float(np.mean(vals)) if vals else None
        counts[name] = len(vals)
    for name in ('coverage', 'accuracy'):
        vals = [v['threshold'][name] for v in heads.values() if v['threshold'][name] is not None]
        means['threshold_' + name] = float(np.mean(vals)) if vals else None
        counts['threshold_' + name] = len(vals)
    missing = [k for k, v in heads.items() if not v['valid_count']]
    return {'primary_metric': 'mean_per_head_macro_f1',
            'primary_score': means['macro_f1'] if not missing else None,
            'participant_count': participants, 'feature_count': len(features),
            'scored_feature_count': len(features) - len(missing), 'complete': not missing,
            'excluded_features': missing, 'mean_per_head': means,
            'mean_denominator_heads': counts, 'threshold': float(threshold), 'features': heads}


def paired_bootstrap(y_by_feature, probs_a, probs_b, n_boot=2000, seed=20261014, *, participant_ids=None):
    """Paired participant bootstrap of mean-head macro-F1(A) - macro-F1(B).

    One shared participant resample is used for every head and both models.
    Head masks (-1) and truth/prediction-union F1 are recomputed per resample.
    All heads need >=1 valid label in the original data. A draw leaving any
    head without truth is discarded, never reweighted across fewer heads.
    n_boot counts attempted draws; returned counts expose sparse-label losses.
    CI is the percentile [2.5%,97.5%] interval. No seed-level inference is implied.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError('n_boot must be an integer >= 2')
    features, n = _aligned_features(y_by_feature, probs_a, probs_b)
    if n < 2:
        raise ValueError('paired bootstrap requires at least two participant rows')
    if participant_ids is not None:
        ids = list(participant_ids)
        if len(ids) != n or len(set(ids)) != n:
            raise ValueError('participant_ids must contain exactly one unique ID per row')
    prepared = []
    for k in features:
        pa, y, mask, classes = _inputs(probs_a[k], y_by_feature[k])
        pb, _, _, _ = _inputs(probs_b[k], y_by_feature[k])
        if not len(y):
            raise ValueError(f'feature {k} has no valid truth; cannot bootstrap the fixed feature set')
        full_y = np.asarray(y_by_feature[k], dtype=np.int64)
        pred_a = np.full(n, -1, dtype=np.int64); pred_b = pred_a.copy()
        pred_a[mask] = pa.argmax(1); pred_b[mask] = pb.argmax(1)
        prepared.append((full_y, pred_a, pred_b, classes))

    def score_pair(indices):
        scores_a, scores_b = [], []
        for y, a, b, classes in prepared:
            valid_indices = indices[y[indices] != -1]
            if not len(valid_indices):
                return None
            truth = y[valid_indices]
            scores_a.append(_confusion_scores(truth, a[valid_indices], classes)['macro_f1'])
            scores_b.append(_confusion_scores(truth, b[valid_indices], classes)['macro_f1'])
        return float(np.mean(scores_a)), float(np.mean(scores_b))

    point_a, point_b = score_pair(np.arange(n))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_boot):
        pair = score_pair(rng.integers(0, n, size=n))
        if pair is not None:
            differences.append(pair[0] - pair[1])
    ci = np.quantile(differences, [0.025, 0.975]).tolist() if len(differences) >= 2 else None
    return {'metric': 'mean_per_head_macro_f1', 'direction': 'A_minus_B',
            'mean_macro_f1_a': point_a, 'mean_macro_f1_b': point_b,
            'mean_macro_f1_difference': point_a - point_b, 'ci95': ci,
            'method': 'paired_participant_percentile_bootstrap',
            'participant_count': n, 'feature_count': len(features), 'features': features,
            'n_boot': int(n_boot), 'valid_bootstrap_samples': len(differences),
            'discarded_bootstrap_samples': int(n_boot) - len(differences),
            'seed': int(seed), 'participant_ids_checked': participant_ids is not None,
            'status': 'ok' if ci is not None else 'insufficient_valid_bootstrap_samples'}
