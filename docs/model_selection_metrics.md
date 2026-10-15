# Backbone selection metrics interface

Module: `scripts/model_selection_metrics.py`. Dependencies: NumPy and SciPy only.
It does not import Torch, the earlier CPU audit, or service code, and does not
modify process environment, network access, device/thread settings, or files.

```python
from scripts.model_selection_metrics import (
    metrics, aggregate_metrics, fit_temperature,
    apply_temperature, paired_bootstrap,
)

head = metrics(test_probabilities, test_truth)  # NxC float, N integer; -1 = missing
fit = fit_temperature(
    val_probabilities, val_truth,
    validation_ids=val_people, test_ids=test_people,  # optional paired arguments
)
if fit['calibrated']:
    corrected = apply_temperature(test_probabilities, fit['temperature'])
    corrected_head = metrics(corrected, test_truth)

# Fifteen feature keys, one row per person, identical participant and class order.
report = aggregate_metrics(y_by_feature, probs_by_feature)
selection_score = report['primary_score']  # mean of 15 head macro-F1s
comparison = paired_bootstrap(
    y_by_feature, probs_a, probs_b, n_boot=2000, seed=20261014,
    participant_ids=test_people,  # optional uniqueness/length check
)
```

## One-head report

`metrics(probabilities, truth, *, threshold=0.6)` returns:

- `total_count`, `valid_count`, `missing_count`, `class_count`.
- `accuracy`, `macro_f1`, `balanced_accuracy`, `nll`, `brier`, `ece`.
- `support`: true-class counts in the original C-column order, including zeros.
  `predicted_support` uses that same order.
- `macro_f1_classes`: union of true and predicted classes on valid rows. A class
  absent from truth but falsely predicted receives F1=0 and stays in the mean.
  A class absent from both does not lower the score.
- `balanced_accuracy_classes`: classes present in truth; balanced accuracy is
  their mean recall. False predictions of other classes still reduce recall.
- `threshold`: `{value, accepted_count, coverage, accuracy}`. Accept when
  `max(prob) >= value`; coverage is accepted / valid_count. Selective accuracy
  is correctness among accepted valid rows. This does not include other runtime
  quality/occlusion gates.
- `ece_bins`: ten fixed equal-width bins `[0,.1), ... [.9,1]`; each includes
  boundaries, count, accuracy, mean confidence. ECE weights bins by valid count.
- `nll_epsilon=1e-15`: only NLL evaluation floors true-class probabilities at
  this value. Brier is the sum over classes of squared probability error,
  averaged over valid people, not divided by class count.

Missing truth `-1` excludes a row from every metric, including its probability
validation. Thus an unknown-label row may contain NaNs. On valid rows probabilities
must be finite, within [0,1], and sum to one within absolute tolerance 1e-6;
accepted summation drift is normalized. Ties use the first maximum class index.
Zero valid labels or zero accepted rows produce `None` for undefined metrics,
never a fabricated zero accuracy or NaN. Reports can be serialized with
`json.dumps(report, allow_nan=False)`.

## Aggregation and model selection

`aggregate_metrics(y_by_feature, probabilities_by_feature, *, threshold=0.6)`
requires identical feature keys and participant row counts. It returns
`features` (individual reports), `mean_per_head`, `mean_denominator_heads`,
`feature_count`, `scored_feature_count`, `complete`, and `excluded_features`.

The primary score is the **unweighted mean of per-head macro-F1**. Pass all 15
heads; neither feature support nor number of classes weights this mean. There
is no pooling of 15 different classification problems. If any requested head
has no valid labels, `primary_score=None` and `complete=False`; available-head
means remain descriptive and must not silently rank an incomplete experiment.
Selective-accuracy means can have fewer contributing heads when a head accepts
no rows; their denominators are explicit.

Choose the backbone/settings on validation, then report held-out test results.
Evaluate the three fixed training seeds separately and report their individual
scores and mean/spread; this participant bootstrap does not estimate training
seed uncertainty. Do not stack repeated predictions of the same people from
three seeds into 3N supposedly independent participants. Averaging probabilities
over seeds creates an ensemble and should be labeled as that separate estimand.

## Temperature fitting

`fit_temperature(val_probabilities, val_truth, *, split='validation',
validation_ids=None, test_ids=None)` minimizes validation NLL using SciPy bounded
`minimize_scalar` on **logT in [-2,2]** (xatol 1e-7). `T=exp(logT)`.
It optimizes the probability matrix supplied, including an ensemble's already
averaged probabilities if that is the deployment candidate.

Application is exactly `softmax(log(p)/T) = p**(1/T) / sum(p**(1/T))`, computed
in log space. Zeros remain zero and positive T preserves argmax. No training
or test arrays are consumed by the fitter. `split='test'` is rejected. Supplying
both ID lists checks participant overlap; array names alone cannot prove genuine
validation provenance or exclusion from model training.

Return fields include `temperature`, `log_temperature`, `n_val`, `support`,
`calibrated`, `status`, `nll_before`, `nll_after`, `optimizer_success`, `at_bound`,
and `participant_separation_checked`. Empty validation yields no fit and
`calibrated=False`. Flat objectives or numerical non-improvements retain T=1
after successful optimization, with `status='identity_no_improvement'`.
`calibrated=True` records a successful validation fit, not proof of improvement
on unseen test data. Report test NLL/ECE before and after without retuning on test.

## Paired interval

`paired_bootstrap(y_by_feature, probs_a, probs_b, n_boot=2000, seed=20261014,
*, participant_ids=None)` returns `mean_macro_f1_a`, `mean_macro_f1_b`,
`mean_macro_f1_difference` (**A minus B**), and `ci95=[lower, upper]`.

Each draw resamples N participant indices with replacement once and applies the
same draw to all heads and both models. Every participant must have one row;
for multiple photos per person, aggregate to that row before calling this API.
Class-column meaning/order must match between candidates; equal shape cannot
detect a semantic class reorder. Optional IDs check uniqueness, not hidden row
ordering in external arrays.

Macro-F1 truth/prediction unions are recomputed on each draw. The original data
must have valid labels in every requested head. Draws with zero valid labels in
any head are discarded to keep the fixed 15-head estimand. `n_boot` is the number
of attempted draws; `valid_bootstrap_samples` and `discarded_bootstrap_samples`
make sparse-label losses explicit. The CI is the ordinary percentile 2.5/97.5
interval; fewer than two valid draws yields `ci95=None`. It is not a confidence
interval over training seeds or a multiple-comparison correction.

The underlying FaceTeller visual reference labels are three-AI-rater consensus,
not independent human ground truth. These scores measure agreement with that
reference. This interpretation belongs in experiment reporting, not metric code.

## Verification and examples

Run CPU-only unit tests:

```text
.venv/Scripts/python.exe -m unittest discover -s tests -p test_model_selection_metrics.py -v
```

Synthetic report examples (not K-FACE results):
`outputs/model-selection-20261014/metrics_examples.json`.
