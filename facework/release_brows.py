"""Visible eyebrow-hair measurements from a face-parsing mask, version 3.

Shared by candidate training and (after main's validation) runtime overlays, so
both use exactly this code. v3 fixes two blockers found by main's independent
review of v2 (outputs/release/brow-mask-independent-review-20261015.json):
endpoint extent no longer drops thin one-pixel tail columns, and a bilateral
subject value requires both contributing sides to be present and repeatable.
Version 1 (scripts/release/audit_brow_masks.py,
"visible-brow-mask-candidate-v1") is kept only as a historical baseline; native
QA of all 1200 frontal masks found failures this version gates instead:
detached fragments inflating extent, ROI truncation, mass-quantile extent that
shrinks with thicker medial hair, paths off the mask, unchecked pose/blink.

Pipeline per frame and subject side (subject's own right/left):
  1. frame gate: frontal pose and blink/expression (same thresholds as
     scripts/kface_labels.quality_ok) when pose/blend are supplied;
  2. 8-connected components of brow classes (2, 3) over the whole mask; the
     guide ROI (FaceMesh brow points +/- 0.3 eye width) only selects candidates;
  3. reject if more than FRAGMENT_LIMIT of the ROI hair mass lies outside the
     largest component (fragmented/ambiguous mask; never bridged);
  4. reject if the chosen component extends beyond the ROI by more than
     TRUNCATION_LIMIT of its pixels, or touches the image border;
  5. extent = first..last aligned column occupied by ANY pixel of the selected
     component (connected, fragments already excluded), so neither thicker body
     nor thin tails change the span; thin endpoint runs are reported
     (thin_tail_columns) rather than silently trimmed;
  6. central path = windowed median row of the component, smoothed (sigma 1);
     reject if fewer than PATH_ON_MASK of a dense polyline probe (endpoints and
     PATH_DENSE points per segment) lies within PATH_DISTANCE aligned px of a
     component pixel. This is a tolerance, not exact containment.
Subject summaries (aggregate): bilateral value only when both sides are
present, each side has >= MIN_REPEATS repeats and passes its own repeat gate.
Ratios use the same-side palpebral fissure width (33-133 / 263-362) in the
eye-centre-isotropic space. tail_angle_deg > 0: lateral end higher than medial.

Automated measurement of visible segmented hair only. Not human-verified,
not concealed anatomical extent; a short visible span is not proof of a short brow.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d, label as connected_components

from facework.morphometrics import BROW, EYE
from facework.release_preprocess import eye_affine

VERSION = 'visible-brow-mask-v3'
BASELINE_VERSION = 'visible-brow-mask-candidate-v1'
PREVIOUS_VERSION = 'visible-brow-mask-v2'
HAIR_CLASSES = (2, 3)  # CelebAMask-HQ left/right eyebrow
SAMPLES = 25
ROI_MARGIN_EYE = 0.3
MIN_PIXELS = 20
MIN_SPAN_EYE = 0.3
MIN_COLUMNS = 20
MIN_COLUMN_PIXELS = 1  # v2 used 2, which trimmed thin tails (review BROW-V2-01)
PATH_DENSE = 8
MIN_REPEATS = 2
FRAGMENT_LIMIT = 0.10
TRUNCATION_LIMIT = 0.05
PATH_DISTANCE = 2.0
PATH_ON_MASK = 0.90
REPEAT_SD_FACTOR = 3.0  # subject repeat SD above 3 x train median single-capture SD -> unstable
METRICS = ('length_eye_ratio', 'horizontal_eye_ratio', 'tail_angle_deg')
GATES = {'version': VERSION, 'roi_margin_eye': ROI_MARGIN_EYE, 'min_pixels': MIN_PIXELS, 'min_span_eye': MIN_SPAN_EYE,
         'min_columns': MIN_COLUMNS, 'min_column_pixels': MIN_COLUMN_PIXELS, 'fragment_limit': FRAGMENT_LIMIT,
         'truncation_limit': TRUNCATION_LIMIT, 'path_distance_aligned_px': PATH_DISTANCE, 'path_on_mask': PATH_ON_MASK,
         'repeat_sd_factor': REPEAT_SD_FACTOR, 'connectivity': 8, 'path_dense_points_per_segment': PATH_DENSE,
         'min_repeats': MIN_REPEATS, 'subject_requires_both_sides_stable': True,
         'frontal_gate': '|yaw|<8, |pitch+5|<16, eyeBlink L/R < 0.5, blend[0] < 0.3'}


def frontal_quality_ok(pose, blend):
    """Same thresholds as scripts.kface_labels.quality_ok (pitch, yaw, roll / blendshape order)."""
    pitch, yaw, _ = pose
    return abs(yaw) < 8 and abs(pitch - (-5.0)) < 16 and blend[1] < 0.5 and blend[2] < 0.5 and blend[0] < 0.3


def _inverse(affine):
    return np.linalg.inv(np.vstack([np.asarray(affine, dtype=np.float64), [0, 0, 1]]))[:2]


def measure_brows(mask, points, *, pose=None, blend=None, include_path=True):
    """Return {'version', 'sides': {side: metrics}, 'rejected': {side: reason}, 'frame_rejected'}.

    Rejected or invisible sides are never imputed. Accepted sides contain the
    metrics, gate diagnostics and, with ``include_path``, the central path
    (medial -> lateral) in aligned and original image pixels.
    """
    out = {'version': VERSION, 'sides': {}, 'rejected': {}, 'frame_rejected': None}
    if pose is not None and blend is not None and not frontal_quality_ok(pose, blend):
        out['frame_rejected'] = 'FRAME_POSE_OR_EXPRESSION'
        return out
    mask = np.asarray(mask)
    hair_mask = np.isin(mask, HAIR_CLASSES)
    labels, _ = connected_components(hair_mask, structure=np.ones((3, 3), int))
    y, x = np.nonzero(labels)
    if len(x) < MIN_PIXELS:
        out['frame_rejected'] = 'NO_VISIBLE_BROW_HAIR'
        return out
    component = labels[y, x]
    q, affine = eye_affine(points)
    aligned = np.c_[x, y, np.ones(len(x))] @ affine.T
    inverse = _inverse(affine)
    h, w = mask.shape[:2]
    for side, e in EYE.items():
        guide = q[BROW[side]['upper'] + BROW[side]['lower']]
        width = float(np.linalg.norm(q[e['outer']] - q[e['inner']]))
        low, high = guide.min(0) - width * ROI_MARGIN_EYE, guide.max(0) + width * ROI_MARGIN_EYE
        in_roi = ((aligned >= low) & (aligned <= high)).all(1)
        if in_roi.sum() < MIN_PIXELS:
            out['rejected'][side] = 'TOO_FEW_PIXELS'; continue
        ids, counts = np.unique(component[in_roi], return_counts=True)
        main = ids[np.argmax(counts)]; main_in_roi = counts.max(); roi_mass = counts.sum()
        diag = {'roi_pixels': int(roi_mass), 'component_pixels': int((component == main).sum()),
                'fragment_fraction': float(1 - main_in_roi / roi_mass)}
        if diag['fragment_fraction'] > FRAGMENT_LIMIT:
            out['rejected'][side] = 'FRAGMENTED_MASK'; continue
        members = component == main
        diag['outside_roi_fraction'] = float(1 - main_in_roi / members.sum())
        if diag['outside_roi_fraction'] > TRUNCATION_LIMIT:
            out['rejected'][side] = 'ROI_TRUNCATED'; continue
        if (x[members].min() == 0 or y[members].min() == 0 or x[members].max() == w - 1 or y[members].max() == h - 1):
            out['rejected'][side] = 'IMAGE_BOUNDARY'; continue
        pts = aligned[members]
        columns = np.floor(pts[:, 0]).astype(int)
        occupied, per_column = np.unique(columns, return_counts=True)
        occupied = occupied[per_column >= MIN_COLUMN_PIXELS]
        if len(occupied) == 0:
            out['rejected'][side] = 'TOO_FEW_PIXELS'; continue
        left, right = float(occupied.min()), float(occupied.max() + 1)
        # Diagnostic only: endpoint runs thinner than half the median column count.
        thin = per_column[per_column >= MIN_COLUMN_PIXELS] < 0.5 * np.median(per_column)
        diag['occupied_columns'] = int(len(occupied))
        diag['thin_tail_columns'] = {'low_x': int(np.argmax(~thin)) if (~thin).any() else int(len(thin)),
                                     'high_x': int(np.argmax(~thin[::-1])) if (~thin).any() else int(len(thin))}
        if members.sum() < MIN_PIXELS or right - left < width * MIN_SPAN_EYE:
            out['rejected'][side] = 'TOO_SHORT'; continue
        xx = np.linspace(left, right, SAMPLES); step = (right - left) / (SAMPLES - 1); yy = []
        for value in xx:
            ys = pts[np.abs(pts[:, 0] - value) <= step, 1]
            yy.append(float(np.median(ys)) if len(ys) >= 2 else np.nan)
        ok = np.isfinite(yy)
        if ok.sum() < MIN_COLUMNS:
            out['rejected'][side] = 'TOO_FEW_COLUMNS'; continue
        yy = gaussian_filter1d(np.interp(xx, xx[ok], np.asarray(yy)[ok]), 1)
        path = np.c_[xx, yy]
        t = np.linspace(0, 1, PATH_DENSE, endpoint=False)
        dense = np.vstack([path[:-1] + (path[1:] - path[:-1]) * f for f in t] + [path[-1:]])
        distance = np.sqrt(((dense[:, None, :] - pts[None, :, :2]) ** 2).sum(-1)).min(1)
        diag['path_on_mask_fraction'] = float((distance <= PATH_DISTANCE).mean())
        diag['path_max_distance'] = float(distance.max())
        diag['path_probe_points'] = int(len(dense))
        if diag['path_on_mask_fraction'] < PATH_ON_MASK:
            out['rejected'][side] = 'PATH_OFF_MASK'; continue
        inner_left = abs(left - q[e['inner'], 0]) < abs(right - q[e['inner'], 0])
        inner_y, outer_y = (yy[0], yy[-1]) if inner_left else (yy[-1], yy[0])
        result = {'length_eye_ratio': float(np.hypot(np.diff(xx), np.diff(yy)).sum() / width),
                  'horizontal_eye_ratio': float((right - left) / width),
                  'tail_angle_deg': float(math.degrees(math.atan2(inner_y - outer_y, right - left))),
                  'hair_pixels': int(members.sum()), 'columns_valid': int(ok.sum()), **diag}
        # Separate observation, not a class: a bottom-quartile "short" is population-relative.
        result['visible_span_shorter_than_eye'] = bool(result['horizontal_eye_ratio'] < 1.0)
        if include_path:
            ordered = path if inner_left else path[::-1]
            result['path_aligned'] = ordered.tolist()
            result['path_original'] = (np.c_[ordered, np.ones(len(ordered))] @ inverse.T).tolist()
        out['sides'][side] = result
    return out


def aggregate(frames, *, single_capture_sd=None, min_repeats=MIN_REPEATS):
    """Subject summaries from per-frame ``measure_brows(...)['sides']`` dicts.

    Per-side value: median of that side's frames; withheld (median None) when it
    has fewer than ``min_repeats`` repeats or, with ``single_capture_sd``
    ({'subject'|'side': {metric: sd}}, train-only), its repeat SD exceeds
    REPEAT_SD_FACTOR x that SD. Subject value: per-frame mean of both sides over
    bilateral frames, median across frames, published only when it passes the
    same gates AND both contributing side summaries are published (a stable
    bilateral mean cannot hide an unstable side). Nothing is imputed.
    """
    def summary(values, limit):
        values = [float(v) for v in values]
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        unstable = bool(limit is not None and sd is not None and sd > REPEAT_SD_FACTOR * limit)
        insufficient = len(values) < min_repeats
        return {'n': len(values), 'median': None if unstable or insufficient else float(np.median(values)),
                'sd': sd, 'unstable': unstable, 'insufficient_repeats': insufficient, 'values': values}
    limits = single_capture_sd or {}
    out = {'version': VERSION, 'subject': {}, 'sides': {'right': {}, 'left': {}}}
    for metric in METRICS:
        for side in ('right', 'left'):
            out['sides'][side][metric] = summary([f[side][metric] for f in frames if side in f], limits.get('side', {}).get(metric))
        both = [np.mean([f[s][metric] for s in ('left', 'right')]) for f in frames if 'left' in f and 'right' in f]
        subject = summary(both, limits.get('subject', {}).get(metric))
        failed_sides = [side for side in ('right', 'left') if out['sides'][side][metric]['median'] is None]
        subject['contributing_side_failed'] = failed_sides
        if failed_sides: subject['median'] = None
        out['subject'][metric] = subject
    spans = [f[s]['visible_span_shorter_than_eye'] for f in frames for s in f]
    out['visible_span_shorter_than_eye_frames'] = {'true': int(sum(spans)), 'observed': len(spans)}
    return out


def overlay_payload(result, metric=None):
    """Adapter for runtime overlays: accepted sides as {'path': original-pixel path, ...}.

    ``path`` is the medial -> lateral central path in original image pixels; a
    consumer must also check ``version`` before presenting mask-derived readings.
    """
    out = {'version': result.get('version', VERSION), 'source': 'visible_brow_mask', 'sides': {}, 'rejected': dict(result.get('rejected', {})),
           'frame_rejected': result.get('frame_rejected')}
    for side, values in result.get('sides', {}).items():
        if 'path_original' not in values: raise ValueError('measure_brows(include_path=True) is required for overlays')
        entry = {'path': values['path_original']}
        for key in (METRICS if metric is None else (metric,)): entry[key] = values[key]
        out['sides'][side] = entry
    return out
