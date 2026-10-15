"""Shared part + whole-face context input (training and runtime must import this).

Some targets keep their design denominators outside the local crop
(face width 234-454 or face height 10-152). Instead of changing those label
definitions, the same ResNet18 receives a 2-tile image: the unchanged local
part input at left and the whole frontal face input at right. Both tiles have
the part's existing network size INPUT_SIZES[part] = (H, W); the face crop
(INPUT_SIZES['face'] = 256x224) is letterboxed into (H, W) with gray 127
padding by facework.release_preprocess.letterbox (isotropic, never stretched).
Result: uint8 H x 2W x 3.

For mirrored left-side crops (eye_left, brow_left) the face tile is mirrored
too, so the measured side keeps the same position relative to its context.
The raw crop contract (registry parts[*].input_size / crops) is unchanged;
callers record CONTEXT_VERSION and model_input_size(part) separately.
"""
import numpy as np
from PIL import Image

from facework.release_preprocess import INPUT_SIZES, letterbox

CONTEXT_VERSION = 'part-face-context-v1'
CONTEXT_PARTS = ('eye', 'brow', 'nose', 'mouth', 'glabella', 'philtrum')
CONTEXT_REASON = {
    'eye': 'eye_size uses face width; eye crop scale is eye-centre distance',
    'brow': 'brow targets use eye width partly outside the brow window; context for parity with eye',
    'nose': 'nose_width uses face width, nose_length uses face height',
    'mouth': 'mouth_width uses face width',
    'glabella': 'brow_gap uses face width',
    'philtrum': 'philtrum_length uses face height'}


def model_input_size(part):
    """Network input (H, W) for a part: [H, 2W] for context parts, else the raw size."""
    h, w = INPUT_SIZES[part]
    return [h, 2 * w] if part in CONTEXT_PARTS else [h, w]


def compose_context(local, whole, part, *, mirror_face=False):
    """local: part network input (H, W, 3) uint8; whole: face network input (256, 224, 3) uint8."""
    if part not in CONTEXT_PARTS:
        raise ValueError(f'part has no context input: {part}')
    local = np.asarray(local); whole = np.asarray(whole)
    h, w = INPUT_SIZES[part]
    if local.dtype != np.uint8 or whole.dtype != np.uint8:
        raise ValueError('context inputs must be uint8 RGB')
    if local.shape != (h, w, 3):
        raise ValueError(f'local crop must be {h}x{w}x3, got {local.shape}')
    if whole.shape != (*INPUT_SIZES['face'], 3):
        raise ValueError(f'whole-face crop must be {INPUT_SIZES["face"]}x3, got {whole.shape}')
    face = np.asarray(letterbox(Image.fromarray(whole), (h, w)), dtype=np.uint8)
    if mirror_face:
        face = face[:, ::-1]
    return np.ascontiguousarray(np.concatenate([local, face], axis=1))
