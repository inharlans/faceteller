"""Shared training/inference crops for design 5.2.

Coordinates are original image pixels. Eye-centre distance sets the scale;
every crop uses one isotropic affine transform and constant padding. No image
is stretched to fit a classifier. Ear boxes must come from a visible-region
detector, never from a guessed Face Mesh ear landmark (there is none).
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image

from facework.morphometrics import EYE, BROW, MOUTH, OVAL

VERSION = 'eye-centre-isotropic-v2'
TARGET_EYE_DISTANCE = 128.0
# Native crop dimensions (height, width) in the common eye-distance space.
WINDOWS = {
    'eye': (80, 128), 'brow': (112, 160), 'nose': (144, 128),
    'mouth': (112, 160), 'face': (416, 352),
    'forehead': (256, 384), 'glabella': (96, 112),
    'under_eye': (80, 128), 'cheekbone': (144, 144),
    'philtrum': (80, 112), 'mole': (416, 352),
}
INPUT_SIZES = {
    'eye': (160, 256), 'brow': (224, 320), 'nose': (256, 224),
    'mouth': (224, 320), 'face': (256, 224),
    'forehead': (192, 288), 'glabella': (192, 224),
    'under_eye': (160, 256), 'cheekbone': (224, 224),
    'ear': (256, 192), 'philtrum': (160, 224), 'mole': (416, 352),
}
PART_CROPS = {
    'eye': ('eye_right', 'eye_left'), 'brow': ('brow_right', 'brow_left'),
    'nose': ('nose',), 'mouth': ('mouth',), 'face': ('face',),
    'forehead': ('forehead',), 'glabella': ('glabella',),
    'under_eye': ('under_eye_right', 'under_eye_left'),
    'cheekbone': ('cheekbone_right', 'cheekbone_left'),
    'ear': ('ear_left_oblique', 'ear_right_oblique'),
    'philtrum': ('philtrum',), 'mole': ('mole',),
}


@dataclass
class Crop:
    image: Image.Image
    transform: np.ndarray
    valid_fraction: float


def eye_affine(points):
    p = np.asarray(points, dtype=np.float64)[:, :2]
    if len(p) < 468 or not np.isfinite(p).all():
        raise ValueError('finite 468-point face landmarks required')
    right, left = (p[33] + p[133]) / 2, (p[263] + p[362]) / 2
    delta = left - right
    distance = float(np.linalg.norm(delta))
    if distance < 4:
        raise ValueError('eye centres are too close to normalize')
    roll = math.degrees(math.atan2(delta[1], delta[0]))
    matrix = cv2.getRotationMatrix2D(tuple((right + left) / 2), roll,
                                   TARGET_EYE_DISTANCE / distance)
    aligned = np.c_[p, np.ones(len(p))] @ matrix.T
    return aligned, matrix


def letterbox(image, size):
    """PIL RGB -> isotropic resize and padding; size is (height, width)."""
    h, w = size
    scale = min(w / image.width, h / image.height)
    nw, nh = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new('RGB', (w, h), (127, 127, 127))
    out.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return out


def frontal_crops(image, points):
    q, affine = eye_affine(points)
    rgb = np.asarray(image.convert('RGB'))
    face_centre = (q[OVAL].min(0) + q[OVAL].max(0)) / 2
    eye_centre = (q[33] + q[133] + q[263] + q[362]) / 4
    brow_centre = (q[107] + q[336]) / 2
    centres = {
        'nose': .4 * q[168] + .6 * q[2],
        'mouth': q[MOUTH].mean(0) - [0, .09 * TARGET_EYE_DISTANCE],
        'face': face_centre, 'mole': face_centre,
        'forehead': brow_centre - [0, 96],
        'glabella': brow_centre,
        'philtrum': (q[2] + q[0]) / 2,
    }
    for side in ('right', 'left'):
        e, b = EYE[side], BROW[side]
        centres['eye_' + side] = q[e['ring']].mean(0)
        centres['brow_' + side] = np.r_[q[b['upper']], q[b['lower']]].mean(0) + [0, .08 * TARGET_EYE_DISTANCE]
        centres['under_eye_' + side] = (q[e['inner']] + q[e['outer']]) / 2 + [0, .20 * TARGET_EYE_DISTANCE]
        edge = 234 if side == 'right' else 454
        centres['cheekbone_' + side] = .6 * q[edge] + .4 * eye_centre + [0, .20 * TARGET_EYE_DISTANCE]
    result = {}
    for name, centre in centres.items():
        kind = name.rsplit('_', 1)[0] if name.endswith(('_right', '_left')) else name
        h, w = WINDOWS[kind]
        matrix = affine.copy()
        matrix[:, 2] += np.array([w / 2, h / 2]) - centre
        pixels = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        valid = cv2.warpAffine(np.ones(rgb.shape[:2], np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
        if name.endswith('_left'):
            pixels = pixels[:, ::-1]
            matrix = (np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]]) @ np.vstack([matrix, [0, 0, 1]]))[:2]
        result[name] = Crop(Image.fromarray(np.ascontiguousarray(pixels)), matrix, float(valid.mean()))
    return result


def ear_crop(image, box, *, landmarks, yaw_degrees, mirror=False):
    """Fixed ear window in yaw-corrected eye units, never fitted to ear size.

    The measured ear_length uses the same e/cos(yaw) reference. Reject a
    truncated visible box instead of shrinking it into the input window.
    """
    x0, y0, x1, y1 = map(float, box)
    if not np.isfinite([x0, y0, x1, y1]).all() or min(x1-x0, y1-y0) < 8:
        raise ValueError('visible ear region is too small')
    if x0 <= 0 or y0 <= 0 or x1 >= image.width or y1 >= image.height:
        raise ValueError('EAR_SOURCE_TRUNCATED')
    if not math.isfinite(yaw_degrees) or not 20 <= abs(yaw_degrees) <= 60:
        raise ValueError('EAR_VIEW_OUTSIDE_OBLIQUE_RANGE')
    _, affine = eye_affine(landmarks)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    linear = affine[:, :2] * math.cos(math.radians(yaw_degrees))
    matrix = np.c_[linear, np.array([96., 128.]) - linear @ [cx, cy]]
    corners = np.array([[x0,y0,1], [x1-1,y0,1], [x0,y1-1,1], [x1-1,y1-1,1]]) @ matrix.T
    # Conservative rotated-box containment, including one pixel interpolation margin.
    if np.any(corners.min(0) < [1,1]) or np.any(corners.max(0) > [190,254]):
        raise ValueError('EAR_WINDOW_TRUNCATED')
    rgb = np.asarray(image.convert('RGB'))
    pixels = cv2.warpAffine(rgb, matrix, (192, 256), borderValue=(127,127,127))
    valid = cv2.warpAffine(np.ones(rgb.shape[:2],np.uint8), matrix, (192,256), flags=cv2.INTER_NEAREST)
    if mirror:
        pixels = pixels[:, ::-1]
        matrix = (np.array([[-1,0,191],[0,1,0],[0,0,1]]) @ np.vstack([matrix,[0,0,1]]))[:2]
    return Crop(Image.fromarray(np.ascontiguousarray(pixels)),matrix,float(valid.mean()))


def network_array(crop, part):
    image = crop.image if isinstance(crop, Crop) else crop
    return np.asarray(letterbox(image, INPUT_SIZES[part]), dtype=np.uint8).copy()
