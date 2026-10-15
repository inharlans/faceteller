"""Morphology v2: landmark ratios, an image-based brow thickness, and scale-normalised crops.

All ratios are unitless and computed after in-plane (roll) alignment. Left/right refer to the
subject. Values describe what a photo shows; they are not physical measurements or fortunes.
"""
import math
import cv2
import numpy as np
from PIL import Image

# MediaPipe Face Mesh indices (subject side).
EYE = {'right': dict(outer=33, inner=133, top=159, bottom=145, ring=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
       'left': dict(outer=263, inner=362, top=386, bottom=374, ring=[263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466])}
BROW = {'right': dict(upper=[70, 63, 105, 66, 107], lower=[46, 53, 52, 65, 55]),
        'left': dict(upper=[300, 293, 334, 296, 336], lower=[276, 283, 282, 295, 285])}
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 49, 279, 129, 358]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
        148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Scale-normalised crop windows: face width (234-454) is resized to FACE_PX; each window is
# (height, width) around the part centre. Relative size information is preserved on purpose.
FACE_PX = 256
WINDOWS = {'eye': (80, 128), 'brow': (112, 160), 'nose': (144, 128), 'mouth': (112, 160), 'face': (208, 176)}
FACE_SCALE = 0.5  # face window is taken at half resolution to include the whole outline

def align(p, w, h):
    """Rotation about the eye-corner midpoint so the outer eye corners are level."""
    v = p[263] - p[33]
    roll = math.degrees(math.atan2(v[1], v[0]))
    M = cv2.getRotationMatrix2D(tuple(map(float, (p[33] + p[263]) / 2)), roll, 1)
    q = np.c_[p[:, :2], np.ones(len(p))] @ M.T
    return q, M, roll

def _d(q, a, b):
    return float(np.linalg.norm(q[a] - q[b]))

def _chord_sag(points):
    """Maximum perpendicular rise of a polyline above its end-point chord, relative to chord length (+ = upward)."""
    a, b = points[0], points[-1]; chord = b - a; n = np.linalg.norm(chord)
    if n < 1e-6:
        return 0.0, 0
    normal = np.array([chord[1], -chord[0]]) / n  # image y grows downward
    if normal[1] > 0:
        normal = -normal
    rise = (points - a) @ normal
    k = int(np.argmax(rise))
    return float(rise[k] / n), k

def landmark_metrics(q):
    """Ratios from aligned landmarks (N x 2). Keys are stable identifiers used by labels and models."""
    fw = _d(q, 234, 454); fh = _d(q, 10, 152)
    m = {'face_width_px': fw,
         'face_length': fh / fw,
         'jaw_width': _d(q, 172, 397) / fw,
         'nose_width': _d(q, 98, 327) / fw,
         'nose_length': _d(q, 168, 2) / fh,
         'mouth_width': _d(q, 61, 291) / fw,
         'lip_thickness': (_d(q, 0, 13) + _d(q, 14, 17)) / _d(q, 61, 291),
         'lip_balance': _d(q, 0, 13) / max(_d(q, 14, 17), 1e-6),
         'mouth_open': _d(q, 13, 14) / _d(q, 61, 291)}
    # Mouth corners relative to the lip contact midpoint; + means corners sit higher.
    centre_y = (q[13, 1] + q[14, 1]) / 2
    m['mouth_corner'] = float((centre_y - (q[61, 1] + q[291, 1]) / 2) / _d(q, 61, 291))
    for side, e in EYE.items():
        ew = _d(q, e['outer'], e['inner'])
        m[f'eye_size_{side}'] = ew / fw
        m[f'eye_roundness_{side}'] = _d(q, e['top'], e['bottom']) / ew  # opening height / width
        m[f'eye_tilt_{side}'] = math.degrees(math.atan2(q[e['inner'], 1] - q[e['outer'], 1], abs(q[e['outer'], 0] - q[e['inner'], 0])))
        b = BROW[side]; up = q[b['upper']]; lo = q[b['lower']]; mid = (up + lo) / 2
        m[f'brow_length_{side}'] = float(sum(np.linalg.norm(mid[i + 1] - mid[i]) for i in range(len(mid) - 1)) / ew)
        # Gap: lowest point of the lower brow row to the upper-lid apex, over eye width.
        m[f'brow_eye_gap_{side}'] = float((q[e['top'], 1] - lo[:, 1].max()) / ew)
        sag, _ = _chord_sag(mid)
        m[f'brow_arch_{side}'] = sag
        # Tail: slope of the lateral third (points ordered lateral -> medial); + means tail rises.
        tail = mid[0] - mid[2]
        m[f'brow_tail_{side}'] = math.degrees(math.atan2(-tail[1], abs(tail[0])))
    for key in ['eye_size', 'eye_roundness', 'eye_tilt', 'brow_length', 'brow_eye_gap', 'brow_arch', 'brow_tail']:
        m[key] = (m[f'{key}_right'] + m[f'{key}_left']) / 2
    m['eye_asymmetry'] = abs(math.log(max(m['eye_roundness_right'], 1e-3) / max(m['eye_roundness_left'], 1e-3)))
    # v3 additions: glabella width, intercanthal spacing, philtrum length, lower/middle face ratio.
    inner_r = (q[107] + q[55]) / 2; inner_l = (q[336] + q[285]) / 2
    m['brow_gap'] = float(np.linalg.norm(inner_r - inner_l) / fw)
    m['eye_spacing'] = _d(q, 133, 362) / ((_d(q, 33, 133) + _d(q, 263, 362)) / 2)
    m['philtrum_length'] = _d(q, 2, 0) / fh
    m['lower_face_ratio'] = _d(q, 2, 152) / _d(q, 168, 2)
    return {k: float(v) for k, v in m.items()}

def brow_thickness(gray_aligned, q, side):
    """Vertical extent of dark hair pixels across the brow, over eye width.

    For three columns (inner, middle, outer third) a local window around the landmark band is
    thresholded by Otsu against surrounding skin. Returns None when contrast is too low
    (dark lighting, sunglasses, hair over brows).
    """
    b = BROW[side]; e = EYE[side]
    ew = _d(q, e['outer'], e['inner'])
    up = q[b['upper']]; lo = q[b['lower']]
    h, w = gray_aligned.shape
    heights = []
    for i in (1, 2, 3):
        cx = (up[i, 0] + lo[i, 0]) / 2
        top = min(up[i, 1], lo[i, 1]) - 0.35 * ew; bottom = max(up[i, 1], lo[i, 1]) + 0.25 * ew
        x0, x1 = int(cx - 0.06 * ew), int(cx + 0.06 * ew) + 1
        y0, y1 = int(max(0, top)), int(min(h, bottom))
        if x0 < 0 or x1 > w or y1 - y0 < 8:
            return None
        strip = gray_aligned[y0:y1, x0:x1].astype(np.float32).mean(1)
        strip = cv2.GaussianBlur(strip.reshape(-1, 1), (1, 5), 0).ravel()
        lo_v, hi_v = np.percentile(strip, 5), np.percentile(strip, 95)
        if hi_v - lo_v < 12:
            return None
        u8 = np.clip((strip - lo_v) / (hi_v - lo_v) * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (u8.ravel() <= t)
        # Longest dark run that overlaps the landmark band centre.
        centre = int((up[i, 1] + lo[i, 1]) / 2) - y0
        if not 0 <= centre < len(dark) or not dark[centre]:
            return None
        a = centre
        while a > 0 and dark[a - 1]:
            a -= 1
        z = centre
        while z < len(dark) - 1 and dark[z + 1]:
            z += 1
        heights.append((z - a + 1) / ew)
    return float(np.median(heights))

def normalized_crops(image, p, parts=('eye_right', 'eye_left', 'brow_right', 'brow_left', 'nose', 'mouth', 'face')):
    """Roll-aligned, face-width-normalised fixed windows. Left eye/brow are mirrored to the right-side orientation."""
    arr = np.asarray(image.convert('RGB'))
    q, M, _ = align(p, arr.shape[1], arr.shape[0])
    fw = _d(q, 234, 454)
    out = {}
    for part in parts:
        kind = part.split('_')[0]
        if kind == 'eye':
            c = q[EYE[part.split('_')[1]]['ring']].mean(0)
        elif kind == 'brow':
            b = BROW[part.split('_')[1]]; c = np.r_[q[b['upper']], q[b['lower']]].mean(0)
            c = c + [0, 0.035 * fw]  # include the upper lid so brow-eye spacing is visible, keep arch headroom
        elif kind == 'nose':
            c = 0.4 * q[168] + 0.6 * q[2]  # full alar base and nostrils inside the window
        elif kind == 'mouth':
            c = q[MOUTH].mean(0) - [0, 0.04 * fw]  # include the philtrum up to the subnasale
        else:
            c = (q[OVAL].min(0) + q[OVAL].max(0)) / 2
        hh, ww = WINDOWS[kind]
        s = FACE_PX / fw * (FACE_SCALE if kind == 'face' else 1.0)
        # Single affine: original -> aligned -> scaled window centred on c.
        A = np.vstack([M, [0, 0, 1]])
        S = np.array([[s, 0, ww / 2 - s * c[0]], [0, s, hh / 2 - s * c[1]], [0, 0, 1]])
        T = (S @ A)[:2]
        crop = cv2.warpAffine(arr, T, (ww, hh), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        if part.endswith('_left'):
            crop = crop[:, ::-1]
        out[part] = Image.fromarray(np.ascontiguousarray(crop))
    return out

def measure(image, p):
    """All v2 metrics for one image given original-pixel landmarks (N x 2 or N x 3)."""
    arr = np.asarray(image.convert('L'))
    q, M, roll = align(p, arr.shape[1], arr.shape[0])
    m = landmark_metrics(q); m['roll_deg'] = roll
    g = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
    for side in ('right', 'left'):
        m[f'brow_thickness_{side}'] = brow_thickness(g, q, side)
    vals = [m[f'brow_thickness_{s}'] for s in ('right', 'left') if m[f'brow_thickness_{s}'] is not None]
    m['brow_thickness'] = float(np.mean(vals)) if vals else None
    return m
