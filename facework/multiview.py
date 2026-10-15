"""Multi-view 3D shape from 2D landmarks of the same moment-equivalent session.

Scaled-orthographic Tomasi-Kanade factorisation with a metric upgrade. Uses only surface
landmarks that stay visible up to about 40 degrees of yaw (midline, eye corners, brows,
nose, lips); silhouette contour points are excluded because they slide across views.
Depths are unitless: divided by the 3D outer-eye-corner distance.
"""
import numpy as np

MIDLINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 0, 13, 14, 17, 18, 200, 199, 175]
PAIRED = [33, 263, 133, 362, 159, 386, 145, 374, 70, 300, 105, 334, 107, 336, 98, 327, 129, 358,
          64, 294, 61, 291, 37, 267, 84, 314, 50, 280]
POINTS = MIDLINE + PAIRED
VIEWS = ['C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C15', 'C16', 'C17', 'C19', 'C20']

def factorize(xy):
    """xy: V x N x 2 image coordinates (same point order in every view). Returns N x 3 metric shape and residual."""
    V, N, _ = xy.shape
    W = (xy - xy.mean(1, keepdims=True)).transpose(0, 2, 1).reshape(2 * V, N)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    M = U[:, :3] * np.sqrt(s[:3]); S = np.sqrt(s[:3])[:, None] * Vt[:3]
    residual = float(np.sqrt((s[3:] ** 2).sum() / (s ** 2).sum()))
    # Metric constraints on L = Q Q^T (symmetric, 6 unknowns): |i|=|j|, i.j=0, and first view |i|=1.
    def row(a, b):
        return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[0]*b[2] + a[2]*b[0], a[1]*b[1], a[1]*b[2] + a[2]*b[1], a[2]*b[2]]
    A = []; y = []
    for v in range(V):
        i, j = M[2 * v], M[2 * v + 1]
        A.append(np.subtract(row(i, i), row(j, j))); y.append(0)
        A.append(row(i, j)); y.append(0)
    A.append(row(M[0], M[0])); y.append(1)
    l = np.linalg.lstsq(np.array(A), np.array(y, float), rcond=None)[0]
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    w, E = np.linalg.eigh(L)
    if (w <= 0).any():
        w = np.clip(w, 1e-9, None)  # nearest PSD; flagged by residual/ratio checks downstream
    Q = E @ np.diag(np.sqrt(w))
    X = (np.linalg.inv(Q) @ S).T
    return X, residual

def canonical_frame(X, idx):
    """Rotate so x runs subject-right->left (outer eye corners), y downward, z toward the viewer (nose positive)."""
    p = {k: X[idx[k]] for k in [33, 263, 10, 175, 1, 133, 362]}
    x = p[263] - p[33]; x /= np.linalg.norm(x)
    down = p[175] - p[10]; down -= x * (down @ x); down /= np.linalg.norm(down)
    z = np.cross(x, down)
    R = np.stack([x, down, z])
    Y = (X - X.mean(0)) @ R.T
    eyes_z = (Y[idx[133], 2] + Y[idx[362], 2]) / 2
    if Y[idx[1], 2] < eyes_z:  # reflection ambiguity: the nose tip must lie in front of the inner canthi
        Y[:, 2] *= -1
    return Y / np.linalg.norm(p[263] - p[33])

def depth_features(xy):
    """Features from V x len(POINTS) x 2. Positive values = further toward the viewer."""
    X, residual = factorize(xy)
    idx = {k: i for i, k in enumerate(POINTS)}
    Y = canonical_frame(X, idx)
    z = lambda k: Y[idx[k], 2]
    return {'nose_projection': float(z(1) - (z(98) + z(327)) / 2),        # tip above alar base
            'nose_bridge': float(z(168) - (z(133) + z(362)) / 2),         # nasal root above inner canthi
            'nose_dorsum': float(z(6) - (z(133) + z(362)) / 2),
            'nose_tip_from_eyes': float(z(1) - (z(133) + z(362)) / 2),
            'lip_projection': float((z(0) + z(17)) / 2 - (z(61) + z(291)) / 2),
            'chin_projection': float(z(175) - z(2)),
            'mid_face_depth': float((z(133) + z(362)) / 2 - (z(33) + z(263)) / 2),
            'residual': residual}, Y
