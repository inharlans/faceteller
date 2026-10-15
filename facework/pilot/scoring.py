"""Three ordered intervals and the user's Gaussian boundary score.

This is a geometric membership score, not a calibrated classification accuracy.
The hard label comes from measured r and the two boundaries, not argmax(score).
"""
import math
import numpy as np
from scipy.special import ndtr

LABELS=('low','medium','high')

def boundary_scores(r, lower, upper, sigma):
    if not all(math.isfinite(x) for x in (r,lower,upper,sigma)):
        raise ValueError('all inputs must be finite')
    if not lower<upper or sigma<=0:raise ValueError('require lower < upper and sigma > 0')
    p1=float(ndtr((r-lower)/sigma));p2=float(ndtr((r-upper)/sigma))
    scores={'low':1-p1,'medium':p1-p2,'high':p2}
    idx=0 if r<lower else (1 if r<upper else 2)
    return {'label':LABELS[idx],'score':scores[LABELS[idx]],'scores':scores,
            'above_lower':p1,'above_upper':p2,'sigma':float(sigma),
            'confidence':None,'score_kind':'gaussian_boundary_membership_not_accuracy'}

def estimate_boundaries(values, quantiles=(.25,.75)):
    a=np.asarray(values,float)
    if a.ndim!=1 or len(a)<3 or not np.isfinite(a).all():raise ValueError('finite 1D cohort required')
    b=np.quantile(a,quantiles)
    if b[0]>=b[1]:raise ValueError('cohort has no usable spread')
    return {'lower':float(b[0]),'upper':float(b[1]),'quantiles':list(quantiles),
            'sigma_population':float(a.std(ddof=1)),'mean':float(a.mean()),'n':len(a)}

def repeat_sigma(repeated):
    """Pooled within-participant SD. Different people are not treated as repeats."""
    residual_ss=0.;df=0
    for values in repeated:
        a=np.asarray(values,float);a=a[np.isfinite(a)]
        if len(a)<2:continue
        residual_ss+=float(((a-a.mean())**2).sum());df+=len(a)-1
    if df==0 or residual_ss<=0:raise ValueError('repeated nonconstant captures required')
    return math.sqrt(residual_ss/df)
