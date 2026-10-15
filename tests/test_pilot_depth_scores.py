import numpy as np
import pytest
from facework.pilot.scoring import boundary_scores,estimate_boundaries,repeat_sigma
from facework.pilot.depth import measurements_from_points,LANDMARK_IDS

@pytest.mark.parametrize('sigma',[.001,.1,1,10])
def test_ordered_memberships_and_hard_intervals(sigma):
    scores=[boundary_scores(float(r),-.2,.3,sigma) for r in np.linspace(-3,3,101)]
    for a in scores:
        assert min(a['scores'].values())>=0
        assert sum(a['scores'].values())==pytest.approx(1)
    assert all(a['scores']['high']<=b['scores']['high'] for a,b in zip(scores,scores[1:]))
    assert boundary_scores(-.2,-.2,.3,sigma)['label']=='medium'
    assert boundary_scores(.3,-.2,.3,sigma)['label']=='high'

def test_no_normality_assumption_for_boundaries():
    a=np.exp(np.linspace(-2,2,400))
    c=estimate_boundaries(a)
    labels=[boundary_scores(v,c['lower'],c['upper'],c['sigma_population'])['label'] for v in a]
    assert [labels.count(k) for k in ['low','medium','high']]==[100,200,100]

def test_sigma_does_not_confuse_population_and_repeat_variation():
    assert repeat_sigma([[100,101,99],[-100,-99,-101]])==pytest.approx(1.)
    with pytest.raises(ValueError):boundary_scores(0,1,1,.1)
    with pytest.raises(ValueError):boundary_scores(0,1,2,0)
    with pytest.raises(ValueError):boundary_scores(float('nan'),1,2,.1)

def test_population_sigma_can_give_middle_score_below_half():
    x=boundary_scores(0,-.67449,.67449,1)
    assert x['scores']['medium']==pytest.approx(.5,abs=1e-5)
    # Hard intervals must not silently switch to argmax membership.
    x=boundary_scores(-.66,-.67449,.67449,1)
    assert x['label']=='medium' and max(x['scores'],key=x['scores'].get)=='low'

def test_measurements_invariant_to_scale_translation_and_common_rotation():
    rng=np.random.default_rng(17)
    p={i:rng.normal(size=3) for i in LANDMARK_IDS}
    p[33]=np.array([-1.,0,0]);p[263]=np.array([1.,0,0]);p[10]=np.array([0.,-1,0]);p[175]=np.array([0.,1,-.1])
    e=np.c_[np.eye(3),np.zeros(3)]
    v,_,_=measurements_from_points(p,e)
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('xyz',[20,-30,10],degrees=True).as_matrix()
    q={i:5*R@x+np.array([10.,20.,-5.]) for i,x in p.items()}
    w,_,_=measurements_from_points(q,np.c_[R.T,np.zeros(3)])
    assert all(v[k]==pytest.approx(w[k],abs=1e-10) for k in v)

def test_design_eye_normalization_preserves_geometry():
    from facework.pilot.design_preprocess import eye_normalization
    p=np.random.default_rng(9).uniform(100,300,(478,2));p[33]=[100,100];p[133]=[120,110];p[263]=[220,170];p[362]=[200,160]
    q,_,meta=eye_normalization(p)
    r=(q[33]+q[133])/2;l=(q[263]+q[362])/2
    assert r[1]==pytest.approx(l[1])
    assert np.linalg.norm(r-l)==pytest.approx(128)
    # Similarity transform: same scale for every segment, no aspect distortion.
    assert np.linalg.norm(q[0]-q[1])/np.linalg.norm(p[0]-p[1])==pytest.approx(meta['scale'])
