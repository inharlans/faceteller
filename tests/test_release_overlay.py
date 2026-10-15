"""The QA overlay must follow the photo, not crop or mirrored-preview coordinates."""
import json
import numpy as np
import pytest

from facework.release_overlay import face_overlay
from facework.morphometrics import BROW


def points():
    p=np.full((478,3),200.);p[:,2]=0
    for index,xy in {234:(80,200),454:(320,200),10:(200,70),152:(200,350),
                     33:(120,155),133:(170,160),263:(280,155),362:(230,160),
                     159:(145,147),145:(145,168),386:(255,147),374:(255,168),
                     168:(200,175),2:(200,240),98:(177,225),327:(223,225),
                     61:(160,275),291:(240,275),0:(200,263),13:(200,274),
                     14:(200,277),17:(200,290),172:(130,310),397:(270,310)}.items():
        p[index,:2]=xy
    for side,b in BROW.items():
        xx=np.linspace(110,178,5) if side=='right' else np.linspace(290,222,5)
        yy=np.array([125,115,112,117,124])
        p[b['upper'],:2]=np.c_[xx,yy-3]
        p[b['lower'],:2]=np.c_[xx,yy+3]
    return p


def by_key(overlay):
    return {f['item_id']:f for f in overlay['features']}


def test_original_pixels_and_brow_curve_are_actual_landmarks():
    p=points();d=face_overlay(p,400,400);f=by_key(d)
    assert len(f)==19
    assert d['width']==d['height']==400
    assert f['eye_tilt']['paths'][0]==p[[133,33],:2].tolist()
    b=BROW['right']
    np.testing.assert_array_equal(f['brow_length']['paths'][0],(p[b['upper'],:2]+p[b['lower'],:2])/2)
    assert f['brow_tail']['value']==pytest.approx(np.degrees(np.arctan2(-1,68)))
    json.dumps(d,allow_nan=False)


def test_photo_scale_rotation_translation_preserve_values_and_transform_paths():
    p=points();a=.25;rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    transformed=p.copy();transformed[:,:2]=p[:,:2]@rot.T*1.8+[37,29]
    before=by_key(face_overlay(p,400,400));after=by_key(face_overlay(transformed,800,800))
    for key in before:
        assert after[key]['value']==pytest.approx(before[key]['value'],abs=1e-10)
        for path,new_path in zip(before[key]['paths'],after[key]['paths']):
            np.testing.assert_allclose(np.asarray(path)@rot.T*1.8+[37,29],new_path,atol=.002)


def test_optional_hair_path_never_claims_old_guide_measurement():
    curve=[[110,122],[130,112],[150,113],[175,120]]
    f=by_key(face_overlay(points(),400,400,visible_brows={'right':{'path':curve}}))
    assert f['brow_length']['source']=='segmentation'
    assert f['brow_length']['paths'][0]==curve
    assert 'value' not in f['brow_length']
    assert f['brow_tail']['paths'][0]==[curve[0],curve[-1]]


def test_failed_hair_measurement_does_not_fall_back_to_mesh_length():
    f=by_key(face_overlay(points(),400,400,visible_brows={}))
    assert 'brow_length' not in f and 'brow_tail' not in f
    assert len(f)==17


def test_hair_paths_show_their_own_bilateral_metrics():
    sides={
        'right':{'path':[[170,120],[145,110],[110,125]],'length_eye_ratio':1.4,'tail_angle_deg':-3.},
        'left':{'path':[[230,120],[255,110],[290,125]],'length_eye_ratio':1.6,'tail_angle_deg':-5.}}
    f=by_key(face_overlay(points(),400,400,visible_brows=sides))
    assert f['brow_length']['value']==1.5
    assert f['brow_tail']['value']==-4.
    assert f['brow_tail']['unit']=='degrees'


@pytest.mark.parametrize('bad',[[],np.zeros((478,3)),np.full((478,3),np.nan),np.zeros((10,2))])
def test_invalid_measurements_omit_overlay_without_breaking_result(bad):
    assert face_overlay(bad,400,400)['features']==[]
