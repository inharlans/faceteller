import numpy as np
import pytest
import torch
from PIL import Image

from facework.release_preprocess import INPUT_SIZES
from facework.release_runtime import RuntimeResources


class RecordingModel:
    def __call__(self,x):
        self.input=x.clone()
        return None,[torch.zeros((len(x),3),device=x.device)]


def resource(context=True):
    r=RuntimeResources.__new__(RuntimeResources)
    r.device='cpu';r.registry={'parts':{'eye':{'features':['eye_size']}}}
    if context:r.registry['parts']['eye']['context_version']='part-face-context-v1'
    r.models={'eye':[RecordingModel()]}
    return r


def test_context_is_supplied_to_same_cnn_with_correct_side_mirror():
    h,w=INPUT_SIZES['eye'];fh,fw=INPUT_SIZES['face']
    local=Image.new('RGB',(w,h),(50,50,50))
    rgb=np.zeros((fh,fw,3),np.uint8);rgb[:,:fw//2]=80;rgb[:,fw//2:]=220
    whole=Image.fromarray(rgb);r=resource()
    p=r.predict_part('eye',[local,local],face_context=whole,crop_names=['eye_right','eye_left'])
    x=r.models['eye'][0].input
    assert tuple(x.shape)==(2,3,h,2*w)
    torch.testing.assert_close(x[0,:,:,:w],x[1,:,:,:w])
    torch.testing.assert_close(x[0,:,:,w:].flip(-1),x[1,:,:,w:])
    # The face tile preserves its actual left/right difference.
    assert x[0,0,h//2,w+w//3] < x[0,0,h//2,w+2*w//3]
    np.testing.assert_allclose(p['eye_size'],[1/3]*3)


def test_context_checkpoint_cannot_silently_run_on_only_a_local_crop():
    h,w=INPUT_SIZES['eye'];local=Image.new('RGB',(w,h))
    with pytest.raises(ValueError,match='Whole-face context'):
        resource().predict_part('eye',[local])


def test_original_checkpoint_keeps_original_input():
    h,w=INPUT_SIZES['eye'];r=resource(context=False)
    r.predict_part('eye',[Image.new('RGB',(w,h))])
    assert tuple(r.models['eye'][0].input.shape)==(1,3,h,w)
