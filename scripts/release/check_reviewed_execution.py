"""Replay cached uint8 inputs with the live service's per-person CUDA execution.
Run after HTTP verification finishes. No fitting, photos or coordinates saved.
The original batched-training comparison is retained, including its strict failures.
"""
import json,gc,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main():
    import torch
    from facework.kface_training import PartNet
    from facework.release_runtime import temperature_probability
    torch.set_num_threads(2)
    candidate=ROOT/'experiments/relabel-20261015/candidate-v3'
    load=lambda p:json.loads(p.read_text(encoding='utf8'))
    report=load(ROOT/'outputs/release/service-verification-reviewed-20261015.json')
    if not report['complete'] or not report['all400']:raise RuntimeError('Finish actual400 before separate GPU replay')
    rows={r['subject_id']:r for r in report['participants'] if r['split']=='selection'}
    manifest=load(candidate/'deployment_candidate.json');registry=load(ROOT/'data/release/training_registry.json')
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    checked=[];differences=[];batch_deltas=[]
    for part,spec in registry['parts'].items():
        cache=torch.load(candidate/f'cache/{part}_selection.pt',map_location='cpu',weights_only=True)
        ck=torch.load(candidate/f'models/resnet18/{part}.pt',map_location='cpu',weights_only=True)
        model=PartNet('resnet18',0,[len(registry['features'][k]['classes']) for k in spec['features']],pretrained=False)
        model.load_state_dict(ck['state_dict']);model.eval().requires_grad_(False);model=model.to('cuda');del ck
        stored=load(candidate/f'models/resnet18/{part}.json')['selection']
        with torch.inference_mode():
            for index,sid in enumerate(cache['ids']):
                confirmed={i['item_id']:i for i in rows[sid]['items'] if i['status']=='CONFIRMED' and i['item_id'] in spec['features']}
                if not confirmed:continue
                mask=cache['owners']==index
                if not bool(mask.any()):continue
                # The API constructs NHWC numpy -> NCHW view, then preserves that layout.
                array=cache['images'][mask].permute(0,2,3,1).contiguous().numpy()
                x=torch.from_numpy(array).permute(0,3,1,2).to('cuda').float()/255
                x=(x-mean)/std;output=model(x)[1]
                for j,key in enumerate(spec['features']):
                    if key not in confirmed:continue
                    raw=output[j].float().softmax(1).mean(0).cpu().numpy()
                    cal=manifest['features'][key];p=temperature_probability(raw,cal['temperature']);top=int(np.argmax(p));item=confirmed[key]
                    delta=abs(float(p[top])-item['confidence']);label=f'{key}:{cal["classes"][top]}'
                    entry={'subject_id':sid,'item_id':key,'label_matches':label==item['label_id'],'confidence_absolute_difference':delta}
                    checked.append(entry)
                    if not entry['label_matches'] or delta>1e-4:differences.append(entry)
                    q=temperature_probability(stored[key]['prob'][index],cal['temperature'])
                    batch_deltas.append(float(np.max(np.abs(q-p))))
                del x,output
        del model,cache;gc.collect();torch.cuda.empty_cache()
    result={'status':'passed' if checked and not differences else 'failed','scope':'same per-person CUDA execution and fixed cached inputs vs real HTTP; not classification accuracy',
        'selection_people':len(rows),'checked_confirmed_predictions':len(checked),'tolerance':1e-4,
        'maximum_confidence_difference':max(x['confidence_absolute_difference'] for x in checked),'differences':differences,
        'maximum_batch16_training_vs_per_person_cuda_probability_difference':max(batch_deltas),
        'photos_or_coordinates_saved':False,'new_model_fitting':False,
        'note':'Batch16 channels-last training evaluation and per-person production convolution can differ numerically. Strict original comparison preserved separately. Tiny ear input changes documented in input-parity.json.'}
    (candidate/'execution-parity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='passed':raise RuntimeError('Per-person execution mismatch')

if __name__=='__main__':main()
