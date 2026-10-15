"""Part CNN training against observed morphology labels; pilot status is explicit."""
import argparse,copy,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from .features import ROOT

def preprocess():
    # Letterboxing preserves aspect ratio, critical for shape classification.
    return transforms.Compose([transforms.Lambda(lambda im:ImageOps.pad(im,(224,224),method=Image.Resampling.BILINEAR,color=(127,127,127))),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class PartsDataset(Dataset):
    def __init__(self,rows,classes):self.rows=rows;self.classes=classes;self.transform=preprocess()
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        im=Image.open(ROOT/r['crop_path']).convert('RGB')
        # Mirror subject-left local parts so both sides share nasal-to-temporal orientation.
        if r['part'].endswith('_left'):im=ImageOps.mirror(im)
        return self.transform(im),self.classes.index(r['manual_label'])

def model_for(classes,pretrained=True):
    m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,len(classes))
    return m

def evaluate(model,loader,device):
    model.eval();ys=[];ps=[];prob=[]
    with torch.inference_mode():
        for x,y in loader:
            p=model(x.to(device)).softmax(1).cpu()
            ys.extend(y.tolist());ps.extend(p.argmax(1).tolist());prob.extend(p.tolist())
    return ys,ps,prob

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['eye','brow','nose','mouth','face','all'],default='all');ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--pilot',action='store_true');ap.add_argument('--device',default='auto');args=ap.parse_args()
    random.seed(20260921);np.random.seed(20260921);torch.manual_seed(20260921);torch.set_num_threads(4)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    schema=json.loads((ROOT/'data/label_schema.json').read_text(encoding='utf-8'))
    rows=list(csv.DictReader((ROOT/'data/derived/training_manifest.csv').open(encoding='utf-8-sig')))
    if not args.pilot and any(r['label_source']!='local_user_review' for r in rows):
        raise SystemExit('Independent human labels are incomplete. Use --pilot explicitly for the supplied AI-reviewed experimental dataset.')
    for a in ['train','val','test']:
        for b in ['train','val','test']:
            if a!=b:assert not ({r['subject_id'] for r in rows if r['split']==a}&{r['subject_id'] for r in rows if r['split']==b})
    output=ROOT/'models/pilot';output.mkdir(exist_ok=True,parents=True)
    report={};start=time.time()
    for task in schema['tasks'] if args.task=='all' else [args.task]:
        classes=list(schema['tasks'][task]['classes']);sub=[r for r in rows if r['task']==task]
        parts={s:[r for r in sub if r['split']==s] for s in ['train','val','test']}
        if any(not v for v in parts.values()):raise RuntimeError(f'{task}: empty split')
        loaders={s:DataLoader(PartsDataset(v,classes),batch_size=args.batch_size,shuffle=s=='train',num_workers=0) for s,v in parts.items()}
        model=model_for(classes).to(device)
        # Freeze convolutional features initially; unfreeze last blocks for mild fine tuning.
        for p in model.features.parameters():p.requires_grad=False
        for block in list(model.features.children())[-3:]:
            for p in block.parameters():p.requires_grad=True
        counts=np.bincount([classes.index(r['manual_label']) for r in parts['train']],minlength=len(classes))
        if (counts==0).any():raise RuntimeError(f'{task}: missing training class {counts}')
        loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(parts['train'])/(len(classes)*counts),dtype=torch.float32,device=device))
        optimizer=torch.optim.AdamW([{'params':[p for p in model.features.parameters() if p.requires_grad],'lr':.0001},{'params':model.classifier.parameters(),'lr':.001}],weight_decay=.01)
        best=-1.;state=None;history=[]
        for epoch in range(args.epochs):
            model.train()
            # Keep pretrained batchnorm statistics fixed for this tiny dataset.
            for module in model.modules():
                if isinstance(module,nn.BatchNorm2d):module.eval()
            total=0.
            for x,y in loaders['train']:
                optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(x.to(device)),y.to(device));loss.backward();optimizer.step();total+=loss.item()*len(y)
            y,p,_=evaluate(model,loaders['val'],device);score=float(f1_score(y,p,labels=list(range(len(classes))),average='macro',zero_division=0))
            history.append({'epoch':epoch+1,'train_loss':total/len(parts['train']),'val_macro_f1':score})
            if score>best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
        model.load_state_dict(state)
        y,p,prob=evaluate(model,loaders['test'],device)
        majority=int(counts.argmax());baseline=float(accuracy_score(y,[majority]*len(y)))
        result={'status':'experimental_ai_single_rater_labels','architecture':'mobilenet_v3_small','classes':classes,'device':device,'epochs':args.epochs,'selected_epoch':best_epoch,'val_macro_f1':best,'test_accuracy':float(accuracy_score(y,p)),'test_macro_f1':float(f1_score(y,p,labels=list(range(len(classes))),average='macro',zero_division=0)),'majority_baseline_test_accuracy':baseline,'confusion_matrix':confusion_matrix(y,p,labels=list(range(len(classes)))).tolist(),'class_report':classification_report(y,p,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),'split_crops':{s:len(v) for s,v in parts.items()},'split_subjects':{s:len({r['subject_id'] for r in v}) for s,v in parts.items()},'train_class_counts':dict(zip(classes,counts.tolist())),'history':history,'caveat':'Test labels are the same AI rater convention, not an independent gold standard; test subjects are disjoint. Paired-eye rows are correlated.'}
        torch.save({'state_dict':state,'classes':classes,'task':task,'schema_version':schema['version'],'metadata':result},output/f'{task}.pt')
        (output/f'{task}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        predictions=[{'annotation_id':r['annotation_id'],'subject_id':r['subject_id'],'expected':classes[a],'predicted':classes[b],'max_softmax':max(c)} for r,a,b,c in zip(parts['test'],y,p,prob)]
        (output/f'{task}_test_predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf-8')
        report[task]=result
        print(json.dumps({k:result[k] for k in ['device','test_accuracy','test_macro_f1','majority_baseline_test_accuracy']}|{'task':task}),flush=True)
    report['elapsed_seconds']=round(time.time()-start,2)
    report['training_manifest_sha256']=hashlib.sha256((ROOT/'data/derived/training_manifest.csv').read_bytes()).hexdigest()
    (output/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
