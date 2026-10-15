"""Versioned morphology targets; automated labels retain their provenance."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from facework.release_preprocess import VERSION, PART_CROPS, INPUT_SIZES


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):return {r['subject_id']:r for r in csv.DictReader(path.open(encoding='utf-8-sig'),delimiter='\t' if path.suffix=='.tsv' else ',')}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'data/release';out.mkdir(exist_ok=True,parents=True)
    schema_path=ROOT/'data/label_schema_kface_v3.json'
    refs_path=ROOT/'models/kface_v4/refs/fold0/reference_labels.csv'
    protocol_path=ROOT/'models/selection-20261014/protocol.json'
    source_features=load(schema_path)['features']
    features={k:dict(v) for k,v in source_features.items() if v['source']!='multiview'}
    # Separate classifier inputs follow the twelve regions listed in design 5.3.
    # Bilateral asymmetry needs both eyes together; averaging single-eye logits
    # cannot compare their shapes. Keep this change outside the frozen benchmark.
    for key,part in {'eye_asymmetry':'face','brow_gap':'glabella','philtrum_length':'philtrum',
                     'cheekbone':'cheekbone','cheek_fullness':'cheekbone'}.items():
        features[key]['part']=part
    features['eye_asymmetry']['region']='eye'
    features['under_eye']={'part':'under_eye','source':'visual','classes':['flat','visible'],
        'ko':'눈 바로 아래의 도톰한 윤곽','classes_ko':['뚜렷하지 않음','뚜렷하게 보임'],
        'annotation_provenance':'existing AI visual v3; 8 positive participants in400; no independent human labels'}
    features['philtrum_definition']={'part':'philtrum','source':'visual','classes':['faint','clear'],
        'ko':'인중 골의 선명도','classes_ko':['희미함','뚜렷함'],
        'annotation_provenance':'existing AI consensus v4; unknown labels excluded; no independent human labels'}
    extension=out/'extended_schema.json'
    if extension.exists():
        for k,f in load(extension)['features'].items():
            if k in features:raise ValueError(f'duplicate extension feature {k}')
            features[k]=f
    refs=rows(refs_path);p=load(protocol_path)
    visual3=rows(ROOT/'data/annotations/kface_visual_v3.tsv')
    visual4=rows(ROOT/'data/annotations/kface_visual_v4.tsv')
    votes=load(ROOT/'data/annotations/kface_visual_v4_soft.json')
    extended_path=out/'extended_labels.json'
    extended=load(extended_path).get('subjects',{}) if extended_path.exists() else {}
    overlay_path=out/'local_marks/merged_mark_labels_vlm_agreement.json'
    overlay=None
    if overlay_path.exists():
        overlay=load(overlay_path)
        if overlay['source_hashes']['original_extended_labels']!=sha(extended_path) or overlay['source_hashes']['protocol']!=sha(protocol_path):
            raise RuntimeError('Local mark annotation provenance changed')
        if overlay['preprocess_version']!=VERSION or overlay['groups']!=p['groups'] or overlay['human_review_count']!=0:
            raise RuntimeError('Local mark annotation contract differs')
        features['pigmented_marks']['annotation_provenance']='One local Qwen3-VL annotator over three lighting repeats plus visible CV agreement; weak automated target; not independent raters or medical truth'
        features['pigmented_marks']['annotation_policy']=overlay['policy']
    landmark_path=out/'landmark_labels.json'
    if not landmark_path.exists():raise RuntimeError('Prepare current train270 landmark targets before registry build')
    landmark=load(landmark_path)
    if landmark['source_hashes']['protocol']!=sha(protocol_path):raise RuntimeError('Landmark split changed')
    for k,definition in landmark['features'].items():
        features[k]['thresholds']=definition
        features[k]['metric']=definition['metric']
        features[k]['annotation_provenance']='Repeated frontal landmark measurements; cutpoints and guard fit train270 only; unknown near boundaries; no human review'
    subjects={}
    for split,ids in p['groups'].items():
        for sid in ids:
            labels={};soft={}
            for k,f in features.items():
                value=refs[sid].get(k)
                if k=='under_eye':value=visual3.get(sid,{}).get(k)
                if k=='philtrum_definition':value=visual4.get(sid,{}).get(k)
                if sid in extended and k in extended[sid]:value=extended[sid][k]
                if overlay is not None and k=='pigmented_marks':value=overlay['subjects'][sid][k]
                if k in landmark['features']:value=landmark['subjects'][sid][k]
                labels[k]=value if value in f['classes'] else None
                dist=votes.get(sid,{}).get(k)
                if dist and labels[k] is not None:
                    vector=[float(dist.get(c,0)) for c in f['classes']]
                    total=sum(vector)
                    if total>0:soft[k]=[v/total for v in vector]
            subjects[sid]={'split':split,'labels':labels,'soft_targets':soft}
    support={}
    for split,ids in p['groups'].items():
        support[split]={}
        for k,f in features.items():
            counts=Counter(subjects[s]['labels'][k] for s in ids)
            support[split][k]={'counts':[counts[c] for c in f['classes']],'unknown':counts[None]}
    for k,f in features.items():
        f['kind']='classification'
        if 'annotation_provenance' not in f:
            if f.get('provenance',{}).get('kind')=='weak_automated':
                f['annotation_provenance']='Automated segmentation/visible-image measurement weak labels; '+f.get('metric','')+'; no human review'
            else:
                f['annotation_provenance']=('AI three-vote visual consensus; no independent human labels' if f['source']=='visual'
                    else 'landmark/image measurement-derived participant labels; train-fitted cutpoints; unknown/borderline excluded')
        assert f['part'] in PART_CROPS and len(f['classes'])>=2
        assert len(f['classes'])==len(f['classes_ko'])
    artifact={'version':'release-morphology-v1','preprocess_version':VERSION,
              'features':features,'depth_features':{k:f for k,f in source_features.items() if f['source']=='multiview'},
              'parts':{part:{'crops':list(names),'input_size':list(INPUT_SIZES[part]),
                              'features':[k for k,f in features.items() if f['part']==part]}
                       for part,names in PART_CROPS.items()},
              'subjects':subjects,'class_support':support,'human_review_count':0,
              'split_sizes':{s:len(ids) for s,ids in p['groups'].items()},
              'source_hashes':{str(path.relative_to(ROOT)):sha(path) for path in
                               [schema_path,refs_path,protocol_path,landmark_path,
                                ROOT/'data/annotations/kface_visual_v3.tsv',
                                ROOT/'data/annotations/kface_visual_v4.tsv',
                                ROOT/'data/annotations/kface_visual_v4_soft.json']+
                               ([extension,extended_path] if extension.exists() and extended_path.exists() else [])+
                               ([overlay_path] if overlay is not None else [])}}
    path=out/'training_registry.json'
    encoded=json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)
    # Never silently change targets underneath a trained release model.
    if (ROOT/'models/release/training_frozen.json').exists() and path.exists() and path.read_text(encoding='utf-8')!=encoded:
        raise RuntimeError('release targets frozen: choose a new release directory')
    # The training queue rechecks the same registry while the corpus finalizer
    # reads it. Do not truncate an identical file or expose a partial update.
    if not path.exists() or path.read_text(encoding='utf-8')!=encoded:
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(encoded,encoding='utf-8')
        temporary.replace(path)
    print(json.dumps({'participants':len(subjects),'features':len(features),'parts':{p:len(d['features']) for p,d in artifact['parts'].items()},'human_review_count':0},ensure_ascii=False))


if __name__=='__main__':main()
