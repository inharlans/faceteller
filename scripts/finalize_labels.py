import csv,json,sys,os
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from facework.features import ROOT
from scripts.prepare_data import write_csv

def main():
    out=ROOT/'data/derived'
    with (ROOT/'data/annotations/visual_v1.tsv').open(encoding='utf-8') as source:
        annotations={r['subject_id']:r for r in csv.DictReader(source,delimiter='\t')}
    schema=json.loads((ROOT/'data/label_schema.json').read_text(encoding='utf-8'))
    parts={p:t for t,v in schema['tasks'].items() for p in v['parts']}
    crops=json.loads((out/'crops.json').read_text(encoding='utf-8'))
    overrides_path=ROOT/'data/annotations/overrides.json'
    overrides=json.loads(overrides_path.read_text(encoding='utf-8')) if overrides_path.exists() else {}
    rows=[]
    for row in crops:
        task=parts[row['part']]
        if row['condition']=='neutral_front':
            ann=annotations[row['subject_id'].removeprefix('frll_')]
            label=ann[row['part']]
            status='unobservable' if label=='unknown' else 'ai_visual_reviewed'
            source='assistant_visual_single_rater_v1'
            notes=ann['quality_notes']
        else:
            label='';status='stress_test_only';source='not_visually_labeled';notes='웃는 표정: 무표정 형태 레이블을 전파하지 않음'
        row.update(task=task,manual_label=label,status=status,notes=notes,label_source=source,schema_version=schema['version'],independent_review=False)
        override=overrides.get(row['annotation_id'])
        if override:
            row.update(manual_label=override['label'],status='human_reviewed' if override['label']!='unknown' else 'unobservable',notes=override['notes'],label_source='local_user_review',independent_review=True)
        assert row['manual_label'] in ['', 'unknown', *schema['tasks'][task]['classes']],row
        rows.append(row)
    (out/'annotations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(out/'annotations.csv',rows)
    # Explicit manifests, with IDs preserved. Train against manifests; no identity mixing.
    training=[r for r in rows if r['condition']=='neutral_front' and r['manual_label'] not in ['', 'unknown']]
    write_csv(out/'training_manifest.csv',training)
    summary={'images':len({r['image_id'] for r in rows}),'subjects':len({r['subject_id'] for r in rows}),'neutral_images':len({r['image_id'] for r in rows if r['condition']=='neutral_front'}),'crop_files':len(rows),'visually_assessed_neutral_parts':sum(r['condition']=='neutral_front' for r in rows),'pilot_labeled_parts':len(training),'unobservable_neutral_parts':sum(r['status']=='unobservable' and r['condition']=='neutral_front' for r in rows),'human_reviewed':sum(r['independent_review'] for r in rows),'stress_test_parts':sum(r['condition']=='smiling_front' for r in rows),'split_subjects':{s:len({r['subject_id'] for r in rows if r['split']==s}) for s in ['train','val','test']},'class_counts':{t:dict(Counter(r['manual_label'] for r in training if r['task']==t)) for t in parts.values()},'annotation_caveat':'AI 단일 검수자의 주관적 형태 레이블; 전문가 정답 및 관상 유형 정답 아님'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
