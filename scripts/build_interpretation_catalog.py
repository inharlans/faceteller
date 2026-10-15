"""Compile authored historical summaries; never derive fortunes from photos."""
import csv, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TOPICS = ['재물','지위·명예','가족·관계','수명·건강','복록','인격·행동','능력에 대한 주장','생활·성취','형벌·재난']
VALUES = {'길','흉','혼합','조건부','해석보류'}

def build():
    sources=json.loads((ROOT/'data/derived/literature.json').read_text(encoding='utf-8'))
    path=ROOT/'data/interpretation_annotations.tsv'
    rows=list(csv.DictReader(path.open(encoding='utf-8'),delimiter='\t'))
    annotations={r['name_zh']:r for r in rows}
    assert len(annotations)==len(rows)==len(sources)==103
    assert set(annotations)=={s['name_zh'] for s in sources}
    records=[]
    for source in sources:
        a=annotations[source['name_zh']]
        evidence=a['evidence'].split('|')
        assert evidence and all(e and e in source['original_section'] for e in evidence),source['name_zh']
        topics={}
        for item in a['topics'].split(';'):
            key,value=item.split('=')
            assert key in TOPICS and value in VALUES
            topics[key]=value
        r={k:source[k] for k in ('record_id','name_zh','name_ko','part','source_url','source_version','original_section','reuse_notice','morphology_ko','limitation')}
        r.update(interpretation_status='draft_not_expert_verified',interpretation_ko='이 문헌은 '+a['summary_ko'],
            interpretation_evidence=evidence,historical_topic_assessment=topics,
            review_note=a['review_note'],independent_review=False,
            application_scope='historical_text_reference_only',personal_prediction_allowed=False,
            complete_type_criteria_status='requires_photo_validation',photo_label_link_status='not_established',
            source_text_sha256=hashlib.sha256(source['original_section'].encode()).hexdigest(),
            display_text=f"『신상전편』의 {source['name_ko']} 항목: {a['summary_ko']} 이는 역사 문헌의 주장에 대한 요약입니다.")
        if '부족' in source['morphology_ko'] or '부족' in source['limitation']:
            r['complete_type_criteria_status']='insufficient_morphology'
        records.append(r)
    catalog={'version':'historical-interpretation-v2-draft','scope':'확보한 633·634권의 눈썹24·눈39·코24·입16 항목. 모든 관상 문헌 전체가 아님.',
        'translation_scope':'한국어 내용 요약 및 핵심 구절 분류. 완역·교감본·독립 전문가 검수 아님.',
        'topic_legend':{'길':'문헌의 긍정적 주장','흉':'문헌의 부정적 주장','혼합':'긍정·부정 혼재','조건부':'다른 형상·시기·상황의 조건부 주장','해석보류':'본문은 관련되나 뜻 또는 평가 보류','미분류':'해당 주제가 요약에서 분류되지 않음; 원문에 없음을 보증하지 않음'},
        'authoring':'assistant_source_reading_single_rater','annotation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'records':records}
    for dest in [ROOT/'data/interpretation_catalog_v1.json',ROOT/'outputs/knowledge-v2/interpretations.json']:
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'outputs/knowledge-v2/문헌해석_103항목.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=['record_id','name_ko','name_zh','part','morphology_ko','interpretation_ko','historical_topic_assessment','interpretation_evidence','review_note','source_url','original_section']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in records:w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(dict,list)) else r[k] for k in keys})
    print(json.dumps({'records':len(records),'draft_summaries':len(records),'pending_summaries':0,'expert_reviews':0},ensure_ascii=False))
    return catalog
if __name__=='__main__':build()
