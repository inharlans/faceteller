"""Prepare source-specific annotation plans, not executable traditional-type classifiers."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Authored acquisition requirements from the sections; no phenotype is inferred here.
SIDE=set('龍鼻 虎鼻 胡羊鼻 獅鼻 懸膽鼻 伏犀鼻 牛鼻 截筒鼻 蒜鼻 盛囊鼻 猴鼻 鷹嘴鼻 狗鼻 鯽魚鼻 三彎三曲鼻 劍鋒鼻 偏凹鼻 孤峰鼻 露脊鼻 露竈鼻 猩鼻 燕眼 羊口 吹火口'.split())
MULTI=set('黃薄眉 獅子眉 新月眉 龍眼 獅眼 鸞眼 伏犀眼 鷺鷥眼 燕眼 鷓鴣眼 貓眼 孤峰鼻 猩鼻 猿鼻 猴口'.split())
BODY=set('伏犀眼 鷺鷥眼 鷓鴣眼 猩鼻'.split())
MOTION=set('猴眼 鴛鴦眼 鳴鳳眼 睡鳳眼 瑞鳳眼 桃花眼 醉眼 鴿眼 鸞眼 狼目 鷺鷥眼 猿眼 鹿眼 熊眼 燕眼 鷓鴣眼 鹿鼻 猿鼻 方口 櫻桃口 龍口 羊口'.split())
COLOR=set('大短促眉 間斷眉 虎眼 孔雀眼 鴛鴦眼 鴈眼 鶴形眼 龍眼 鶴眼 鵝眼 醉眼 羊眼 豬眼 蛇眼 鴿眼 狼目 鷺鷥眼 猿眼 鹿眼 燕眼 鷓鴣眼 貓眼 猴鼻 四字口 方口 仰月口 彎弓口 櫻桃口'.split())
TEXT_REVIEW=set('柳葉眉 間斷眉 象眼 鵲眼 孔雀眼 鳴鳳眼 陰陽眼 猿眼 熊眼 蝦眼 燕眼 貓眼 虎鼻 狗鼻 三彎三曲鼻 偏凹鼻 櫻桃口 猴口 覆船口'.split())
INSUFFICIENT=set('疏散眉 羅漢眉 交加眉 蝦眼'.split())

def main():
 catalog=json.loads((ROOT/'data/interpretation_catalog_v1.json').read_text(encoding='utf8'))
 records=[]
 for r in catalog['records']:
  n=r['name_zh'];needs=['정면 무표정 전체 얼굴','해당 부위 고해상도 crop']
  if r['part']=='brow':needs.append('눈썹 털 영역·방향의 독립 주석')
  if n in SIDE:needs.append('측면 또는 보정된 3D')
  if n in MULTI:needs.append('여러 얼굴 부위 동시 검수')
  if n in BODY:needs.append('전신·귀 등 별도 관찰; 현재 서비스 범위 밖')
  if n in MOTION:needs.append('동작·표정·시선 조건 별도 관찰; 단일 사진으로 전체 조건 확인 불가')
  if n in COLOR:needs.append('색·명암 통제; 현재 형태 레이블에서는 해당 조건 미판정')
  blockers=['독립 검수 예시와 유형 간 구별 기준 미확정']
  if n in TEXT_REVIEW:blockers.append('원문·판본·전고 또는 형태 용어 대조 필요')
  if n in INSUFFICIENT:blockers.append('해당 절의 구체적 형태 조건 부족')
  records.append({'record_id':r['record_id'],'name_zh':n,'name_ko':r['name_ko'],'part':r['part'],
   'candidate_morphology_conditions':r['morphology_ko'],'condition_status':'human_readable_draft_not_executable_rule',
   'required_inputs':needs,'limitations':r['limitation'],'blockers':blockers,
   'single_crop_sufficient':False,'type_gold_label_ready':False,'photo_to_interpretation_enabled':False,
   'annotation_action':'개별 관찰 항목부터 레이블하고 전체 유형은 보류. 문헌 이름만으로 형상 보충 금지.',
   'interpretation_record_id':r['record_id'],'source_url':r['source_url']})
 path=ROOT/'data/type_requirements.json';path.write_text(json.dumps({'version':'type-requirements-v1-draft','records':records},ensure_ascii=False,indent=2),encoding='utf8')
 with (ROOT/'outputs/knowledge-v2/유형별_레이블링_요건.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=list(records[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in records:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
 print('Type requirements:',len(records))
if __name__=='__main__':main()
