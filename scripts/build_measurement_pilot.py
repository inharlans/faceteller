"""Descriptive FRLL reference from train-neutral images only, never Korean thresholds."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from facework.measurement_rules import pilot_reference

def main():
 spec=json.loads((ROOT/'data/measurement_definitions.json').read_text(encoding='utf8'))
 images=json.loads((ROOT/'data/derived/images.json').read_text(encoding='utf8'))
 rows=[]
 for image in images:
  if image['split']=='train' and image['condition']=='neutral_front' and image['landmark_status']=='ok':
   metrics=json.loads((ROOT/f"data/derived/landmarks/{image['image_id']}.json").read_text(encoding='utf8'))['metrics']
   rows.append(image|{'metrics':metrics})
 keys=[k for f in spec['features'] for k in f['metric_keys']]
 ref=pilot_reference(rows,keys,'FRLL 학습 72명 무표정; 한국인 기준 아님; 품질 수동 검수 전')
 ref['input_sha256']=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
 ref['limitations']=['독립 반복 촬영 자료 없음','표정 비교 사진을 측정 오차 추정에 사용하지 않음','촬영거리·렌즈 일반화 미검증','백분위 후보를 서비스 레이블로 사용하지 않음']
 out=ROOT/'outputs/knowledge-v2';out.mkdir(exist_ok=True,parents=True)
 (out/'FRLL_학습집단_비율분포.json').write_text(json.dumps(ref,ensure_ascii=False,indent=2),encoding='utf8')
 print('Reference subjects:',ref['subjects'],'metrics:',len(keys),'deployment_ready:',ref['deployment_ready'])
if __name__=='__main__':main()
