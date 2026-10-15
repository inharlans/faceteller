"""Aggregate recorded experimental results, without new training/inference."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/experiment-report-2026-10-01'
read=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
A=read('outputs/experiment-report-2026-10-01/실험_재집계.json')
S=read('data/label_schema_kface_v3.json')['features'];E=read('models/kface_v4/evaluation.json');O=read('models/kface_v4/evaluation_oof.json')
V=read('models/kface_v4/visual_confidence.json');P=read('models/kface_v4/paired_comparison.json')
R=read('data/derived/kface/reference_summary.json');K=read('data/annotations/kface_rater_agreement.json')
refs=list(csv.DictReader((ROOT/'data/derived/kface/reference_labels.csv').open(encoding='utf-8-sig')))
test=[r for r in refs if r['split']=='test'];parts={'eye':'눈','brow':'눈썹','nose':'코','mouth':'입','face':'얼굴'}
G='service(C7,bright)'
def writecsv(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(v):return '—' if v is None else f'{v*100:.1f}%'
def num(v):return '—' if v is None else f'{v:.3f}'
def fmt(v):return f'{v:,}'
def short(k):return S[k]['ko'].split('(')[0]
# Count the actual non-reference landmark evaluation denominator from local metrics.
lm_counts=Counter();lm_people={k:set() for k,f in S.items() if f['source']=='landmark'}
for r in test:
 sid=r['subject_id']
 with np.load(ROOT/'data/derived/kface/subjects'/f'{sid}.npz') as d:keys=d['keys'];status=d['status']
 metrics=read(f'data/derived/kface/subjects/{sid}.json')
 for i,key in enumerate(keys):
  acc,light,expr,cam,purpose=str(key).split('|')
  if status[i]!=1 or purpose!='core' or cam!='C7' or light not in ('L4','L14','L23'):continue
  for k,f in S.items():
   if f['source']!='landmark' or r.get(k) in ('borderline','',None):continue
   for side in (['right','left'] if f['part'] in ('eye','brow') else [None]):
    rk=f'{k}_{side}_value' if f.get('per_side') and side else k+'_value'
    mk=f"{f['metric']}_{side}" if f.get('per_side') and side else f['metric']
    if r.get(rk) not in ('',None) and metrics[i] is not None and metrics[i].get(mk) is not None:
     lm_counts[k]+=1;lm_people[k].add(sid)
features=[];rng=np.random.default_rng(20261001)
for part,pr in E['parts'].items():
 for k,groups in pr['features'].items():
  f=S[k];g=groups[G];og=O['parts'][part]['features'][k][G];visual=f['source']=='visual'
  lm=og.get('landmark_accuracy') is not None and og['landmark_accuracy']>og['cnn_accuracy']+.02
  method='랜드마크' if lm else 'CNN'
  acc=g.get('cnn_accuracy',g.get('accuracy'));f1=g.get('cnn_macro_f1',g.get('macro_f1'))
  chosen=g['landmark_accuracy'] if lm else acc
  chosenf1=g['landmark_macro_f1'] if lm else f1
  pp=P['per_participant']['landmark' if lm else 'new'][k]
  vals=np.array(list(pp.values()));b=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1)
  lo,hi=np.quantile(b,[.025,.975]);conf=next((v for v in V.get(k,[]) if v['threshold']==.7),{})
  row=dict(feature_id=k,feature=f['ko'],part=parts[part],source=f['source'],method=method,
   test_people=len(pp),cnn_accuracy=acc,cnn_macro_f1=f1,cnn_evaluated_crops=g.get('n_class_eval',g['n']),
   landmark_accuracy=g.get('landmark_accuracy'),landmark_macro_f1=g.get('landmark_macro_f1'),
   landmark_evaluated_crops=lm_counts.get(k,0) if f['source']=='landmark' else None,
   selected_accuracy=chosen,selected_macro_f1=chosenf1,selected_evaluated_crops=lm_counts[k] if lm else g.get('n_class_eval',g['n']),
   participant_mean_accuracy=round(float(vals.mean()),6),participant_bootstrap_ci95_low=round(float(lo),6),participant_bootstrap_ci95_high=round(float(hi),6),
   cnn_pearson_r=g.get('cnn_pearson_r'),landmark_pearson_r=g.get('landmark_pearson_r'),cnn_mae_sd_units=g.get('cnn_mae_sd_units'),
   cnn_confident_accuracy=g.get('cnn_confident_accuracy'),cnn_confident_coverage=g.get('cnn_confident_rate'),
   visual_photo_accuracy=V[k][0]['accuracy'] if visual else None,
   visual_photo_confident_accuracy=conf.get('accuracy'),visual_photo_confident_coverage=conf.get('coverage'),
   majority_baseline=g.get('majority_baseline_accuracy'),icc=R['features'].get(k,{}).get('icc_1_1_train'),fleiss_kappa=K.get(k,{}).get('fleiss_kappa'),
   oof_cnn_accuracy=og.get('cnn_accuracy',og.get('accuracy')),oof_landmark_accuracy=og.get('landmark_accuracy'),
   denominator_note='비참조 정면 조명 3종' if lm else ('정면 조명 11종; 눈/눈썹은 좌우 crop 별도' if part in ('eye','brow') else '정면 조명 11종'))
  features.append(row)
assert len(features)==38 and sum(r['method']=='랜드마크' for r in features)==10
assert all(len(P['per_participant']['new'][r['feature_id']])==r['test_people'] for r in features)
writecsv('전체_38피처_실험성능.csv',features)
conditions=[]
for part,pr in E['parts'].items():
 for k,groups in pr['features'].items():
  for condition,g in groups.items():
   conditions.append({'feature_id':k,'feature':S[k]['ko'],'part':part,'condition':condition,'n_total':g['n'],'n_class_eval':g.get('n_class_eval',g['n']),'cnn_accuracy':g.get('cnn_accuracy',g.get('accuracy')),'cnn_macro_f1':g.get('cnn_macro_f1',g.get('macro_f1')),'cnn_pearson_r':g.get('cnn_pearson_r'),'landmark_accuracy':g.get('landmark_accuracy'),'landmark_macro_f1':g.get('landmark_macro_f1'),'cnn_confident_accuracy':g.get('cnn_confident_accuracy'),'cnn_confident_coverage':g.get('cnn_confident_rate'),'majority_baseline':g.get('majority_baseline_accuracy')})
writecsv('촬영조건별_피처성능.csv',conditions)
meta_rows=[];epoch_rows=[]
for m in A['checkpoints']:
 meta_rows.append({'checkpoint':m['checkpoint'],'arch':m['arch'],'part':m['part'],'fold':int(m['checkpoint'].split('/')[-2][4:]),'train_people':m['train_participants'],'val_people':m['val_people_recounted'],'train_crop_pool':m['train_captures'],'val_crops':m['val_crops_recounted'],'epochs_run':len(m['history']),'selected_epoch':m['selected_epoch'],'selected_val_score':m['val_score'],'input_height':m['input_size'][0],'input_width':m['input_size'][1],'epoch_seconds_total':round(sum(h['seconds'] for h in m['history']),1),'data':m['data'],'soft_labels':m['soft_labels'],'labels_sha256':m['labels_sha256']})
 for h in m['history']:epoch_rows.append({'checkpoint':m['checkpoint'],'arch':m['arch'],'part':m['part'],'epoch':h['epoch'],'train_loss':h['train_loss'],'val_score':h['val_score'],'seconds':h['seconds'],'selected':h['epoch']==m['selected_epoch']})
writecsv('모델_50개_학습설정.csv',meta_rows);writecsv('모델_에포크별_학습이력.csv',epoch_rows)
visual_rows=[]
for k,rows in V.items():
 for v in rows:visual_rows.append({'feature_id':k,'feature':short(k),'threshold':v['threshold'],'accuracy':v['accuracy'],'coverage':v['coverage'],'selected_photos':sum(v['truth_dist']),'all_labeled_photos':sum(rows[0]['truth_dist'])})
writecsv('육안피처_확률기준별_성능.csv',visual_rows)
pages=[]
def page(title,lead,*blocks):pages.append({'title':title,'lead':lead,'blocks':list(blocks)})
def p(t):return {'kind':'p','text':t}
def table(h,r):return {'kind':'table','head':h,'rows':r}
def bullets(*a):return {'kind':'bullets','items':list(a)}
page('실험 결과 요약','2026-10-01 현재 디스크에 보관된 실험 산출물을 다시 집계한 보고서다. 주요 결과는 2026-09-29 재평가 파일이며, 이번 작업에서는 모델을 재학습하거나 얼굴 추론을 다시 실행하지 않았다.',
 table(['항목','확인 결과'],[['서로 다른 사람','400명. 초기 학습 276 / 검증 62 / 시험 62명'],['최종 교차학습','개발 338명, 5-fold. fold별 학습 270~271 / 검증 67~68명'],['공통 시험','62명. 모든 fold 학습·검증 명단과 겹침 없음'],['최종 모델 파일','5부위 × 2백본 × 5-fold = 형태 모델 50개. 부위마다 10개 앙상블'],['특징','38개: 랜드마크 기준 19 + 다중 시점 깊이 4 + AI 시각 합의 15'],['현재 방법 선택','OOF 결과 기준 직접 측정 10개, CNN 28개'],['정면 시험 사진','62명 × 조명 11종 = 682장. 눈·눈썹은 좌우 crop 1,364개'],['중요 한계','피처별 유효 사람 수가 다름. 인간 독립 정답 검수 0건. 스마트폰 외부 시험 미확인']]),
 p('이 보고서의 정확도는 얼굴 형태 레이블과의 일치율이다. 실제 성격·재산·운세의 정확도가 아니며, 문헌 유형 전체를 정답으로 검증한 결과도 아니다.'),
 p('핵심 결과: 입술 두께 CNN 92.7%, 아래턱 폭 직접 측정 90.4%, 얼굴형 CNN 73.2%, 턱 돌출 CNN 58.1%. 흰자 노출 94.4%는 최빈값 기준 93.5%와 차이가 작아 높은 정확도만으로 우수하다고 해석할 수 없다.'))
tot=A['capture_totals']
page('데이터 규모와 사진 수','원본 보유량, 특징점 처리량, 학습 후보량, 한 epoch 투입량은 서로 다른 수다. 아래 인원·처리량은 reference_labels.csv와 400개 NPZ의 keys/status/pose를 재집계했다.',
 table(['구분','학습 집단','검증 집단','시험 집단','합계'],[[label]+[fmt(tot[sp][key]) for sp in ('train','val','test')]+[fmt(sum(tot[sp][key] for sp in tot))] for label,key in [('처리 요청 사진','planned'),('검출 성공 사진','detected'),('기본 core 성공','detected_core'),('기본 학습 필터 후보','train_pool'),('epoch 검증 필터 후보','epoch_val_pool'),('정면 11조명 성공','service_pool')]]),
 p('최초 처리 계획은 1인 266장, 106,400장이다. 확장 후 현재 처리 계획은 1인 386장, 154,400장이고 검출 성공은 147,525장이다. 최초 계획에 해당하는 성공 사진은 101,865장으로, 확장 추가분과 구분한다.'),
 p('원본 매니페스트는 400명 × 10,800장 = 4,320,000 JPEG를 기록한다. 원본 용량은 89,493,929,688 byte(83.35GiB)다. 이 전체를 CNN에 투입한 것이 아니다. 최종 체크포인트의 data는 모두 base이며, 확장 촬영 전체를 학습한 모델이 아니다.'),
 p('표의 “기본 학습 필터 후보” 시험 열 10,216장은 같은 조건으로 추린 시험 집단 사진 수다. 학습에 사용했다는 뜻이 아니다. 초기 실제 학습 후보는 학습 집단 44,990장이고, 최종 개발 집단 338명의 합집합 후보는 55,235장이다.'))
page('분할과 최종 validation 인원','초기 고정 분할과 최종 5-fold 분할을 함께 보아야 학습·검증 수를 정확하게 설명할 수 있다. 338명을 한 모델에 모두 학습한 것은 아니다.',
 table(['fold','학습 사람','검증 사람','학습 후보 사진','검증 사진'],[[str(f['fold']),fmt(f['train_people']),fmt(f['val_people']),fmt(f['train_photo_pool']),fmt(f['val_photo_pool'])] for f in A['folds']]),
 bullets('초기 비교 실험: 학습 276명 / validation 62명 / test 62명.','최종 모델: 기존 train+validation 338명을 5개 fold로 재편. 각 사람은 4개 fold의 학습과 1개 fold의 검증에 참여한다.','시험 62명은 모든 fold의 학습·검증 명단에서 제외된다. 저장된 fold CSV와 frozen split, 체크포인트 인원·입력 수를 대조했다.','검증 사진은 C6/C7/C8/C16 × L1/L2/L3/L8/L12/L19/L25/L4의 검출 성공 core 사진이다. 최종 정면 시험 조건과 동일하지 않다.'),
 p('최종 OOF 평가는 338명 각각에 대해 자신을 학습하지 않은 해당 fold의 백본 2개만 사용한다. 최종 시험 평가는 부위별 백본 2개 × 5-fold, 총 10개의 예측을 평균한다. OOF와 시험은 앙상블 크기도 다르다.'),
 p('저장된 50개 체크포인트에는 참여자 ID 해시 필드가 없어, 당시 명단의 암호학적 증명까지 할 수는 없다. 현재 보관된 fold 명단의 비중복과 메타데이터 인원·crop 수의 일치는 확인했다.'))
page('학습 입력과 epoch당 투입량','한 사진에서 눈과 눈썹은 좌우 2개 crop을 만들고, 코·입·얼굴은 각각 1개를 만든다. 체크포인트의 train_captures는 원본 사진 수가 아니라 해당 부위 crop 후보 수다.',
 table(['부위','입력 H×W','회귀 / 분류 head','fold별 학습 crop 후보','fold별 검증 crop'],[[parts[part],f"{next(m for m in A['checkpoints'] if m['part']==part)['input_size'][0]} × {next(m for m in A['checkpoints'] if m['part']==part)['input_size'][1]}",f"{len(next(m for m in A['checkpoints'] if m['part']==part)['regression_features'])} / {len(next(m for m in A['checkpoints'] if m['part']==part)['classification_features'])}",f"{min(m['train_captures'] for m in A['checkpoints'] if m['part']==part):,}~{max(m['train_captures'] for m in A['checkpoints'] if m['part']==part):,}",f"{min(m['val_crops_recounted'] for m in A['checkpoints'] if m['part']==part):,}~{max(m['val_crops_recounted'] for m in A['checkpoints'] if m['part']==part):,}"] for part in parts]),
 p('현재 코드·실행 기본값 기준: 사람마다 epoch당 코/입/얼굴 36개, 눈/눈썹 72개 crop을 샘플링하고 마지막 불완전 batch를 버린다(batch=64, drop_last=True). 눈/눈썹의 72개는 좌우 후보에서 뽑으므로 정확히 같은 36장 사진의 양쪽을 항상 쓰는 구조는 아니다.'),
 table(['학습 인원','코/입/얼굴 샘플 → 소비','눈/눈썹 샘플 → 소비'],[['270명','9,720 → 9,664 crop/epoch','19,440 → 19,392 crop/epoch'],['271명','9,756 → 9,728 crop/epoch','19,512 → 19,456 crop/epoch']]),
 p('이 epoch 투입량은 보관된 메타데이터의 인원과 현재 sampler·batch 설정으로 재구성한 값이다. 원본 미니배치 로그가 없어 당시 실제 순서와 여러 epoch 동안 한 번 이상 소비한 고유 사진 수는 확정하지 않았다. 6 epoch의 입력 수를 6배 더해 서로 다른 얼굴 수라고 보고하면 안 된다.'))
page('모델 구조와 학습 설정','ImageNet 사전학습 모델의 backbone을 가져와 부위별 회귀·분류 head를 함께 학습했다. 숫자 형태 특징은 연속값 회귀 후 경계값으로 등급화하고, 시각 특징은 분류한다.',
 table(['설정','내용 / 확인 범위'],[['백본','ConvNeXt-Tiny 25개 + EfficientNet-B0 25개 — 체크포인트 확인'],['학습 완료 epoch','ConvNeXt 각 6회, B0 각 5회 — history 확인'],['저장 epoch','각 모델의 validation 혼합 점수가 가장 큰 epoch. 1~6회 중 선택'],['손실','현재 코드: Smooth L1 회귀 + 분류 0.3 가중. 3 AI 투표 soft label; hard label 보완'],['최적화','현재 코드: AdamW lr 3e-4, weight decay 0.05; OneCycle max lr 6e-4'],['규제·증강','현재 코드: dropout 0.3, 작은 회전·이동, 색·밝기·흐림. 크기 변화 증강 제외'],['정규화·경계','기본 학습 276명 기준 통계를 모든 fold가 공유한다. fold별 완전 독립 전처리 아님'],['선택 점수','회귀 Pearson r과 시각 macro-F1의 평균. val_score는 정확도가 아니다.']]),
 p('메타데이터는 data=base, soft_labels=True, cls_loss_weight=0.3을 기록한다. 코드가 학습 뒤 수정된 이력이 있으므로 체크포인트에서 확인한 값과 현재 코드로 복원한 설정을 구분했다.'),
 p('기존 보고서는 RTX 3070 사용을 기록하지만 체크포인트에 GPU 모델은 없다. 이 보고서에서 4080 학습 결과로 바꾸어 표기하지 않는다. 학습 seed는 fold별 20260927~20260931의 정수값이다(날짜 자체가 아니라 난수 seed).'))
time_rows=[]
for arch in ('convnext_tiny','efficientnet_b0'):
 ms=[m for m in A['checkpoints'] if m['arch']==arch]
 sec=sum(h['seconds'] for m in ms for h in m['history'])
 time_rows.append([arch,str(len(ms)),str(len(ms[0]['history'])),f'{sec/3600:.2f}시간',', '.join(f'{e}회:{n}개' for e,n in sorted(Counter(m['selected_epoch'] for m in ms).items()))])
page('학습 시간과 선택된 epoch','시간은 체크포인트 history.seconds를 더한 값이다. 각 epoch의 학습·검증은 포함하지만 모델 초기화, 저장, 전체 파이프라인 준비 시간은 포함하지 않는다.',
 table(['백본','모델 수','완료 epoch','기록 시간 합','최적 epoch 분포'],time_rows),
 p('두 백본의 기록 시간 합은 약 8.47시간이다. 기존 보고서의 약 3시간+6시간과 다른 것은 벽시계 전체 시간과 epoch 기록 시간의 범위가 다르기 때문이다.'),
 table(['백본 / 부위','fold0','fold1','fold2','fold3','fold4'],[[arch+' / '+parts[part]]+[str(next(m['selected_epoch'] for m in A['checkpoints'] if m['arch']==arch and m['part']==part and '/fold'+str(f)+'/' in m['checkpoint'])) for f in range(5)] for arch in ('convnext_tiny','efficientnet_b0') for part in parts]),
 p('전체 275개의 epoch 기록과 50개 모델별 입력 수·검증 점수·선택 epoch는 동봉 CSV에 수록했다. EfficientNet-B0는 25개 중 다수가 마지막 epoch를 선택했다는 사실은 확인되지만, 추가 학습이 반드시 개선된다는 증거는 아니다.'))
page('정답 레이블과 피처별 실효 인원','400명 모두에게 38개 특징의 확정 정답이 있는 것은 아니다. 경계·unknown·uncertain 레이블을 제외한 분류 평가 인원이 특징마다 달라진다.',
 table(['레이블 출처','특징 수','정답 작성 방식','한계'],[['랜드마크','19','여러 기준 사진의 측정 중앙값, 학습 P25/P75 등으로 등급화','같은 측정기 기반 기준. 경계 인원은 분류 평가 제외'],['다중 시점 깊이','4','여러 고정 카메라의 상대 깊이 복원','실측 mm 정답이나 스마트폰 검증이 아님'],['AI 육안 합의','15','3 AI의 판정·다수결 및 투표 비율','동일 계열 AI 오류 상관 가능. 인간 전문가 검수 0건']]),
 p('형태 분류 평가에 포함된 시험 인원은 특징별 29~62명이다. 예를 들어 눈썹 길이는 29명, 눈 크기는 34명, 코 폭은 35명이다. 반복 사진 638개·748개·385개가 각각 수백 명을 뜻하는 것은 아니다.'),
 p('시각 특징 16개를 검토했으나 인중 골의 뚜렷함은 Fleiss κ=0.173으로 제외되어 현재 15개다. 남은 시각 특징의 κ도 약 0.299~0.489이므로 전문가 정답 수준으로 간주하지 않는다.'),
 p('합의 레이블에서 각진 눈썹의 시험 인원은 0명이다. 흰자 위쪽 노출도 0명이며, 눈 깊이의 움푹함/돌출은 시험 2명/1명이다. 이런 소수 클래스는 전체 정확도가 높아도 클래스별 성능을 충분히 검증하지 못했다.'),
 p('피처별 train/validation/test의 모든 클래스 및 경계·미확인 인원은 피처별_클래스_인원.csv에 수록했다. 초기 단일 AI 레이블의 예전 분포표를 최신 합의 레이블 분포로 재사용하지 않았다.'))
page('평가 조건과 수치 읽는 법','이후 표의 기본 평가 조건은 S001·E01·정면 C7 × 밝은/중간 정면 조명 11종이다. 이는 서비스 입력을 근사한 촬영 조건이며, 실제 서비스 품질 검사를 모두 통과시킨 종단 평가라는 뜻은 아니다.',
 table(['지표 / 표기','정의'],[['CNN 정확도','분류 평가 대상 crop의 정답 레이블 일치율. 회귀 특징은 예측 연속값을 등급으로 바꾸어 비교'],['macro-F1','스키마에 정의된 모든 클래스의 F1 평균. 해당 시험에 없는 클래스도 0으로 포함'],['Pearson r','연속 기준값과 예측값 상관. 0.9를 정확도 90%로 읽지 않음'],['유효 사람 / N','피처 정답이 있는 시험 인원 / 실제 채점한 crop 수. 좌우는 별도 crop'],['직접 측정','정답 중앙값 생성에 쓰이지 않은 L4/L14/L23만 채점. CNN의 11조명 결과와 분모 다름'],['확신 정확도 / coverage','보류 조건을 제외한 선택 예측의 정확도 / 원래 평가 대상 중 선택 비율'],['OOF','자신을 학습하지 않은 fold의 예측. 해당 fold로 epoch를 골랐으므로 완전 독립 시험은 아님']]),
 p('방법 선택은 OOF에서 직접 측정 정확도가 CNN보다 2%p 초과 높으면 직접 측정을 쓰는 규칙이다. 시험에서 더 높은 방법을 보고 사후 교체하지 않았기 때문에 얼굴 세로비처럼 시험 CNN이 더 높아도 직접 측정이 선택된 항목이 있다.'),
 p('아래 CNN 표는 평가 파일과 동일하게 눈·눈썹을 좌우 crop별 채점한다. 서비스는 좌우 예측을 평균하므로, 시각 피처의 사진 단위 결과를 별도 절에 제공한다. 두 평가 단위를 섞어 같은 정확도로 표시하지 않는다.'))
for part,ko in parts.items():
 rows=[r for r in features if S[r['feature_id']]['part']==part]
 page('피처별 시험 결과 · '+ko,'CNN은 11조명, 직접 측정은 비참조 3조명 결과다. 최종 열의 정확도·F1은 선택된 방법의 해당 평가값이며 실제 서비스 전체의 종단 정확도는 아니다.',
  table(['특징','CNN 정확도','직접 측정','선택 방법','선택 정확도 / F1','유효 사람 / N'],[[short(r['feature_id']),pct(r['cnn_accuracy']),pct(r['landmark_accuracy']),r['method'],pct(r['selected_accuracy'])+' / '+num(r['selected_macro_f1']),str(r['test_people'])+' / '+fmt(r['selected_evaluated_crops'])] for r in rows]),
  table(['특징','CNN r','CNN 확신 정확도 / 적용률','정답 출처'],[[short(r['feature_id']),num(r['cnn_pearson_r']),pct(r['cnn_confident_accuracy'])+' / '+pct(r['cnn_confident_coverage']) if r['source']!='visual' else '다음 시각 피처 표 참조',{'landmark':'측정 중앙값','multiview':'다중 시점 복원','visual':'3 AI 합의'}[r['source']]] for r in rows]),
  p({'eye':'흰자 노출 CNN 94.4%는 최빈값 기준 93.5%와 유사하다. 눈 깊이는 CNN 94.4%보다 최빈값 기준 95.2%가 높다. 좌우 눈 차이는 CNN r=0.131로 연속값 예측력이 낮다.','brow':'눈썹 모양 정확도는 81.8%지만 macro-F1은 0.547이다. 시험에 각진 클래스가 없어 전체 형태를 잘 구분한다고 결론 내릴 수 없다. 눈썹 길이는 62명 중 29명만 분류 평가 대상이다.','nose':'코 폭은 얼굴 대비 콧방울 폭의 상대 등급이다. 물리적 cm/mm 크기나 “복코” 전체 조건의 정확도가 아니다. 코끝·코뿌리 깊이는 단일 사진 CNN의 상대 깊이 등급이며 다중 시점 입력을 쓴 서비스 성능과 구분한다.','mouth':'입술 두께는 CNN 92.7%, macro-F1 0.924로 비교적 높다. 입 돌출은 68.4%다. 입술 상하비의 기준 측정 ICC=0.673으로 정답 반복성에도 한계가 있다.','face':'턱 돌출은 CNN 58.1%, 확신 구간도 55.8%로 약하다. 얼굴형은 73.2%이며 최빈값 기준 35.1%보다 높다. 하정/중정 비는 직접 측정 87.0%다.'}[part]))
page('시각 피처 · 사진 단위 정확도와 보류','visual_confidence.json은 좌우 눈·눈썹의 확률을 같은 사진끼리 평균한 뒤 채점한다. 여기서는 항상 답할 때와 최대 softmax 확률 0.7 이상일 때를 비교한다.',
 table(['피처','전체 정확도','확신 정확도','coverage','선택 / 전체 사진'],[[short(k),pct(v[0]['accuracy']),pct(next(t for t in v if t['threshold']==.7)['accuracy']),pct(next(t for t in v if t['threshold']==.7)['coverage']),f"{sum(next(t for t in v if t['threshold']==.7)['truth_dist'])} / {sum(v[0]['truth_dist'])}"] for k,v in V.items()]),
 p('예: 눈꺼풀은 전체 74.8%에서 확신 구간 99.3%로 오르지만, 출력 대상은 44.1%로 줄어든다. 99.3%를 모든 입력에 대한 정확도로 제시하면 안 된다. coverage는 정답이 존재하는 평가 사진 안에서의 비율이며 모든 서비스 업로드의 반환율이 아니다.'),
 p('확률 0.7 정책은 시험 결과를 본 뒤 결정된 이력이 있다. 위 결과는 현재 프로토타입의 사후 분석으로 보고하며, 정책 선택에 사용하지 않은 새 사람 집단에서 재확인이 필요하다.'))
page('개발 비교 실험과 v3→v4 변화','모델 비교 실험은 초기 validation 62명에서 실행했다. 비교 점수는 회귀 r과 시각 macro-F1을 섞은 선택 점수이므로 정확도(%)로 환산하지 않는다.',
 table(['실험','눈','눈썹','코','입','얼굴'],[[name]+[num(v[part]['_score']) for part in parts] for name,v in read('models/exp/comparison.json').items()]),
 p('기록에서 hard label→soft label 변경은 5부위의 평균 선택 점수를 모두 개선했다. 데이터 확대(base→more)는 같은 4 epoch 비교에서 개선되지 않았다. 이 비교만으로 확대 촬영이 일반적으로 무용하다고 결론 내릴 수는 없다. B2 실험은 중단되어 완료 비교표에서 제외한다.'),
 table(['동일 시험 사람 대응 비교','평균 향상','95% bootstrap 구간'],[['측정·깊이 23특징','+3.8%p','+2.3~+5.3%p'],['시각 15특징 / 현재 합의 정답','+9.7%p','+7.0~+12.3%p']]),
 p('위 향상은 저장된 paired_comparison.json의 사람 단위 대응 부트스트랩 10,000회 결과다. v3와 v4 CNN을 같은 정면 조건·현재 정답으로 비교했다. 직접 측정 10개를 섞은 서비스 전체 향상 수치가 아니다. 여러 변경을 동시에 적용했으므로 개선 원인을 앙상블 하나로만 돌릴 수 없다.'),
 p('동봉 38피처 CSV에는 기존 사람별 점수에서 새로 재집계한 절대 정확도의 사람 단위 bootstrap 95% 구간도 포함했다(seed=20261001, 10,000회). 이것은 새 추론 결과가 아니며, 다중 비교 보정 또는 모델 학습 변동성 평가를 수행한 구간도 아니다.'))
occ=read('models/kface_v2/occlusion_evaluation.json')
page('보조 실험 · 가림 검출과 깊이','가림 검출은 형태 CNN과 별개인 MobileNetV3-Large 모델이다. 같은 액세서리 종류를 모든 참여자가 착용한 촬영 데이터의 내부 시험이다.',
 table(['항목','시험 인원 / 사진','정확도','macro-F1'],[[name+' / '+label,str(occ['test_participants'])+' / '+fmt(occ['test_captures']),pct(occ['test'][name]['accuracy']),num(occ['test'][name]['macro_f1'])] for name,label in [('eyewear','안경·선글라스'),('hat','모자')]]),
 p('최신 저장 파일은 시험 1,481장, 두 head 정확도 100.0%다. 오래된 보고서의 1,484장·모자 99.9%는 이전 결과이므로 이번 보고서에 그대로 옮기지 않았다. 외부 형태의 안경·모자에 대한 100% 성능 보증은 아니다.'),
 p('다중 시점 깊이의 정답은 고정 카메라 복원값이다. 기존 fewview_depth.json의 소수 시점 실험은 별도 촬영 조합과 보정 절차를 사용하므로, 본문의 단일 정면 CNN 깊이 정확도와 합치지 않았다. 스마트폰 이동 촬영에 대한 독립 실험 결과는 확인하지 못했다.'),
 p('표정·조명·액세서리별 세부 결과는 기존 evaluation.json에 남아 있다. 본문은 사용자 요청의 핵심인 최종 정면 조건에 집중했으며, 동일 사진에 여러 조건 그룹을 중복 집계한 전체 결과를 독립 데이터 수로 더하지 않았다.'))
for part,ko in parts.items():
 fkeys=[k for k,f in S.items() if f['part']==part]
 rows=[]
 for k in fkeys:
  f=S[k]
  for j,cl in enumerate(f['classes']):
   rows.append([short(k) if j==0 else '↳',f.get('classes_ko',f['classes'])[j]+' / '+cl]+[str(sum(r['split']==sp and r.get(k)==cl for r in refs)) for sp in ('train','val','test')])
 page('클래스별 얼굴 인원 · '+ko,'현재 합의 레이블과 고정된 초기 train/validation/test 분할 기준이다. 각 행은 사진 수가 아니라 서로 다른 사람 수다. 같은 사람이 여러 특징 행에 반복 등장하므로 행을 합쳐 총인원으로 쓰지 않는다.',table(['피처','클래스','학습','검증','시험'],rows),p('경계(borderline), unknown, uncertain, missing은 이 표에서 생략하고 동봉 클래스 인원 CSV에 별도 수록했다. 최종 교차학습은 이 표의 학습+검증 338명을 다시 나누어 사용한다.'))
page('해석의 한계와 정정해야 할 표현','실험은 실제 실행된 흔적과 수치를 갖고 있지만, 그 수치가 뜻하는 범위를 지켜야 한다.',
 bullets('“338명을 한 모델에 학습” 대신 “338명 개발 집단의 5-fold, 모델당 270~271명 학습”으로 설명한다.','“모델이 총 10개” 대신 “각 부위 10개, 5부위 전체 50개”로 설명한다.','“CNN이 거의 모든 측정에서 우수”는 옛 사선 혼합 평가의 해석이다. 정면 OOF에서는 직접 측정 10개를 선택했다.','“시험 누수가 전혀 없다”로 단정하지 않는다. 시험 명단의 직접 중복은 없으나, 시험을 본 뒤 정책을 정했고 OOF epoch 선택과 평가 대상이 겹친다.','특징 경계·정규화는 초기 학습 276명에서 계산한 값을 fold 전체가 공유하므로 완전히 fold 내부에서만 학습한 전처리 평가가 아니다.','일부 반복 오차 통계는 학습 사람만으로 계산하도록 수정하는 작업이 보류되어 있다. 현 결과의 완전 독립성을 과장하지 않는다.','형태 정답은 측정기·AI 판정으로 만든 기준이며 인간 독립 검수는 0건이다. 관상 풀이의 진실성을 검증한 실험이 아니다.','테스트 희소 클래스, 동일 인물 반복 사진, 촬영 장치·조명 편향 때문에 실제 일반 사용자 성능을 바로 확정할 수 없다.'),
 p('확인된 결과와 새로 검증해야 할 부분을 구별하면 현재 프로젝트의 설명은 다음과 같다: “400명 K-FACE 기반 프로토타입에서 38개 형태 특징을 학습·평가했고, 최종 62명 내부 시험과 피처별 보류 성능을 확보했다. 전문가 정답과 별도 스마트폰 사용자 집단 검증은 남아 있다.”'))
paths=['models/kface_v4/evaluation.json','models/kface_v4/evaluation_oof.json','models/kface_v4/visual_confidence.json','models/kface_v4/paired_comparison.json','models/kface_v4/folds.json','data/derived/kface/reference_labels.csv','data/derived/kface/reference_summary.json','data/derived/kface/split_frozen.json','data/label_schema_kface_v3.json','data/annotations/kface_rater_agreement.json','models/exp/comparison.json','models/kface_v2/occlusion_evaluation.json','facework/kface_training.py','scripts/kface_cv_train.py','scripts/kface_evaluate.py','scripts/kface_visual_confidence.py','scripts/kface_paired_compare.py','facework/analysis_v2.py','docs/K-FACE_v2_구축보고서.md']
hashes={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
(OUT/'근거파일_SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf8')
page('근거 파일과 재집계 절차','보고서 수치는 외부 추정이나 새 설계 목표가 아니라 작업 폴더에 존재하는 실험 결과·학습 메타데이터에서 가져왔다.',
 table(['근거','사용 내용'],[['models/kface_v4/evaluation.json','정면 조건의 38피처 시험 정확도·F1·r·분모'],['models/kface_v4/evaluation_oof.json','338명 OOF와 방법 선택'],['models/kface_v4/visual_confidence.json','사진 단위 시각 정확도·확률별 적용률'],['models/kface_v4/*/fold*/*.pt','50개 모델의 실제 학습 인원·crop 수·epoch·시간·레이블 해시'],['models/kface_v4/refs/fold*/reference_labels.csv','fold별 학습·검증·시험 명단 대조'],['data/derived/kface/subjects/*.npz, *.json','400명의 처리 사진·성공 상태·학습 후보·비참조 평가 분모 재계수'],['data/derived/kface/reference_labels.csv','현재 38피처 클래스별 사람 수'],['models/kface_v4/paired_comparison.json','기존 사람별 결과·대응 개선 구간'],['모델/평가/분할 코드 및 기존 보고서','수치의 생성 방식과 정정 이력']]),
 p('모든 최종 모델과 평가 파일의 labels_sha256은 현재 레이블 CSV의 실제 SHA-256과 일치했다. 출력 사본(outputs/kface-v4)의 주요 평가 파일도 models/kface_v4와 동일함을 확인했다. 얼굴 원본과 참여자별 예측 명단은 보고서에 첨부하지 않았다.'),
 p('재집계 스크립트: scripts/reporting/audit_experiments.py → build_experiment_report.py. 이번 검증은 보관 기록의 일관성·수량 확인이며, 학습 전체 재현이나 새 얼굴 추론 시험이 아니다. 모델 가중치의 성능을 새 실행으로 재확인했다고 주장하지 않는다.'))
doc={'title':'K-FACE 얼굴 형태 분석 — 실제 실험 결과 보고서','date':'2026-10-01','version':'v1.0 / 보관 실험 재집계','pages':pages,'features':features,'source_hashes':hashes}
(OUT/'보고서_구조화원본.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# '+doc['title'],'',doc['date']+' · '+doc['version'],'']
for i,pg in enumerate(pages,1):
 lines += [f"## {i:02d}. {pg['title']}",'',pg['lead'],'']
 for b in pg['blocks']:
  if b['kind']=='p':lines += [b['text'],'']
  elif b['kind']=='bullets':lines += ['- '+v for v in b['items']]+['']
  elif b['kind']=='table':
   clean=lambda t:str(t).replace('|','/').replace('\n','<br>')
   lines += ['| '+' | '.join(map(clean,b['head']))+' |','| '+' | '.join('---' for _ in b['head'])+' |']+['| '+' | '.join(map(clean,row))+' |' for row in b['rows']]+['']
(OUT/'실제_실험결과_보고서.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps({'sections':len(pages),'features':len(features),'epoch_records':len(epoch_rows),'label_people_range':[min(r['test_people'] for r in features),max(r['test_people'] for r in features)],'selected_methods':dict(Counter(r['method'] for r in features))},ensure_ascii=False))
