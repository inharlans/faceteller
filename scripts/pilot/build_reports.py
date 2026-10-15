"""Build human-readable Korean pilot reports from actual machine-readable results."""
import json,hashlib,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/pilot-20261014';D=ROOT/'data/derived/pilot-20261014'
def load(name):return json.loads((O/name).read_text(encoding='utf-8'))
def pct(v):return '-' if v is None else f'{100*v:.1f}%'
def num(v,n=4):return '-' if v is None else f'{v:.{n}f}'
NAMES={'nose_projection':'코끝 돌출','nose_bridge':'코뿌리 높이(콧대 대리지표)','lip_projection':'입술 돌출','chin_projection':'턱끝 돌출'}

def pdf(path,text,chart=None):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,PageBreak
    from reportlab.lib.enums import TA_LEFT
    from xml.sax.saxutils import escape
    pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('MalgunBold','C:/Windows/Fonts/malgunbd.ttf'))
    body=ParagraphStyle('body',fontName='Malgun',fontSize=9.5,leading=15,spaceAfter=5,wordWrap='CJK')
    heading=ParagraphStyle('heading',parent=body,fontName='MalgunBold',fontSize=12,leading=18,spaceBefore=13,spaceAfter=8,keepWithNext=True)
    title=ParagraphStyle('title',parent=heading,fontSize=18,leading=25,spaceAfter=14)
    story=[]
    for i,line in enumerate(text.splitlines()):
        if not line.strip():story.append(Spacer(1,5));continue
        style=title if i==0 else (heading if line.startswith('[') and line.endswith(']') else body)
        story.append(Paragraph(escape(line.strip('[]') if style is heading else line),style))
    if chart:
        story+=[PageBreak(),Paragraph('400명 깊이 지표 분포',heading),Image(str(chart),width=480,height=349)]
    def footer(canvas,doc):
        canvas.setFont('Malgun',8);canvas.drawCentredString(297.6,24,f'{doc.page}')
    SimpleDocTemplate(str(path),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42,title=text.splitlines()[0],author='FaceTeller').build(story,onFirstPage=footer,onLaterPages=footer)

def main():
    depth=load('depth_distribution_results.json');cnn=load('cnn_results.json');integ=load('integration_results.json');contract=load('contract_results.json');ui=load('ui_results.json')
    src=Path('C:/Users/user/Downloads/FaceTeller_상세설계서_제출본.pdf')
    source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    warm=integ['warm_timing'];cal=cnn['calibration']['test_summary']
    summary=[
    'FaceTeller 실제 파일럿 검증 보고서','2026-10-14 | 기준: FaceTeller_상세설계서_제출본.pdf (23쪽)',
    '[1. 확인된 결과]',
    f'실제 K-FACE 400명 × 정면·좌우 사선 3장 = 1,200장으로 VGGT를 실행했다. 네 가지 상대 돌출도는 {depth["cohort_measured"]}명에서 산출됐다. 모델을 새로 학습한 것이 아니라 공개된 VGGT 가중치를 고정해 추론한 결과다.',
    'MediaPipe를 import하지 않은 프로세스에서 RGB 사진만 입력해 VGGT 깊이 출력을 얻었다. 따라서 VGGT 신경망 자체는 MediaPipe에 의존하지 않는다. 코끝 등 얼굴 위치를 찾아 상대 돌출도를 계산하는 단계에서는 MediaPipe 특징점과 VGGT 결과를 결합한다.',
    f'고정된 추론 작업자를 재사용한 통합 API 시험은 시험 참여자 10명에서 평균 {warm["mean"]:.3f}초, P95 {warm["p95"]:.3f}초, 최대 {warm["max"]:.3f}초였다. {warm["within_5s_count"]}/10건이 5초 이내였다. 모델 사전 로딩·최초 예열 후 단일 요청 기준이며 외부 네트워크·HTTPS·브라우저 표시 시간은 포함하지 않았다.',
    '실행 성공과 3D 측정 정확도를 구분한다. 실제 3D 스캔 정답이 없으므로 코 높이가 실제로 맞는지에 대한 mm 오차나 정답 정확도는 이번 실험에서 산출하지 않았다.',
    '[2. 사진·환경·재현 조건]',
    '하드웨어: Ryzen 5 5600, NVIDIA RTX 3070 8GB, Windows. Python 3.12.14, PyTorch 2.11.0+cu128.',
    '사진: K-FACE S001(가림 없음), E01(무표정), L1, C7/C4/C10. 기존 프로젝트에서 방향을 약 0°/-40°/+40°로 확인한 카메라를 사용했다. 여러 카메라의 정적인 사진 실험이며 스마트폰 한 대로 고개를 돌려 촬영한 새 영상의 검증은 아니다.',
    '촬영 가이드 모사: 제공기관 얼굴 박스를 1.5배로 넓혀 자르고 종횡비를 보존해 518×518로 맞췄다. 원본 전체 화면 입력은 별도 비교했다. VGGT에는 CNN용 부위 crop이나 정면 회전 정렬 이미지를 넣지 않았다.',
    '주 실험 400명 외에 40명을 사전에 일정 간격으로 선택해 L2/L3 사진 80세트를 추가했다. L2에는 얼굴 박스 TXT가 없어 같은 카메라의 L1 박스를 재사용했다. 최초 주석 누락 오류 40건은 시도 기록에 남기고 재실행했으며, 성공한 같은 촬영을 중복 표본으로 세지 않았다.',
    'VGGT: facebook/VGGT-1B 공개 연구용 가중치. 코드 commit a288dd0f14786c93483e45524328726ab7b1b4ce. transformer는 BF16, camera/depth head는 FP32, point/track head는 불필요해 사용하지 않았다. 깊이와 카메라 행렬로 3D 좌표를 복원했다.',
    '가중치 SHA256: d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0. 공식 Hugging Face LFS 해시와 일치한다. 상용 배포에는 별도 commercial checkpoint 조건 확인이 필요하며 이번 파일럿은 원본 연구용 checkpoint를 사용했다.',
    '[3. 설계서 4.2의 의존 관계]',
    '현재 그림의 DepthEstimator가 “깊이 추론 + 얼굴 부위별 측정”을 모두 뜻한다면 특징점을 받는 인터페이스는 맞다. 다만 이 그림을 “VGGT에 MediaPipe 출력이 입력된다”로 해석하면 틀리다.',
    '입력 검증 → MediaPipe 특징점 → 정면 정렬·부위 추출 → CNN 형태 분류',
    '입력 검증 → 사진 3장 → VGGT → 깊이·카메라·3D 형상',
    'MediaPipe 2D 특징점 + VGGT 3D 형상 → 상대 돌출도 → 두 경계·CDF 점수 → 항목별 보류·문헌 선정',
    '두 모델은 독립 호출할 수 있지만 얼굴 부위별 측정은 두 결과가 모두 필요하다. 독립 호출이 반드시 GPU 동시 실행을 뜻하지는 않는다. 이번 통합 서버는 모델을 미리 로딩하고 고정 추론 작업자에서 순차 실행했다.',
    '권장 인터페이스: VGGTGeometry.infer(images) → geometry, DepthMeasurement.measure(geometry, landmarks) → measurements. 기존 DepthEstimator.estimate(images, landmarks)는 이 둘을 감싼 모듈로 유지할 수 있다.',
    '[4. 실제 경계와 반복 측정]',
    '400명 전체의 L1 측정값에서 구한 경험 P25/P75다. 각 값은 3D 외안각 사이 거리로 나눈 비율이며 단위는 mm가 아니다. 각 항목의 낮음·보통·높음 인원은 100·200·100명이다.',
    '항목 | 낮음/보통 경계 b1 | 보통/높음 경계 b2 | 같은 사람 3회 레이블 모두 일치',
    ]
    for k,v in depth['features'].items():summary.append(f'{NAMES[k]} | {v["lower"]:.6f} | {v["upper"]:.6f} | {pct(v["repeat"]["all_three_label_agreement"])}')
    summary += ['연속 측정값의 반복성은 ICC(1,1)로 계산했다. 구간 레이블의 일치율과 다른 지표이며 정확도라고 부르지 않는다.']
    for k,v in depth['features'].items():summary.append(f'{NAMES[k]}: ICC {v["repeat"]["icc_1_1"]:.3f}; L1↔L2 {pct(v["repeat"]["label_agreement_L1_L2"])} / L1↔L3 {pct(v["repeat"]["label_agreement_L1_L3"])}.')
    summary += ['경계가 안정적인지 보려면 위 반복 결과를 함께 봐야 한다. 경계 근처에서 점수가 낮다는 정보는 유용하지만 CDF 점수를 붙이는 것만으로 촬영 변동이 사라지지는 않는다.','정규성 검정(Shapiro-Wilk, 보정 전 p값):']
    for k,v in depth['features'].items():summary.append(f'{NAMES[k]} p={v["normality"]["shapiro_p"]:.4f}, 왜도 {v["normality"]["skew"]:.3f}.')
    summary += ['큰 p값은 정규분포라는 증명이 아니다. 콧대 지표는 p≈0.026으로 정규 가정을 당연하게 둘 수 없다. 경계는 정규분포 가정 없이 실제 분위수로 계산했다. 별도로 평균±0.67449×표준편차 경계와의 판정 차이도 JSON에 기록했다.',
    '원본 화면·정면 1장·정면 복제 3장·각도 변경·crop 변경을 8명씩 비교했다. 아래는 기본 3장 입력과의 레이블 일치율이며 정답 정확도가 아니다.']
    for setting,fs in depth['controls'].items():summary.append(setting+' → '+', '.join(f'{NAMES[k]} {pct(v["label_agreement"])}' for k,v in fs.items()))
    summary += ['기존 Tomasi–Kanade 12시점 측정값과도 비교했지만 그것 역시 실측 3D 정답은 아니다. 두 알고리즘의 값이 비슷하다는 이유만으로 정확하다고 결론내리지 않았다.',
    '[5. CNN 실측과 신뢰도 보정]',
    '기존 v4 앙상블 50개를 시험 참여자 62명의 정면 사진 62장에 실제 실행했다. 한 사진에서 7개 부위 입력을 만들어 총 434개 crop을 처리했다. 동일인 좌우 부위를 별도 사람으로 세지 않았다.',
    '원래 분할은 train 276 / val 62 / test 62명이다. v4는 train+val 338명을 5겹으로 나누므로 개별 모델의 학습 인원은 270 또는 271명, 검증 인원은 68 또는 67명이다. 시험 62명은 모든 fold 학습에서 제외돼 있다.',
    '통합 파일럿은 fold0의 두 백본 × 5부위 = 10개 모델을 사용했다. 해당 후보는 학습 270명, 검증 68명, 시험 62명이다. 검증 68명에서 15개 시각 분류 항목의 temperature를 맞추고 시험 62명에서 보정 결과를 확인했다. 전체50 모델에 이 보정값을 적용하지 않았다.',
    f'15항목 평균 NLL: {cal["mean_head_nll_before"]:.5f} → {cal["mean_head_nll_after"]:.5f}; 평균 ECE: {cal["mean_head_ece_before"]:.5f} → {cal["mean_head_ece_after"]:.5f}. 일부 항목은 악화됐으며 원본 결과에 함께 기록했다.',
    '검증 집단은 early stopping에도 쓰였으므로 별도의 독립 보정 전용 집단은 아니다. 참조 레이블은 AI 판정자 합의 또는 자동 측정 기반이다. 독립 인간 검수 정답은 0건이므로 아래 수치를 전문가 정답에 대한 정확도로 해석하지 않는다.',
    '전체50 모델의 부위별 CNN 원출력 일치율(시험62명, 유효 참조가 있는 특징만 집계):']
    for part,v in cnn['fresh_measurement']['by_legacy_part'].items():
        x=v['cnn_micro'];summary.append(f'{part}: {x["n_correct"]}/{x["n_scored"]}={pct(x["accuracy"])}. 보류 기준 통과 항목만의 일치율 {pct(x["accepted_accuracy"])}; 유효 참조 중 통과율 {pct(x["coverage_on_scored"])}.')
    summary += ['[6. 특징별 실측 결과 — 기존 50개 CNN]',
    '각 행: 특징 / 평가 가능한 사람 수 / 전체 일치율 / 보류 기준 통과율 / 통과한 항목의 일치율. 각 특징의 참조 unknown·borderline은 분모에서 제외돼 n이 다르다. 아래는 통합10모델 결과와 혼합하지 않는다.']
    for k,v in cnn['fresh_measurement']['features'].items():
        x=v['cnn_only'];summary.append(f'{v["name_ko"]} ({k}) / n={x["n_scored"]} / {pct(x["accuracy"])} / {pct(x["coverage_on_scored"])} / {pct(x["accepted_accuracy"])}')
    summary += ['[7. 설계대로 된 부분과 아직 다른 부분]',
    '5.2 정렬: 기존 CNN 입력은 외안각 수평 정렬·얼굴 폭 기준이다. 설계서는 눈 중심·눈 사이 거리 기준이다. 별도 눈 중심 정규화 함수를 구현해 눈 사이 128px와 종횡비 보존을 수치 검증했다. 기존 모델에 그대로 바꿔 넣는 비교는 아래와 같으며 재학습한 결과는 아니다.']
    total=sum(x['n'] for x in integ['design_eye_normalized_vs_training_crop']);equal=sum(x['labels_equal'] for x in integ['design_eye_normalized_vs_training_crop'])
    summary += [f'5명 × 15 시각 항목 비교: 입력 정규화만 변경했을 때 {equal}/{total}개 레이블이 같았다. 기존 가중치의 정확도가 새 정규화에서도 유지된다고 단정할 수 없다.',
    '5.3 범위: 현재 전용 CNN은 눈·눈썹·코·입·얼굴형 5개 부위다. 이마·미간·애교살·광대·귀·인중·점의 전용 crop 분류 모델은 없다. 기존 미간 너비·광대·인중 특징 일부가 있다는 것과 전용 부위 모델이 있다는 것은 다르다.',
    '기존 모델은 시각 분류 15항목과 연속값 회귀 23항목으로 구성돼 있다. 회귀값을 경계로 나눠 얻은 레이블에는 보정된 CNN 분류 신뢰도가 없다. 통합 파일럿에서는 해당 회귀 분류를 억지로 확정하지 않았고, 깊이 4항목은 VGGT 측정으로 대체했다.',
    '문헌 103개를 ID와 원문·설명·규칙으로 대조했다. 완전 조건 full 15개, partial 65개, none 23개다. 단일값 AND 조건으로 처리 가능한 초벌 14개만 실험용으로 사용했다. 독립 검수 완료 문헌은 0개이며 결과에 draft 상태를 표시한다.',
    '[8. 기능별 시험]',
    'FR_001/002: 촬영 조건 유지·이탈 초기화, 오래된 판정 무시, 방향별 1회 캡처, 3장 후 1회 제출 상태 테스트 통과. 권한 거부 시 업로드 전환은 실제 브라우저에서 확인. 실제 사람·웹캠의 조건 검출은 미검증이다.',
    'FR_003/004: 3장 필수, 파일 형식·용량·디코딩, 누락 방향, 잘린 요청, 메모리 정리, 오류·취소·응답 시작 실패 경로를 검증했다. 실제 얼굴 없는 측면 사진은 방향을 포함한 422 응답으로 확인했다.',
    'FR_005: 실제 얼굴 검출·특징점·부위 crop 실행. 설계 눈 중심 정규화는 별도 구현·검증했지만 기존 CNN 학습 전처리와 차이가 있다.',
    'FR_006/007: 실제 CNN/VGGT 실행, 미학습 부위 보류, 보정 안 된 분류 보류, 측면 방향 오류 시 깊이만 보류. 탐색 모드의 400명 기준 판정과 엄격 모드의 미검증 깊이 보류를 구분했다.',
    'FR_008: 전체 조건 AND, 중복 제거, 분류 신뢰도의 기하평균 G, 동률 순서, 관련 모든 부위별 3개/전체5개, 측정만 있는 조항의 후순위, 문헌 없음과 DB 실패 구분을 테스트했다.',
    'FR_009/010: NDJSON 분할 수신·UTF-8·중단 오류·결과·보류·문헌 없음·다시 시작. A4 실제 PDF에서 원문·직역·근거·출처를 추출 대조했다. 아주 긴 문헌은 1장에 들어가지 않아 경고 후 여러 장 출력하도록 시험했다. 이 대체 출력은 “항상 A4 1장” 요구를 달성한 것으로 세지 않는다.',
    f'계약/API 자동화: {contract["tests"]["run"]} 통과. PostgreSQL 실제 검증 {len(contract["postgresql"]["checks"])}개 통과. 독립 임시 DB를 사용했으며 종료·제거했다. 실제 모델 통합 시에는 초벌 문헌을 메모리에 읽어 조회했고, PostgreSQL 어댑터는 별도 실제 DB 시험이다.',
    f'UI 시험 {len(ui["checks"])}개 기록. UI는 vanilla JS 기능 파일럿이며 설계의 React/TypeScript 구현 완료가 아니다. 실제 추론 결과 재생과 합성 API fixture를 구분한다.',
    '[9. 통합 응답시간과 실제 결과]',
    '모델 로딩·최초 예열 후 사진 3장 요청부터 NDJSON 최종 응답까지 실제 모델로 실행했다. 기존 프로토타입의 추정시간을 재인용한 것이 아니다.',
    f'평균 {warm["mean"]:.3f}초 / 중앙값 {warm["median"]:.3f}초 / P95 {warm["p95"]:.3f}초 / 최대 {warm["max"]:.3f}초 / 5초 이내 {warm["within_5s_count"]}/10건.',
    '요청별 수치:']
    for x in integ['records']:summary.append(f'시험 {x["case"]}: {x["seconds_local_testclient"]:.3f}초; 확정 {x["confirmed"]}, 보류 {x["held"]}, 깊이 확정 {x["depth_confirmed"]}, 선정 조항 {x["passages"]}.')
    summary += [f'최초 예열 요청은 {integ["warmup_seconds"]:.3f}초였다. 서버는 예열까지 마친 뒤 요청을 받도록 준비해야 한다.',
    '초기 통합 시험에서는 요청마다 새 작업 스레드가 생성되는 실행 방식에서 약 6초대가 나왔다. 고정 작업자 재사용 후 위 결과를 다시 측정했다. 실제 GPU 모델은 동일하고 정밀도도 유지했다. 최초 실행·동시 요청·저사양 장치·외부망 시간을 보장하는 수치는 아니다.',
    '[10. 남은 작업]',
    '새로 추가된 7개 부위의 레이블·전용 모델과 검수 자료 확보. 연속값 회귀 항목을 설계의 분류 신뢰도 체계로 전환할지 확정.',
    '눈 기준 정규화로 새로 학습하거나 기존 얼굴 폭 정규화를 설계에 명시적으로 채택. 신뢰도 보정용 별도 자료 확보 및 희소 클래스 검수.',
    '실제 3D 스캔 또는 신뢰 가능한 측정 정답으로 깊이 오차 검증. 스마트폰 한 대의 순차 촬영, 표정 변화, 가림, 카메라별 재현성 검증.',
    '경계 점수가 낮은 항목의 보류 기준 선택. 현재 400명 경험 구간은 관상 문헌 유형의 정답 경계라는 뜻이 아니다.',
    '문헌 번역·형태 조건 독립 검수, React/TypeScript 통합, 실제 웹캠 검증, 장문 조항의 A4 한 장 정책 확정.',
    '[11. 산출물과 재실행]',
    '코드: facework/pilot/{depth,scoring,runtime,design_preprocess}.py, facework/pilot_contract.py.',
    '실험: scripts/pilot/run_vggt.py, analyze_vggt.py, run_integration.py, scripts/pilot_cnn_check.py, scripts/pilot_contract_check.py, scripts/pilot/ui-check.cjs.',
    '숫자: outputs/pilot-20261014/*results.json. 400명 개별 측정과 점수: data/derived/pilot-20261014/cohort_scores.csv. 보고서에는 제공 얼굴 사진을 넣지 않았다.',
    '원본 상세설계서 SHA256: '+source_hash+'. 이 PDF 원본은 수정하지 않았다. 필요한 수정 문구는 별도 TXT에 정리했다.',
    '공식 근거: https://github.com/facebookresearch/vggt (RGB-only forward, camera/depth heads); https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html (표준정규 누적분포); https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html (정규성 검정).',
    ]
    method=['FaceTeller 깊이 3단계 분류·점수 설계안','2026-10-14 | 실제 400명 VGGT 측정 및 40명 반복 촬영 파일럿 반영',
    '[1. 적용 범위]','사진 3장으로 얻은 상대 돌출도 r를 낮음·보통·높음으로 나누고 경계에서 얼마나 떨어져 있는지 점수로 표시한다. 얼굴 크기·코 높이를 mm로 측정하거나 성격·복·운명을 추정하는 점수가 아니다.',
    '[2. 측정값 r 정의]','VGGT는 사진으로 깊이·카메라를 추정한다. 깊이를 역투영한 3D 좌표에서 MediaPipe의 2D 특징점 위치를 찾아 표면점을 얻는다. MediaPipe의 z값을 실제 깊이로 쓰지 않는다.','가로축은 외안각 33→263, 세로축은 이마10→턱175를 가로축에 직교화한 방향, 앞쪽 축은 카메라 방향이다. 기준 길이 W는 3D 외안각 사이 거리다. 각 상대 앞뒤 차이를 W로 나눈다.',
    '코끝 = [앞뒤좌표(1) − 평균(98,327)] / W. 콧방울 기준에 대한 코끝 돌출이다.',
    '코뿌리 = [앞뒤좌표(168) − 평균(133,362)] / W. 콧대 전체 높이의 직접 측정값으로 부르지 않는다.',
    '입술 = [평균(0,17) − 평균(61,291)] / W. 입꼬리 기준의 입술 표면 돌출이다.',
    '턱끝 = [앞뒤좌표(175) − 앞뒤좌표(2)] / W. 코밑 기준이므로 음수도 가능하다.',
    '영상 좌표와 깊이 출력의 resize/pad 변환을 일치시킨다. 이번 경계 실험은 518×518 입력의 특징점 검출을 기준으로 했다. 처리 해상도·crop·측정점·모델을 바꾸면 경계를 재검증해야 한다.',
    '[3. 두 경계와 레이블]','사람마다 한 값씩 사용한다. 한 사람의 사진이 많다고 분포에서 더 큰 비중을 주지 않는다. 주 기준은 400명 L1의 각 1개 측정값이다.',
    'b1=P25, b2=P75. r<b1이면 낮음, b1≤r<b2이면 보통, b2≤r이면 높음이다. 이번 파일럿은 항목별100/200/100명이다. P25/P75는 파일럿의 운영 선택이며 문헌이 이 비율을 정해 준 것은 아니다.',
    '정규성이 성립하지 않아도 경험 분위수를 사용한다. 정규분포를 가정하면 b1=μ−0.67449σ사람, b2=μ+0.67449σ사람이지만 실제 데이터에서 두 방법의 경계와 판정은 다를 수 있다.',
    '항목 | b1 | b2 | 사람 간 표준편차 | 반복 오차 σ(학습집단)']
    for k,v in depth['features'].items():method.append(f'{NAMES[k]} | {v["lower"]:.6f} | {v["upper"]:.6f} | {v["sigma_population"]:.6f} | {v["sigma_repeat_train"]:.6f}')
    method += ['[4. P = Φ((r−경계값)/σ)의 적용]',
    'A = Φ((r−b1)/σ), B = Φ((r−b2)/σ). A는 아래 경계를 넘는 점수이고 B는 위 경계를 넘는 점수다.',
    '낮음 점수 S낮음 = 1−A', '보통 점수 S보통 = A−B', '높음 점수 S높음 = B',
    '세 점수는 0~1이고 합은 1이다. 화면에서는 판정된 레이블의 해당 점수를 100배해 “경계점수”로 표시한다. 분류 신뢰도 confidence와 구분해 measurement_score 필드에 넣는다.',
    '레이블 자체는 r와 b1,b2로 결정한다. 점수의 argmax로 바꾸지 않는다. 특히 σ가 크면 보통 구간의 일부에서도 낮음/높음 점수가 더 클 수 있기 때문이다.',
    '이 값은 정답률로 검증된 확률이 아니다. 정규분포의 측정 오차와 추가 가정을 두면 확률적으로 해석할 수 있지만, Φ를 사용했다는 이유만으로 실제 정답 확률이 되는 것은 아니다.',
    '[5. σ를 어떻게 정할 것인가]',
    '사람 간 표준편차 σ사람은 사람들의 얼굴 차이를 나타낸다. 이를 분모로 사용하면 점수는 넓고 완만해진다. 정규분포의 P25/P75를 경계로 하고 σ사람을 넣으면 보통 구간 중앙의 보통 점수도 약 0.5다. CNN의 0.60 보류 기준을 그대로 적용하면 보통 항목이 대거 보류될 수 있다.',
    '반복측정 표준편차 σ반복은 같은 사람의 촬영 조건 변화에 따른 흔들림이다. 서로 다른 사람의 차이를 제거한 뒤 구한다.',
    'σ반복 = sqrt[ Σ사람 Σ촬영 (r사람,촬영 − 해당사람평균)^2 / Σ사람(촬영수−1) ].',
    '40명 각각 L1/L2/L3 세 조건에서 측정했다. 이 중 기존 train에 속하는 26명으로 구한 σ를 통합 탐색 파일럿에 사용했다. 나머지 14명은 별도 반복 평가에 사용했다. 조명 반복만으로 모든 웹캠 오차가 설명되지는 않는다.',
    '독립 비교용 경계는 원래 train 276명만으로도 별도로 만들었다. 아래 14명 비교는 train 경계와 train 반복 σ를 고정한 결과다. 400명 전체 경계는 집단 전체를 설명하는 탐색용 기준이며 별도로 유지한다.',
    '점수가 0.7 이상인 두 촬영 쌍만 비교한 결과(정답 정확도가 아니라 반복 레이블 일치율):']
    for k,v in depth['features'].items():
        x=v['repeat_selective_agreement']['heldout_nontrain']['0.7'];method.append(f'{NAMES[k]}: {x["accepted_pairs"]}/{x["pairs"]}쌍 유지, 유지율 {pct(x["pair_coverage"])}, 같은 레이블 {pct(x["label_agreement_on_accepted_pairs"])}.')
    method += ['0.7은 비교 실험용 값이며 서비스의 최종 보류 임계값으로 확정하지 않았다. 0.6/0.7/0.8의 결과를 JSON에 모두 남겼다.',
    '[6. 판정 실패와 문헌 연결]',
    '얼굴 위치를 찾지 못했거나 값이 유한하지 않거나 촬영 방향이 맞지 않으면 해당 깊이 항목을 보류한다. VGGT depth_conf는 자체 점수로 1보다 클 수 있으므로 퍼센트 정확도로 해석하지 않는다. 독립적인 품질 기준은 추가 검증해야 한다.',
    '숫자가 나왔다는 것만으로 정확한 깊이를 측정한 것은 아니다. 파일럿의 탐색 모드에서는 400명 기준 레이블과 경계점수를 보여 주되, 엄격 모드에서는 독립 품질 검증 전 깊이 항목을 보류한다.',
    '문헌 G는 기본적으로 기존 설계대로 CNN의 보정된 신뢰도만 사용한다. 깊이 점수까지 포함한 기하평균은 별도 비교 모드로 구현·시험했다. 경계점수와 CNN 신뢰도가 같은 의미라고 검증되지 않았으므로 기본 정렬을 자동 변경하지 않았다.',
    '[7. 코드와 결과]',
    'facework/pilot/scoring.py: 경계 추정, 세 구간 점수, 반복 σ. tests/test_pilot_depth_scores.py: 합1, 단조성, 경계 포함 규칙, σ오류, 비정규 표본, 3D 축/크기 불변성 검증.',
    'data/derived/pilot-20261014/boundary_config.json: 실제 경계·σ·분포·반복 결과.',
    'data/derived/pilot-20261014/cohort_scores.csv: 400명 항목별 r·레이블·σ종류별 점수.',
    'outputs/pilot-20261014/depth_distribution_results.json: 모든 통계 및 비교.',
    '근거: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html . 점수 구성과 경계·σ 선택은 본 프로젝트 파일럿의 정의다.'
    ]
    changes='''FaceTeller 상세설계서 수정 문구 — 실제 파일럿 반영 제안
원본 PDF는 변경하지 않았다.

1. 4.2 실제 처리 순서 설명
기존: 실제 처리 순서는 전처리 → 형태·깊이 분석 → 판정 보류 → 문헌 선정 → 결과 응답이다.
교체: 입력 검증 후 MediaPipe 특징점 추출과 VGGT 깊이 추정을 각각 수행한다. 특징점으로 정면 부위를 추출해 CNN 분류를 수행하고, VGGT의 3D 형상과 특징점을 결합해 상대 돌출도를 계산한다. 두 분석 결과를 판정 보류와 문헌 선정 모듈로 전달한다.
그림: MediaPipe와 VGGT로 들어가는 사진 경로를 나누고, 상대 돌출도 계산에서 두 결과가 합쳐지는 화살표를 표시한다.
주의: 기존 DepthEstimator가 추론+측정을 모두 포함한다면 '사진3장·특징점' 입력은 맞다. 특징점을 VGGT 신경망 입력으로 표시하지만 않으면 된다. 독립 경로가 GPU 동시 실행을 강제하는 것은 아니다.

2. 5.4 측정 항목 문장
기존: 상대 돌출도는 분류 신뢰도로 변환하지 않는다. 품질 기준을 충족한 측정값을 경계값과 비교하여 레이블을 확정한다.
교체: 품질 기준을 충족한 상대 돌출도를 두 경계값과 비교해 낮음·보통·높음으로 판정한다. 각 경계에 P=Φ((r−경계값)/σ)를 적용해 경계점수를 계산하고, 측정값·판정 근거와 함께 표시한다. 경계점수는 CNN 분류 신뢰도와 구분한다.
추가: 두 경계와 σ의 계산 방법은 별도 '깊이 3단계 분류·점수 설계안'을 따른다.

3. 6.1 결과 자료형 추가
confidence: CNN 분류 신뢰도. 깊이 측정 항목은 null.
measurement_score: 깊이 측정값의 경계점수(0~1). 해당하지 않으면 null.
measurement: 기존 지표명·비율값·단위(ratio) 유지.
6.2 결과 표시: 측정 항목은 비율값과 경계점수를 구분해 표시한다.

4. 5.5 G 정렬
현 단계에서는 문구 유지. 깊이 경계점수를 G에 포함하는 실험 모드는 따로 검증했으나, 두 점수의 의미가 같다고 확인되지 않아 기본 정렬을 임의로 바꾸지 않았다.

5. 구현 전 확정할 차이(이번 문서에 자동 반영하지 않음)
- 5.2 눈 사이 거리 정규화 vs 기존 CNN의 얼굴 폭 정규화. 기존 가중치를 재사용하려면 기존 방식 유지와 새 방식 재학습 중 선택해야 한다.
- 5.3 기존5부위 외7부위 전용 CNN은 미학습. 기존에 일부 관련 특징을 출력한다는 것과 구분한다.
- 5.3 기존 연속값 회귀 항목의 분류 신뢰도 처리 방식을 정해야 한다.
- 6.2 장문 문헌을 항상 A4한장에 원문그대로 넣는 조건은 실제 분량에 따라 불가능할 수 있다. 한장초과시 안내/다중페이지 등 정책 확정이 필요하다.
'''
    for name,lines in [('FaceTeller_파일럿_검증보고서',summary),('FaceTeller_깊이_3단계분류와점수_설계안',method)]:
        text='\n\n'.join(lines)
        (O/(name+'.txt')).write_text(text,encoding='utf-8-sig')
        pdf(O/(name+'.pdf'),text,O/'depth_distributions.png' if '검증보고서' in name else None)
    (O/'FaceTeller_설계서_수정문구.txt').write_text(changes,encoding='utf-8-sig')
    (O/'provenance.json').write_text(json.dumps({'design_sha256':source_hash,'design_pages':23,'weights_sha256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0','photos_in_reports':False,'report_inputs':['depth_distribution_results.json','cnn_results.json','integration_results.json','contract_results.json','ui_results.json']},indent=2))
    print('REPORTS_WRITTEN',O)
if __name__=='__main__':main()
