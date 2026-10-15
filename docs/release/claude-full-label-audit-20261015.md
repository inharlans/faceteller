# FaceTeller 전체 특징·레이블·문헌대응 1차 감사 (Claude, 2026-10-15)

작성: Claude Opus 5.5 (AI 감사). 사람 검수 0명이고 독립 정답도 없다. 이 문서는 자동 산출물이 서로 맞는지, 그리고 정의가 의미에 맞는지를 점검한 기록이며, 정답 정확도를 증명하지 않는다.
상태: main 2차 독립 검토 전의 1차 감사다. 동결 산출물, 데이터, 모델, 프론트, 파이프라인은 하나도 바꾸지 않았다. GPU와 서버 재시작은 쓰지 않았고 사진도 외부로 보내지 않았다.
기계 판독용 전체 행렬: `outputs/release/claude-full-label-audit-20261015.json` (41 CNN head + 4 VGGT, 129 class, 적격 73조항, 입력 산출물 SHA256 포함).

## 0. 요약 (가장 큰 근본 원인)

1. **강조형·명명형 문헌 표현을 집단의 일반 방향 클래스에 연결했다 (C1).**
   - 「目尾朝天」(許負相法 相目篇)을 `eye_tilt:up`에 연결했는데, 이 클래스는 기울기가 +1.665°만 넘으면 된다. train270의 중앙값은 +4.48°이고, 400명 중 88.7%가 이 기준을 넘는다.
   - NIH Elements of Morphology는 눈틈 기울기의 평균 자체가 "약간 위로 기운" 상태이고, 객관적 upslanted는 평균+2SD 초과라고 정의한다.
   - 따라서 'up'이 많은 것은 **실제 출현 빈도이지 오류가 아니다.** 오류는 강조형 명칭을 일반 클래스에 대응시킨 것이다.
   - 같은 유형이 더 있다: 「下垂」 3조항 → 입꼬리 down(64%), 「十分昂露」 → 콧구멍 visible(39%), 「尾垂·頭高尾低·八字」 → brow_tail descending(99.6%).
2. **한 클래스에 거의 몰린 보정 자료로 temperature를 적합했고, 포화된 확률이 기하평균 순위를 지배했다 (C2).**
   - eye_tilt의 보정 지원은 [down0, level1, up30]인데 "관측 클래스 2개 이상" 규칙을 통과했고, T=0.370으로 확률이 날카로워졌다.
   - 그 결과 서비스 신뢰도 중앙값은 0.9999999999, 94.7%가 0.99 이상이다. XF01-eye-006은 400명 중 357명에게 선택됐고 338명에게 1위였다(test62에서는 48명).
   - sclera_show도 지원 [32,2,0]에서 T=0.329로 같은 형태다.
3. **CNN 입력에 보이지 않는 분모로 목표값을 정의했다 (C3).**
   - eye_size, nose_width, mouth_width, brow_gap은 얼굴 폭으로, nose_length는 얼굴 높이로 나눈다. 그런데 crop은 눈중심거리로 정규화되어 있다.
   - crop에서 보이는 양이 목표의 분산을 설명하는 비율(R²)은 eye_size .22, brow_gap .25, nose_length .40, nose_width .53, mouth_width .69다.
4. **경계 보류로 경계 근처 사람을 학습·보정·시험에서 빼고, 서비스는 모든 사람에게 적용한다 (C4).**
   - train unknown은 eye_size 151/270, brow_length 160/270이다.
   - 서비스에서는 기준 레이블이 '보통'인 사람이 대부분 보류된다(mouth_width는 train 92명 중 85명). 결과적으로 확정 레이블이 극단 클래스로 치우치고, 문헌 선택도 극단 쪽으로 치우친다.
5. **방향 정의가 퇴화했거나 기준점이 맞지 않는다 (C5).**
   - brow_tail: 바깥-안쪽 끝점 기울기가 400/400명 모두 음수다(최대 −0.21°). 끝점 순서가 바뀐 문제는 아니다(400/400 순서 확인).
   - lip_balance: 기준점은 1.0인데 train 중앙값은 0.786이다.
   - eye_tilt의 down은 train 지원이 0인데, 이 클래스에 연결된 조항이 2개 있다.
6. **문헌 정책 자체의 위반이 있다 (C7).**
   - 여성 전용 장(〈專論女人之說〉)에서 온 조항 2개가 남아 있다. 성별은 관찰할 수 없는 조건이다.
   - 「顴高反好爲非」(악행 주장)와 「兩眼雌雄 豈由仁由義之子」(도덕 판단)가 적격이다.
7. **AI 시각 레이블의 일치도가 낮다 (C6).** 같은 계열 AI 판정자 3명의 Fleiss κ가 0.17~0.49다. philtrum_definition은 κ 0.173에서 'withhold'로 결정됐는데도 release에서 다시 학습 목표로 쓰였다.

판정 집계:
- **특징 45개**: PASS 2 / PASS_QUALIFIED 23 / CHANGE 15 / BLOCKED 5.
- **적격 73조항 의미 판정**: PASS 2 / PASS_QUALIFIED 52 / CHANGE 15 / BLOCKED 4.
- **다음 release에서의 서비스 판정**: 의존하는 특징의 판정까지 반영하면 PASS 2 / PASS_QUALIFIED 19 / CHANGE 39 / BLOCKED 13.

누설(leakage)은 이 실패의 원인이 아니다. 시험군에서도 1위 비율이 48/62(77%)다.

## 1. 범위·방법·한계

- **입력**: training_registry.json, landmark_labels.json, extended_labels.json, deployment.json, test_metrics.json, service-verification.json(실측 400명 HTTP), score-influence.json, reference_corpus_v1.json, kface_rater_agreement.json, depth_config.json, depth-stability.json, protocol.json. 해시는 JSON의 `source_sha256`에 있다.
- **CPU 재계산**: 400명 NPZ에서 기준 조명 C7 사진으로 19개 landmark 측정값을 다시 계산했다(분포, 측면 차이, 끝점 순서). train270에서 단일 촬영 ICC(1,1)도 계산했다. 입력 crop에서 보이는 양과 목표값의 R²도 구했다.
- **문헌**: 적격 73조항 모두 원문, 앞뒤 원문 문맥(source_sections의 char_span 주변), 직역, 조건 evidence를 읽고 대응이 정확한지 판정했다. 외부 근거로 NIH Elements of Morphology의 Palpebral Fissure Upslanted/Downslanted, Mouth Downturned Corners, Mouth Narrow 정의를 확인했다(문헌 출처는 9절).
- **한계**:
  - 고문 판독은 AI 독해이며 전문가 검수가 없다.
  - 'marked' 기준(±2SD)은 Elements of Morphology의 객관 정의를 빌린 제안이며, 관상 용어와 같다는 증명이 아니다.
  - 기존 CNN의 오류율과 진짜 형태 사이의 관계는 정답이 없어 판정할 수 없다.

## 2. XF01-eye-006 실패 해부 (사실 순서)

1. **측정**: `eye_tilt = atan2(y_inner − y_outer, |dx|)`이며 눈중심 정렬 좌표를 쓴다. 양수면 바깥 눈꼬리가 위에 있다. 부호와 landmark(33/133, 263/362) 순서는 400/400명 모두 맞다.
2. **클래스**: 기준점 0°에, 수평(level) 폭 = 1.96 × 반복 SD 중앙값(0.85°) = ±1.665°를 두었다. 이 폭은 측정 잡음 폭이지 '수평'이라는 의미 범주가 아니다.
3. **분포(train270)**:
   - p05 +0.64°, p25 +2.97°, 중앙값 +4.48°, p75 +6.10°, p95 +7.88°.
   - 0°를 넘는 사람 97.5%, +1.665°를 넘는 사람 88.7%, +8°를 넘는 사람 4.5%.
   - 학습 레이블은 down0 / level21 / up226 / unknown23이다.
4. **학습**: down logit은 학습 표본이 0개다. 클래스 가중치는 up 0.36, level 3.9, down 5(상한)였다.
5. **보정**: calibration34 중 유효 31명이 [0,1,30]이었다. NLL이 0.0526에서 0.0440으로 줄면서 T=0.370이 됐고, 기준값 0.65가 선택됐다. 사실상 한 클래스 자료로 확률을 날카롭게 만든 것이다.
6. **서비스**: 400명 중 395명이 확정됐다(up 366). 신뢰도 중앙값은 1−1e−10이다.
   - 기하평균 G는 특징 1개 조항에서 그 점수와 같다. G≈1인 조항이 사실상 모든 사람에게 생긴다.
   - 그 결과 XF01-eye-006이 선택 357/400, 1위 338/400이다(train 229, selection 32, calibration 29, test 48).
7. **문헌 의미**: 許負相法 相目篇의 나열(「目秀而長」 「龍睛鳳目」 「目如臥弓」 … 「目尾朝天，福祿綿綿」)에서 '朝天'(하늘을 향함)은 특정한 강조형 명칭이다. 일반적인 '양의 기울기'와 같지 않다.

**올바른 교정 방향**:
- up/down 비율을 억지로 맞추지 않는다.
- 연속 기울기와 서술형 방향 클래스는 유지한다.
- 별도의 사전 등록 marked 범주를 둔다. train270 개인간 분포에서 평균+2SD(+8.91°, 400명 중 1.5%)를 marked_up, 평균−2SD(+0.03°, 2.5%)를 marked_down으로 한다.
- 朝天은 marked_up의 '근사 명명형 대리'로만 검토한다. 승인 전까지는 부적격이다.
- 보정은 최소 지원 규칙으로 막는다(C2).

## 3. 공통 근본 원인과 교정

- **C1 강조형·명명형 대응.** 대응 상태를 exact / approximate / emphatic / named_type / normative(宜·欲) / sex_specific으로 나눈다. emphatic과 named_type은 사전 등록 marked 범주로만 연결하거나 부적격으로 둔다. 점수에 임의의 감점을 주지 않고, 대응 자체를 고친다.
- **C2 보정.**
  - 클래스별 최소 지원을 요구한다. 확정할 수 있는 모든 클래스가 보정 자료에 5명 이상 있어야 하고, 관측 클래스가 2개 이상이어야 한다. 모자라면 그 클래스를 보류한다.
  - train270을 인물 단위 5-fold로 나눠 out-of-fold 예측으로 T와 θ를 적합하고(n≈270), calibration34는 확인용, test62는 동결 뒤 보고용으로 쓴다.
  - T에 bootstrap 신뢰구간을 붙이고, 포화 비율(≥0.99)을 보고한다.
- **C3 분모.** 국소 부위 목표를 crop 척도인 눈중심거리로 정규화한다. eye_size는 눈 폭/눈중심거리, nose_width·mouth_width·brow_gap·nose_length·(선택) philtrum_length도 같은 방식이다. 의미가 바뀌므로 한국어 명칭과 문헌 qualifier("이 집단 대비")를 함께 갱신한다. 또는 얼굴 전체 맥락 입력을 추가한다.
- **C4 경계 보류.**
  - hard 제외 대신, 측정값과 반복 SD로 만든 Φ 구간 소속 확률을 soft target으로 쓴다. P(class)=Φ((v−b1)/s)… 형태이며 s=반복SD/√n이고, 깊이 점수와 같은 형태다.
  - landmark 특징은 회귀 head(값 + 로그분산)로 학습하고, 예측 분포에서 클래스 확률을 구한다.
  - 평가는 모든 참가자를 대상으로 한다. 연속 MAE/ICC와 class 일치를 함께 보고한다.
- **C5 정의.**
  - brow_tail은 face parser의 눈썹 mask(class 2/3) 양 끝점으로 머리와 꼬리를 정한다. 서술형 클래스와 marked_descending을 둔다.
  - lip_balance는 연속 비율로 두고, 서술형 명칭을 쓴다("아랫입술이 뚜렷이 두꺼움(흔함)/비슷함/윗입술이 두꺼움"). 여기에 marked 범주를 더한다.
  - eye_tilt와 mouth_corner는 서술형 클래스와 marked 범주를 함께 둔다.
- **C6 시각 레이블.** philtrum_definition과 under_eye는 BLOCKED로 둔다. 동률(tie) 94건은 soft target으로 유지하고, hard 지표에서 제외하거나 별도로 집계한다. 모든 지표에 κ를 함께 표시한다. soft와 hard가 엄격하게 불일치한 경우는 0건이다(main이 확인한 대로 모두 동률이다).
- **C7 문헌 정책.** 장(section) 범위 검사(여성·소아 장)를 추가하고, 민감 주제를 다시 검사한다.
  - BLOCKED: LZC63880-women_body-015, LZC63880-remaining_forms-021, LZC70286-verse-010, CX638-eye-0020.
  - 재검토: CX639-chin-0188(孤寡), ZD646A-eye-062(貪愛).
- **C8 좌우.** 측면별 목표를 쓴다. eye_size는 train 270명 중 69명이 좌우 클래스가 다르다.
- **C9 분할.** 서비스 400명에는 학습한 270명이 들어 있으므로 test62 지표만 독립 성능으로 쓴다. 이미 split별로 보고하고 있으니 유지한다.

## 4. 빈도와 오류의 구분 (오류가 아닌 것)

다음은 실제 출현 빈도이며, 그 자체로는 오류가 아니다:
- eye_tilt up 다수(평균 기울기가 원래 약간 위로 기운다)
- 입꼬리가 휴식 상태에서 중앙선보다 낮음
- 아랫입술이 윗입술보다 두꺼움
- nose_tip rounded 75%
- pigmented_marks present 다수(제한된 관찰 범위 안에서)

이런 항목의 레이블 비율을 균형 맞추거나 점수를 깎지 않는다. 바꿀 것은 대응 의미, 보정 지원, 정의, 분모다.

## 5. 확인했고 문제없는 것

- 분할 270/34/34/62가 서로 겹치지 않고 400명이 모두 고유하다.
- landmark 경계는 train270에서만 적합했다.
- 눈썹·눈 끝점 순서가 400/400 맞다.
- soft/hard 엄격 불일치 0건(동률 94건).
- 깊이 경계는 all400 기술 집단으로 명시되어 있다.
- 런타임에 PREDICTED_CLASS_NOT_TRAINED 게이트가 있다.
- 미보정 4개 head(brow_tail, lip_balance, under_eye, pigmented_marks)는 400/400 보류된다.

## 6. 특징별 판정 (CNN 41개 + VGGT 4개 = 129 class)

test62 지표는 기준 레이블이 unknown이 아닌 사람만 평가한 것이다(C4 참조). 교정 정의와 검증 계획의 전체 문구는 JSON의 `features[].corrected_definition`과 `validation_training_plan`에 있다.

| # | feature | part | 판정 | train 지원(unknown) | 보정 T / 지원 | test62 macroF1 (평가대상 n) | 서비스 확정/보류 | 핵심 근거 |
|---|---|---|---|---|---|---|---|---|
| 1 | eye_size | eye | **CHANGE** | [30, 49, 40] (151) | 2.831 / [8, 7, 5] | 0.623 (33) | 204/196 | target = eye width / FACE width (234-454) but eye crop is scaled by eye-centre distance; crop-visible eye width/eye-centre distance explains only R2=0.22 of the target (train270); single-capture ICC 0.643; middle class width = 1.99 within-subject SD; boundary guard leaves 151/270 train unknown |
| 2 | eye_roundness | eye | **PASS_QUALIFIED** | [56, 84, 45] (85) | 1.104 / [8, 14, 5] | 0.812 (44) | 351/49 | ICC 0.877; test macroF1 0.812; per-side label copied to both crops (minor) |
| 3 | eye_tilt | eye | **CHANGE** | [0, 21, 226] (23) | 0.370 / [0, 1, 30] | 0.730 (54) | 395/5 | sign/landmarks verified: + = lateral canthus (33/263) higher; 400/400 landmark order correct; centre 0 deg with "level" band = 1.96 x median repeat SD (0.85 deg) -> +/-1.665 deg: a measurement-noise band, not a semantic category |
| 4 | eyelid | eye | **PASS_QUALIFIED** | [66, 59, 145] (0) | 0.855 / [9, 7, 18] | 0.725 (62) | 175/225 | AI three-rater Fleiss kappa 0.381 (same model family raters, not independent humans); 27 three-way ties resolved by ordinal median |
| 5 | sclera_show | eye | **CHANGE** | [248, 22, 0] (0) | 0.329 / [32, 2, 0] | 0.650 (62) | 379/21 | class upper has 0 support in all splits; calibration support [32,2,0] -> T=0.329 sharpening; service median conf 0.99999 |
| 6 | eye_depth | eye | **CHANGE** | [15, 252, 3] (0) | 0.773 / [4, 29, 1] | 0.512 (62) | 399/1 | frontal 2D AI label of orbital depth, kappa 0.344; support deep15/average252/prominent3 (train); prominent 1 in calibration |
| 7 | brow_shape | brow | **PASS_QUALIFIED** | [138, 102, 5] (25) | 1.028 / [22, 11, 0] | 0.554 (58) | 225/175 | kappa 0.332; angled train5/cal0/test0 -> BLOCK angled |
| 8 | brow_thickness | brow | **PASS_QUALIFIED** | [52, 156, 43] (19) | 1.442 / [8, 19, 7] | 0.588 (60) | 168/232 | kappa 0.35; 11 ties; held 232/400 |
| 9 | brow_length | brow | **CHANGE** | [33, 40, 37] (160) | 3.233 / [5, 3, 7] | 0.646 (28) | 207/193 | arc length through MediaPipe brow points 70..107 / 46..55 divided by eye width; MediaPipe brow endpoints are not validated against actual hair extent; range all400 1.58-1.91 eye widths: 0/400 brows shorter than the eye, so literature "短不及目/眉短於目" cannot be expressed; class short = bottom quartile (<1.7186), i.e. relative |
| 10 | brow_eye_gap | brow | **CHANGE** | [27, 72, 41] (130) | 1.817 / [4, 9, 5] | 0.656 (40) | 372/28 | ICC 0.619; guard 130/270 unknown; test 11/36 confirmed disagree; upper-lid apex 159 is affected by eyelid fold type |
| 11 | brow_tail | brow | **BLOCKED** | [269, 1, 0] (0) | 미보정 / [34, 0, 0] | 1.000 (62) | 0/400 | outer-inner endpoint tilt negative for 400/400 (max -0.21 deg, median -8.84); labels descending269/level1/ascending0; uncalibrated, held 400/400; landmark order verified (70/300 lateral), so not an index swap; zero reference is non-discriminative with MediaPipe lateral brow points |
| 12 | brow_density | brow | **PASS_QUALIFIED** | [52, 143, 56] (19) | 1.198 / [11, 17, 6] | 0.707 (60) | 161/239 | kappa 0.367; 13 ties |
| 13 | brow_tail_clarity | brow | **PASS_QUALIFIED** | [94, 155] (21) | 1.042 / [13, 21] | 0.879 (58) | 194/206 | kappa 0.383; test macroF1 0.879 |
| 14 | nose_width | nose | **CHANGE** | [41, 70, 42] (117) | 2.064 / [4, 8, 7] | 0.477 (35) | 147/253 | alar width / FACE width; crop-visible R2=0.534; test macroF1 0.477 (lowest landmark head); held 253/400 |
| 15 | nose_length | nose | **CHANGE** | [43, 96, 48] (83) | 0.873 / [3, 12, 7] | 0.580 (43) | 130/270 | nasion(168)-subnasale(2) / face height (10-152); crop-visible R2=0.403; service confirmed long 96/130 vs base 25%; medium 76/94 train references held |
| 16 | nose_tip | nose | **PASS_QUALIFIED** | [203, 67] (0) | 0.403 / [25, 9] | 0.834 (62) | 285/115 | kappa 0.384; rounded 75% train (frequency, not error); T=0.403 on [25,9] |
| 17 | nostril_show | nose | **CHANGE** | [9, 155, 106] (0) | 0.739 / [2, 22, 10] | 0.449 (62) | 299/101 | kappa 0.313; test macroF1 0.449; frontal nostril visibility depends strongly on head pitch; quality gate allows pitch -5+/-16 deg |
| 18 | alar_fullness | nose | **PASS_QUALIFIED** | [20, 164, 86] (0) | 0.612 / [0, 19, 15] | 0.708 (62) | 380/20 | kappa 0.412; calibration flat=0 |
| 19 | mouth_width | mouth | **CHANGE** | [48, 92, 52] (78) | 1.617 / [5, 11, 8] | 0.731 (44) | 107/293 | commissure width / FACE width; crop-visible R2=0.691; medium 85/92 train references held; held 293/400 |
| 20 | lip_thickness | mouth | **PASS** | [39, 75, 47] (109) | 1.353 / [6, 10, 3] | 0.881 (35) | 365/35 | ICC 0.810; denominator mouth width visible; test macroF1 0.881 |
| 21 | lip_balance | mouth | **CHANGE** | [214, 10, 0] (46) | 미보정 / [30, 0, 0] | 0.323 (51) | 0/400 | centre fixed at 1.0 but train median upper/lower = 0.786 (p95 0.933): upper_thin214/balanced10/upper_full0; class name "upper_thin" frames the typical morphology as deviant; uncalibrated, held 400/400 |
| 22 | mouth_corner | mouth | **CHANGE** | [172, 44, 19] (35) | 1.093 / [14, 11, 4] | 0.832 (49) | 393/7 | measurement reliable (ICC 0.917); down = corners >1.86% of mouth width below inner-lip midpoint (noise band) -> 64% of known train; EoM "downturned corners" is subjective with no magnitude |
| 23 | lip_bow | mouth | **PASS_QUALIFIED** | [23, 181, 66] (0) | 0.979 / [1, 28, 5] | 0.568 (62) | 339/61 | kappa 0.299 (low); test macroF1 0.568; no eligible literature |
| 24 | eye_asymmetry | face | **BLOCKED** | [166, 30] (74) | 4.167 / [18, 6] | 0.570 (47) | 357/43 | /log(roundness_R/roundness_L)/ top ~15% cut 0.046 (4.7% difference); ICC 0.413 (between SD 0.018 < within SD 0.022) |
| 25 | face_shape | face | **PASS_QUALIFIED** | [87, 72, 60, 26] (25) | 1.141 / [9, 12, 6, 3] | 0.712 (57) | 230/170 | kappa 0.388; 37 uncertain excluded |
| 26 | face_length | face | **PASS_QUALIFIED** | [49, 85, 48] (88) | 2.525 / [7, 9, 6] | 0.727 (50) | 144/256 | landmark 10 is mesh top, not hairline (already labelled 헤어라인 제외); ICC 0.858 |
| 27 | jaw_width | face | **PASS** | [55, 104, 58] (53) | 2.713 / [7, 10, 8] | 0.715 (45) | 303/97 | ICC 0.920; face crop shows both landmarks |
| 28 | eye_spacing | face | **PASS_QUALIFIED** | [40, 55, 36] (139) | 2.378 / [4, 10, 7] | 0.627 (31) | 123/277 | ICC 0.666; guard 139/270 unknown; held 277/400 |
| 29 | chin_shape | face | **PASS_QUALIFIED** | [103, 87, 68] (12) | 1.576 / [9, 12, 10] | 0.702 (57) | 167/233 | kappa 0.489 |
| 30 | lower_face_ratio | face | **PASS_QUALIFIED** | [50, 99, 52] (69) | 1.480 / [7, 18, 3] | 0.664 (45) | 309/91 | subnasale-menton / nasion-subnasale; classical 三停 uses 準頭 and brow, not subnasale/nasion; ICC 0.877 |
| 31 | forehead_height | forehead | **PASS_QUALIFIED** | [59, 58, 59] (94) | 2.474 / [7, 8, 6] | 0.725 (47) | 231/169 | visible hairline only (bangs/hat -> unknown); tertiles without boundary guard; 114 LABEL_UNAVAILABLE + 62 hairline holds |
| 32 | forehead_width | forehead | **PASS_QUALIFIED** | [56, 59, 63] (92) | 2.114 / [5, 9, 7] | 0.732 (44) | 258/142 | visible skin width (temporal hair covers); tertiles without boundary guard; 3 passages |
| 33 | brow_gap | glabella | **CHANGE** | [48, 79, 49] (94) | 2.345 / [5, 11, 9] | 0.583 (34) | 43/357 | inner-brow distance / FACE width; glabella crop 112px wide; crop-visible R2=0.249; held 357/400 |
| 34 | under_eye | under_eye | **BLOCKED** | [264, 6] (0) | 미보정 / [34, 0] | 0.492 (62) | 0/400 | single-pass AI visual v3, 8 positives in 400 (train6/sel0/cal0/test2); no kappa; uncalibrated |
| 35 | cheekbone | cheekbone | **PASS_QUALIFIED** | [150, 120] (0) | 0.817 / [21, 13] | 0.750 (62) | 264/136 | kappa 0.454; frontal 2D proxy of zygomatic prominence |
| 36 | cheek_fullness | cheekbone | **PASS_QUALIFIED** | [36, 152, 82] (0) | 0.552 / [2, 22, 10] | 0.599 (62) | 273/127 | kappa 0.412; T=0.552 |
| 37 | ear_length | ear | **CHANGE** | [87, 87, 87] (9) | 0.353 / [11, 11, 12] | 0.755 (59) | 334/66 | weak-perspective ear-mask height / (eye distance/cos yaw); tertile cuts medium band 0.914-0.983 (7%) while repeat acceptance tolerance is 15% relative range; service confirmed short 131/334; T=0.353 (n=34) |
| 38 | ear_width | ear | **CHANGE** | [87, 86, 87] (10) | 0.967 / [12, 8, 14] | 0.646 (57) | 138/262 | mask width/height; same guard issue; service confirmed wide 107/138 vs tertile base; narrow/medium mostly held |
| 39 | philtrum_length | philtrum | **PASS_QUALIFIED** | [42, 82, 43] (103) | 3.382 / [2, 12, 7] | 0.817 (45) | 200/200 | face-height denominator but crop-visible R2=0.869; ICC 0.849; T=3.38 |
| 40 | philtrum_definition | philtrum | **BLOCKED** | [141, 51] (78) | 1.153 / [21, 2] | 0.774 (40) | 253/147 | AI Fleiss kappa 0.173 with recorded decision "withhold", re-included in release registry; 7 eligible passages (深/淺) depend on it; 2D shading proxy of groove depth |
| 41 | pigmented_marks | mole | **BLOCKED** | [4, 50] (216) | 미보정 / [0, 6] | 0.448 (16) | 0/400 | VLM+CV agreement: absent train4/cal0; uncalibrated, held 400/400; literature 痣/黑子 mapping intentionally unmapped |
| 42 | nose_projection | nose | **PASS_QUALIFIED** | all400 P25/P75 | σ=0.0123 | ICC 0.896, 3조명일치 0.65 | 387/13 | VGGT relative ratio, quartile boundaries on all400 (user-approved descriptive cohort, includes test), sigma from 26 train repeat people; ICC 0.896; all-three-lights label agreement 0.650 |
| 43 | nose_bridge | nose | **PASS_QUALIFIED** | all400 P25/P75 | σ=0.0066 | ICC 0.889, 3조명일치 0.62 | 387/13 | VGGT relative ratio, quartile boundaries on all400 (user-approved descriptive cohort, includes test), sigma from 26 train repeat people; ICC 0.889; all-three-lights label agreement 0.625 |
| 44 | lip_projection | mouth | **PASS_QUALIFIED** | all400 P25/P75 | σ=0.0065 | ICC 0.914, 3조명일치 0.70 | 385/15 | VGGT relative ratio, quartile boundaries on all400 (user-approved descriptive cohort, includes test), sigma from 26 train repeat people; ICC 0.914; all-three-lights label agreement 0.700 |
| 45 | chin_projection | face | **PASS_QUALIFIED** | all400 P25/P75 | σ=0.0106 | ICC 0.948, 3조명일치 0.75 | 387/13 | VGGT relative ratio, quartile boundaries on all400 (user-approved descriptive cohort, includes test), sigma from 26 train repeat people; ICC 0.948; all-three-lights label agreement 0.750 |


### 6.1 주요 교정 정의 (CHANGE/BLOCKED 특징)

- **eye_tilt**: 연속 각도(°)를 쓰고, 정렬 기준선은 안쪽 눈구석선(133–362, Farkas/EoM)으로 한다. 서술형 down/level/up은 유지한다. 별도로 marked_up(평균+2SD 초과)과 marked_down(평균−2SD 미만)을 사전 등록한다.
- **eye_size, nose_width, nose_length, mouth_width, brow_gap**: 분모를 눈중심거리로 바꾼다. eye_size는 측면별로 정의한다.
- **brow_length**: parser 눈썹 mask 양 끝 길이/눈 폭을 측면별로 쓴다. '눈보다 짧음' 불리언을 따로 보존한다. 0/400명이므로 지금은 해당 문헌을 쓸 수 없다.
- **brow_eye_gap**: 측면별 값으로 바꾸고, 눈썹 mask 하단 사용을 검토한다.
- **brow_tail**: mask 끝점 기준 꼬리 하강량/눈 폭으로 정의하고 marked_descending을 둔다. 검증 전에는 BLOCKED다.
- **lip_balance**: 서술형 명칭과 marked 범주로 바꾸고, 강제 균형은 하지 않는다.
- **mouth_corner**: 측정은 유지하고(ICC .917) marked_down을 추가한다. 「下垂」 조항은 marked_down에만 연결한다.
- **sclera_show**: upper를 보류하고 보정 최소 지원 규칙을 적용한다.
- **eye_depth**: prominent를 BLOCKED로 둔다(train 3명). deep은 qualifier를 붙여 유지한다.
- **brow_shape**: angled를 보류한다(train 5, cal/test 0).
- **nostril_show**: pitch 공변량을 쓰거나 이 head만 pitch gate를 좁힌다. test를 pitch 층으로 나눠 보고한다.
- **ear_length / ear_width, forehead_***: 반복 SD 기반 경계 보류를 적용한다(지금은 tertile만 있고 보류 폭이 없다). 반복 허용 15%가 ear_length 보통 폭 7%보다 넓다.
- **eye_asymmetry**: ICC .41이므로 classifier 재학습을 보류한다.
- **philtrum_definition / under_eye / pigmented_marks**: BLOCKED를 유지한다.

## 7. 적격 73조항 대응 판정 (원문·문맥 직독 + 기계 행렬)

괄호 안 숫자는 train270의 known 레이블 중 해당 클래스 비율이다. VGGT는 정의상 0.25다.
- **의미 판정**: 원문과 레이블 정의 사이의 의미 대응만 본다.
- **다음 release 서비스 판정**: 의존하는 특징의 CHANGE/BLOCKED를 반영한다. 특징이 고쳐지기 전에는 서비스 적격이 아니라는 뜻이다.
- 고친 뒤에도 approximate 대응을 서비스할지는 정책 결정이다. 현재 corpus 정책은 exact만 허용하므로, 엄격히 적용하면 approximate인 PASS_QUALIFIED 조항도 "근사 대응" 표시 정책을 승인해야 서비스할 수 있다.

| # | passage | 원문 | 요구 레이블 (train 비율) | 선택/1위 (all400) | 의미 판정 | 정확도 | 다음 release 서비스 | 근거 |
|---|---|---|---|---|---|---|---|---|
| 1 | SQ632A-brow-005 | 《短促》兄弟非宜。 | brow_length:short (0.3) | 14/0 | CHANGE | approximate | **CHANGE** | context 《廣鑑集》「若短不及目者，難為兄弟」 = absolute (shorter than eye); brow_length:short is bottom quartile of 1.58-1.91 eye widths |
| 2 | SQ632A-brow-016 | 郭林宗曰：「眉如新月樣，名譽四方聞。」 | brow_shape:arched (0.416) | 42/0 | PASS_QUALIFIED | approximate_named_type | **PASS_QUALIFIED** | 眉如新月 crescent named type ~ arched (42% of known train); kappa 0.33 |
| 3 | SQ632B-eye-009 | 眼大者多攻藝業。 | eye_size:large (0.336) | 20/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眼大 overall eye size; label is horizontal width only; relative population class; display "이 집단 대비" |
| 4 | SQ632B-eye-026 | 《月波洞中經》云：「莫交眼突，往往見災迍。」 | eye_depth:prominent (0.011) | 1/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眼突 ~ prominent; feature class prominent BLOCKED (train 3) |
| 5 | SQ632B-nose-041 | 昂露鼻孔：仰孔為二庫之門，十分昂露者，謂戶門開闊，內無積也，難為平生不利，五九大 | nostril_show:visible (0.393) | 55/2 | CHANGE | approximate_emphatic | **CHANGE** | 「十分昂露」 extreme upturned/exposed nostrils mapped to generic visible (39% train) |
| 6 | SQ632B-nose-046 | 《照膽經》云：「鼻孔外仰成惡敗。」 | nostril_show:visible (0.393) | 23/0 | PASS_QUALIFIED | approximate | **CHANGE** | 鼻孔外仰 -> nostrils visible frontally; pitch confound |
| 7 | SQ632C-mouth-043 | 親曾見，低垂兩角，常被世人嫌。 | mouth_corner:down (0.732) | 222/10 | CHANGE | approximate_emphatic | **CHANGE** | 低垂兩角 (drooping corners) -> generic down (64% known train); needs marked_down |
| 8 | SQ632C-mouth-044 | 兩角下垂者，無衣食也。最招人憎嫌。 | mouth_corner:down (0.732) | 202/0 | CHANGE | approximate_emphatic | **CHANGE** | 兩角下垂 same as above |
| 9 | SQ632C-mouth-052 | 口小為五極，無有盛衣食。 | mouth_width:narrow (0.25) | 15/0 | PASS_QUALIFIED | approximate | **CHANGE** | 口小 in 《墳經》 五極 list -> narrow mouth; relative population class; display "이 집단 대비" |
| 10 | SQ632C-chin-004 | 下停長，少吉祥， | lower_face_ratio:long (0.259) | 9/1 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 下停長 in 三停訣; metric uses subnasale/nasion not 準頭/眉; reading of 少吉祥 ambiguous (already noted) |
| 11 | SQ633FB-face-020 | 面欲長而方， | face_length:long (0.264), face_shape:angular (0.245) | 0/0 | PASS_QUALIFIED | approximate_normative | **PASS_QUALIFIED** | 面欲長而方 is a normative "should be" statement, not if-then; 方 ~ angular |
| 12 | SQ633FB-face-025 | 下停長者無下梢。」 | lower_face_ratio:long (0.259) | 5/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 下停長者無下梢; same 三停 qualifier |
| 13 | SQ633FB-brow-009 | 尾垂眼者性懦。 | brow_tail:descending (0.996) | 0/0 | CHANGE | approximate_emphatic | **BLOCKED** | 尾垂眼 (tail drooping onto eye); feature brow_tail degenerate (all descending) |
| 14 | SQ633FB-brow-036 | 眉若高直。身當清職。 | brow_eye_gap:wide (0.293), brow_shape:straight (0.563) | 1/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眉若高直; 高 ambiguous (high-set vs raised); straight ok |
| 15 | SQ633FB-brow-059 | 眉短家無兄弟真， | brow_length:short (0.3) | 9/0 | CHANGE | approximate | **CHANGE** | 眉短 in verse contrasting 「眉長過眼」「不過兩目」 -> eye-relative absolute criterion |
| 16 | SQ633E-eye-036 | 目短眉長，愈益田莊。 | brow_length:long (0.336), eye_size:small (0.252) | 2/0 | PASS_QUALIFIED | approximate | **CHANGE** | 目短眉長; note brow_length uses eye width denominator, so the two conditions are mechanically coupled |
| 17 | SQ633E-eye-045 | 目尾相垂，夫妻相離。」 | eye_tilt:down (0.0) | 0/0 | CHANGE | approximate_emphatic | **CHANGE** | 目尾相垂 -> eye_tilt:down (0 support); needs marked_down definition |
| 18 | SQ633E-eye-051 | 細深多是無心腹， | eye_depth:deep (0.056), eye_roundness:narrow (0.303) | 2/0 | PASS_QUALIFIED | approximate | **CHANGE** | 細深 -> narrow + deep (deep train 15) |
| 19 | SQ633E-eye-059 | 眼深定是乏資糧， | eye_depth:deep (0.056) | 4/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眼深 -> deep |
| 20 | SQ634N-nose-001 | 「山根為孛星」，鼻梁上也。宜高不宜低折。 | nose_bridge:high (0.25) | 49/13 | PASS_QUALIFIED | approximate_normative | **PASS_QUALIFIED** | 山根宜高不宜低折 normative; VGGT top quartile relative ratio |
| 21 | SQ634N-nose-035 | 準頭圓肥，足食豐衣。 | nose_tip:rounded (0.752) | 159/28 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 準頭圓肥 -> rounded (75% train: frequency not error) |
| 22 | SQ634N-nose-045 | 孔仰家無隔宿糧。 | nostril_show:visible (0.393) | 10/0 | PASS_QUALIFIED | approximate | **CHANGE** | 孔仰 -> visible; pitch confound |
| 23 | SQ634M-mouth-116 | 脣厚少語 | lip_thickness:full (0.292) | 52/1 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 脣厚 -> full; relative population class; display "이 집단 대비" |
| 24 | SQ634M-mouth-117 | 薄多訟，依此言之，定不差。 | lip_thickness:thin (0.242) | 19/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 薄 -> thin; relative population class; display "이 집단 대비" |
| 25 | SQ637-nose-085 | 鼻若仰兮竅不祥，〈鼻仰孔露也〉 | nostril_show:visible (0.393) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 鼻仰 glossed 〈鼻仰孔露也〉 -> visible |
| 26 | SQ637Y-brow-024 | 眉粗眼細不相當，寅年吃了卯年糧， | brow_thickness:thick (0.171), eye_roundness:narrow (0.303) | 1/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 眉粗 (coarse/thick ambiguity) + 眼細 |
| 27 | SQ637RA-brow-092 | 眉尾毫毛脫落而疏稀，主財物破散。初運二十六至二十九，財散 | brow_tail_clarity:fading (0.622) | 35/0 | PASS | close | **PASS** | 眉尾毫毛脫落而疏稀 -> tail fading |
| 28 | SQ638CX-nose-009a | 「《準頭》隆」者誠信， | nose_projection:high (0.25) | 49/4 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 準頭隆 (raised/full) -> VGGT tip projection high quartile |
| 29 | SQ638CX-nose-009c | 夫準頭者，為面部之標本。準頭高隆，其人誠懇而篤信 | nose_projection:high (0.25) | 38/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 準頭高隆 -> projection high |
| 30 | SQ638CX-philtrum-013a | 夫《人中》者，溝洫之態，深則疏通， | philtrum_definition:clear (0.266) | 10/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 人中深/淺 groove depth -> 2D visual definition proxy; feature philtrum_definition BLOCKED (kappa 0.173) |
| 31 | SQ638CX-philtrum-013b | 淺則遲延。 | philtrum_definition:faint (0.734) | 57/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 人中深/淺 groove depth -> 2D visual definition proxy; feature philtrum_definition BLOCKED (kappa 0.173) |
| 32 | SQ638CX-philtrum-013d | 深則必致亨通， | philtrum_definition:clear (0.266) | 4/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 人中深/淺 groove depth -> 2D visual definition proxy; feature philtrum_definition BLOCKED (kappa 0.173) |
| 33 | SQ638CX-philtrum-013e | 淺則應當困滯。 | philtrum_definition:faint (0.734) | 40/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 人中深/淺 groove depth -> 2D visual definition proxy; feature philtrum_definition BLOCKED (kappa 0.173) |
| 34 | SQ638CX-philtrum-017a | 深長者誠信著， | philtrum_definition:clear (0.266), philtrum_length:long (0.257) | 0/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 深長 -> clear + long; philtrum_definition BLOCKED |
| 35 | CX638-brow-0205 | 素無兄弟眉粗短， | brow_length:short (0.3), brow_thickness:thick (0.171) | 0/0 | CHANGE | approximate | **CHANGE** | 眉粗短: short is relative quartile, literature 短 eye-relative |
| 36 | CX638-eye-0020 | 兩眼雌雄，豈由仁由義」之子。 | eye_asymmetry:asymmetric (0.153) | 1/0 | BLOCKED | approximate_named_type | **BLOCKED** | 兩眼雌雄 named type (obvious disparity) vs 4.7% opening-ratio top cut with ICC 0.41; claim 「豈由仁由義之子」 is a moral-character judgement (sensitive malice/character) |
| 37 | CX638-brow-0023 | 眉如一字，豈能潤屋肥家， | brow_shape:straight (0.563) | 35/0 | PASS_QUALIFIED | approximate_named_type | **PASS_QUALIFIED** | 眉如一字 named type ~ straight (and horizontal) |
| 38 | CX638-brow-0102 | 濃長必豐資之相； | brow_density:dense (0.223), brow_length:long (0.336) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 濃長 -> dense + long |
| 39 | CX638-brow-0104 | 細長定慳劣之徒。 | brow_length:long (0.336), brow_thickness:thin (0.207) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 細長 -> thin + long |
| 40 | CX638-brow-0058 | 眉粗眼小福薄， | brow_thickness:thick (0.171), eye_size:small (0.252) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眉粗眼小 -> thick + small |
| 41 | CX638-mouth-0069 | 口角下垂無祿。 | mouth_corner:down (0.732) | 170/0 | CHANGE | approximate_emphatic | **CHANGE** | 口角下垂 -> generic down |
| 42 | SQ639R-nose-011 | 孔昂露而《甑釜》塵。 | nostril_show:visible (0.393) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 孔昂露 -> visible |
| 43 | CX639-chin-0188 | 頤尖額窄，但為孤寡之徒。 | chin_shape:pointed (0.399), forehead_width:narrow (0.315) | 3/0 | CHANGE | approximate | **CHANGE** | 頤尖額窄 但為孤寡之徒: 孤寡 (orphaned/widowed) is kinship-harm adjacent -> sensitive-policy review; 頤 = jaw region not only chin tip |
| 44 | CX639-glabella-0286 | 碧油紅斾，因印堂廣而封衛； | brow_gap:wide (0.278) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 印堂廣 -> brow_gap wide (feature denominator CHANGE) |
| 45 | CX639-brow-0308 | 眉濃多滯， | brow_density:dense (0.223) | 10/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 眉濃 -> dense |
| 46 | CX639-brow-0215 | 眉高則名高， | brow_eye_gap:wide (0.293) | 21/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眉高 -> brow-eye gap wide |
| 47 | CX639-cheekbone-0555 | 顴骨高起眼帶深，終是作猿形。 | cheekbone:prominent (0.444), eye_depth:deep (0.056) | 2/0 | PASS_QUALIFIED | approximate_named_type | **CHANGE** | 顴骨高起眼帶深 -> 猿形 animal-form gestalt; display as form-type description |
| 48 | CX639-nose-0117 | 何知幼小災厄故，但看鼻頭井竈露， | nostril_show:visible (0.393) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 井竈露 -> visible |
| 49 | CX639-eye-0147 | 何知主有美貌妻，看他兩眼長而細， | eye_roundness:narrow (0.303), eye_size:large (0.336) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 兩眼長而細 -> large width + narrow (coupled ratios) |
| 50 | SQ642F-forehead-059 | 額寬終是貴， | forehead_width:wide (0.354) | 39/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 額寬 -> visible forehead width wide |
| 51 | XF01-eye-006 | 目尾朝天，福祿綿綿。 | eye_tilt:up (0.915) | 357/338 | CHANGE | approximate_emphatic | **CHANGE** | 許負相法 相目篇 「目尾朝天，福祿綿綿」 (in list with 目秀而長/龍睛鳳目) = eye tail pointing to the sky, emphatic; mapped to generic up (88.7% of people); rank1 338/400. Ineligible until a pre-registered marked_up proxy is approved |
| 52 | XF02-nose-017 | 鼻相廣長，必多伎倆。 | nose_length:long (0.257), nose_width:wide (0.275) | 1/0 | PASS_QUALIFIED | approximate | **CHANGE** | 鼻相廣長 -> wide + long |
| 53 | TQ5-brow-023 | 眉如彎弓性善不雄 | brow_shape:arched (0.416) | 20/0 | PASS_QUALIFIED | approximate_named_type | **PASS_QUALIFIED** | 眉如彎弓 -> arched |
| 54 | ZD645-chin-035 | 《地閣》尖不得產業力。 | chin_shape:pointed (0.399) | 11/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 地閣尖 -> pointed |
| 55 | ZD646A-eye-038 | 眼尾兩邊垂，一生多別離。 | eye_tilt:down (0.0) | 0/0 | CHANGE | approximate_emphatic | **CHANGE** | 眼尾兩邊垂 -> eye_tilt:down (0 support); needs marked_down |
| 56 | ZD646A-eye-062 | 目大多貪愛， | eye_size:large (0.336) | 11/0 | PASS_QUALIFIED | approximate | **CHANGE** | 目大多貪愛; 貪愛 (craving) sensitivity review recommended |
| 57 | ZD646A-eye-069 | 目小終無祿。 | eye_size:small (0.252) | 12/0 | PASS_QUALIFIED | approximate | **CHANGE** | 目小 -> small |
| 58 | ZD646A-nose-010 | 鼻廣而長，伎藝非常。 | nose_length:long (0.257), nose_width:wide (0.275) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 鼻廣而長 -> wide + long |
| 59 | ZD646B-philtrum-004 | 人中淺，多破財； | philtrum_definition:faint (0.734) | 32/0 | PASS_QUALIFIED | approximate_proxy | **BLOCKED** | 人中深/淺 groove depth -> 2D visual definition proxy; feature philtrum_definition BLOCKED (kappa 0.173) |
| 60 | ZD646B-brow-009 | 眉疏而散，暫富終貧。 | brow_density:sparse (0.207) | 7/0 | PASS | close | **PASS** | 眉疏而散 -> sparse |
| 61 | LZC63880-women_body-015 | 眉散又疎手足如同陌路 | brow_density:sparse (0.207) | 6/0 | BLOCKED | qualifier_missing_sex | **BLOCKED** | section 柳庄進永樂乾坤賦〈專論女人之說〉 (women only); sex is an unobservable condition under the corpus own eligibility policy |
| 62 | LZC63880-remaining_forms-021 | 溝洫又云人中最要長深 | philtrum_definition:clear (0.266), philtrum_length:long (0.257) | 0/0 | BLOCKED | qualifier_missing_sex | **BLOCKED** | same women-only section; also philtrum_definition BLOCKED |
| 63 | LZC63880-palace-012 | 莫教孔仰主無隔宿之糧 | nostril_show:visible (0.393) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 莫教孔仰 -> visible |
| 64 | LZC63880-palace-016 | 形如新月和睦永遠超羣 | brow_shape:arched (0.416) | 13/0 | PASS_QUALIFIED | approximate_named_type | **PASS_QUALIFIED** | 形如新月 -> arched |
| 65 | LZC63880-palace-017 | 若是短粗手足難同定別 | brow_length:short (0.3), brow_thickness:thick (0.171) | 0/0 | CHANGE | approximate | **CHANGE** | 短粗; same section says 「眉長過目」 -> eye-relative criterion |
| 66 | LZC63880-palace-020 | 頭高尾低兄弟不如 | brow_tail:descending (0.996) | 0/0 | CHANGE | approximate | **BLOCKED** | 頭高尾低; brow_tail degenerate |
| 67 | LZC63880-palace-034 | 頦圓額窄應知苦在初年 | chin_shape:rounded (0.337), forehead_width:narrow (0.315) | 3/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 頦圓額窄 -> rounded chin + narrow forehead |
| 68 | LZC70286-questions_a-007 | 下停若長一生蹇滯 | lower_face_ratio:long (0.259) | 3/0 | PASS_QUALIFIED | approximate | **PASS_QUALIFIED** | 下停若長 in 三停 Q&A (上停中停俱長下停宜短) |
| 69 | LZC70286-questions_b-020 | 口小者一世無糧 | mouth_width:narrow (0.25) | 10/0 | PASS_QUALIFIED | approximate | **CHANGE** | 口小 -> narrow |
| 70 | LZC70286-questions_a-035 | 眉分八字可作軍官 | brow_tail:descending (0.996) | 0/0 | CHANGE | approximate_named_type | **BLOCKED** | 眉分八字 named 八字眉 (marked down-splay) -> degenerate brow_tail:descending |
| 71 | LZC70286-verse-010 | 顴高反好爲非 | cheekbone:prominent (0.444) | 64/2 | BLOCKED | sensitive_claim | **BLOCKED** | 顴高反好爲非 = "likes to do wrong": malice/wrongdoing claim excluded by corpus own sensitive-topic policy |
| 72 | LZC70286-verse-012 | 眉低眼陷一世辛勤 | brow_eye_gap:close (0.193), eye_depth:deep (0.056) | 3/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眉低眼陷 -> close + deep |
| 73 | LZC70286-secret-004 | 眼大常招陰人口舌 | eye_size:large (0.336) | 0/0 | PASS_QUALIFIED | approximate | **CHANGE** | 眼大 -> large |

## 8. 레이블 재생성·전체 ResNet18 재학습 계획 (main 2차 검토 뒤 실행)

0. **보존**: 기존 training_frozen/inference_frozen/deployment/test_metrics/registry를 바꾸지 않는다. 새 release 디렉터리와 새 registry 버전을 만든다.
1. **사전 등록**: 시험 전에 정책 JSON으로 동결한다.
   - 특징 정의, 분모, marked 기준(±2SD), 최소 보정 지원(클래스당 5명 이상), fold 구성, 승인 기준.
   - 문헌 대응 taxonomy와 각 조항의 상태.
2. **레이블 재생성 (CPU)**:
   - landmark 19개: 측면별 연속값, 새 분모, 서술형 클래스 + marked 범주, Φ soft target.
   - brow mask 끝점 길이·꼬리와 MediaPipe의 일치율을 train270에서 보고한다.
   - 확장 4개 target에 경계 보류를 적용한다.
   - 시각 레이블은 soft로 유지하고 동률을 표시한다.
   - BLOCKED 항목은 학습 목표에서 빼거나 보류 head로 둔다.
3. **학습 (GPU, main만)**:
   - ResNet18 12부위로, 시각 항목은 분류 head, 기하 항목은 회귀 head(μ, log s²)로 학습한다.
   - train270 전 조명, 인물 단위 5-fold로 out-of-fold 예측을 만든다. selection34는 checkpoint 선택용이다.
   - 최종 모델은 fold 앙상블 또는 전체 재학습 중 하나를 사전에 정한다.
4. **보정**: OOF(train270)로 T와 θ를 적합하고 calibration34로 확인한다. 최소 지원에 못 미치는 클래스는 확정하지 않는다. 포화 비율과 T의 신뢰구간을 보고한다.
5. **시험 (test62, 동결 뒤 한 번)**:
   - 모든 참가자에 대해 연속 MAE/ICC, class 일치, 보류율, ECE를 보고한다. known-only 지표는 별도로 낸다.
   - 반복 40명의 L1/L2/L3 일관성도 보고한다.
6. **문헌 재빌드와 감사**:
   - 대응 taxonomy를 적용한다. 여성 전용 장과 민감 조항을 제외하고, emphatic 조항은 marked 범주로만 연결한다.
   - train 기준 레이블로 조항별 발화율(firing rate)을 사전 감사한다. 기준은 "강조형인데 발화율 > 20%면 의미를 재검토"이며, 점수 감점이 아니라 대응을 재검토하는 기준이다.
7. **회귀 검사**:
   - XF01-eye-006과 같은 "단일 조항 1위 독점"을 진단한다. test62에서 조항별 1위 분포를 보되, 기준값으로 선택에 쓰지 않는다.
   - 실측 400명 HTTP를 다시 실행하고 split별로 보고한다.

## 9. 외부 근거

- NIH Elements of Morphology, Palpebral Fissure, Upslanted: objective "more than 2 SD above the mean for age"; comment는 Farkas 1994 inclination 정의. https://elementsofmorphology.nih.gov/index.cgi?tid=54784a7f792013f6db1c0419a5db1c0a
- 같은 사이트, Downslanted: "Note that the mean inclination is slightly upslanting at all ages"; norms가 제한적이라는 언급. https://elementsofmorphology.nih.gov/index.cgi?tid=bfab2c5cc87959adc34dbef8d5cc98aa
- Mouth, Downturned Corners of: subjective 정의만 있고 객관 기준이 없다. https://elementsofmorphology.nih.gov/index.cgi?tid=5c723b1624611f01765e9a80de6b5b31
- Mouth, Narrow: objective "more than 2 SD below the mean". https://elementsofmorphology.nih.gov/index.cgi?tid=10b0833efede21449fd514c4267cdf57
- Hall BD, Graham JM Jr, Cassidy SB, Opitz JM. Elements of morphology: standard terminology for the periorbital region. Am J Med Genet A 2009;149A:29-39.
- MediaPipe 눈썹 끝점이 해부학적 정답과 맞는지 검증한 연구는 찾지 못했다. 그래서 parser mask와의 교차 검증을 제안한다.

## 10. main 2차 검토 요청 사항

1. marked ±2SD 기준과 기준 집단(train270)을 채택할지 결정해 주세요.
2. approximate 대응을 "근사" 표시와 함께 서비스할지, 아니면 exact만 허용할지 정책을 결정해 주세요.
3. 분모를 바꿔 의미를 교정할지(눈중심거리), 아니면 얼굴 맥락 입력을 추가할지 정해 주세요.
4. OOF 5-fold 보정에 드는 GPU 비용을 승인할지 정해 주세요.
5. brow mask 기반 재정의의 실현 가능성을 검토해 주세요. 사용할 parser 클래스(눈썹 2/3)가 C7 정면에 모두 있는지 확인이 필요합니다.
6. LZC63880 두 조항, LZC70286-verse-010, CX638-eye-0020의 제외를 확인해 주세요.
