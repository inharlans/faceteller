# 얼굴 형태 연구 작업대

실제 FRLL 얼굴 사진으로 **데이터 확보 → 부위 추출 → 육안 형태 레이블 → 부위별 CNN → 사진 분석**을 실행한 로컬 연구용 프로젝트입니다. 현재 상용 관상 서비스나 전문가 정답 데이터셋이 완성된 것은 아닙니다.

## 지금 사용하기

- 웹 작업대: <http://127.0.0.1:8765>
- 실행: PowerShell에서 `./start-workbench.ps1`
- 검수 엑셀: `outputs/face-dataset-v1/얼굴형태_데이터셋_검수.xlsx`
- 구축·조사·제약 보고서: `docs/구축보고서.md`
- 학습 및 확장: `docs/학습과확장.md`

원본 204장(102명), 부위 이미지 1,428개, 무표정 부위 판정 714건 중 사용 가능한 외형 레이블 706건을 포함합니다. 8건은 가림 때문에 `unknown`입니다. 웃는 사진의 714개 부위는 표정 비교용으로 분리했습니다. 문헌 사전은 눈썹 24·눈 39·코 24·입 16개, 총 103항목입니다.

**레이블 출처는 AI 단일 육안 검수자입니다. 인간 전문가의 독립 검수는 0건입니다.** 자동 랜드마크 측정값과 육안 레이블은 분리되어 있습니다. 문헌 이름을 그대로 103개 CNN 정답 클래스로 사용하지 않았습니다.

## 파일 구조

```text
data/raw/frll/                공식 ZIP·메타데이터·JPEG·189점 템플릿
data/sources/                 원문 페이지와 SHA256
data/annotations/visual_v1.tsv 사진을 보고 작성한 변경 전 1차 레이블
data/annotations/overrides.json 사용자가 수정한 내용 (첫 저장 때 생성)
data/label_schema.json        외형 클래스 정의와 한계
data/literature_morphology.tsv 문헌별 한국어 형태 설명
data/literature_links.json    현대 외형과 문헌의 부분적 표현 연결
data/derived/                이미지·부위·측정·주석·학습 CSV/JSON
data/derived/classified/      버전별 부위/분할/레이블 이미지 폴더
models/pilot/                5개 CNN 체크포인트·평가·오분류 목록
outputs/face-dataset-v1/      검수 엑셀·데이터 검증·보고 자료
facework/                   특징 추출·CNN 학습·API
web/index.html              로컬 검수/분석 UI
```

사진과 모델은 Git에서 제외했습니다. 파일을 지운 경우 `scripts/download_frll.py`, `scripts/prepare_data.py`, `scripts/finalize_labels.py` 순으로 복원할 수 있습니다. 이 폴더 전체를 다른 환경으로 옮기려면 `.venv`를 복사하기보다 Python 3.12 환경을 새로 설치하세요.

## 검수와 저장

웹 화면에서 부위별로 레이블을 선택하고 **저장**하면 JSON과 CSV에 반영됩니다. 변경 전 AI 레이블은 TSV에 유지하며 수정 이력은 별도 JSONL에 기록합니다. 동시에 같은 행을 수정한 경우 오래된 저장 요청을 거절합니다.

엑셀은 생성 당시의 스냅샷입니다. `레이블 검수` 시트의 노란 칸에 수정 레이블, 검수자, 메모를 기입한 뒤 가져올 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/import_workbook.py "수정본.xlsx"
.\.venv\Scripts\python.exe scripts/import_workbook.py "수정본.xlsx" --apply
```

첫 명령은 검증만 하며, 두 번째는 실행 중인 로컬 API로 수정 내용을 저장합니다. 레이블을 바꿔도 학습된 가중치는 자동으로 바뀌지 않습니다. 재학습 후 서버를 다시 시작해야 새 모델이 사용됩니다.

## 출처

사진: Lisa DeBruine & Benedict Jones, [Face Research Lab London Set v5](https://doi.org/10.6084/m9.figshare.5047666.v5), CC BY 4.0. 부위 추출·정렬·레이블은 이 프로젝트에서 만든 변경분입니다. 저자의 연구 참여 동의 설명을 상용 서비스 전체 이용 승인으로 확대 해석하지 않았습니다.

문헌: [『欽定古今圖書集成』藝術典 第633卷](https://zh.wikisource.org/wiki/欽定古今圖書集成/博物彙編/藝術典/第633卷), [第634卷](https://zh.wikisource.org/wiki/欽定古今圖書集成/博物彙編/藝術典/第634卷). 원저작은 공공영역, 전사본의 편집·구두점은 해당 사이트의 CC BY-SA 4.0 안내를 보존합니다. 한국어 설명은 AI가 작성한 초벌 형태 설명입니다.


## 문헌 해석·측정 정의 v2 (2026-09-22)

서버 없이 `outputs/knowledge-v2/관상문헌_측정사전.html`을 열면 103항목 해석 요약, 주제별 필터, 원문, 판정 요건, 17개 측정 정의를 볼 수 있습니다. 기존 작업대에도 문헌 사전과 측정 기준 탭을 연결했습니다.

- `data/interpretation_annotations.tsv`: 출처를 읽어 작성한 103건 초벌 해석. 독립 전문가 검수 아님.
- `data/measurement_definitions.json`: 계산 가능한 대리 지표 11개와 추가 구현이 필요한 정의 6개.
- `data/type_requirements.json`: 103유형의 입력·조건·보류 사유. 현재 전통 유형 정답 0건.
- `docs/촬영과레이블링_작업지침.md`: 촬영 통제·독립 주석·반복성·수치 경계의 검증 절차.
- `docs/문헌측정_v2_완료내역.md`: 실제 완료와 외부 자료·검토가 필요한 작업 구분.

재생성 순서: `python scripts/build_interpretation_catalog.py`, `python scripts/build_type_requirements.py`, `python scripts/build_measurement_pilot.py`, `python scripts/build_knowledge_page.py`. 기존 v1 데이터·모델은 그대로 사용하며 v2 문헌 유형을 학습했다고 주장하지 않습니다.


## K-FACE v2: 한국인 400명 기반 관상 분석 (2026-09-26)

제공기관 승인을 받은 K-FACE High_Resolution(400명, 432만 장)으로 다시 구축한 버전입니다. 작업대 실행 후 <http://127.0.0.1:8765/v2> 에서 사용합니다.

- **현재 모델(v4)**: AI 판정자 3명 합의 레이블 + ConvNeXt·EfficientNet 5겹 교차학습 앙상블(모델 10개, `models/kface_v4`). 방법·결과는 보고서 13장.
- **사진 → 형태 특징 38개**(v3 39개에서 일치도가 낮은 인중 뚜렷함 제외): 눈(크기·개방·눈꼬리·좌우차·쌍꺼풀·흰자 노출·깊이·눈 사이 간격), 눈썹(모양·굵기·밀도·꼬리 선명도·길이·눈과의 간격·꼬리 방향·미간), 코(콧방울 폭·풍만함·길이·코끝 모양·콧구멍 노출·코끝 돌출·코뿌리 높이), 입(너비·입술 두께·상하비·입꼬리·활 모양·인중 길이·인중 골·입 돌출), 얼굴(형태·세로비·턱 폭·턱끝·광대·볼살·하정 비율·턱 돌출). 각 항목에 시험 정확도 기반 신뢰도 A~D를 표시하고 D는 유형 판정에 쓰지 않습니다. 한국인 참여자 분포의 하위 25%/중간/상위 25%로 표시하고, 경계 근처는 `경계`로 보류합니다.
- **다중 시점 깊이**: K-FACE 12개 카메라로 코끝 돌출·코뿌리 높이를 복원했습니다. 서비스에서는 정면과 좌우 약 40° 사선 사진을 함께 올리면 같은 방식으로 측정합니다.
- **관상 유형과 풀이**: 『고금도서집성』 103유형마다 한 줄 요약, 전통 평가(길/흉/길흉혼재), 재물·명예·건강·가족·성품 영역별 풀이, 원문 번역을 제공합니다(`data/interpretation_service_v1.json`). 사진의 형태 특징이 문헌 조건(`data/type_rules_v2.json`)과 맞는 유형을 보여줍니다. 사진으로 구별할 수 없는 유형은 묶음으로 보여주고 하나로 단정하지 않습니다.
- **가림 검출**: 안경·선글라스·모자를 K-FACE 촬영 조건으로 학습해, 감지되면 해당 부위 결과를 보류합니다.
- **직접 레이블링**: `tmp/kface_review/kface_labeling_v3_80.html`(16개 육안 항목, JSON 내보내기). 간단 검수는 `/v2/review`에서 AI 레이블을 보지 않고 검수하고, `python scripts/kface_agreement.py score outputs/kface-v2/독립검수_응답.csv`로 일치도(Cohen κ)를 계산합니다.

상세 방법·성능·한계: `docs/K-FACE_v2_구축보고서.md`. 전체 재현: `./scripts/run_kface_pipeline.ps1`.

**K-FACE 자료 취급**: 원본 ZIP·파생 crop·모델은 `data/raw`, `data/derived`, `models`에 있고 Git에서 제외됩니다. 검수 시트(`tmp/kface_review`)도 제외됩니다. 보고서·발표 자료에는 제공 사진을 넣지 않습니다(서약 조건). 결과 활용 14일 전 통보와 출처(한국과학기술연구원, 한국지능정보사회진흥원) 표기 의무가 있습니다.
