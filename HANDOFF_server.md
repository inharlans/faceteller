# FaceTeller 서버 이관 안내 (deploy/v1, 2026-11-16)

연구실 구조(기준 구조 v10: C안, 신뢰도 하한 0.6)를 사진 3장으로 돌리는 실행기와 임시 서버입니다.
데이터셋이나 레이블 파일 없이, 이 저장소와 모델 파일만으로 돕니다. 브랜치는 `lab/deploy-v1-handoff`입니다.

> **2026-11-17 변경 (구조 v10).** 팀이 측정값 순위로 경계를 정한 특징 5개를 반영했습니다.
> - 눈썹-눈 간격, 입술 두께, 인중 길이: CNN 판정 → 측정값 판정(경계 근처는 보류).
> - 광대 돌출, 입 돌출: 식은 그대로, 경계값만 팀이 정한 값으로 바꿈(보류 없음).
> - 측정 특징 7개(눈 크기, 눈썹-눈 간격, 코 길이, 코 높이, 입술 두께, 인중 길이, 턱 폭)의 보류 기준을 0.755에서 0.6으로 바꿈. 이제 보류 하한은 모두 0.6 하나입니다.
> - 새 설정 파일 `deploy/v1/config/cuts5_measures.json` 하나가 늘었고, 모델 파일은 추가가 없습니다. 인중 CNN 5개는 더 이상 읽지 않습니다.
> - (11월 21일) 조건 묶음이 48개에서 50개로 늘었습니다(미간: 좁음 / 코 폭: 좁음 + 코끝: 좁아지는 편). report에 `region_phrases`(부위별 한 문장: `region`, `text`, `tone`, `id`, `features`, `groups`)가 추가됐습니다. 기존 필드는 그대로입니다.
> - 서버에서는 이 브랜치를 다시 받고 `serve_person.py`를 다시 띄우면 됩니다. report 형식은 그대로입니다(`judgements`의 `method`, `measure` 값만 달라짐).

- 실행 코드: `deploy/v1/*.py`
- 설정(경계값, 보정값, 하한, 선정 규칙, 문헌 조항): `deploy/v1/config/*.json`
- 프론트의 연구용 상세 결과 화면(임시): `frontend/src/ResearchReport.tsx` 외

## 1. 실행 명령어

EC2 기준 위치(전달받은 대로): 저장소 `~/faceteller`, 우리 모델 `~/deploy_v1/experiments/feature17-20261109/`, 공개 모델 `~/faceteller/models/`.

```bash
cd ~/faceteller
git fetch origin && git checkout lab/deploy-v1-handoff
pip install -r requirements-release.txt        # torch / torchvision(cu128)는 README 순서대로 먼저 설치

# (1) 사진 3장으로 한 번 실행: 정면, 사선, 사선 (사선 두 장의 순서는 상관없음)
python deploy/v1/run_person.py front.jpg oblique_a.jpg oblique_b.jpg \
    --models-dir ~/deploy_v1/experiments/feature17-20261109 \
    --public-models-dir ~/faceteller/models \
    --vggt-dtype fp16 --out report.json

# (2) 서버: 모델을 GPU에 올려 두고 요청을 받음
python deploy/v1/serve_person.py \
    --models-dir ~/deploy_v1/experiments/feature17-20261109 \
    --public-models-dir ~/faceteller/models \
    --vggt-dtype fp16 --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/healthz             # {"status":"ok","vggt_dtype":"fp16","check_cnn":false,...}
```

인자 대신 환경변수도 됩니다(인자가 우선).

| 인자 | 환경변수 | 기본값 | 내용 |
|---|---|---|---|
| `--models-dir` | `FACETELLER_MODELS_DIR` | `<저장소>/experiments/feature17-20261109` | 우리 모델(S3) 폴더 |
| `--public-models-dir` | `FACETELLER_PUBLIC_MODELS_DIR` | `<저장소>/models` | 공개 모델 폴더 |
| `--config-dir` | `FACETELLER_CONFIG_DIR` | `deploy/v1/config` | 설정 JSON 폴더 |
| `--vggt-dtype` | `FACETELLER_VGGT_DTYPE` | `auto` | `auto` / `bf16` / `fp16` / `fp32`. `auto`는 GPU가 BF16을 지원하면 `bf16`, 아니면(T4) `fp16` |
| `--check` | `FACETELLER_CHECK_CNN=1` | 꺼짐 | 확인용 CNN도 실행(4장 참고) |
| `--check-models-dir` | `FACETELLER_CHECK_MODELS_DIR` | 모델 폴더와 같음 | 확인용 모델 폴더 |
| `--host`, `--port` (서버) | `FACETELLER_HOST`, `FACETELLER_PORT` | `127.0.0.1`, `8777` | |
| `--frontend-dist` (서버) | `FACETELLER_FRONTEND_DIST` | `<저장소>/frontend/dist` | 있으면 `/`에서 프론트를 같이 서빙 |

필요한 파일이 없으면 시작할 때 어떤 파일이 없는지 경로와 함께 알려 주고 멈춥니다.

## 2. 필요한 파일과 위치

### git (이 브랜치)

| 위치 | 내용 |
|---|---|
| `deploy/v1/run_person.py` | 실행기: `Runner(settings).analyze([정면, 사선, 사선])` → 결과(report) |
| `deploy/v1/serve_person.py` | 임시 서버(FastAPI): `POST /api/v1/analysis`, `GET /healthz` |
| `deploy/v1/settings.py` | 경로·옵션(인자, 환경변수) |
| `deploy/v1/structure.py` | 판정 규칙, 조항 조건 맞추기, C안 선정 |
| `deploy/v1/cnn.py` | CNN 입력 만들기, 5개 모델 평균 |
| `deploy/v1/measures.py` | 정면 사진 측정(눈 크기, 코 길이, 턱 폭, 미간, 입 폭) |
| `deploy/v1/vggt_depth.py` | VGGT 3차원 점, 코 높이, 광대·입 돌출 식, FP16 전환 |
| `deploy/v1/posenorm.py` | 자세 보정(정면화) |
| `deploy/v1/config/*.json` | 아래 표 |
| `facework/`, `third_party/vggt/` | 기존 서비스 코드(얼굴 점 검출, 크롭, 얼굴 분할, 눈썹 경로, VGGT 엔진). 이미 저장소에 있던 것을 그대로 불러 씁니다 |

| 설정 파일 | 내용 |
|---|---|
| `features.json` | 특징 19개의 이름, 부위, 칸(순서), 판정 방법, CNN 부위별 출력 목록 |
| `cnn_calibration.json` | CNN 특징의 보정값(온도)과 하한 0.6 |
| `numeric_cuts.json` | 미간·입 폭의 경계값, 측정 오차 |
| `eye_measures.json` | 눈 크기 경계값, 표준화 상수, 측정 오차 |
| `nose_measures.json` | 코 길이·코 높이 경계값(순위 33% / 67%), 측정 오차, 보류 기준(0.6) |
| `jaw_measures.json` | 턱 폭(턱 각도) 경계값, 측정 오차 |
| `cuts5_measures.json` | 11월 17일에 팀이 정한 순위 경계 5개: 눈썹-눈 간격·입술 두께·인중 길이의 경계값과 측정 오차, 광대 돌출·입 돌출 식 점수의 경계값. 광대·입 돌출은 보류하지 않음(`not_held`) |
| `depth_models.json` | 광대 돌출·입 돌출 식의 계수(F10, F19 항목은 쓰지 않음). 경계값은 `cuts5_measures.json`의 것을 씁니다 |
| `literature_representatives.json` | 조건 묶음 48개: 조건, 풀이 방향(다수결), 내용 종류별 줄(`lines`)과 그 조항들(127개)의 원문, 번역, 해설, 주제, 출처 위치(책, 권·편, 절, 글자 위치, 스캔 쪽, 주소) |
| `region_phrases.json` | 부위별 문구: 부위마다 확정된 형태 조합의 한 문장(596개). report의 `region_phrases`로 나감. 조합에 맞는 문헌이 없으면 그 부위는 문구가 없음 |
| `selection_rules.json` | C안 선정 규칙, 하한, 부위 표 |
| `manifest.json` | 전체 파일 목록(경로, 크기, sha256, git / s3 구분) |
| `check_models.json` | 확인용 CNN 목록(선택 사항) |
| `depth_cuts.json` | 11월 12일 깊이 경계(기록용, 쓰지 않음) |

### S3 (우리 모델, 27개, 1.04 GiB) → `--models-dir`

| 경로(모델 폴더 기준) | 개수 | 내용 |
|---|---|---|
| `candidate-v2-posenorm/models/{eye,brow,nose,mouth,philtrum}_fold{0..4}.pt` | 25 | CNN(ResNet18), 각 42.7 MiB. 11월 17일부터 `philtrum` 5개는 구조에서 쓰지 않습니다(없어도 실행됨, `--check`일 때만 읽음) |
| `posenorm_template.npy` | 1 | 자세 보정 기준 얼굴(학습 270명 평균, 개인 정보 아님) |
| `vggt3/vggt_template.npy` | 1 | VGGT 좌표의 기준 얼굴(같음) |

받은 뒤 `manifest.json`의 sha256과 대조해 주세요.

### 공개 모델 → `--public-models-dir`

| 경로 | 내용 |
|---|---|
| `vggt/model.pt` | VGGT-1B 가중치(4.7 GiB) |
| `face_landmarker.task` | MediaPipe Face Landmarker |
| `face_parsing/resnet18.onnx` | 얼굴 분할(CPU, onnxruntime). 코드가 sha256을 검사합니다 |

### 라이브러리 (연구실에서 쓴 버전, `requirements-release.txt`와 같음)

Python 3.12.14 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · mediapipe 1.0.1 · onnxruntime 1.31.0 · opencv 5.0.0 · numpy 2.5.3 · scipy 1.18.1 · pillow 12.3.0 · fastapi 0.141.1 · uvicorn 0.53.0 · python-multipart 0.0.32 · einops 0.8.2 · huggingface-hub 2.2.0

서버에는 화면이 없으니 OpenCV는 headless 빌드를 쓰면 됩니다. 연구실 PC는 onnxruntime을 `models/face_parsing/python`에 따로 두고 썼는데, 서버에서는 pip으로 설치한 것이 그대로 쓰입니다.

## 3. 판정 흐름

입력은 정면 1장과 고개를 좌우로 돌린 사진 2장입니다. 사진은 메모리에서만 처리하고 저장하지 않습니다.

1. **얼굴 점 검출** (`facework.analysis_v2.detect`, MediaPipe): 세 장 각각 478점. 얼굴이 하나가 아니면 오류로 끝냅니다.
2. **CNN 8개 특징** (`cnn.py`): 정면 사진을 자세 보정(`posenorm.warp`) → 부위 크롭 → 얼굴 전체 타일과 나란히 붙임 → 부위별 5개 모델 평균 → 보정(`p^(1/T)`) → 가장 큰 칸의 확률이 신뢰도. **0.6 미만이면 보류.**
   - 눈: 눈 가늘기, 눈꼬리 기울기 / 눈썹: 꼬리 길이, 휨, 숱 / 코: 코 폭, 코끝 둥근 정도 / 입: 입꼬리
   - 모델에는 눈 크기, 코 길이, 입 폭, 눈썹-눈 간격, 입술 두께 출력도 있지만 쓰지 않습니다. 인중 모델(`philtrum`)은 읽지 않습니다.
3. **수치 2개** (`measures.py`): 미간(얼굴 분할로 찾은 두 눈썹 안쪽 끝 사이 ÷ 얼굴 폭), 입 폭(입꼬리 사이 ÷ 얼굴 폭). 경계점수 **0.6 미만이면 보류.**
4. **측정 7개** (경계점수 **0.6 미만이면 보류**. CNN·수치와 같은 하한입니다. 11월 17일 전에는 0.755였습니다)
   - 눈 크기: 눈 가로 폭과 사진에서 읽은 눈 뜬 높이(÷ 눈썹 아래 얼굴 넓이의 제곱근)를 표준화해 평균
   - 코 길이: 두 눈 위쪽 끝의 가운데 ~ 코밑 ÷ 얼굴 폭
   - 턱 폭: 턱 끝에서 양쪽 턱 모서리를 본 각도
   - 코 높이: VGGT 3차원 점에서 코끝이 미간~턱 앞 선에서 떨어진 거리 ÷ 두 눈 바깥 끝 거리
   - 눈썹-눈 간격: 눈썹 아래 가장자리(얼굴 분할) ~ 눈 위쪽 끝 ÷ 눈 가로 (11월 17일부터, 이전에는 CNN)
   - 입술 두께: 윗입술 + 아랫입술 두께 ÷ 입 폭, 자세 보정한 점에서 (11월 17일부터, 이전에는 CNN)
   - 인중 길이: 코밑 ~ 윗입술 ÷ 눈썹 사이 ~ 턱 끝, 자세 보정한 점에서 (11월 17일부터, 이전에는 CNN)
5. **VGGT 식 2개** (`vggt_depth.py`, 보류 없음): 광대 돌출, 입 돌출. 사진 3장을 얼굴 중심으로 잘라 VGGT에 넣고, 정면 사진의 얼굴 점 위치에서 3차원 점을 읽어 `depth_models.json`의 계수로 점수를 냅니다. 경계값은 11월 17일에 팀이 정한 값(`cuts5_measures.json`)입니다: 점수가 경계 이상이면 "높음" / "나옴", 미만이면 "보통".
6. **조항 맞추기** (`structure.py`): 확정된 판정만으로 조건 묶음 48개의 조건을 확인합니다. 한 묶음은 같은 조건을 다룬 문헌 조항들입니다. 화면에는 내용 종류마다 한 줄(문장 하나와 같은 말을 하는 책 이름들)로 보여 주고, 좋은 풀이가 더 많거나 같은 묶음에서는 나쁜 풀이를 보여 주지 않습니다. 조항의 신뢰도(G)는 쓰인 특징 신뢰도의 기하평균입니다.
7. **C안 선정**: 신뢰도 순으로 줄 세운 뒤 부위별로 하나씩 돌아가며 고릅니다. 나쁜 풀이는 2개까지, 전체 5개.

눈을 감은 사진이면 눈 크기와 코 길이는 판정하지 않습니다. CNN의 "판정 불가" 출력이 0.5를 넘는 특징도 판정하지 않습니다.

## 4. 확인용 CNN (기본 꺼짐)

수치·VGGT로 판정하는 특징에 대해, 예전 CNN과 시험 CNN의 결과를 참고용으로 같이 보는 기능입니다. 조항에는 쓰지 않습니다.
이 모델들(40개)은 S3에 올리지 않았습니다. `--check`를 켜도 모델 파일이 없으면 조용히 건너뜁니다. 서비스에는 필요 없습니다.

## 5. research_report 형식

`run_person.py --out`으로 저장되는 JSON이고, 서버 응답의 `data.research_report`에 그대로 들어갑니다. 사진과 얼굴 점 좌표는 없습니다.

```jsonc
{
  "schema": "faceteller-research-report-v1",
  "structure": "구조 v10 · C안 · 하한 0.6",
  "photos": [{"view": "front", "size": [1200, 1600], "yaw": 0.5, "pitch": -3.0, "roll": 0.2, "blink": 0.05, "smile": 0.0}],   // 예시 값(실제 결과 아님)
  "face_width_px": 520.0,
  "judgements": {                     // 특징 19개, 키는 F01 ~ F21
    "F09": {"name": "코 길이", "region": "코", "method": "정면 수치", "value": 0.3712,
            "class": "보통", "confidence": 0.81, "threshold": 0.6, "status": "확정",
            "measure": {"label": "두 눈 위쪽 끝의 가운데 ~ 코밑 ÷ 얼굴 폭", "value": 0.3712, "unit": "비율",
                        "used_for_judgement": true, "cuts": [0.355, 0.377], "cut_classes": ["짧음", "보통", "김"]}},
    "F05": {"name": "눈썹 휨", "region": "눈썹", "method": "CNN", "value": null,
            "class": "일자", "confidence": 0.91, "threshold": 0.6, "status": "확정",
            "probabilities": {"일자": 0.91, "곡선": 0.09},
            "reference": {"label": "눈썹 휨: 양 끝을 이은 선에서 가장 솟은 높이 ÷ 눈썹 길이", "value": 0.042, "unit": "비율",
                          "components": [{"label": "가장 솟은 곳에서 꺾인 각도", "value": 7.5, "unit": "도"}]}}
    // status: "확정" | "보류" | "판정 불가(…)" | "측정 불가" 등.  threshold가 null이면 보류 규칙 없음
    // measure: 판정에 쓴 측정값(무엇을 쟀는지, 단위, 경계값과 칸). 눈 크기는 components에 눈 가로 폭·눈 뜬 높이가 들어감
    // probabilities: CNN 특징의 칸별 확률.  reference: CNN 특징 옆에 보여 주는 참고 측정값(판정에는 쓰지 않음)
  },
  "plan_C_selected": [                // 화면에 내보낼 조항, 최대 5개
    {"representative": "조항 id", "requirement": ["F18 눈썹 숱:성김"], "valence": "나쁨", "G": 0.77,
     "regions": ["눈썹"], "features": {"F18": {"class": "성김", "confidence": 0.77, "held": false}},
     "translation": "문헌 번역", "source": "문헌 이름",
     "original_text": "문헌 원문(한문)", "commentary": "해설", "split_of": null,
     "annotation_status": "ai_annotated_unreviewed", "translation_kind": "korean_literal_ai_draft",
     "source_detail": {"title": "神相全編", "title_ko": "신상전편", "chapter": "권·편 이름", "section": "절 제목",
                       "char_span": [10, 22], "scan_page": "스캔 쪽", "scan_page_end": "스캔 쪽", "attributed_to": null,
                       "url": "원문 주소(고정된 판)", "license": "이용 조건", "text_witness": "저본"},
     // 이 조건에 대해 문헌이 하는 말. 내용 종류마다 한 줄: 첫 조항의 문장 + 같은 종류의 말을 하는 책 전부 + 그 조항들
     "lines": [{"topic": "형제·가족", "translation": "한 줄로 보여 줄 번역", "valence": "좋음", "sources": ["신상전편", "조담경"],
                "entries": [{"id": "조항 id", "valence": "좋음", "topics": ["형제·가족"], "translation": "번역", "source": "책 이름",
                             "original_text": "원문", "commentary": "해설", "source_detail": {…}, "annotation_status": "…"}]}],
     "entry_count": 13, "hidden_unfavourable": 1, "omitted_entries": 0}
     // hidden_unfavourable: 좋은 풀이가 더 많거나 같아서 보여 주지 않는 나쁜 풀이 수.  omitted_entries: 떼어 보여 줄 수 없어 뺀 조항 수
  ],
  "all_matching": [],                 // 조건이 맞는 조항 전부(같은 형식, 신뢰도 순)
  "blocked_by_hold": [],              // 보류된 판정까지 넣으면 맞았을 조항(같은 형식, held = true인 특징이 있음)
  "held_judgements": [{"feature": "F15", "name": "인중 길이", "class": "보통", "confidence": 0.55, "threshold": 0.6,
                       "reason": "이 칸을 조건으로 쓰는 조항이 없음"}],
  "cnn_check": {},                    // 확인용 CNN을 켰을 때만: {"F09": {"모델 이름": {"class": "김", "confidence": 0.70, …}}}
  "vggt": {"dtype": "fp16", "network_seconds": 0.7},
  "seconds": 1.7
}
```

서버 응답(NDJSON, 프론트가 이미 쓰는 형식):

```
{"type":"progress","stage":"얼굴 정렬"}      ← 부위 추출, 형태 분류, 깊이 분석, 문헌 선정 순서로 이어짐
{"type":"result","data":{"status":"COMPLETE","items":[],"passages":[],"notice":"…","research_report":{…}}}
{"type":"error","message":"얼굴을 찾지 못했습니다.","direction":"front","code":"FACE_NOT_FOUND"}   ← 실패했을 때
```

프론트는 `research_report`가 있으면 기존 카드 화면 대신 "연구용 상세 결과" 화면을 보여 줍니다. 형식이 틀리면 버리고 기존 화면을 그립니다.

## 6. 연구실 실측 (RTX 3070 8GB, 확인용 CNN 꺼짐)

| 항목 | 값 |
|---|---|
| 서버가 모델을 올리는 시간 | 15.6초 |
| 모델을 올린 뒤 GPU 메모리(torch 기준) | 3.69 GiB |
| nvidia-smi 증가분 | 올린 직후 약 4.1 GiB, 요청을 몇 번 처리한 뒤 약 5.4 GiB |
| 요청 1건 (모델이 올라가 있을 때) | 첫 요청 3.9초, 이후 1.7~1.8초 (그중 VGGT 0.7초) |
| 실행기를 한 번만 돌릴 때 | 20~26초 (대부분 모델을 읽는 시간), torch 최대 3.35 GiB |
| BF16과 FP16의 차이 | 같은 사진에서 판정 19개와 조항이 모두 같음. 신뢰도 차이 최대 0.009 |

T4에서는 값이 다를 수 있으니 올린 뒤 같은 표로 한 번 재 주세요.

## 7. 연동 작업 9개

1. **환경 준비:** 브랜치를 받고 `requirements-release.txt`로 설치합니다. `torch.cuda.is_available()`과 T4 지원 여부를 확인합니다.
2. **우리 모델 받기:** S3의 27개 파일을 모델 폴더 구조 그대로 받고 `deploy/v1/config/manifest.json`의 sha256과 대조합니다.
3. **공개 모델 확인:** `vggt/model.pt`, `face_landmarker.task`, `face_parsing/resnet18.onnx`가 공개 모델 폴더에 있는지 봅니다.
4. **단독 실행:** `run_person.py`를 `--vggt-dtype fp16`으로 돌려 report가 나오는지, 판정이 19개인지 확인합니다.
5. **서버 상주:** `serve_person.py`를 서비스로 등록하고(systemd 등) `/healthz`로 상태를 봅니다. 분석은 한 번에 한 건씩 처리합니다.
6. **프론트 배치:** `frontend`에서 `npm ci && npm run build`. `dist`를 서버가 같이 서빙하거나, 다른 곳에 두고 `/api`만 이 서버로 넘깁니다.
7. **경로와 통신:** CloudFront에서 EC2로 `/api/v1/analysis`를 넘깁니다. 카메라는 HTTPS에서만 열립니다. 응답은 줄 단위로 흘려보내므로 중간에서 버퍼링이나 압축으로 묶이지 않게 하고, 업로드 크기(서버는 장당 12 MiB까지)와 타임아웃(첫 요청은 더 걸림)을 맞춥니다.
8. **사진 취급:** 사진은 메모리에서만 다룹니다. 디스크, 로그, S3에 사진이나 개인 결과가 남지 않게 해 주세요. S3 버킷에는 모델과 배포 파일만 둡니다.
9. **결과 대조와 실측:** 같은 사진 3장을 연구실과 서버에서 돌려 report JSON(사진 없음)을 비교합니다. 판정과 조항이 같고 신뢰도 차이가 0.01 안팎이면 정상입니다. T4에서의 시간과 메모리도 같이 알려 주세요.

## 8. 알아둘 점

- **임시 화면입니다.** 기존 서비스(`facework/release_runtime.py`)는 손대지 않았습니다. 이 서버는 그것과 별개로 뜨며, 응답에는 기존 형식의 `items`·`passages`가 비어 있습니다.
- **프론트의 사진 검사가 먼저 걸러 냅니다.** 고개 각도 등이 기준을 벗어나면 서버로 보내기 전에 다시 찍으라고 합니다.
- **가까이서 찍은 셀카는 값이 치우칩니다.** 경계값은 멀리서 찍은 사진으로 정했습니다. 가까이서 찍으면 코 길이·눈 크기는 크게, 턱 폭은 좁게 나오는 방향입니다(크기는 아직 재지 않았습니다).
- **턱 폭은 고개 각도 보정이 없습니다.** 턱을 들면 넓게, 숙이면 좁게 나옵니다.
- **눈 크기는 조명에 따라 값이 달라집니다.** 그래서 보류가 많습니다.
- **문헌의 번역과 해설은 자동 작성한 초안입니다.** 원문과 출처 위치는 `data/release/literature/reference_corpus_v1.json`에서 가져왔고, 사람 검수 전이라는 표시(`annotation_status`)를 화면에 같이 보여 줍니다.
- **참고 측정값은 판정에 쓰지 않습니다.** CNN으로 판정하는 특징 옆에 보여 주는 값(눈썹 휨 정도, 눈꼬리 각도 등)은 설명용입니다.
- **신뢰도는 정답 확률이 아닙니다.** 판정이 경계에서 얼마나 떨어져 있는지를 나타냅니다.
- 전통 문헌 조회 결과이며, 개인의 성격·지능·건강·운명을 판정하거나 예측하지 않습니다.
