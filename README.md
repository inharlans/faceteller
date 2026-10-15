# FaceTeller · 페이스텔러

정면·좌측 사선·우측 사선 사진 3장에서 얼굴 형태를 분류하고, 해당 형태를 설명하는 전통 문헌을 찾아 보여주는 연구용 웹서비스입니다.

React 화면, FastAPI 서버, ResNet-18 분류 모델, MediaPipe 얼굴 특징점, VGGT 깊이 추정, PostgreSQL 문헌 DB를 사용합니다. 사진에서 개인의 성격·지능·건강·운명을 판정하는 서비스는 아닙니다.

## 구현된 기능

- 사진 3장 업로드 또는 카메라 안내에 따른 순차 촬영
- 촬영 방향·표정·밝기·가림 검사와 항목별 판정 보류
- 12부위 ResNet-18 모델로 41개 형태 항목 분류
- VGGT와 얼굴 특징점을 이용한 코끝·코뿌리·입·턱의 상대적 돌출 측정
- 일치하는 문헌을 조건별 점수의 기하평균으로 정렬해 최대 5개 표시
- 원문·번역·풀이·출처 조회와 A4 결과지 인쇄
- **실제 정면 사진과 측정선 선택 표시**: 눈썹 길이, 눈꼬리, 코 너비, 입 너비 등 최대 19개 항목. 눈썹 선은 추출한 눈썹털 영역을 따릅니다.

서비스 입력 사진은 메모리에서 처리합니다. 결과 화면의 정면 사진도 브라우저 메모리에만 보관하며, 다시 시작하거나 페이지를 벗어나면 정리합니다. 사용자가 사진 표시를 켠 채 결과지를 저장하면 해당 PDF에는 사진이 포함됩니다.

## 최신 검증 결과 · 2026-10-15

| 항목 | 결과 |
| --- | --- |
| 대상 | 400명: 학습 270 / 모델 선택 34 / 점수 보정 34 / 검사 62 |
| 재학습 | ResNet-18 12모델, 학습·반복별 선택 검사 합계 약 57분 |
| 실행 환경 | Windows, RTX 3070 8GB, Python 3.12.14, PyTorch 2.11.0+cu128 |
| 실제 사진 3장 HTTP 검사 | 400명 완료, 응답 오류 0건 |
| 로컬 요청부터 결과까지 | 평균 1.43초, 최대 1.70초. 촬영 대기·외부 업로드 시간 제외 |
| 측정선 검사 | 400명 좌표 구조·사진 범위 검사 통과. 395명은 19개, 5명은 눈썹 선을 생략한 17개 |
| 카메라/업로드 비교 | 같은 녹화 프레임에 대해 45개 항목 판정·점수 동일 |
| 문헌 연결 | 적격 73개 중 의미 검토를 거친 37개 연결 활성화 |

**분류 정확도가 완성됐다는 뜻은 아닙니다.** 평가는 자동 측정·AI 레이블과의 일치율이며 사람의 독립 정답과 비교한 정확도가 아닙니다. 눈썹 길이는 유효 검사 55명에서 52.7% 일치했고, 41개 분류 항목 중 9개는 보정 근거가 부족하거나 레이블이 불충분해 확정하지 않습니다. 평균 보류는 45개 중 20.54개입니다. 과거에 사용했던 검사 집합이며 새로운 외부 검증은 아닙니다.

- [재학습 결과 TXT](docs/reports/FaceTeller_수정_재학습_결과_20261015.txt)
- [항목별 결과 CSV](docs/reports/FaceTeller_재학습_항목별결과_20261015.csv)
- [사진·측정선 사용 안내](docs/release/정면사진_측정선_검증_20261015.txt)

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| `frontend/` | React·TypeScript 화면, 촬영, 결과지, 화면 테스트 |
| `facework/` | 전처리, 분류, 깊이 측정, 문헌 조회, API |
| `scripts/release/` | 자료 준비, 학습, 평가, DB 설정, 서버 실행 |
| `tests/` | 서버·측정·문헌 선정·학습 정책 검사 |
| `data/release/literature/` | 문헌 원문·풀이·출처·의미 검토 자료 |
| `data/release/migrations/` | PostgreSQL 스키마 |
| `third_party/vggt/` | 고정 버전 VGGT 소스와 원 라이선스·파일 해시 |
| `docs/` | 설계·구현 기록과 집계 보고서. 과거 문서는 작성 시점 기준 |

얼굴 원본, 파생 사진, 개인별 레이블·측정값, 가중치, 실험 캐시, DB 비밀번호는 Git에 포함하지 않습니다. **clone만으로 실제 추론 서버를 실행할 수는 없습니다.** 승인된 데이터로 만든 로컬 배포 산출물과 DB가 별도로 필요합니다.

## 환경 준비

현재 확인한 실행 환경은 Windows와 NVIDIA CUDA GPU입니다. 새 PC에서의 전체 재설치·실행은 아직 검증하지 않았습니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-release.txt

# Node.js 22.12+; 개발 환경은 Node 24 사용
Push-Location frontend
npm ci
npm run build
Pop-Location
```

빌드 시 MediaPipe·글꼴 자산을 로컬로 준비합니다. 실행 중에는 같은 서버의 자산을 사용합니다. `requirements-lock.txt`는 과거 환경 기록이며 현재 설치 목록은 `requirements-release.txt`입니다.

실제 추론에 필요한 로컬 자료:

- `models/release/`: 12개 ResNet-18 가중치, 배포·보정·학습 설정
- `models/vggt/`, `models/face_landmarker.task`, `models/face_parsing/`: 깊이·특징점·부위 분할 모델
- `data/release/training_registry.json`, `landmark_labels.json`, `depth_config.json`: 배포와 일치하는 정의·참조 자료
- PostgreSQL 및 문헌 DB

파일명만 맞춘 임의 산출물로 대체하면 해시·계약 검증에서 중단됩니다. 학습·배포 준비 코드는 `scripts/release/`에 있고, 기존 실험의 동결 기록을 요구하는 스크립트도 있습니다. 가중치 다운로드와 데이터 승인까지 자동으로 해결하는 설치 프로그램은 아닙니다.

## DB와 서버 실행

PostgreSQL과 필요한 로컬 산출물이 준비된 뒤 실행합니다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py local-init
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py import --corpus data/release/literature/reference_corpus_v1.json
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py sync-labels
.\start-release.ps1 -ValidateOnly
.\start-release.ps1
```

접속 주소는 **http://127.0.0.1:8767/** 입니다. 이미 로컬 DB가 실행 중이면 `start-release.ps1 -SkipLocalInit`을 사용할 수 있습니다. 종료는 `Ctrl+C`입니다.

연결 정보는 `%LOCALAPPDATA%/FaceTeller/postgresql/connection.json` 또는 `FACETELLER_DATABASE_URL`에서 읽습니다. 서비스는 읽기 전용 DB 계정을 사용합니다. 상세 내용은 [DB 안내](docs/release/database-report.txt)를 참고하세요.

프런트엔드 단위 검사는 `frontend/`에서 `npm test`로 실행합니다. 실제 모델·사진을 사용하는 통합 검사는 로컬 자료와 실행 중인 서버가 필요합니다. 시험용 K-FACE 사진을 외부 서버로 보내지 않도록 실제 사진 통합 검사 주소는 loopback으로 제한되어 있습니다.

## 출처와 이용 조건

K-FACE는 제공기관의 이용 승인·약정에 따라 별도로 확보해야 합니다. 이 저장소에는 해당 얼굴 데이터와 개인별 파생 자료를 올리지 않습니다.

현재 사용 중인 원 연구 `facebook/VGGT-1B` 가중치에는 비상업용 제한이 남아 있습니다. 코드 공개와 상용 서비스 이용 허가는 별개입니다. [VGGT 소스 라이선스](third_party/vggt/LICENSE.txt), [모델 카드](https://huggingface.co/facebook/VGGT-1B), [모델별 이용 조건 기록](docs/model-commercial-status.txt)을 확인하세요.

문헌의 출처·전사본 이용 조건은 [source_registry.json](data/release/literature/source_registry.json)과 각 조항의 출처에 보존했습니다. Wikisource 전사 자료의 CC BY-SA 조건 등 원출처 조건을 따릅니다. 제3자 코드·자료에는 각자의 라이선스가 적용됩니다.
