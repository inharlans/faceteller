# FaceTeller — 현재 release

정면·좌측 사선·우측 사선 **사진 3장**에서 관찰 가능한 얼굴 외형을 분류하고, 일치하는 역사적 문헌을 PostgreSQL에서 조회하는 로컬 연구 프로젝트입니다. 현재 실행 진입점은 **`start-release.ps1`**, 화면은 **http://127.0.0.1:8767/** 입니다.

**2026-10-15 현재: 분류 배포는 사용자 지시에 따라 ResNet-18 단일 계열 12모델(앙상블 없음)이며, 보정34에서 고른 헤드별 기준값으로 보정34·역사적 시험62, 실제 400명 HTTP, 실제 브라우저 UI, 녹화 입력 HTTPS 카메라 경로 검증까지 완료됐습니다. 물리 웹캠·여러 기기 촬영, 공개 배포, 상업 이용 조건은 검증하지 않았습니다.** 필수 산출물이 없으면 시작 스크립트가 중단합니다. 아래 준비 절차와 사전 검사 통과를 실제 추론 성능 검증 완료로 해석하지 마세요.

## 실행

프로젝트 루트의 Windows PowerShell에서 실행합니다. 학습/검증 작업이 GPU를 사용 중이면 일반 실행을 하지 않습니다.

```powershell
# CPU 사전 검사만: DB 변경·모델 로드·GPU 연산·서버 기동 없음
.\start-release.ps1 -ValidateOnly

# 모든 산출물이 준비되고 GPU가 비어 있을 때 실제 서비스 시작
.\start-release.ps1
```

기본 실행은 Windows `.venv`와 필수 파일을 확인하고, 배포/가중치/전처리 해시를 검증한 뒤 로컬 PostgreSQL의 `local-init`을 호출합니다. 이 명령은 전용 DB를 만들거나 기존 DB를 다시 시작하며 기존 corpus를 초기화하지 않습니다. **문헌 import는 자동 수행하지 않습니다.** 읽기 전용 DB 계정과 현재 corpus 해시까지 확인한 뒤 `scripts/release/serve.py`를 실행합니다.

- `-ValidateOnly`: DB가 이미 준비되어 있어야 마지막 읽기 전용 DB 검사까지 통과합니다. CUDA 사용 가능 여부나 실제 추론 성공은 확인하지 않습니다.
- `-SkipLocalInit`: 이미 실행 중인 로컬 DB를 사용합니다. DB 읽기 전용 검사는 생략하지 않습니다.
- `-Port 8768`: 다른 로컬 포트 사용. 기본은 `8767`입니다.
- `FACETELLER_DATABASE_URL`이 설정되면 해당 reader를 사용하고 로컬 DB 초기화를 생략합니다. 외부 DB의 스키마/권한 준비는 [DB 안내](docs/release/database-report.txt)를 따릅니다. 관리자 DSN을 reader로 사용하지 마세요.
- 종료는 `Ctrl+C`입니다. 기동 후 `/api/v1/service-info`의 `ready`와 배포·깊이·corpus 해시를 확인합니다.

사진 업로드와 카메라 자동 3단계 촬영은 동일 API를 사용합니다. 카메라 권한이 없으면 업로드를 사용합니다. 촬영 품질 기본값은 취득을 위한 설정이며 과학적으로 검증된 정확도 기준이라는 뜻은 아닙니다. 실제 사람 웹캠·여러 기기에서의 검증 범위는 [프런트엔드 보고](docs/release/frontend-report.txt)를 확인하세요.

## Windows 환경 준비

관측 환경은 **Windows 64비트 / CPython 3.12.14**입니다. PostgreSQL은 별도 설치가 필요하며 현재 전용 인스턴스는 기본 `127.0.0.1:55432`를 사용합니다. 이미 준비된 개발 환경에서는 아래 설치를 다시 실행할 필요가 없습니다. 실행 중인 학습 venv를 변경하지 마세요.

```powershell
# 새 환경에서만 실행
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 관측된 CUDA 12.8 PyTorch 조합: 일반 Python 의존성과 별도 설치
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-release.txt
.\.venv\Scripts\python.exe -m pip check

# React/TypeScript: Node.js 22.12+ 또는 현재 사용 중인 Node 24 계열
Push-Location frontend
npm ci
npm run build
Pop-Location
```

CUDA 패키지 조합은 [PyTorch 공식 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/)와 실제 NVIDIA 드라이버 환경을 확인해야 합니다. CPU 전용 PyTorch로 전체 VGGT 서비스를 대체하지 않습니다. 이번 정리는 현재 환경의 버전을 확인한 것이며 **새 PC에서의 깨끗한 재설치·GPU 실행까지 검증한 기록은 아닙니다**.

[`requirements-release.txt`](requirements-release.txt)는 release 서버·전처리·평가·보고서의 직접 의존성과 실제 확인한 버전을 담습니다. 전체 전이 의존성을 동결한 lock은 아닙니다. 기존 [`requirements-lock.txt`](requirements-lock.txt)는 과거 환경 스냅샷으로 보존하며 현재 release 설치 목록과 혼용하지 않습니다. ONNX Runtime 1.31.0은 현재 `models/face_parsing/python`에 별도 설치되어 있고, 새 환경에서는 requirements 목록으로 venv에 설치할 수 있습니다.

프런트엔드 빌드는 MediaPipe WASM·모델·글꼴을 `/assets/`에 준비합니다. 최초 준비 때 모델 다운로드가 발생할 수 있지만 런타임에는 동일 origin의 로컬 자산만 사용합니다. Python 모델 가중치, VGGT 소스, 승인된 원자료, DB corpus는 pip/npm 설치만으로 만들어지지 않습니다.

## 필요한 로컬 산출물과 DB

| 대상 | 위치 / 역할 |
| --- | --- |
| 전체 특성·분할·참조 지지수 | `data/release/training_registry.json` |
| 배포 manifest·온도 (ResNet-18 12가중치) | `models/release/deployment.json` |
| 추론 계열 선택 기록 | `models/release/inference_frozen.json` (`release-inference-choice-v1`, `architectures=[resnet18]`) |
| 원 학습 설정(3후보 계열, 변경 금지) | `models/release/training_frozen.json` |
| 12부위 가중치·학습 기록 (배포는 `resnet18/`, 나머지 두 계열은 역사적 후보) | `models/release/{architecture}/{part}.pt`, `.json` |
| 400명 깊이 경계·반복 sigma·품질 기준 | `data/release/depth_config.json` |
| 원 연구 VGGT 가중치 | `models/vggt/model.pt` 또는 depth config가 지정한 프로젝트 내 경로 |
| 고정 VGGT 코드·검증 해시 | `third_party/vggt/`, `provenance.json` |
| 로컬 얼굴 검출·부위 파싱 모델 | `models/face_landmarker.task`, `models/face_parsing/resnet18.onnx` |
| 원문·직역·해설·조건·출처 corpus | `data/release/literature/reference_corpus_v1.json` |
| React 배포 앱·카메라 자산 | `frontend/dist/` |

가중치를 임의 파일로 대체하거나 파일럿 경계를 복사해 사전 검사를 통과시키지 않습니다. 새 자료 구축/학습 순서는 메인이 관리하는 `scripts/release/run_training_queue.py`와 `outputs/release/training_queue.json`에서 확인합니다. GPU 큐와 별도로 학습·깊이 추론·검증 서버를 중복 실행하지 않습니다.

PostgreSQL 설치와 corpus 준비 후, 최초 설정 또는 재부팅 후에는 다음 순서로 진행합니다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py local-init
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py import --corpus data/release/literature/reference_corpus_v1.json
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py sync-labels
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/setup_database.py status
.\start-release.ps1 -ValidateOnly
```

`local-init`은 문헌을 넣지 않습니다. `import`는 지정한 corpus를 활성 스냅샷으로 지정하므로 의도한 파일을 선택해야 합니다. 연결 비밀은 `%LOCALAPPDATA%/FaceTeller/postgresql/connection.json`에 두며 README·보고서·Git에 복사하지 않습니다. 서비스는 제한된 읽기 전용 계정만 사용합니다. 현재 corpus의 canonical JSON SHA256과 활성 DB revision이 달라도 기동을 중단합니다.

상세설계 4.3의 레이블·조항 컬럼은 `data/release/migrations/004_label_passage_metadata.sql`로 **추가만** 합니다. `label`에는 `region`, `label_name`, `criteria`, `measurement_indicator`(깊이 항목만), `measurement_method`, `metadata_revision`이 생기고, `passage`의 `chapter`·`passage_no`는 기존 `source.chapter`·`source.locator`에서 자동 생성되는 읽기 전용 컬럼입니다. 원문에 통일된 조번호가 없으므로 새 번호를 만들지 않고 기존 locator(절·문자 범위·스캔 쪽)를 그대로 씁니다. 식별자는 설계의 BIGINT 대신 기존 TEXT 자연키(`feature:class` 레이블, corpus `passage_id`)를 의도적으로 유지하며 번호를 다시 매기지 않습니다. 레이블 메타데이터는 `sync-labels`(또는 `scripts/release/sync_label_metadata.py sync`)가 동결된 `training_registry.json`·`landmark_labels.json`·`depth_config.json`에서 129개 클래스 전체를 채웁니다. 반복 실행해도 바뀐 행만 갱신하며 corpus·적격 조항·revision 해시는 바꾸지 않습니다. 기준값은 파일에 기록된 값만 쓰고, AI 합의 시각 레이블은 "수치 물리 경계 없음"으로 표시합니다. `sync_label_metadata.py check`는 서비스 읽기 계정으로 DB 내용이 동결 정의와 같은지 확인합니다. `local-init`/`start-release.ps1`은 migration을 자동 적용하지만 레이블 동기화는 하지 않습니다.

## 현재 release의 범위

- **입력 3장:** `front`, `left_oblique`, `right_oblique`, `input_mode(camera|upload)`를 `POST /api/v1/analysis` multipart로 전송합니다. `progress`, `result`, `error`를 NDJSON으로 받습니다.
- **12부위:** 눈, 눈썹, 코, 입, 얼굴형, 이마, 미간, 눈 밑, 광대, 귀, 인중, 점. 분류 헤드 수와 클래스는 registry/배포 manifest를 따르며, 과거 15헤드 비교나 이전 v4 특성 수를 그대로 적용하지 않습니다.
- **4개 깊이:** 코끝 돌출, 코뿌리 높이, 입 돌출, 턱 돌출. VGGT 결과의 400명 기준 분포에서 두 경계로 낮음/보통/높음을 구분합니다. `Pj = Phi((r − bj) / sigma)`이며 경계점수는 각각 `1−P1`, `P1−P2`, `P2`입니다. sigma는 학습군 반복 측정 산포이고 품질 실패는 보류합니다. 깊이 `measurement_score`는 CNN `confidence`와 다르며 정답확률이 아닙니다.
- **문헌:** 모든 관찰 조건 AND 일치, 중복 제거, 부위당 최대 3개·전체 최대 5개. CNN 보정 신뢰도와 깊이 경계점수를 동일 가중 기하평균 `G`로 결합해 순위를 정합니다. 사전 DB의 원문·직역·해설·출처를 조회하며 분석 중 새 해석을 생성하지 않습니다. 확보한 판본·AI 주석·서비스 적격 범위와 접근하지 못한 지정 문헌을 구분합니다.
- **화면/인쇄:** React+TypeScript, FastAPI, PostgreSQL. 결과와 보류 사유를 표시하고 원문·직역·출처 중심 A4 한 장을 제공합니다. 상세해설은 인쇄에서 제외하며 본문이 넘치면 삭제·말줄임 대신 넘침과 다중 페이지 대안을 알립니다.

## 데이터·실험을 읽는 방법

K-FACE **400명**을 인물 단위로 **학습270 / 선택34 / 보정34 / 역사적 시험62**로 분리합니다. 선택군으로 체크포인트를 정한 뒤 보정군에서 헤드별 온도를 맞추고, 같은 보정군에서만 헤드별 확정 기준값을 고른 다음 고정 배포 모델의 시험 지표를 기록합니다. 기준값 정책(`models/release/threshold_policy_frozen.json`, `validation-reject-cost-v1`)은 시험 전에 고정되며 오확정 비용 1·보류 비용 0.25, 후보 0.50~1.00(0.05 간격), 2클래스 하한 0.60·다중 클래스 하한 0.50을 씁니다. 같은 보정군의 개발 기준이므로 독립 정확도가 아닙니다. 실제 crop 가시성이 부족하면 특성별 유효 학습 인물·클래스 지지수가 원본 참조 지지수보다 적을 수 있습니다.

보정군의 클래스 지지수가 부족해 유효 정답이 **한 클래스뿐인 헤드**는 최적화가 수치적으로 성공해도 보정을 승인하지 않습니다. 새 release 보정은 `calibrated=false`, `temperature=null`, `status=single_class_calibration_support`로 기록하고 raw 지표만 보존합니다. 해당 헤드는 런타임에서 `CONFIDENCE_NOT_CALIBRATED`로 보류하며 문헌 조건 확정에 사용하지 않습니다. 이는 모든 클래스가 반드시 나타나야 한다는 규칙과는 다르며, 두 클래스 이상이 관측되어도 누락 클래스의 성능 검증을 뜻하지 않습니다. 현재 ResNet-18 배포에서는 41헤드 중 37헤드가 보정되고 4헤드가 이 규칙으로 보류됩니다(`deployment.json` 기준).

자동 측정 참조는 `data/release/landmark_labels.json`에 기록하며 400명 중 학습군270명만 경계 적합에 사용합니다. 눈꼬리·눈썹 꼬리·입꼬리의 방향은 0 중심, 입술 상하 비는 1 중심으로 정의합니다. 눈썹 꼬리는 `brow_outer_inner_tilt`(양 끝점 전체 기울기)를 사용하고, 나머지 크기/비율형 항목은 학습군 P25/P75 기준을 따릅니다. 경계 부근은 보류하며 이 허용 폭은 검증된 신뢰구간이 아닙니다. 유형에 따라 학습 지지수가 0일 수 있어 해당 유형의 양성 검출 능력이 검증되었다고 주장하지 않습니다. 실제 기준·유효 인물 수·클래스 지지수는 registry를 확인합니다.

레이블은 기존 AI 합의와 자동 외형/기하 측정으로 만든 **weak labels**입니다. 인간 전문가의 독립 정답이 아니며 ImageNet 사전학습 모델이 관상을 이미 아는 것도 아닙니다. 분류 정확도는 이 참조 레이블과의 일치도를 뜻합니다. 성격·지능·건강·운명을 예측하는 도구로 해석하지 않습니다.

62명은 앞선 개발에 사용된 역사적 시험 집합으로 신규 외부 검증이 아닙니다. 400명 HTTP 실행에는 학습군도 포함되므로 그 실행 완료를 400명 독립 정확도라고 부르지 않습니다. 기존 `models/selection-20261014`의 45회(3계열×5부위×3시드) 비교와 이번 확대 범위·전처리의 새 배포 성능도 구분합니다. 현재 ResNet-18 단일 배포는 사용자의 계열 지정이며(`inference_frozen.json`의 `selection_basis` 원문 참조), 계열 간 우열이나 앙상블 대비 우열을 새로 검증한 결과가 아닙니다. 원 3계열 학습 설정(`training_frozen.json`)과 비교 기록은 바꾸지 않았습니다.

귀 입력은 검출 bbox를 입력 크기로 확대·축소하면 귀 길이의 상대적 기준이 사라지는 문제를 수정했습니다. 수정 입력은 눈 사이 거리와 yaw를 기준으로 정규화한 고정 창을 사용합니다. 당시 3계열 배포에서 **귀 3모델만 재학습하고 비귀 33모델은 유지**했으며, 근거는 `outputs/release/ear-input-correction/verification.json`의 통과 여부·전후 33개 가중치 해시·귀 전처리 버전입니다. 이 기록은 역사적 자료로 그대로 둡니다. 현재 ResNet-18 배포는 그중 비귀 11모델(해시가 수정 기록과 동일)과 수정된 귀 1모델을 사용하며, 보고서는 이 부분집합 일치를 검사합니다. 자료가 없으면 보고서는 미제공/대기로 표시하며 재학습 완료를 주장하지 않습니다.

보고서의 **현재 배포 12모델 학습 시간은 해당 metadata만 합산**하고, 미배포 두 계열 24모델의 학습 기록은 역사적 후보로 별도 표에 둡니다. ResNet-18 단일 선택 뒤 재학습하지 않았으므로 배포 모델 학습 시간은 원 학습(및 귀 수정 재학습) 기록입니다. 폐기된 이전 귀 3모델의 학습 및 이전 배포의 보정·시험 비용은 이 합계에 포함되지 않습니다. 해당 실측 비용과 입력 수정 전체 경과는 검증 JSON이 제공될 때 TXT/XLSX/PDF에 별도로 기록합니다. 최종 보정·시험도 별도 집계하며, 수정 전체 경과와 단계 비용은 중첩될 수 있으므로 더해서 전체 작업 시간으로 제시하지 않습니다.

## 완료 여부와 결과 갱신

아래 파일은 **생성 예정/검증 근거 위치**이며 존재 자체가 통과를 뜻하지 않습니다. 실제 생성 후 배포 해시·400명 포함 여부·실패·보류·5초 초과를 확인해 이 표를 갱신합니다.

| 확인 대상 | 실제 근거 | 이 README 작성 시 상태 |
| --- | --- | --- |
| 새 배포 모델·깊이 설정 | `models/release/deployment.json`, `data/release/depth_config.json` | 깊이400명+80회 반복 측정 완료; ResNet-18 단일 12모델 배포·보정 완료(이전 3계열 배포는 `outputs/release/before-resnet-only/`에 보관) |
| 문헌 DB | `data/release/literature_ready.json`, `outputs/release/literature-finalization.json` | 5문헌·5304조항·925원문구간·적격73개 실제 반영/검색 검사 완료 |
| 새 분류 성능·raw/T NLL·ECE | `outputs/release/test_metrics.json` | ResNet-18 단일 배포·헤드별 기준값(`validation-reject-cost-v1`)으로 산출 완료; 수치는 최종 보고서 |
| 실제 400명·분할·속도·DB 응답 | `outputs/release/service-verification.json` | 400명 완료(오류 0, 5초 초과 0). 품질 게이트 v3·최종 기준값으로 다시 실행했고 이전 실행은 보관본. Windows 파일 교체 오류 뒤 397명 기록을 보존하고 1회 재개함(보고서에 공개) |
| 최종 PDF/XLSX/TXT | `outputs/release/FaceTeller_전체구현_실험보고서.*` | 필수 자료 완비, `--validate-only` 통과. main이 생성·복사 |
| 프런트엔드·DB·런타임 범위 | `docs/release/frontend-report.txt`, `database-report.txt`, `runtime-report.txt` | 각 문서의 합성/실측 범위 참조 |

최종 보고서 생성은 CPU 작업입니다. 필수 400명·배포·깊이·corpus 검증에 실패하면 생성이 중단됩니다.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/build_report.py --validate-only
.\.venv\Scripts\python.exe -B -X utf8 scripts/release/build_report.py
```

정상 생성 시 `C:/Users/user/Downloads/FaceTeller_완성프로젝트_공유자료`에 보고서 3종을 복사합니다. 처리시간은 평균뿐 아니라 p95·최대·5초 초과 수를 보고하고 예열을 분리합니다. CNN 단계 시간과 전체 HTTP 시간을 혼동하지 않습니다. 상세 계약은 [보고서 생성기 안내](docs/release/report-builder.txt)를 참고하세요.

CPU 시작점 검증: Windows PowerShell 문법, 내장 Python 구문, 직접 의존성15개 버전 대조를 확인했습니다. `-ValidateOnly`는 미완성 배포 manifest·깊이 설정을 감지해 DB 변경/모델 로드 전에 중단했습니다. 별도 읽기 전용 DB 검사는 활성 corpus 일치와 변경 후 불일치 차단을 확인했으며, 실제 서버/GPU는 기동하지 않았습니다.

## 개인정보·공유·라이선스

서비스 입력 사진은 메모리에서만 처리합니다. 분석 사진·원시 좌표·개인별 분석 이력을 저장/로그/외부 AI 전송하는 경로를 제공하지 않습니다. 오프라인 학습 자료와 실서비스의 사진 처리 정책은 구분합니다.

**공유 보고서와 배포 자료에 K-FACE 원본 얼굴 파일, crop, 원시 좌표, 인물 식별자, 개인별 예측을 포함하지 않습니다.** 원본 ZIP·파생 이미지·비공개 레이블/체크포인트·DB 비밀은 자료 재배포 목록에서 제외합니다. 코드 공유가 승인된 연구 자료나 가중치 재배포 권한까지 부여하지 않습니다. K-FACE 이용은 제공기관 승인·약정의 적용 범위를 따릅니다.

현재 사용하는 원 연구 **`facebook/VGGT-1B` 체크포인트에는 비상업용 제한이 남아 있습니다**. 별도 `VGGT-1B-Commercial`의 조건을 현재 가중치에 적용하지 않으며 상용 준비 완료라고 표현하지 않습니다. [VGGT 공식 안내](https://github.com/facebookresearch/vggt)와 [원 체크포인트 모델 카드](https://huggingface.co/facebook/VGGT-1B)를 확인하세요. 문헌 원저작·Wikisource 전사본·모델/글꼴의 각 출처와 라이선스도 별도로 보존합니다.

## 과거 버전

과거 FRLL102명 작업대·K-FACE v4 설명은 [이전 README 원본](docs/archive/README-before-release.md)에 보존했습니다. `start-workbench.ps1`, `start-pilot.ps1`, `web/v2.html`, 기존 모델과 과거 보고서는 연구 이력이며 이번 release 시작 경로와 별개입니다. 기존 실행 스크립트는 변경하지 않았습니다.
