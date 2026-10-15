# K-FACE v2 전체 재현 (프로젝트 루트에서 실행). 원본 ZIP 경로가 다르면 -Source 로 지정.
param([string]$Source = "$HOME\Downloads\High_Resolution.zip", [switch]$SkipTraining)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:TORCH_HOME = (Join-Path (Get-Location) 'models/torch_cache')
$py = '.\.venv\Scripts\python.exe'
& $py scripts/kface_extract.py --source $Source            # 1. 참여자별 ZIP 400개, CRC 검증 (약 3분)
& $py scripts/kface_probe.py                               # 2. 카메라·조명·표정 조건 확인
& $py scripts/kface_landmarks.py --workers 8               # 3. MediaPipe 랜드마크·측정·정규화 crop (약 9분)
& $py scripts/kface_depth.py                               # 4. 12시점 깊이 + 재현성
& $py scripts/kface_add_metrics.py                         # 5a. 랜드마크 지표 추가(v3)
& $py scripts/kface_labels.py                              # 5. 사람별 기준 레이블 (시각 레이블 v3 TSV 병합)
& $py scripts/kface_fewview_eval.py                        # 6. 정면+사선 3장 깊이 보정·검증
if (-not $SkipTraining) {
  & $py -m facework.kface_training --part all --epochs 6 --per-subject 36 --workers 6   # 7. 부위별 CNN (RTX 3070 약 75분)
}
& $py scripts/kface_evaluate.py                            # 8. 시험 참여자 조건별 평가
& $py scripts/kface_visual_confidence.py                   # 8b. 육안 특징 확률-정확도 표
& $py scripts/kface_external_check.py                      # 9. FRLL 외부 일관성
& $py -m unittest discover -s tests -v                     # 10. 자동 검사
