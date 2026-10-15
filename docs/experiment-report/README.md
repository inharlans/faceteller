# 2026-10-01 실제 실험 보고서

산출물: `outputs/experiment-report-2026-10-01/`.

재생성 순서:

1. 프로젝트 `.venv/Scripts/python.exe`로 `scripts/reporting/audit_experiments.py` 실행. 보관된 체크포인트 50개와 NPZ 400개를 읽으며 학습·이미지 추론은 실행하지 않는다.
2. 같은 Python으로 `scripts/reporting/build_experiment_report.py` 실행. 지표·클래스 인원·모델 이력 CSV와 Markdown·JSON을 생성한다.
3. python-docx가 설치된 Python으로 `scripts/reporting/render_experiment_docx.py` 실행.
4. `cscript.exe //nologo docs/requirements-v1/export_word.vbs <DOCX 절대경로> <PDF 절대경로>`로 Word 목차 갱신·PDF 내보내기. Microsoft Word가 필요하다.
5. pypdf·pypdfium2·Pillow가 설치된 Python으로 `scripts/reporting/verify_experiment_report.py` 실행 후 `tmp/experiment-report/`의 모든 페이지를 육안 검수한다.
6. 먼저읽기·검수 기록·ZIP을 갱신한다. 기존 보고서와 원본 학습 자료는 변경하지 않는다.

보관 실험을 재집계하는 스크립트다. 보고서의 인원·crop 수 대조를 새 모델 평가 실행으로 표현하지 않는다.
