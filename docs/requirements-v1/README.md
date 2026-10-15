# 설계서 생성 원본

`build_content.py`가 검토용 v0.9의 본문 원본이다. 같은 원본에서 Markdown, 구조화 JSON, 요구사항·시험 계획 CSV를 만든다. `render_docx.py`는 JSON으로 Word를 만들며 `export_word.vbs`가 Word의 실제 페이지 배치와 목차를 갱신해 PDF로 내보낸다.

생성 순서:

1. Python으로 `build_content.py` 실행.
2. python-docx가 설치된 Python으로 `render_docx.py` 실행.
3. Windows `cscript.exe //nologo export_word.vbs <DOCX 절대경로> <PDF 절대경로>` 실행. Microsoft Word가 필요하다.
4. pypdf, pypdfium2, Pillow가 설치된 Python으로 `verify_render.py` 실행.
5. `tmp/requirements-v1/`의 전체 페이지·축소판을 육안 검수한다. 검수 기록과 ZIP은 변경 후 다시 작성한다.

참조 파일은 기존 프로토타입의 스키마·보고서이다. Word를 직접 수정했다면 원본에도 변경을 반영해야 다음 생성에서 덮어쓰지 않는다. 문서 검수는 실제 서비스·모델의 수용시험 실행과 다르다.
