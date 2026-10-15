$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& '.\.venv\Scripts\python.exe' -X utf8 'scripts/pilot/serve.py' --exploratory-depth
