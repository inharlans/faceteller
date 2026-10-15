$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn facework.server:app --host 127.0.0.1 --port 8765
