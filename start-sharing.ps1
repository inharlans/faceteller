param()
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$health = Invoke-RestMethod 'http://127.0.0.1:8767/api/v1/health' -TimeoutSec 15
if (-not $health.ready) { throw 'Start start-release.ps1 and wait until the service is ready.' }
$install = Get-Content (Join-Path $PSScriptRoot 'outputs/release/cloudflared-install.json') -Raw | ConvertFrom-Json
$tool = Join-Path $env:LOCALAPPDATA 'FaceTeller/tools/cloudflared-windows-amd64.exe'
if (-not (Test-Path $tool)) { throw 'The verified cloudflared executable is missing.' }
$actualHash = 'sha256:' + (Get-FileHash $tool -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $install.sha256) { throw 'cloudflared executable checksum changed.' }
$logRoot = Join-Path $env:LOCALAPPDATA 'FaceTeller/sharing'
New-Item -ItemType Directory -Force $logRoot | Out-Null
$logFile = Join-Path $logRoot ('tunnel-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Write-Host 'Starting an external HTTPS link for FaceTeller. Keep this terminal and the GPU service running.'
Write-Host 'Press Ctrl+C here to stop external access. A restart creates a different URL.'
Write-Host ('Tunnel log: ' + $logFile)
& $tool tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:8767 --http-host-header localhost --logfile $logFile
if ($LASTEXITCODE -ne 0) { throw "Tunnel exited with code $LASTEXITCODE" }
