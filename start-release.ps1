[CmdletBinding()]
param(
    [ValidateRange(1, 65535)] [int] $Port = 8767,
    [switch] $Https,
    [switch] $ValidateOnly,
    [switch] $SkipLocalInit
)

# Windows PowerShell 5.1 compatible. CPU preflight never loads model weights.
# Normal startup uses CUDA; do not run it while training owns the GPU.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$releaseRoot = $PSScriptRoot
$releasePython = Join-Path $releaseRoot '.venv\Scripts\python.exe'

function Invoke-ReleasePython {
    param([string[]] $Arguments)
    & $releasePython -B -X utf8 @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Release command failed (exit $LASTEXITCODE). Service was not started."
    }
}

Push-Location $releaseRoot
try {
    if ($Https) {
        if ($PSBoundParameters.ContainsKey('Port') -and $Port -ne 8768) {
            throw 'Https uses separate loopback port 8768. The HTTP coordinator on 8767 is unchanged.'
        }
        $Port = 8768
    }
    if (-not (Test-Path -LiteralPath $releasePython -PathType Leaf)) {
        throw 'Missing .venv\Scripts\python.exe. Follow the Windows setup in README.md.'
    }
    Write-Host 'Checking Windows venv, direct runtime dependencies and required local files...'
    $preflight = @'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

root = Path.cwd()
if sys.platform != 'win32' or Path(sys.prefix).resolve() != (root / '.venv').resolve() or sys.prefix == sys.base_prefix:
    raise SystemExit('A Windows project .venv is required; no global Python fallback.')
names = ['fastapi', 'uvicorn', 'python-multipart', 'numpy', 'pillow', 'mediapipe',
         'opencv-contrib-python', 'psycopg', 'psycopg-binary', 'scipy',
         'huggingface-hub', 'einops', 'torch', 'torchvision']
missing_packages = []
for name in names:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        missing_packages.append(name)
vendor = root / 'models/face_parsing/python'
ort_versions = [d.version for d in metadata.distributions(path=[str(vendor)])
                if (d.metadata.get('Name') or '').lower() == 'onnxruntime']
if not ort_versions:
    try:
        metadata.version('onnxruntime')
    except metadata.PackageNotFoundError:
        missing_packages.append('onnxruntime')
if missing_packages:
    raise SystemExit('Missing runtime packages: ' + ', '.join(missing_packages) + '. See requirements-release.txt and the separate CUDA instructions.')
if '+cu' not in metadata.version('torch'):
    raise SystemExit('CUDA-enabled PyTorch is required for the real VGGT service. See README.md; no CUDA operation was attempted.')
files = [
    'scripts/release/serve.py', 'scripts/release/setup_database.py',
    'data/release/training_registry.json', 'models/release/deployment.json',
    'data/release/depth_config.json', 'data/release/literature/reference_corpus_v1.json',
    'models/face_landmarker.task', 'models/face_parsing/resnet18.onnx',
    'third_party/vggt/provenance.json', 'third_party/vggt/vggt/models/vggt.py',
    'frontend/dist/index.html', 'frontend/dist/assets/mediapipe/manifest.json',
    'frontend/dist/assets/mediapipe/face_landmarker.task',
    'frontend/dist/assets/mediapipe/vision-worker.js',
    'frontend/dist/assets/mediapipe/wasm/vision_wasm_internal.js',
    'frontend/dist/assets/mediapipe/wasm/vision_wasm_internal.wasm',
    'frontend/dist/assets/mediapipe/wasm/vision_wasm_nosimd_internal.js',
    'frontend/dist/assets/mediapipe/wasm/vision_wasm_nosimd_internal.wasm',
]
missing = [name for name in files if not (root / name).is_file()]
if missing:
    raise SystemExit('Required release artifacts are not ready:\n  ' + '\n  '.join(missing) + '\nNo DB initialization or model loading was performed.')
depth = json.loads((root / 'data/release/depth_config.json').read_text(encoding='utf-8-sig'))
checkpoint = (root / depth.get('checkpoint_path', 'models/vggt/model.pt')).resolve()
if not checkpoint.is_relative_to(root.resolve()) or not checkpoint.is_file():
    raise SystemExit('Configured local VGGT checkpoint is missing/invalid. No automatic download or pilot fallback.')
print('Windows venv and required files present. CUDA availability has not been probed.')
'@
    Invoke-ReleasePython -Arguments @('-c', $preflight)
    $releaseTlsArguments = @()
    if ($Https) {
        $releaseTlsRoot = Join-Path $env:LOCALAPPDATA 'FaceTeller\tls'
        if (-not $ValidateOnly) {
            Write-Host 'Preparing private localhost development TLS files; no trust-store or firewall changes...'
            Invoke-ReleasePython -Arguments @('scripts/release/setup_local_tls.py')
        }
        $releaseTlsArguments = @('--https', '--ssl-certfile', (Join-Path $releaseTlsRoot 'localhost-cert.pem'),
                                 '--ssl-keyfile', (Join-Path $releaseTlsRoot 'localhost-key.pem'))
        Write-Host 'Self-signed development certificate: browsers do not trust it by default; this is not public deployment.'
    }
    Write-Host 'Checking deployment manifests and weight/source hashes (CPU only)...'
    Invoke-ReleasePython -Arguments (@('scripts/release/serve.py', '--validate-only') + $releaseTlsArguments)

    $externalReader = -not [string]::IsNullOrWhiteSpace($env:FACETELLER_DATABASE_URL)
    if (-not $externalReader -and -not [string]::IsNullOrWhiteSpace($env:FACETELLER_ADMIN_DATABASE_URL)) {
        throw 'An external admin DSN is set without FACETELLER_DATABASE_URL for the runtime reader. Configure the reader explicitly.'
    }
    if ($ValidateOnly) {
        Write-Host 'ValidateOnly: skipping DB local-init, downloads, model loading and server startup.'
    } elseif ($externalReader) {
        Write-Host 'Using the configured external reader. DB local-init is skipped.'
    } elseif ($SkipLocalInit) {
        Write-Host 'SkipLocalInit: using the existing local DB configuration.'
    } else {
        Write-Host 'Ensuring the dedicated local PostgreSQL instance is initialized/running...'
        Invoke-ReleasePython -Arguments @('scripts/release/setup_database.py', 'local-init')
    }

    Write-Host 'Checking the DB reader and active corpus (read-only)...'
    $databaseCheck = @'
import hashlib
import json
from pathlib import Path
import sys
import psycopg
from facework.release_repository import database_dsn

class PreflightError(RuntimeError):
    pass

try:
    corpus = json.loads(Path('data/release/literature/reference_corpus_v1.json').read_text(encoding='utf-8-sig'))
    canonical = json.dumps(corpus, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    expected = hashlib.sha256(canonical).hexdigest()
    with psycopg.connect(database_dsn(), connect_timeout=5) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        role = conn.execute('SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname=current_user').fetchone()
        if any(role) or conn.execute('SHOW default_transaction_read_only').fetchone()[0] != 'on':
            raise PreflightError('Reader must be unprivileged and read-only')
        for table in ('literature', 'passage', 'label', 'label_passage', 'active_corpus'):
            if conn.execute('SELECT has_table_privilege(current_user,%s,%s)', ('face_release.' + table, 'INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER')).fetchone()[0]:
                raise PreflightError('Reader has write privileges')
        active = conn.execute('SELECT canonical_sha256 FROM face_release.active_corpus WHERE singleton=1').fetchone()
        eligible = conn.execute('SELECT count(*) FROM face_release.passage WHERE service_eligible').fetchone()[0]
        if not active or active[0] != expected or eligible <= 0 or eligible != sum(p['service_eligible'] for p in corpus['passages']):
            raise PreflightError('Active DB corpus is absent, empty or differs from the local corpus. Import the intended current corpus explicitly.')
    print('DB reader and active corpus verified; eligible passages: ' + str(eligible))
except PreflightError as exc:
    print('DB preflight failed: ' + str(exc), file=sys.stderr)
    raise SystemExit(2)
except Exception as exc:
    # Do not expose a DSN, password, participant data or server exception details.
    print('DB preflight failed: ' + type(exc).__name__ + '. Check local-init, reader configuration and explicit corpus import; see README.md.', file=sys.stderr)
    raise SystemExit(2)
'@
    Invoke-ReleasePython -Arguments @('-c', $databaseCheck)
    if ($ValidateOnly) {
        Write-Host 'CPU preflight passed. No CUDA check, inference, server startup or 400-person verification was performed.'
    } else {
        $releaseScheme = if ($Https) { 'https' } else { 'http' }
        Write-Host "Starting FaceTeller release at ${releaseScheme}://127.0.0.1:$Port/ (CUDA model loading follows)..."
        Invoke-ReleasePython -Arguments (@('scripts/release/serve.py', '--port', "$Port") + $releaseTlsArguments)
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
} finally {
    Pop-Location
}
