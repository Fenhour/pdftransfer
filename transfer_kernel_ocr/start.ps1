$ErrorActionPreference = 'Stop'
$AppArguments = @($args)
$setupOnly = $AppArguments -contains '--setup-only'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$port = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json).port
if ($port -isnot [int] -or $port -lt 1024 -or $port -gt 65535) { throw 'config.json port must be an integer between 1024 and 65535.' }
if ($AppArguments -contains '--port') {
    $portIndex = [Array]::IndexOf($AppArguments, '--port')
    if ($portIndex + 1 -ge $AppArguments.Count) { throw '--port requires a value.' }
    $port = [int]$AppArguments[$portIndex + 1]
}
if ($port -lt 1024 -or $port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
$dataRoot = $PSScriptRoot
if ($env:OCR_DATA_ROOT) { $dataRoot = [IO.Path]::GetFullPath($env:OCR_DATA_ROOT) }
$expectedOutput = Join-Path $dataRoot 'output'
$recognitionBackend = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'config.json') -Raw | ConvertFrom-Json).recognition_backend
if ($env:PDFTRANSFER_RECOGNITION) { $recognitionBackend = $env:PDFTRANSFER_RECOGNITION }
if (-not $recognitionBackend) { $recognitionBackend = 'v3' }
# V3 defaults to a lazy local Python child. No Docker or HTTP worker startup.
# Install it once with services/structure_v3/setup-native.ps1; model paths are configurable.
# File conversion and installation must run even if the server is already open.
if ($AppArguments.Count -eq 0 -or ($AppArguments | Where-Object { $_ -notin @('--no-browser', '--port', [string]$port) }).Count -eq 0) {
try {
    $healthResponse = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 2
    $health = [Text.Encoding]::UTF8.GetString($healthResponse.RawContentStream.ToArray()) | ConvertFrom-Json
    if ($health.status -eq 'ok' -and $health.version -eq '1.0.0' -and
        $health.storage.output -eq $expectedOutput -and $health.ocr.backend -eq $recognitionBackend) {
        if ($AppArguments -notcontains '--no-browser') { Start-Process "http://127.0.0.1:$port" }
        exit 0
    }
} catch { }
}

function Test-Python([string]$Executable) {
    $ErrorActionPreference = 'Continue'
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { return $false }
    & $Executable -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)' 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Python $python)) {
    $basePython = $null
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        foreach ($version in @('-3.12', '-3.13')) {
            $candidate = $null
            try { $candidate = & py.exe $version -c 'import sys; print(sys.executable)' 2>$null } catch { continue }
            if ($LASTEXITCODE -eq 0 -and $candidate -and (Test-Python $candidate)) {
                $basePython = $candidate
                break
            }
        }
    }
    if (-not $basePython) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
        $candidates = @($bundledPython)
        if ($systemPython) { $candidates += $systemPython.Source }
        foreach ($candidate in $candidates) {
            if (Test-Python $candidate) { $basePython = $candidate; break }
        }
    }
    if (-not $basePython) { throw 'Please install Python 3.12 or 3.13 and try again.' }

    # Virtual environments copied from another computer must be recreated.
    $environmentPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.venv'))
    if (Test-Path -LiteralPath $environmentPath) {
        $backupPath = Join-Path $PSScriptRoot ('.venv.backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6))
        $backupPath = [IO.Path]::GetFullPath($backupPath)
        $projectPath = [IO.Path]::GetFullPath($PSScriptRoot)
        if ([IO.Path]::GetDirectoryName($environmentPath) -ne $projectPath -or
            [IO.Path]::GetDirectoryName($backupPath) -ne $projectPath) {
            throw 'Environment paths must stay inside the project directory.'
        }
        Move-Item -LiteralPath $environmentPath -Destination $backupPath
        Write-Host "Preserved the previous environment: $backupPath"
    }
    Write-Host "Creating local Python environment with $basePython"
    & $basePython -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create local Python environment.' }
}

function Test-Pip {
    $ErrorActionPreference = 'Continue'
    & $python -m pip --version 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-Pip)) {
    & $python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize pip in the local environment.' }
}
function Test-Dependencies {
    $ErrorActionPreference = 'Continue'
    & $python -c 'import pymupdf, fastapi, uvicorn, rapidocr_onnxruntime, cv2, numpy, PIL, python_multipart; import backend.app' 2>$null
    return ($LASTEXITCODE -eq 0)
}
if (-not (Test-Dependencies)) {
    & $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check your network and run again.' }
    & $python -c 'import backend.app; import rapidocr_onnxruntime'
    if ($LASTEXITCODE -ne 0) { throw 'Application dependencies could not be loaded.' }
}
if ($setupOnly) { Write-Host 'Environment ready.'; exit 0 }
Write-Host "PDF OCR Parser: http://127.0.0.1:$port" -ForegroundColor Cyan
Write-Host 'Keep this window open. Press Ctrl+C to stop.'
& $python (Join-Path $PSScriptRoot 'run.py') @AppArguments
exit $LASTEXITCODE




