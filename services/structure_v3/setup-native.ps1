param([string]$Python = '', [string]$IndexUrl = 'https://pypi.org/simple')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $Python) {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $Python = & py.exe -3.12 -c 'import sys; print(sys.executable)'
    }
    if (-not $Python) { throw 'Install Python 3.12 or supply -Python with its executable path.' }
}
$nativePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $nativePython -PathType Leaf)) {
    & $Python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create native V3 environment.' }
}
& $nativePython -X utf8 -m pip install --index-url $IndexUrl -r (Join-Path $PSScriptRoot 'requirements-native.txt')
if ($LASTEXITCODE -ne 0) { throw 'Native V3 dependency installation failed.' }
& $nativePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Native V3 dependency validation failed.' }
New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot '..\..\log') -Force | Out-Null
Write-Host 'Native V3 is installed. Configure native_config.json model_dir or PDFTRANSFER_V3_MODEL_DIR.'
Write-Host 'Applications load the models on demand in a resident Python child; Docker is not required.'
