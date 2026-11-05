param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
if ($PSScriptRoot -match '[^\x00-\x7F]') { throw 'Use an ASCII-only installation directory.' }
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) { throw 'Run setup.ps1 first.' }
$env:PYTHONUTF8 = '1'
$env:PDFTRANSFER_V3_PYTHON = Join-Path $PSScriptRoot 'services\structure_v3\.venv\Scripts\python.exe'
$env:PDFTRANSFER_V3_MODEL_DIR = Join-Path $PSScriptRoot 'V3\models'
$env:PDFTRANSFER_V3_TRANSPORT = 'native'
Set-Location -LiteralPath $PSScriptRoot
$appArguments = @('-X','utf8',(Join-Path $PSScriptRoot 'run.py'),'--port',"$Port")
if ($NoBrowser) { $appArguments += '--no-browser' }
& $appPython @appArguments
exit $LASTEXITCODE
