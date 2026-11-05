param([string]$Python = '', [switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if ($taskRoot -match '[^\x00-\x7F]') {
    throw 'Install in an ASCII-only directory, for example C:\pdftransfer.'
}
if (-not $Python) {
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $Python = & py.exe -3.12 -c 'import sys; print(sys.executable)'
    } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        $Python = (Get-Command python.exe).Source
    }
}
if (-not $Python) { throw 'Install Python 3.12 (64-bit), then run setup.ps1 again.' }
& $Python -c "import sys,struct; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8, 'Python 3.12 64-bit is required'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
function Invoke-CheckedPython([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE). Fix the error above and rerun setup.ps1." }
}
$appDirectory = Join-Path $taskRoot '.venv'
$nativeDirectory = Join-Path $taskRoot 'services\structure_v3\.venv'
if (-not (Test-Path -LiteralPath (Join-Path $appDirectory 'Scripts\python.exe'))) {
    Invoke-CheckedPython $Python @('-m','venv',$appDirectory)
}
if (-not (Test-Path -LiteralPath (Join-Path $nativeDirectory 'Scripts\python.exe'))) {
    Invoke-CheckedPython $Python @('-m','venv',$nativeDirectory)
}
$appPython = Join-Path $appDirectory 'Scripts\python.exe'
$nativePython = Join-Path $nativeDirectory 'Scripts\python.exe'
Invoke-CheckedPython $appPython @('-m','pip','install','--upgrade','pip')
Invoke-CheckedPython $appPython @('-m','pip','install','-r',(Join-Path $taskRoot 'requirements.txt'))
Invoke-CheckedPython $appPython @('-m','pip','check')
Invoke-CheckedPython $nativePython @('-m','pip','install','--upgrade','pip')
Invoke-CheckedPython $nativePython @('-m','pip','install','-r',(Join-Path $taskRoot 'services\structure_v3\requirements-native.txt'))
Invoke-CheckedPython $nativePython @('-m','pip','check')
if (-not $SkipModels) {
    Invoke-CheckedPython $nativePython @('-X','utf8',(Join-Path $taskRoot 'services\structure_v3\prepare_models.py'))
}
Write-Host ''
Write-Host 'Installation complete. Start the app:' -ForegroundColor Green
Write-Host 'powershell -NoProfile -ExecutionPolicy Bypass -File .\start.ps1'
