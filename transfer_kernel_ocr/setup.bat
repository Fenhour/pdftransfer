@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" --setup-only
if errorlevel 1 (
  pause
  exit /b 1
)
