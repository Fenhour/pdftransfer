@echo off
cd /d "%~dp0"
set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 set "PYTHON=py -3"
%PYTHON% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Double-click start.bat.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.10 or newer with pip, then try again.
pause
exit /b 1
