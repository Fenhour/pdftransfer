@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 goto failed
)
python -m pip --python ".venv\Scripts\python.exe" install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Double-click start.bat.
pause
exit /b 0
:failed
echo Setup failed. Install Python 3.10 or newer with pip, then try again.
pause
exit /b 1
