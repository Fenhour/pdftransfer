@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto failed
".venv\Scripts\python.exe" -c "import pymupdf, rapidocr_onnxruntime" >nul 2>nul
if errorlevel 1 goto failed
".venv\Scripts\python.exe" launch.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Please run setup.bat to create or repair the local environment first.
pause
exit /b 1
