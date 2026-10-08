@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch.py
) else (
    python launch.py
)
if errorlevel 1 (
    echo.
    echo Please run setup.bat to install dependencies first.
    pause
)
