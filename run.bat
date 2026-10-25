@echo off
title BlinkCV - Digital Eye Strain Monitor
cd /d "%~dp0"
echo ==================================================
echo             Starting BlinkCV Monitor...
echo ==================================================
echo.
.venv\Scripts\python.exe main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Program exited with error code %ERRORLEVEL%.
    pause
)
