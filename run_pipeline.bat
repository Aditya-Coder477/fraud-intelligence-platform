@echo off
REM ─────────────────────────────────────────────────────────────
REM  Mule Account Feature Engineering — launcher
REM  Run this from the project folder:
REM      run_pipeline.bat
REM ─────────────────────────────────────────────────────────────
echo.
echo === Mule Account Feature Engineering Pipeline ===
echo.

REM Activate the existing venv
call "%~dp0venv\Scripts\activate.bat"

REM Run the pipeline
python "%~dp0feature_engineering_pipeline.py"

echo.
echo === Done ===
pause
