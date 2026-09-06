@echo off
title PD Detect - Backend
color 0A

echo.
echo  =====================================================
echo    PD DETECT - Backend API Server
echo    URL: http://localhost:8000
echo  =====================================================
echo.

cd /d "%~dp0backend"

REM ── Check Python ──────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During install, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version

REM ── Create venv if missing ────────────────────────────
if not exist venv\Scripts\python.exe (
    echo.
    echo [INFO] Setting up virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM ── Install dependencies ──────────────────────────────
echo.
echo [INFO] Installing Python packages (first time takes 2-3 minutes)...
venv\Scripts\pip install --quiet fastapi==0.111.0 uvicorn==0.30.1 python-multipart==0.0.9 numpy==1.26.4 scikit-learn==1.5.0 xgboost==2.0.3 joblib==1.4.2 pandas==2.2.2 reportlab==4.2.2 scipy==1.13.0 soundfile==0.12.1 gunicorn==22.0.0
if errorlevel 1 (
    echo [ERROR] Package installation failed. Check internet connection.
    pause
    exit /b 1
)
echo [OK] Packages installed.

REM ── Check/train models ───────────────────────────────
if not exist models\ensemble.pkl (
    echo.
    echo [INFO] Training ML models on UCI Parkinson Dataset...
    echo [INFO] This takes about 2-3 minutes (downloads ~3KB dataset)...
    echo.
    venv\Scripts\python model_trainer.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Models trained and saved!
)

REM ── Start server ──────────────────────────────────────
echo.
echo [OK] Starting backend server...
echo.
echo  =====================================================
echo    Backend is running at: http://localhost:8000
echo    API Docs:              http://localhost:8000/docs
echo    Press Ctrl+C to stop.
echo  =====================================================
echo.
venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000 --reload
echo.
echo Server stopped.
pause
