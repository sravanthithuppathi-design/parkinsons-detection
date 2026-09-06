@echo off
title PD Detect - Frontend
color 0B

echo.
echo  =====================================================
echo    PD DETECT - Frontend Web App
echo    URL: http://localhost:5173
echo  =====================================================
echo.

cd /d "%~dp0frontend"

REM ── Check Node.js ──────────────────────────────────────
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo.
    echo Please install Node.js 18+ from:
    echo   https://nodejs.org/en/download
    echo.
    echo After installing, restart this file.
    echo.
    pause
    exit /b 1
)

echo [OK] Node.js found:
node --version

REM ── Install npm packages ───────────────────────────────
if not exist node_modules (
    echo.
    echo [INFO] Installing npm packages (first time takes 1-2 minutes)...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed. Check internet connection.
        pause
        exit /b 1
    )
    echo [OK] Packages installed.
)

REM ── Start frontend ────────────────────────────────────
echo.
echo [OK] Starting frontend server...
echo.
echo  =====================================================
echo    App is running at: http://localhost:5173
echo    Open this URL in your browser.
echo    Press Ctrl+C to stop.
echo  =====================================================
echo.
call npm run dev
echo.
echo Server stopped.
pause
