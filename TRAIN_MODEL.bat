@echo off
title PD Detect - Model Training
color 0E
echo.
echo  ========================================
echo    PD Detect ^| ML Model Training
echo  ========================================
echo.
echo  This will:
echo    1. Download UCI Parkinson's Dataset (195 samples)
echo    2. Train Random Forest (200 trees)
echo    3. Train XGBoost (200 estimators)
echo    4. Train Calibrated SVM
echo    5. Build Voting Ensemble
echo    6. Save all models to backend\models\
echo.
echo  Expected accuracy: ~92-97%%
echo  Expected time:     ~2-3 minutes
echo.
pause

cd /d "%~dp0backend"
venv\Scripts\python model_trainer.py

echo.
if errorlevel 1 (
    echo [ERROR] Training failed. Make sure backend dependencies are installed.
    echo [INFO]  Run START_BACKEND.bat first to install dependencies.
) else (
    echo [SUCCESS] Models trained and saved to backend\models\
    echo [INFO]    Restart START_BACKEND.bat to load the new models.
)
echo.
pause
