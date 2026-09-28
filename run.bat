@echo off
title GoldSight AI - Gold Price Prediction
color 0E

echo ============================================
echo    GoldSight AI - Gold Price Prediction
echo ============================================
echo.

:: Pre-flight checks
echo [0/4] Checking environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found. Please install Python and ensure it's on PATH.
    pause
    exit /b 1
)
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Node.js not found. Please install Node.js and ensure it's on PATH.
    pause
    exit /b 1
)
echo       Environment OK.
echo.

:: ── Backend Setup ──
echo [1/4] Installing backend dependencies...
pushd "%~dp0backend"
python -m pip install -r requirements.txt --quiet 2>"%~dp0backend\backend-install.log"
if %errorlevel% neq 0 (
    echo [!] pip install failed. See "%~dp0backend\backend-install.log" for details.
    popd
    pause
    exit /b 1
)
popd
echo       Done.
echo.

:: ── Frontend Setup ──
echo [2/4] Installing frontend dependencies...
pushd "%~dp0frontend"
if exist "%~dp0frontend\node_modules" (
    echo       Skipping install — node_modules already present.
    set "FRONTEND_SKIPPED=1"
)
if not defined FRONTEND_SKIPPED (
    if exist "%~dp0frontend\package-lock.json" (
        npm ci --prefer-offline --no-audit --progress=false >"%~dp0frontend\frontend-install.log" 2>&1
    ) else (
        npm install --no-audit --progress=false >"%~dp0frontend\frontend-install.log" 2>&1
    )
    if %errorlevel% neq 0 (
        echo [!] npm install failed. See "%~dp0frontend\frontend-install.log" for details.
        popd
        pause
        exit /b 1
    )
)
popd
echo       Done.
echo.

:: ── Start Backend ──
echo [3/4] Starting backend API on http://localhost:7860 ...
start "GoldSight API" /D "%~dp0backend" cmd /k "title GoldSight API - Backend && color 0A && python run_server.py"
echo       Waiting for API to become ready (up to 45s)...
set /a WAIT_COUNT=0
:wait_api
timeout /t 3 /nobreak >nul
powershell -NoProfile -Command "try { (Invoke-WebRequest -Uri 'http://127.0.0.1:7860/health' -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200 } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 goto api_ready
set /a WAIT_COUNT+=1
if %WAIT_COUNT% lss 15 goto wait_api
echo [!] API not responding yet — open http://127.0.0.1:7860/health manually, then refresh the dashboard.
goto api_done
:api_ready
echo       Backend API is ready.
:api_done
echo.

:: ── Start Frontend ──
echo [4/4] Starting frontend on http://localhost:3000 ...
start "GoldSight Dashboard" /D "%~dp0frontend" cmd /k "title GoldSight Dashboard - Frontend && color 0B && npm run dev"
timeout /t 3 /nobreak >nul
echo       Frontend started.
echo.

echo ============================================
echo    All services running!
echo.
echo    Dashboard:  http://localhost:3000
echo    API:        http://127.0.0.1:7860
echo    API Docs:   http://127.0.0.1:7860/docs
echo ============================================
echo.
echo Press any key to open the dashboard...
pause >nul
start http://localhost:3000
