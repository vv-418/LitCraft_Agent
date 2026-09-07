@echo off
chcp 65001 >nul
setlocal EnableExtensions
title LitCraft Agent 一键启动

set "ROOT=%~dp0"
cd /d "%ROOT%"
if errorlevel 1 (
  echo [ERROR] Cannot cd to project root
  goto :fail
)

echo ============================================
echo   LitCraft Agent - start
echo ============================================
echo.
echo   ROOT: %ROOT%
echo   Config: .env
echo.

set "PYTHONUNBUFFERED=1"

REM Do not clear system proxy. Only prefer direct for localhost.
REM Avoid "::1" here: inside bat files "::" starts a comment and breaks parsing.
set "NO_PROXY=localhost,127.0.0.1"
set "no_proxy=localhost,127.0.0.1"

if defined LITCRAFT_PYTHON (
  set "CONDA_PYTHON=%LITCRAFT_PYTHON%"
) else (
  set "CONDA_PYTHON=d:\Users\vv\Anaconda3\envs\litcraft\python.exe"
)

if not exist "%CONDA_PYTHON%" (
  echo [ERROR] Python not found: "%CONDA_PYTHON%"
  echo         Set LITCRAFT_PYTHON to your python.exe
  goto :fail
)

if not exist "%ROOT%.env" (
  echo [WARN]  .env missing - copy from .env.example
  echo.
)

where npm >nul 2>&1
if errorlevel 1 (
  echo [ERROR] npm not found - install Node.js first
  goto :fail
)

if not exist "%ROOT%frontend\package.json" (
  echo [ERROR] missing frontend\package.json
  goto :fail
)

REM Do NOT write ...\node_modules\  (trailing backslash escapes the quote)
if not exist "%ROOT%frontend\node_modules" (
  echo [SETUP] npm install ...
  pushd "%ROOT%frontend"
  call npm install
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed
    goto :fail
  )
  popd
  echo.
)

echo [0/2] Stop previous LitCraft windows / ports if still running
call :stop_old
echo.

echo [1/2] Backend  http://localhost:8000
start "LitCraft Backend" /D "%ROOT%" cmd /k ""%CONDA_PYTHON%" api\server.py"
echo       window: LitCraft Backend
echo.

timeout /t 2 /nobreak >nul

echo [2/2] Frontend http://localhost:5173
start "LitCraft Frontend" /D "%ROOT%frontend" cmd /k "npm run dev"
echo       window: LitCraft Frontend
echo.

echo ============================================
echo   Started
echo   API:  http://localhost:8000
echo   UI:   http://localhost:5173
echo   Python: %CONDA_PYTHON%
echo ============================================
echo.
pause
endlocal
exit /b 0

:stop_old
REM Close the cmd windows this script creates (kills python / uvicorn / node children).
taskkill /FI "WINDOWTITLE eq LitCraft Backend*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq LitCraft Frontend*" /T /F >nul 2>&1

REM Leftover python/node if the window title already changed.
set "ROOT_NS=%ROOT%"
if "%ROOT_NS:~-1%"=="\" set "ROOT_NS=%ROOT_NS:~0,-1%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -like '*api\server.py*' -or $_.CommandLine -like '*api.server:app*' -or ($_.Name -eq 'node.exe' -and $_.CommandLine -like '*vite*')) -and $_.CommandLine -like '*%ROOT_NS%*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

call :kill_listen_port 8000
call :kill_listen_port 5173
timeout /t 1 /nobreak >nul
exit /b 0

:kill_listen_port
REM %~1 = port. Kill whatever is still LISTENING (uvicorn child / leftover node).
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%~1 .*LISTENING"') do (
  if not "%%P"=="0" if not "%%P"=="4" (
    echo       stop PID %%P on port %~1
    taskkill /F /PID %%P /T >nul 2>&1
  )
)
exit /b 0

:fail
echo.
echo Start failed. See messages above.
pause
endlocal
exit /b 1

exit /b 1