@echo off
chcp 65001 >nul
title LitCraft Agent 一键启动

echo ============================================
echo   LitCraft 智能文献综述 Agent - 一键启动
echo ============================================
echo.

set CONDA_PYTHON=d:\Users\vv\Anaconda3\envs\litcraft\python.exe

REM Python 无缓冲模式，确保 print() 实时显示在终端
set PYTHONUNBUFFERED=1

REM 清除 HTTP_PROXY，避免 CopilotHub 代理干扰本地通信
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=
set NO_PROXY=localhost,127.0.0.1,::1,0.0.0.0

REM ---------- 1. 启动 Ollama 服务 ----------
echo [1/3] 正在启动 Ollama 本地大模型服务...
start "Ollama Serve" cmd /c "C:\Users\vv041\AppData\Local\Programs\Ollama\ollama.exe serve"
echo        Ollama 已启动（新窗口），等待 5 秒加载模型...
echo.
ping -n 5 127.0.0.1 >nul

REM ---------- 2. 启动后端 API ----------
echo [2/3] 正在启动后端 API 服务 (http://localhost:8000)...
start "LitCraft Backend" cmd /c "cd /d %~dp0 && %CONDA_PYTHON% api/server.py"
echo.
echo        后端启动中（新窗口）...
echo.

REM ---------- 3. 启动前端 ----------
echo [3/3] 正在启动前端页面 (http://localhost:8501)...
start "LitCraft Frontend" cmd /c "cd /d %~dp0 && %CONDA_PYTHON% -m streamlit run frontend/app.py"
echo.
echo        前端启动中（新窗口）...
echo.

echo ============================================
echo   ✅ 所有服务已全部启动！
echo.
echo   📌 Ollama API:  http://localhost:11434
echo   📌 后端 API:    http://localhost:8000
echo   📌 前端页面:    http://localhost:8501
echo.
echo   💡 关闭窗口 = 停止对应服务
echo ============================================
echo.
pause
