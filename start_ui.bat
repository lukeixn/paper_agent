@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\py\Anaconda3\envs\py311\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python environment not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

echo Starting Paper Agent UI at http://localhost:8502
echo Keep this window open while using the application.
echo.

"%PYTHON_EXE%" -m streamlit run ui.py ^
    --server.port 8502 ^
    --server.address 127.0.0.1 ^
    --server.headless true ^
    --server.fileWatcherType poll

pause
