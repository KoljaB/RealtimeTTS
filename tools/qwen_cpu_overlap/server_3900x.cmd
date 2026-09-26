@echo off
setlocal
rem Opt-in profile for Windows Ryzen 9 3900X: 12 physical cores / 24 threads.
set "QWEN_PYTHON="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "QWEN_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined QWEN_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "QWEN_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined QWEN_PYTHON if exist "%~dp0..\..\.venv\Scripts\python.exe" set "QWEN_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not defined QWEN_PYTHON (
  echo Activate your Qwen CPU server venv, or create .venv next to this file.
  exit /b 2
)
echo Qwen Studio: http://127.0.0.1:8080/studio
echo Preparing the CPU profile and public demo voice; first start may download files.
"%QWEN_PYTHON%" -I -B -m RealtimeTTS.qwen_server --preset windows-3900x --demo-voice %*
exit /b %errorlevel%
