@echo off
chcp 65001 >nul
pushd "%~dp0"
".venv\Scripts\python.exe" -X utf8 tools\qwen_prefix_splice.py --play-only %*
if errorlevel 1 pause
popd
