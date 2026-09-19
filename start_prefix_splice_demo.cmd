@echo off
chcp 65001 >nul
pushd "%~dp0"
".venv\Scripts\python.exe" -X utf8 -u tools\prefix_splice_keyboard_demo.py %*
if errorlevel 1 pause
popd
