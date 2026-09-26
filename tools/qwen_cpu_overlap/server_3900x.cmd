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
echo Windows Ryzen 3900X: 6+6 workers, separate cores, AboveNormal, 80 ms server buffer.
"%QWEN_PYTHON%" -I -B -c "import ctypes,os,runpy; assert (os.cpu_count() or 0)>=24,'This 3900X profile requires at least 24 logical CPUs'; k=ctypes.windll.kernel32; k.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]; result=k.SetPriorityClass(ctypes.c_void_p(-1),0x8000); assert result,ctypes.WinError(); runpy.run_module('RealtimeTTS.qwen_server',run_name='__main__')" --device cpu --host 127.0.0.1 --port 8080 --cpu-threads 6 --cpu-codec-threads 6 --cpu-stream-frames 2 --cpu-affinity 0x555 --cpu-codec-affinity 0x555000 --startup-buffer-ms 80 --clone-mode speaker_only --no-clamp-fp16 --onset-silence-profile qwen3_tts_12hz_0_6b_base_q8_v1 --onset-silence-recovery %*
exit /b %errorlevel%
