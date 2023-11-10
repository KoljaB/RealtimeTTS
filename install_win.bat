@echo off
echo Please first install NVIDIA CUDA Toolkit 11.8, NVIDIA cuDNN 8.7.0 for CUDA 11.x and ffmpeg before running this

cd /d %~dp0

if not exist test_env\Scripts\python.exe (
    echo Creating VENV
    python -m venv test_env
) else (
    echo VENV already exists
)


echo Activating VENV
call test_env\Scripts\activate.bat

REM upgrade pip
REM -----------------------------------

python.exe -m pip install --upgrade pip


REM install realtime libraries
REM -----------------------------------

pip install RealtimeTTS>=0.1.9


REM torch with GPU support
REM -----------------------------------
REM https://pytorch.org/get-started/locally/

pip3 install torch>=2.0.1 torchaudio>=2.0.2 --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu118