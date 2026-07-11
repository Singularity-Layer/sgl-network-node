@echo off
setlocal enabledelayedexpansion
title Singularity Node - Windows Beta Setup

REM ===========================================================================
REM  Singularity Node - Windows Beta one-click setup.
REM  Prereq: a Solana wallet with >= 50,000 SGL staked (you approve login in a
REM  browser with THAT wallet). Windows 10/11 x64. A GPU is nice but optional.
REM ===========================================================================

set "SGL_DIR=%LOCALAPPDATA%\sgl-node"
set "BIN=%SGL_DIR%\sgl.exe"
set "MODEL_DIR=%SGL_DIR%\models"
set "MODEL=%MODEL_DIR%\gemma-2-2b.gguf"
set "MODEL_NAME=gemma-2-2b"

REM If the exe is hosted, put its link here. Otherwise leave blank and drop the
REM sgl.exe you were given into %SGL_DIR% before running.
set "EXE_URL="
set "MODEL_URL=https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"

if not exist "%SGL_DIR%" mkdir "%SGL_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if not exist "%BIN%" (
  if defined EXE_URL (
    echo [1/4] Downloading Singularity Node...
    curl -L -o "%BIN%" "%EXE_URL%" || goto :fail
  ) else (
    echo Place the sgl.exe you were given at:
    echo    %BIN%
    echo then re-run this script.
    goto :done
  )
) else (
  echo [1/4] Singularity Node already present.
)

echo [2/4] Installing inference backend ^(llama.cpp, hash-verified^)...
"%BIN%" setup || goto :fail

if not exist "%MODEL%" (
  echo [3/4] Downloading model gemma-2-2b ^(~1.6 GB, one time^)...
  curl -L -o "%MODEL%" "%MODEL_URL%" || goto :fail
) else (
  echo [3/4] Model already present.
)

echo [4/4] Logging in - a browser window opens. Approve with your STAKED wallet.
"%BIN%" login --models %MODEL_NAME% || goto :fail

echo.
echo Starting the node. It will register, serve jobs, and earn. Keep this window OPEN.
echo Watch it live at: https://cloud.x402compute.cc/network/console
echo.
"%BIN%" start --model-path "%MODEL%" --model-name %MODEL_NAME%
goto :done

:fail
echo.
echo *** Something failed. Copy everything above and send it to the team. ***
:done
pause
