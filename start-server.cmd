@echo off
rem Builds and starts syrinx-server, Vireo's own engine serving a model on this PC for the
rem Vireo Prototype app. Leave this window open while you chat; close it to stop.
rem   start-server.cmd                       Gemma 4 E2B from models\
rem   start-server.cmd models\other.gguf     any Llama, Qwen 3.5 or Gemma 4 GGUF
rem In the app: Settings, connect to "localhost" with the code printed below.
setlocal
cd /d "%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=models\gemma-4-E2B_q4_0-it.gguf"
if not exist "%MODEL%" (
    echo No model at %MODEL%.
    echo Give a GGUF file: start-server.cmd path\to\model.gguf
    exit /b 1
)
echo Building the server...
cargo build --release -p syrinx-server -q
if errorlevel 1 (
    echo The build failed.
    exit /b 1
)
target\release\syrinx-server.exe "%MODEL%" %2 %3 %4 %5 %6
