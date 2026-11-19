@echo off
rem Builds and starts Syrinx on its own, apart from Vireo: the model served over the
rem OpenAI chat API, with a chat page in the browser. Leave this window open; close it to stop.
rem   serve.cmd                              Gemma 4 E2B (Q6_K) from models\
rem   serve.cmd models\other.gguf            any Llama, Qwen 3.5 or Gemma 4 GGUF
rem   serve.cmd models\other.gguf --port 9000 --api-key secret
rem Then open http://127.0.0.1:8088/ or point any OpenAI client at http://127.0.0.1:8088/v1
setlocal
cd /d "%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=models\gemma-4-E2B-it-Q6_K.gguf"
if not exist "%MODEL%" (
    echo No model at %MODEL%.
    echo Give a GGUF file: serve.cmd path\to\model.gguf
    exit /b 1
)
echo Building Syrinx...
cargo build --release -p syrinx-server -q
if errorlevel 1 (
    echo The build failed.
    exit /b 1
)
target\release\syrinx-server.exe --standalone "%MODEL%" %2 %3 %4 %5 %6 %7 %8 %9
