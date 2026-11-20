#!/bin/bash
# Builds and starts Syrinx on its own, apart from Vireo: the model served over the OpenAI
# chat API, with a chat page in the browser. The Mac's serve.cmd. Ctrl-C stops it.
#   ./serve.sh                              Gemma 4 E2B (Q6_K) from models/
#   ./serve.sh models/other.gguf            any Llama, Qwen 3.5 or Gemma 4 GGUF
#   ./serve.sh models/other.gguf --port 9000 --api-key secret
# Then open http://127.0.0.1:8088/ or point any OpenAI client at http://127.0.0.1:8088/v1
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${1:-models/gemma-4-E2B-it-Q6_K.gguf}"
if [[ ! -f "$MODEL" ]]; then
    echo "No model at $MODEL."
    echo "Give a GGUF file: ./serve.sh path/to/model.gguf"
    exit 1
fi
CARGO="$(command -v cargo || true)"
[[ -z "$CARGO" && -x /opt/homebrew/opt/rustup/bin/cargo ]] && CARGO=/opt/homebrew/opt/rustup/bin/cargo
echo "Building Syrinx..."
PATH="$(dirname "${CARGO:-cargo}"):$PATH" "${CARGO:-cargo}" build --release -p syrinx-server -q
exec target/release/syrinx-server --standalone "$MODEL" "${@:2}"
