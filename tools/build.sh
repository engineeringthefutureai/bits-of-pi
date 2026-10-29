#!/usr/bin/env bash
# Build tf_dump against the pinned llama.cpp build named in config.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA=$(.venv/bin/python -c 'import yaml;print(yaml.safe_load(open("config.yaml"))["llama_cpp"]["dir"])')
BUILD=$(.venv/bin/python -c 'import yaml;print(yaml.safe_load(open("config.yaml"))["llama_cpp"]["build"])')
LLAMA=$(readlink -f "$LLAMA")
BUILD=$(readlink -f "$BUILD")

mkdir -p tools/bin
g++ -O2 -std=c++17 tools/tf_dump.cpp \
    -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
    -L"$BUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin" \
    -o tools/bin/tf_dump
echo "built tools/bin/tf_dump against $BUILD"
