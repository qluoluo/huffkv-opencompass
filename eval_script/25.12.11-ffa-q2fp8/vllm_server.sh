#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6,7

python -m vllm.entrypoints.openai.api_server \
    --model /inspire/hdd/global_user/liuzhigeng-253108120105/models/Qwen2.5-32B-Instruct \
    --served-model-name Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000 \
    --trust-remote-code
