#!/bin/bash



export WANDB_API_KEY="xxxxx"
export OPENAI_API_BASE="https://xxxxx/v1"
export OPENAI_API_KEY="sk-xxxxx"
export HOSTED_VLLM_API_BASE="http://0.0.0.0:xxxx/v1/"
export TOGETHER_API_KEY="sk-xxxx"

# # litellm_proxy/your-model-name
export LITELLM_PROXY_API_BASE="http://0.0.0.0:xxxx/v1/"
export LITELLM_PROXY_API_KEY="sk-xxxx"


python main.py \
    --attack-model Qwen2.5-32B-Instruct \
    --target-model Meta-Llama-3-8B-Instruct \
    --judge-model Qwen2.5-32B-Instruct \
    --data-path "xxx/data/llama8b_gsm8k_train-successful.json" \
    --n-streams 5 \
    --n-iterations 3 \
    --attacker-prompt-type "wo_guidelines" \
    -vv \
    --max-samples 1000 \
    2>&1 | tee log/output_qwen_gsm8k.log

