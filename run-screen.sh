#!/bin/bash
export WANDB_API_KEY="xxxxx"
export HOSTED_VLLM_API_BASE="http://0.0.0.0:xxxx/v1/"
export TOGETHER_API_KEY="sk-xxxx"

# # litellm_proxy/your-model-name
export LITELLM_PROXY_API_BASE="http://0.0.0.0:xxxx/v1/"
export LITELLM_PROXY_API_KEY="sk-xxxx"



python screening.py --input_file "data/gsm8k.json"  --output_file "data/gsm8k-xxxxx-output.json" --model "xxx/model/xxxx"

python post_judge.py "data/gsm8k-xxxxx-output.json" "data/gsm8k-xxxxx-output-successful.json" "data/gsm8k-xxxxx-output-failed.json"


