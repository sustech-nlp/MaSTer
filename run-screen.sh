#!/bin/bash
export WANDB_API_KEY="xxxxx"
export OPENAI_API_BASE="http://0.0.0.0:xxxx/v1"
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_CLIENT_BASE_URL="http://0.0.0.0:xxxx/v1"
export OPENAI_CLIENT_API_KEY="sk-xxxxx"


python screening.py --input_file "data/gsm8k.json"  --output_file "data/gsm8k-xxxxx-output.json" --model "xxx/model/xxxx"

python post_judge.py "data/gsm8k-xxxxx-output.json" "data/gsm8k-xxxxx-output-successful.json" "data/gsm8k-xxxxx-output-failed.json"


