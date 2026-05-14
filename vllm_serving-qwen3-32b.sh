
MODEL_PATH="xxx/model/Qwen3-32B"


echo "Model path: $MODEL_PATH"

# Set the API key
API_KEY="sk-xxxx"
echo "API key $API_KEY"

PORT=xxxx


export CUDA_VISIBLE_DEVICES=0,1
vllm serve $MODEL_PATH --api-key $API_KEY --port $PORT --tensor-parallel-size 2 --gpu-memory-utilization 0.9  
# --chat-template ./qwen3_nonthinking.jinja

