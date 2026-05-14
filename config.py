from enum import Enum
import os
import sys


VICUNA_PATH = f"models/lmsys/vicuna-13b-v1.5"
LLAMA_PATH = f"models/meta-llama/Llama-2-7b-chat-hf"

ATTACK_TEMP = 1
TARGET_TEMP = 0.6
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4o-2024-11-20"
    gpt_4omini = "gpt-4o-mini"
    gpt_5 = "gpt-5-2025-08-07"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    llama_3_2 = "Llama-3.2-1B-Instruct"
    qwen2_5="Qwen2.5-32B-Instruct"
MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.llama_3_2: "meta-llama/Llama-3.2-1B-Instruct",
    Model.qwen2_5:"Qwen/Qwen2.5-32B-Instruct"
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: f"hosted_vllm/meta-llama/llama-2-7b-chat",
    Model.vicuna: f"hosted_vllm/lmsys/vicuna-13b-v1.5",
    Model.llama_3_2: f"hosted_vllm//xxx/model/Llama-3.2-1B-Instruct",
    Model.qwen2_5:f"litellm_proxy//xxx/model/Qwen2.5-32B-Instruct"
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4",
    Model.gpt_4omini: "gpt-4o-mini",
    Model.gpt_5: "gpt-5",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2-7b-chat-hf",
    Model.llama_3_2: "Llama-3.2-1B-Instruct",
    Model.qwen2_5: "Qwen2.5-32B-Instruct"
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.gpt_4omini:    "OPENAI_API_KEY",
    Model.gpt_5:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.llama_3_2: "TOGETHER_API_KEY",
    Model.qwen2_5: "TOGETHER_API_KEY"
}

LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]         
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]  
            },
    Model.llama_3_2: {"roles":{
                    "system": {
                        "pre_message": "<|start_header_id|>system<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    },
                    "user": {
                        "pre_message": "<|start_header_id|>user<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    },
                    "assistant": {
                        "pre_message": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                        "post_message": "<|eot_id|>"
                    }
                },
                "post_message": "<|eot_id|>",
                "initial_prompt_value": "",
                "eos_tokens": ["<|eot_id|>"]
    },
    Model.qwen2_5: {"roles": {
        "system": {
            "pre_message": "<|im_start|>system\n",
            "post_message": "<|im_end|>\n"
        },
        "user": {
            "pre_message": "<|im_start|>user\n",
            "post_message": "<|im_end|>\n"
        },
        "assistant": {
            "pre_message": "<|im_start|>assistant\n",
            "post_message": "<|im_end|>\n"
        }
    },
        "post_message": "",
        "initial_prompt_value": "",
        "eos_tokens": ["<|im_end|>"]
    }

}