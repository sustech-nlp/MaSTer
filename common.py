import ast
from loggers import logger
from fastchat.model import get_conversation_template
from system_prompts import get_attacker_system_prompts
from config import API_KEY_NAMES
import os 
import re
import json

import json

FORMAT_REMINDER = "NOTE: Format your responses in JSON, with two elements 'improvement' and 'prompt'."


def extract_json(s, data_path=None):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `adversarial prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.
        data_path (str): The path of the dataset being processed.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    if data_path == "data/math-500-deepseek-successful.json":
        # Use the new code for this specific dataset
        start_pos = s.find("{") 
        end_pos = s.rfind("}") + 1  # Use rfind to ensure we get the last '}'
        if start_pos == -1 or end_pos == 0:
            logger.error("Error extracting potential JSON structure")
            logger.error(f"Input:\n {s}")
            return None, None

        json_str = s[start_pos:end_pos].replace("\n", "")  # Remove all line breaks

        # Key modification: replace single backslashes with double backslashes
        json_str = json_str.replace("\\", "\\\\")

        try:
            parsed = json.loads(json_str)
            if not all(x in parsed for x in ["improvement", "prompt"]):
                logger.error("Error in extracted structure. Missing keys.")
                logger.error(f"Extracted:\n {json_str}")
                return None, None
            return parsed, json_str
        except json.JSONDecodeError as e:
            logger.error("Error parsing extracted structure")
            logger.error(f"Extracted:\n {json_str}")
            logger.error(str(e))
            return None, None
    else:
        # Original code for other datasets
        start_pos = s.find("{") 
        # end_pos = s.find("}") + 1  # +1 to include the closing brace
        end_pos = s.rfind("}") + 1  
        if end_pos == -1:
            logger.error("Error extracting potential JSON structure")
            logger.error(f"Input:\n {s}")
            return None, None

        json_str = s[start_pos:end_pos]
        json_str = json_str.replace("\n", "")  # Remove all line breaks

        try:
            parsed = ast.literal_eval(json_str)
            if not all(x in parsed for x in ["improvement", "prompt"]):
                logger.error("Error in extracted structure. Missing keys.")
                logger.error(f"Extracted:\n {json_str}")
                return None, None
            return parsed, json_str
        except (SyntaxError, ValueError):
            logger.error("Error parsing extracted structure")
            logger.error(f"Extracted:\n {json_str}")
            return None, None



def get_init_msg(goal, target_str):
    return f"Your goal is to manipulate the question in such a way that, while keeping the core meaning and answers unchanged, other LLMs are more likely to provide incorrect answers to these modified questions. The original question is {goal}. **{FORMAT_REMINDER}** Begin." 

def process_target_response(target_response, score, goal, target_str):
    attacker_task = f"""Your task is to manipulate the mathematical question in such a way that, while keeping the core meaning and answers unchanged, other LLMs are more likely to provide incorrect answers to these modified questions.\nThe original question is '{goal}'"""

    if score == 'A':
        judge_output = f"""The core meaning of the question has been altered and the language model's answer is not `{target_str}`."""
    elif score == 'B':
        judge_output = f"""The core meaning of the question has been altered and the language model's answer is `{target_str}`."""
    elif score == 'C':
        judge_output = f"""The core meaning of the question remains unchanged and the language model's answer is `{target_str}`."""
    elif score == 'D':
        judge_output = f"""The core meaning of the question remains unchanged and the language model's answer is not `{target_str}`."""
    else:
        judge_output = f"""Unknown score."""

    return judge_output + "\n" + attacker_task + "\n" + FORMAT_REMINDER

def conv_template(template_name):
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template

def set_system_prompts(system_prompts, convs_list):
    """Set the system prompts for each conversation in the list. 
        The number of system prompts should divide the number of conversations evenly.   
    """

    num_system_prompts = len(system_prompts)
    num_convs = len(convs_list)
    if num_convs % num_system_prompts != 0:
        logger.warning("Warning: Number of system prompts does not divide the number of conversations evenly.")
    for i,conv in enumerate(convs_list):
        conv.set_system_message(system_prompts[i%num_system_prompts])
        

def initialize_conversations(n_streams: int, goal: str, target_str: str, attacker_template_name: str, prompt_type: str):
    batchsize = n_streams
    init_msg = get_init_msg(goal, target_str)
    processed_response_list = [init_msg for _ in range(batchsize)]
    convs_list = [conv_template(attacker_template_name) for _ in range(batchsize)]

    # Set system prompts
    system_prompts = get_attacker_system_prompts(goal, target_str, n_streams, prompt_type)
    set_system_prompts(system_prompts, convs_list)
    return convs_list, processed_response_list, system_prompts

def get_api_key(model):
    environ_var = API_KEY_NAMES[model]
    try:
        return os.environ[environ_var]  
    except KeyError:
        raise ValueError(f"Missing API key, for {model.value}, please enter your API key by running: export {environ_var}='your-api-key-here'")
        