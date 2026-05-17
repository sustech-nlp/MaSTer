# -*- coding: utf-8 -*-

import argparse
import json
import os
import psutil
from datetime import datetime
from tqdm import tqdm

from loggers import WandBLogger, logger
from judges import load_judge
from conversers import load_attack_and_target_models
from common import process_target_response, initialize_conversations


def memory_usage_psutil():
    # Returns the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)  # bytes to MB
    return mem

def load_data(json_path):
    """Load questions and answers from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main(args):
    memory_before = memory_usage_psutil()
    starttime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_dir = os.path.join("results", starttime)
    os.makedirs(run_dir)

    # Load data from JSON file
    data = load_data(args.data_path)
    
    if args.max_samples > 0:
        data = data[:args.max_samples]

    #  start_index 
    if args.start_index >= len(data):
        raise ValueError(f"The starting index {args.start_index} is out of range (max index:{len(data)-1})")

    for index, item in tqdm(enumerate(data[args.start_index:], start=args.start_index), 
                            desc="Processing", total=len(data) - args.start_index):
        logger.info(f"\n{'='*36}\nIndex: {index + 1}/{len(data)}\n{'='*36}\n")
        goal, target_str = item["question"], item["answer"]

        # Initialize models and judge
        attackLM, targetLM = load_attack_and_target_models(args)
        judgeLM = load_judge(args)
        judgeLM.update_judge_system_prompt(goal, target_str)

        # Initialize conversations
        convs_list, processed_response_list, system_prompts = initialize_conversations(
            args.n_streams, goal, target_str, attackLM.template, args.attacker_prompt_type
        )
        if index == args.start_index:
            wandb_logger = WandBLogger(args, system_prompts, len(data), run_dir)

        batchsize = args.n_streams
        target_response_list, judge_scores = None, None

        # Begin PAIR iterations
        for iteration in range(1, args.n_iterations + 1):
            logger.debug(f"\n{'='*36}\nIteration: {iteration}\n{'='*36}\n")

            if iteration > 1:
                processed_response_list = [
                    process_target_response(target_response, score, goal, target_str)
                    for target_response, score in zip(target_response_list, judge_scores)
                ]
            print(f"processed_response_list: {processed_response_list}")

            # Generate adversarial prompts and improvements
            extracted_attack_list = attackLM.get_attack(convs_list, processed_response_list)
            logger.debug("Finished getting adversarial prompts.")
        
            adv_prompt_list = [attack["prompt"] for attack in extracted_attack_list]
            improv_list = [attack["improvement"] for attack in extracted_attack_list]

            # Get target responses
            target_response_list = targetLM.get_response(adv_prompt_list)
            logger.debug("Finished getting target responses.")
            
             # Get judge scores
            judge_scores = judgeLM.score(adv_prompt_list, target_response_list)
            logger.debug("Finished getting judge scores.")

            # Logging iteration details
            for i, (prompt, improv, response, score) in enumerate(
                zip(adv_prompt_list, improv_list, target_response_list, judge_scores)
            ):
                logger.debug(f"{i+1}/{batchsize}\n[IMPROVEMENT]: {improv}\n[PROMPT]: {prompt}\n[RESPONSE]: {response}\n[SCORE]: {score}\n")

            # WandB log values
            wandb_logger.log(index, iteration, extracted_attack_list, target_response_list, judge_scores)

            # Truncate conversation to avoid context length issues
            for i, conv in enumerate(convs_list):
                conv.messages = conv.messages[-2*(args.keep_last_n):]

            # Stop early if any score is 'D'
            if 'D' in judge_scores:
                logger.info("Detected interference. Exiting iteration.")
                break

    wandb_logger.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument for JSON data file
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the JSON file containing questions and answers.",
    )

    # Attack model parameters
    parser.add_argument(
        "--attack-model",
        default = "vicuna-13b-v1.5",
        help = "Name of attacking model.",
        choices=["vicuna-13b-v1.5","gpt-4o-mini", "llama-2-7b-chat-hf", "gpt-3.5-turbo-1106", "gpt-4-0125-preview", "claude-instant-1.2", "claude-2.1", "gemini-pro", 
        "mixtral","vicuna-7b-v1.5", "Meta-Llama-3.1-8B-Instruct", "Llama-3.2-1B-Instruct", "DeepSeek-R1-Distill-Llama-8B", "Meta-Llama-3-8B-Instruct", "Mistral-Small-24B-Instruct-2501","Qwen2.5-32B-Instruct"]
    )
    parser.add_argument(
        "--attack-max-n-tokens",
        type = int,
        default = 1024,
        help = "Maximum number of generated tokens for the attacker."
    )
    parser.add_argument(
        "--max-n-attack-attempts",
        type = int,
        default = 10,
        help = "Maximum number of attack generation attempts, in case of generation errors."
    )
    # Target model parameters
    parser.add_argument(
        "--target-model",
        default = "vicuna-13b-v1.5", #TODO changed
        help = "Name of target model.",
        choices=["vicuna-13b-v1.5","Qwen2.5-Math-7B-Instruct","gpt-4o-mini","Llama-3.2-1B-Instruct", "llama-2-7b-chat-hf", "gpt-3.5-turbo-1106", "gpt-4-0125-preview", "claude-instant-1.2", "claude-2.1",
                 "gemini-pro", "Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3-8B-Instruct","DeepSeek-R1-Distill-Llama-8B","Mistral-Small-24B-Instruct-2501", "gemma-2-2b-it","Qwen2.5-32B-Instruct"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type = int,
        default = 1024,
        help = "Maximum number of generated tokens for the target."
    )
    # Judge model parameters
    parser.add_argument(
        "--judge-model",
        default="no-judge", #TODO changed
        help="Name of judge model.",
        choices=["no-judge", "gpt-4o-mini", "gpt-4-0125-preview", "Mistral-Small-24B-Instruct-2501", "Meta-Llama-3-8B-Instruct","Qwen2.5-32B-Instruct"]
    )
    parser.add_argument(
        "--judge-max-n-tokens",
        type = int,
        default = 1024,
        help = "Maximum number of tokens for the judge."
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0,
        help="Temperature to use for judge."
    )
    # PAIR parameters
    parser.add_argument(
        "--n-streams",
        type = int,
        default = 2, #TODO changed
        help = "Number of concurrent jailbreak conversations. If this is too large, then there may be out of memory errors when running locally. For our experiments, we use 30."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Specify the starting index of the data processing."
    )

    parser.add_argument(
        "--keep-last-n",
        type = int,
        default = 4,
        help = "Number of responses to save in conversation history of attack model. If this is too large, then it may exceed the context window of the model."
    )
    parser.add_argument(
        "--n-iterations",
        type = int,
        default = 5,
        help = "Number of iterations to run the attack. For our experiments, we use 3."
    )
    parser.add_argument(
        "--evaluate-locally",
        action = 'store_true',
        help = "Evaluate models locally rather than through Together.ai. We do not recommend this option as it may be computationally expensive and slow."
    )

    parser.add_argument(
        '-v', 
        '--verbosity', 
        action="count", 
        default = 0,
        help="Level of verbosity of outputs, use -v for some outputs and -vv for all outputs.")
    ##################################################

    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--attacker-prompt-type",
        type=str,
        default="wo_guidelines",
        choices=["wo_guidelines", "w_guidelines", "w_guidelines_per_stream"],
    )
    
    args = parser.parse_args()
    logger.set_level(args.verbosity)
    main(args)
