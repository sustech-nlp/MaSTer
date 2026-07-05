import json
import re
import os
import logging
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


base_url = os.getenv("OPENAI_API_BASE")
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI


client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


model = "xxx/model/Qwen2.5-32B-Instruct"


PROMPT_TEMPLATE = (
    "You are a meticulous and analytical assistant. Evaluate the answer contained in the response below and "
    "determine whether it is consistent with the expected answer \"{answer}\". If they are consistent, "
    "output [[1]]; if not, output [[0]].\n\nResponse: {response}"
)

def process_output(raw_output):
    pattern = r'\[\[(\d)\]\]'
    match = re.search(pattern, raw_output)
    if match:
        score = match.group(1)
        if score in ['0', '1']:
            return score
    logger.warning(f"Invalid judge output: {raw_output}")
    return '0'

def call_llm(prompt):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""

def evaluate_record(record):
    answer = record.get("answer", "").strip().upper()
    response = record.get("response", "").strip()


    answer_patterns = [
        rf"answer\s*[:：]?\s*{re.escape(answer)}",
        rf"the\s+correct\s+answer\s+is\s+{re.escape(answer)}",
        rf"correct\s+answer\s*[:：]?\s*{re.escape(answer)}",
        rf"the\s+answer\s+is\s+{re.escape(answer)}",
        rf"my\s+answer\s+is\s+{re.escape(answer)}",
        rf"{re.escape(answer)}\s+is\s+the\s+correct\s+answer",
    ]

    for pattern in answer_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            record["final_output"] = f"string-match: {answer}"
            record["evaluation"] = "1"
            return "1"


    prompt = PROMPT_TEMPLATE.format(answer=answer, response=response)
    llm_output = call_llm(prompt)
    verdict = process_output(llm_output)
    record["final_output"] = llm_output
    record["evaluation"] = verdict
    return verdict

def process_dataset(input_file, success_file, failed_file):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return

    successful = []
    failed = []

    for idx, record in enumerate(data):
        record["index"] = idx
        verdict = evaluate_record(record)
        if verdict == "1":
            successful.append(record)
        else:
            failed.append(record)


    with open(success_file, "w", encoding="utf-8") as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)

    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    logger.info(f"Total records: {len(data)}")
    logger.info(f"Successful (output [[1]]): {len(successful)}")
    logger.info(f"Failed (output [[0]]): {len(failed)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logger.error("Usage: python post_judge.py <input_file> <success_file> <failed_file>")
        sys.exit(1)

    input_filepath = sys.argv[1]
    success_output_filepath = sys.argv[2]
    failed_output_filepath = sys.argv[3]

    if not os.path.exists(input_filepath):
        logger.error(f"Input file {input_filepath} not found.")
    else:
        process_dataset(input_filepath, success_output_filepath, failed_output_filepath)
