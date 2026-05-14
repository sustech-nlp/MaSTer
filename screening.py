import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import os  # 新增导入

def process_file(input_file_path, output_file_path, model):
    # 从环境变量获取 base_url 和 api_key
    base_url = os.getenv("HOSTED_VLLM_API_BASE")
    api_key = os.getenv("TOGETHER_API_KEY")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
            open(output_file_path, 'a', encoding='utf-8') as output_file:

        data = json.load(input_file)  # Load the entire JSON file at once
        for entry in tqdm(data, desc="Processing", unit="entry"):  # Iterate over entries
            question = entry.get('question', '')  # Updated key to match new structure
            answer = entry.get('answer', '')

            prompt = f"{question}"

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1024
            )

            try:
                response = f"{completion.choices[0].message.content.strip()}"
            except Exception as e:
                print(f"Error processing question: {question}\n{str(e)}")
                response = "Error processing the question."  # Fixed variable name

            result = {"question": question, "answer": answer, "response": response }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files using OpenAI API.')  # Updated description
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')  # Updated help text
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    parser.add_argument('--model', required=True, help='Path to the OpenAI model.')

    args = parser.parse_args()

    process_file(args.input_file, args.output_file, args.model)
