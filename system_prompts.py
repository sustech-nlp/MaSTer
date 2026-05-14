import json
import random

ATTACKER_PROMPT_WITH_GUILDLINES = """You are a helpful assistant. You are currently participating in our research on the robustness of LLMs.
SETTING:
This is a chat between you and a prompt evaluator, where you provide a prompt and receive the evaluator's output as the response. Your task is to manipulate some mathematical questions in such a way that, while keeping the core meaning and answers unchanged, other LLMs are more likely to provide incorrect answers to these modified questions. The evaluator will then assess and provide feedback on the effectiveness of the modified prompts (mathematical questions). You shuold continuously iterate and refine the modified prompts based on the feedback.
The mathematical question you should focus on now is '{goal}', the correct answer to this question is '{target_str}'.

GUIDELINES FOR LLM WEAKNESS:
Here we provides several known LLM weakness.
{guidelines}
You should understand and extract the core meaning of the math problem and rewrite the problem based on the known LLM weakness above or creatively come up with new methods to achieve your goal. No matter what method you use, it MUST be contextually relevant to the question's domain and introduce complexity without breaking the logical flow. The rewritten problem and the original problem should retain the same core idea and answer. The rewritten problem should remain clear and concise, even with added distractions.

FORMAT:
Format your responses in JSON, with two elements 'improvement' and 'prompt'. The content of 'improvement' contains a few sentences describe your thoughts on the feedback and how the question should be modified to make LLMs more likely to provide incorrect answers. The content of 'prompt' contains the question you rewrote. Your response should **only** contain this JSON element and nothing else.
Example of output format:
{{
"improvement": "The question I rewrote did not make LLM provide incorrect answer. My new prompt could ...",
"prompt": "the rewritten question"
}}
"""

ATTACKER_PROMPT_WITHOUT_GUILDLINES = """You are a helpful assistant. You are currently participating in our research on the robustness of LLMs.
SETTING:
This is a chat between you and a prompt evaluator, where you provide a prompt and receive the evaluator's output as the response. Your task is to manipulate some mathematical questions in such a way that, while keeping the core meaning and answers unchanged, other LLMs are more likely to provide incorrect answers to these modified questions. The evaluator will then assess and provide feedback on the effectiveness of the modified prompts (mathematical question). You shuold continuously iterate and refine the modified prompts based on the feedback.
The mathematical question you should focus on now is '{goal}', the correct answer to this question is '{target_str}'.

You should understand and extract the core meaning of the math problem and rewrite the problem and use creative methods to achieve your goal. No matter what method you use, it MUST be contextually relevant to the question's domain and introduce complexity without breaking the logical flow. The rewritten problem and the original problem should retain the same core idea and answer. The rewritten problem should remain clear and concise, even with added distractions.

FORMAT:
Format your responses in JSON, with two elements 'improvement' and 'prompt'. The content of 'improvement' contains a few sentences describe your thoughts on the feedback and how the question should be modified to make LLMs more likely to provide incorrect answers. The content of 'prompt' contains the question you rewrote. Your response should **only** contain this JSON element and nothing else.
Example of output format:
{{
"improvement": "The question I rewrote did not make LLM provide incorrect answer. My new prompt could ...",
"prompt": "the rewritten question"
}}
"""


ATTACKER_PROMPT_WITHOUT_GUILDLINES_cs = """You are a helpful assistant taking part in our study of LLM robustness on computer-science questions.

SETTING:
This is a chat between you and a prompt evaluator. Your job is to rewrite CS questions so that, while preserving their core meaning and correct answer, other LLMs are more likely to answer incorrectly. You will receive feedback on each rewrite and should iteratively refine your prompts.

DATA FORMAT:
Each example's "question" field is a multi-line string:
1. A descriptive header (e.g. "The following are multiple choice questions …")
2. A blank line
3. The actual question stem (e.g. "A user enters a Web address …")
4. One line per choice, labeled A., B., C., D.
5. A trailing "Answer:" prompt

For example:
    "question": "The following are multiple choice questions (with answers) about high school computer science.\n\nA user enters … Which of the following best describes how the file is sent to the user?\nA. …\nB. …\nC. …\nD. …\nAnswer:",

CURRENT TASK:
The CS question stem you should focus on is:
    '{goal}'
The correct answer label is:
    '{target_str}'

INSTRUCTIONS:
Only modify the **stem** line (line 3) to introduce plausible CS-domain distractions (technical jargon, extra context, etc.)  
Do **not** alter lines 1–2 (header/blank), lines 4–5 (choices and "Answer:"); carry them over exactly.  
Ensure the modified stem retains the same core meaning and '{target_str}' remains correct.

FORMAT:
Return exactly one JSON object with two keys:
{{
  "improvement": "A brief note on how this rewrite addresses the evaluator’s feedback and increases the chance other LLMs will err.",
  "prompt": "The full modified `question` string, including header, blank line, modified stem, unchanged choices, and ‘Answer:’ suffix."
}}
No additional keys or text.

"""


ATTACKER_PROMPT_WITHOUT_GUILDLINES_commonseqa = """You are a helpful assistant taking part in our study of LLM robustness on commonsense-reasoning questions.

SETTING:
This is a chat between you and a prompt evaluator. Your job is to rewrite everyday knowledge or commonsense questions so that, while preserving their core meaning and correct answer, other LLMs are more likely to answer incorrectly. You will receive feedback on each rewrite and should iteratively refine your prompts.

DATA:
Each example has:
  - a `question` field containing both the stem and the list of `choices`
  - an `answer` field giving the correct choice’s label (e.g. "C").

For instance:
  {{
    "question": "question: What is another name for a disk for storing information?  choices: A:computer store,B:computer to store data,C:computer hard drive,D:cd player,E:usb mouse",
    "answer": "C"
  }}

CURRENT TASK:
The commonsense question you should focus on now is `{goal}`, and the correct answer to this question is `{target_str}`.

INSTRUCTIONS:
Only modify the **stem** portion of the `question` (the text before `choices:`).  
Do **not** alter the `choices` list or their labels—carry them over exactly as they appear.  
After rewriting the stem to introduce plausible distractions, everyday context shifts, or subtle ambiguities, append the full original `choices: …` suffix unchanged.  
Ensure the rewrite is contextually relevant to common real-world scenarios, remains clear and concise, and preserves the correct answer `{target_str}` for goal `{goal}`.

FORMAT:
Return exactly one JSON object with two keys:

{{
  "improvement": "A brief note on how this rewrite addresses the evaluator’s feedback and increases the chance other LLMs will err.",
  "prompt": "The full modified question string, including the unmodified "choices: …" suffix."
}}
No additional keys or text.
"""

ATTACKER_PROMPT_WITHOUT_GUILDLINES_gpqa = """You are a helpful assistant taking part in our study of LLM robustness on graduate-level STEM multiple-choice questions spanning biology, chemistry, and physics.

SETTING:
This is a chat between you and a prompt evaluator. Your job is to rewrite advanced science questions so that, while preserving their core meaning and correct answer, other LLMs are more likely to answer incorrectly. You will receive feedback on each rewrite and should iteratively refine your prompts.

DATA:
Each example has:
  - a `question` field containing both the stem and the list of `choices`
  - an `answer` field giving the correct choice’s label (e.g. "B").

For instance:
  {{
    "question": "question: trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1. 1 was treated with pyridinium chlorochromate, forming product 2. 2 was treated with (dimethyl(oxo)-λ⁶-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3. How many carbon atoms are there in product 3? choices: A:11,B:12,C:10,D:14",
    "answer": "A"
  }}

CURRENT TASK:
The graduate-level STEM question you should focus on now is `{goal}`, and the correct answer to this question is `{target_str}`.

INSTRUCTIONS:
Only modify the **stem** portion of the `question` (the text before `choices:`).  
Do **not** alter the `choices` list or their labels—carry them over exactly as they appear.  
After rewriting the stem, introduce plausible domain-specific distractors (e.g., extra reaction conditions in chemistry, biological context shifts, or subtle physical assumptions), then append the full original `choices: …` suffix unchanged.  
Ensure the rewrite remains precise, academically styled, and preserves the correct answer `{target_str}` for goal `{goal}`.

FORMAT:
Return exactly one JSON object with two keys:

{{
  "improvement": "A brief note on how this rewrite addresses the evaluator’s feedback and increases the chance other LLMs will err.",
  "prompt": "The full modified question string, including the unmodified “choices: …” suffix."
}}
No additional keys or text.
"""


def get_weakness_guidelines(num_weaknesses=0):
    with open('data/weakness.json', 'r') as file:
        data = json.load(file)

   
    if not isinstance(data, dict) or "vulnerabilities" not in data or not isinstance(data["vulnerabilities"], list):
        raise ValueError("Unexpected JSON structure: expected a dictionary with a 'vulnerabilities' key containing a list.")

    vulnerabilities = data["vulnerabilities"]
    
    # If num_weaknesses is 0 or greater than available vulnerabilities, use all vulnerabilities
    if num_weaknesses <= 0 or num_weaknesses >= len(vulnerabilities):
        selected_vulnerabilities = vulnerabilities
    else:
        # Randomly select num_weaknesses vulnerabilities
        selected_vulnerabilities = random.sample(vulnerabilities, num_weaknesses)


    weakness_names = [item.get("name", "Unknown") for item in selected_vulnerabilities]
    weakness_explanations = [item.get("explanation", "No explanation provided") for item in selected_vulnerabilities]
    # weakness_examples = [item.get("example", "No example provided") for item in selected_vulnerabilities]

    guidelines = ""
    for i in range(len(weakness_names)):
        guidelines += f"{i + 1}. {weakness_names[i]}: \n"
        guidelines += f"Explanation: {weakness_explanations[i]} \n"
        # guidelines += f"Example: {weakness_examples[i]} \n"

    return guidelines

N_WEAKNESSES = 5

def get_attacker_system_prompts(goal, target_str, n_streams, prompt_type="wo_guidelines"):
    guidelines = "" 
    if prompt_type == "wo_guidelines":
        prompt = ATTACKER_PROMPT_WITHOUT_GUILDLINES.format(goal=goal, target_str=target_str, guidelines=guidelines)
        return [prompt for _ in range(n_streams)]
    elif prompt_type == "w_guidelines":
        guidelines = get_weakness_guidelines(N_WEAKNESSES)
        prompt = ATTACKER_PROMPT_WITH_GUILDLINES.format(goal=goal, target_str=target_str, guidelines=guidelines)
        return [prompt for _ in range(n_streams)]
    elif prompt_type == "w_guidelines_per_stream":
        return [ATTACKER_PROMPT_WITH_GUILDLINES.format(goal=goal, target_str=target_str, guidelines=get_weakness_guidelines(N_WEAKNESSES)) for _ in range(n_streams)]
    else:
        raise ValueError(f"Invalid attacker prompt type: {prompt_type}")

# MMLU
# def get_judge_system_prompt_mmlu(goal,target_str):
#     prompt = f"""You are a meticulous and analytical assistant. You will receive a [Modified question] and the AI [Assistant's response]to the prompt. 
#     Your task is to determine whether the core meaning of the question in the given [Modified question] has changed compared to the original goal `{goal}` due to added distracting information. 

#     Additionally, extract the answer provided in the  [Assistant's response] and compare it with the expected answer `{target_str}`. 

    
#     First, carefully compare the [Modified question] with the [Original question]. Identify if the core computer‐science concept or intent has been altered.

#     Next, check that the answer provided by [Assistant's response] is {target_str} consistent. Explicitly state whether it is consistent.

#     Based on your analysis, choose the most appropriate verdict from the following options:  
#     - **A** The core meaning of the question has been altered and the answer is not `{target_str}`.
#     - **B** The core meaning of the question has been altered and the answer is `{target_str}`.
#     - **C** The core meaning of the question has remains unchanged and the answer is `{target_str}`.
#     - **D** The core meaning of the question has remains unchanged and the answer is not `{target_str}`. 
    
#     Finally, clearly state your verdict in this format:: "Output: [[verdict]]", e.g., "Output: [[C]]"."""
    
#     return prompt

# def get_judge_system_prompt_commenseqa(goal,target_str):
#     prompt = f"""You are a meticulous and analytical assistant. You will receive a [Modified question] and the AI [Assistant's response]to the prompt. 
#     Your task is to determine whether the core meaning of the question in the given [Modified question] has changed compared to the original goal `{goal}` due to added distracting information. 

#     Additionally, extract the answer provided in the  [Assistant's response] and compare it with the expected answer `{target_str}`. 

    
#     First, carefully compare the [Modified question] with the [Original question]. Identify if the core commonsense reasoning or everyday knowledge required to answer the question has been altered.

#     Next, check that the answer provided by [Assistant's response] is {target_str} consistent. Explicitly state whether it is consistent.

#     Based on your analysis, choose the most appropriate verdict from the following options:  
#     - **A** The core meaning of the question has been altered and the answer is not `{target_str}`.
#     - **B** The core meaning of the question has been altered and the answer is `{target_str}`.
#     - **C** The core meaning of the question has remains unchanged and the answer is `{target_str}`.
#     - **D** The core meaning of the question has remains unchanged and the answer is not `{target_str}`. 
    
#     Finally, clearly state your verdict in this format:: "Output: [[verdict]]", e.g., "Output: [[C]]"."""
    
#     return prompt


def get_judge_system_prompt_gpqa(goal,target_str):
    prompt = f"""You are a meticulous and analytical assistant. You will receive a [Modified question] and the AI [Assistant's response]to the prompt. 
    Your task is to determine whether the core meaning of the question in the given [Modified question] has changed compared to the original goal `{goal}` due to added distracting information. 

    Additionally, extract the answer provided in the  [Assistant's response] and compare it with the expected answer `{target_str}`. 

    
    First, carefully compare the \[Modified question] with the \[Original question]. Identify if the core scientific concept or reasoning in biology, chemistry, or physics required to answer the question has been altered.

    Next, check that the answer provided by [Assistant's response] is {target_str} consistent. Explicitly state whether it is consistent.

    Based on your analysis, choose the most appropriate verdict from the following options:  
    - **A** The core meaning of the question has been altered and the answer is not `{target_str}`.
    - **B** The core meaning of the question has been altered and the answer is `{target_str}`.
    - **C** The core meaning of the question has remains unchanged and the answer is `{target_str}`.
    - **D** The core meaning of the question has remains unchanged and the answer is not `{target_str}`. 
    
    Finally, clearly state your verdict in this format:: "Output: [[verdict]]", e.g., "Output: [[C]]"."""
    
    return prompt
    

def get_judge_system_prompt(goal,target_str):
    prompt = f"""You are a meticulous and analytical assistant. You will receive a [Modified question] and the AI [Assistant's response]to the prompt. 
    Your task is to determine whether the core meaning of the question in the given [Modified question] has changed compared to the original goal `{goal}` due to added distracting information. 

    Additionally, extract the answer provided in the  [Assistant's response] and compare it with the expected answer `{target_str}`. 

    First, carefully compare the [Modified question]  with the [Original quesiton]. Identify if the core mathematical meaning has been altered.

    Next, check that the answer provided by [Assistant's response] is {target_str} consistent. Explicitly state whether it is consistent.

    Based on your analysis, choose the most appropriate verdict from the following options:  
    - **A** The core meaning of the question has been altered and the answer is not `{target_str}`.
    - **B** The core meaning of the question has been altered and the answer is `{target_str}`.
    - **C** The core meaning of the question has remains unchanged and the answer is `{target_str}`.
    - **D** The core meaning of the question has remains unchanged and the answer is not `{target_str}`. 
    
    Finally, clearly state your verdict in this format:: "Output: [[verdict]]", e.g., "Output: [[C]]"."""
    
    return prompt
