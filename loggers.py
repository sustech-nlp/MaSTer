import os
import wandb
import pandas as pd
import logging 
from collections import Counter
import json

def setup_logger():
    logger = logging.getLogger('auto-weakness')
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    return logger

def set_logger_level(logger, verbosity):
    if verbosity == 0:
        level=logging.CRITICAL # Disables logging
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    
logger = setup_logger()
logger.set_level = lambda verbosity : set_logger_level(logger, verbosity)

class WandBLogger:
    """WandB logger."""

    def __init__(self, args, system_prompts, total_count, save_dir):

        self.logger = wandb.init(
            project = "auto-weakness",
            config = {
                "attack_model" : args.attack_model,
                "target_model" : args.target_model,
                "judge_model": args.judge_model,
                "keep_last_n": args.keep_last_n,
                "system_prompt": system_prompts,
                "n_iter": args.n_iterations,
                "n_streams": args.n_streams,
                "total_count": total_count,
            }
        )
        self.total_count = total_count
        self.save_dir = save_dir
        self.table = pd.DataFrame()
        self.batch_size = args.n_streams
        self.num_successful_jailbreaks = 0

    def _extract_qa_from_item(self, item):
       
        if not isinstance(item, dict):
            return None, None
        q = item.get("question") or item.get("goal") or item.get("prompt") or None
        a = item.get("answer") or item.get("target_str") or item.get("target") or item.get("response") or None
        return q, a

    def log(self, index: int, iteration: int, attack_list: list, response_list: list, judge_scores: list):

        df = pd.DataFrame(attack_list)  
       
        df["target_response"] = response_list
        df["judge_scores"] = judge_scores
        df["iter"] = iteration
        df["index"] = index
        df["conv_num"] = [i+1 for i in range(len(response_list))]

        
        questions = []
        answers = []
        for item in attack_list:
            q, a = self._extract_qa_from_item(item)
            questions.append(q)
            answers.append(a)
        df["question"] = questions
        df["answer"] = answers
       
        self.table = pd.concat([self.table, df], ignore_index=True)

        if 'D' in judge_scores:
            self.num_successful_jailbreaks += 1

        self.logger.log({
            "num_successful_jailbreaks": self.num_successful_jailbreaks},
            )
       
        self.table.to_csv(os.path.join(self.save_dir, "results.csv"), index=False, encoding='utf-8')

        self.print_summary_stats(index, iteration)

    def finish(self):
        self.print_final_summary_stats()
        self.logger.finish()

    def print_summary_stats(self, index, iter):
        df = self.table 

        # Get data for current index and iteration
        current_iter_data = df[(df['index'] == index) & (df['iter'] == iter)]

        # Calculate score proportions
        score_counts = Counter(current_iter_data['judge_scores'])
        total_samples = len(current_iter_data)
        
        logger.info(f"{'='*14} SUMMARY STATISTICS for Iteration {iter} {'='*14}")
        logger.info("Score distribution:")
        for score, count in score_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
            logger.info(f"Score {score}: {count}/{total_samples} ({percentage:.1f}%)")

    def print_final_summary_stats(self):
        logger.info(f"{'='*8} FINAL SUMMARY STATISTICS {'='*8}")
        logger.info("Upload all data to wandb......")
        self.logger.log({"data": wandb.Table(data=self.table)})

        logger.info(f"Total number of successful jailbreaks: {self.num_successful_jailbreaks}/{self.total_count}")
