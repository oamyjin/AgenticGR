import os
import json
import re
import string
import random
import pandas as pd
import argparse

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    # print match info
    # print(f"Extracting answer from solution_str: {solution_str}")

    matches = list(match)
    # print(f"Matches found: {len(matches)}")
    # If there are 0 or exactly 1 matches, return None # Jiajin comments: why at least 2
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    do_print = 0
    if do_print:
        print(f"Golden answers: {ground_truth}")
        print(f"Solution string: {solution_str}")
        print(f"Extracted answer: {answer}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score




def cal_em_accuracy(dataset, jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"[Error] jsonl_path {jsonl_path} does not exist!")
        return

    true_labels = {}
    num_test = 0
    with open(f'/scratch/jl11523/projects/AgenticGR/dataset/{dataset}/{dataset}_corpus.json', 'r') as f:
        corpus = json.load(f)
        for node_id, record in corpus.items():
            true_labels[node_id] = record['label']
            if record['split'] == 'test':
                num_test += 1
    print(f"len(true_labels): {len(true_labels)}", f"num_test: {num_test}")

    print(f"jsonl_path: {jsonl_path}")
    n_examples = 0
    total_score = 0
    search_time = 0
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                result = json.loads(line)
                nid = result["node_id"]
                #ground_truth = result["true_label"]
                ground_truth = true_labels[nid]
                response = result["pred"]
                # print(f"\nNode ID: {nid}, Ground Truth: {ground_truth}, Answer: {extract_solution(response)}")
                compute_score = compute_score_em(solution_str=response, ground_truth=ground_truth)
                n_examples += 1
                total_score += compute_score
                search_time += result["search_times"]
    accuracy = (total_score / n_examples) if n_examples > 0 else 0
    avg_search_time = (search_time / n_examples) if n_examples > 0 else 0
    print(f"[{dataset}] EM accuracy: {total_score}/{n_examples} = {accuracy:.3f}", f"\t Average search times: {avg_search_time:.3f} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--result_file", type=str, default="results/arxiv_32b_raw-similarity_updatedprompt.json")
    args = parser.parse_args()
    
    datastes = None #["cora", "arxiv", "pubmed", "products", "computers", "sports"]
    output_file_names = ["_32b_raw-similarity_updatedprompt", "_32b_hop-similarity_updatedprompt", "_32b_hop-global-similarity_updatedprompt", "_32b_raw-similarity_enhancedsearchprompt"]
    do_print = 0

    if datastes is None:
        cal_em_accuracy(args.dataset, args.result_file)
    else:   
        for dataset in datastes:
            print("=" * 30, f"Dataset: {dataset}", "=" * 30)
            for output_file_name in output_file_names:
                output_result_path = f"results/{dataset}{output_file_name}.json"
                print(f"Calculating EM accuracy for {dataset}: {output_result_path} ...")
                cal_em_accuracy(dataset, output_result_path)
            

