from vllm import LLM, SamplingParams
import transformers
import torch
import random
from datasets import load_dataset
import requests
import argparse
import os
import json
from typing import List
from tqdm import tqdm

import pdb

CATEGORIES = {
    'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'],
    'arxiv': ['cs.NA', 'cs.MM', 'cs.LO', 'cs.CY', 'cs.CR', 'cs.DC', 'cs.HC', 'cs.CE', 'cs.NI', 'cs.CC', 'cs.AI', 'cs.MA', 'cs.GL', 'cs.NE', 'cs.SC', 'cs.AR', 
                'cs.CV', 'cs.GR', 'cs.ET', 'cs.SY', 'cs.CG', 'cs.OH', 'cs.PL', 'cs.SE', 'cs.LG', 'cs.SD', 'cs.SI', 'cs.RO', 'cs.IT', 'cs.PF', 'cs.CL', 'cs.IR', 
                'cs.MS', 'cs.FL', 'cs.DS', 'cs.OS', 'cs.GT', 'cs.DB', 'cs.DL', 'cs.DM'],
    'pubmed': ['Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes'],
    'products': ['Home & Kitchen', 'Health & Personal Care', 'Beauty', 'Sports & Outdoors', 'Books', 'Patio, Lawn & Garden', 'Toys & Games', 'CDs & Vinyl', 
                    'Cell Phones & Accessories', 'Grocery & Gourmet Food', 'Arts, Crafts & Sewing', 'Clothing, Shoes & Jewelry', 'Electronics', 'Movies & TV', 
                    'Software', 'Video Games', 'Automotive', 'Pet Supplies', 'Office Products', 'Industrial & Scientific', 'Musical Instruments', 'Tools & Home Improvement', 
                    'Magazine Subscriptions', 'Baby Products', 'label 25', 'Appliances', 'Kitchen & Dining', 'Collectibles & Fine Art', 'All Beauty', 'Luxury Beauty', 'Amazon Fashion', 
                    'Computers', 'All Electronics', 'Purchase Circles', 'MP3 Players & Accessories', 'Gift Cards', 'Office & School Supplies', 'Home Improvement', 'Camera & Photo', 
                    'GPS & Navigation', 'Digital Music', 'Car Electronics', 'Baby', 'Kindle Store', 'Buy a Kindle', 'Furniture & Decor', '#508510'],
    'computers': ['Tablet Replacement Parts', 'Monitors', 'Networking Products', 'Computers & Tablets',
                            'Computer Accessories & Peripherals', 'Tablet Accessories', 'Laptop Accessories',
                            'Computer Components', 'Data Storage', 'Servers'],
    'sports': ['Other Sports', 'Exercise & Fitness', 'Hunting & Fishing', 'Accessories', 'Leisure Sports & Game Room',  
                        'Team Sports', 'Boating & Sailing', 'Swimming', 'Tennis & Racquet Sports', 'Golf', 'Airsoft & Paintball', 
                        'Clothing', 'Sports Medicine'],
}



def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str, node_record: dict, search_cnt: int, curr_nb_ids: List[str], candidate_filter: str, ranker: str):
    if ranker == "SemQA":
            title = node_record["text"].split(";")[0]
            query = f"{title}; Query: {query}"
    elif ranker == "SemA": # ignore the query, only use the anchor text
            query = node_record["text"]    

    payload = {
            "queries": [query],
            "curr_nb_ids": [curr_nb_ids],
            "node_ids": [str(node_record["id"])],
            "search_hops": [search_cnt],
            "topk": 3,
            "candidate_filter": candidate_filter, # "Sem" or "Hop" or "PPR" or "Sem+Hop" or "Sem+PPR" or "Hop+PPR" or "Sem+Hop+PPR"
            "score_method": ranker # ranker can be "SemQ", "SemA", "SemQA", "WeightedSemQA", "StrucSemQA"(not clear yet)
        }

    # print("[POST] payload:", payload)
    results = requests.post(f"http://127.0.0.1:{args.port}/retrieve", json=payload).json()['result']

    def _passages2string(retrieval_result):
        format_reference = ''
        nb_ids = []
        for idx, node_info in enumerate(retrieval_result):
            node_text = " ".join(node_info["text"].split()[:50]) # truncate to 50 words
            nb_ids.append(str(node_info["id"]))
            format_reference += f"Node {idx+1}: {node_text}\n"
        # Jiajin add:
        if format_reference == '':
            format_reference = 'No relevant information found.\n'
        return format_reference, nb_ids

    return _passages2string(results[0])



def make_prompt(dataset, record):
    if dataset.lower() in ["cora", "pubmed", "arxiv"]:
        platform = dataset.capitalize()
        node_type = "paper"
        relation_type = "citation"
    elif dataset.lower() in ["computers", "sports", "products"]:
        platform = "Amazon"
        node_type = "product"
        relation_type = "co-purchase" 

    task_desc = (
        f"You are a reasoning assistant for node classification on an {platform} dataset graph."
        f"Your goal is to select the most likely category for the target node from the provided list.\n"
        "Tools:\n"
        "- To perform a search, write <search> your query here </search>.\n" 
        "- The graph retriever considers both the graph structure and the semantic similarity of your query, and returns the most relevant data inside <information> ... </information>.\n"
        "- You can repeat the search process multiple times if necessary.\n"
    )
    reasoning_policy = (
        "Reasoning protocol:\n"
        "- Whenever you receive new information, first reason inside <think> ... </think>.\n" 
        "- If no further information is needed, output only your final choice inside <answer> ... </answer> (no extra explanation).\n"
        "Example:\n"
        "Question: ...\n"
        "Assistant:"
        "<think> …your reasoning...</think>"
        "<search> …your query… </search>"
        "<information>...retriever results...</information>"
        "<think>...reasoning with the new information...</think>"
        "<answer>Movies</answer>\n\n"
        "Use the following information for the node classification task:\n"
    )

    node_text = record['text']
    node_info = f"- The target {node_type}'s information: {node_text}\n"

    degree = record['degree']
    avg_degree = record['dataset_avg_degree']
    domain_info = (
        "- The domain knowledge:"
        f"Each node represents a {node_type} and connected to other {node_type}s through {relation_type} relationships."
        f"The degree of target node is {degree}, while the average degree of the dataset is {avg_degree}.\n"
    )

    categories = CATEGORIES[dataset]
    category_list = f"- The category list: {'; '.join(categories)}.\n"

    # final prompt
    prompt = task_desc + reasoning_policy + node_info + domain_info + category_list
    return prompt


def make_prompt_old(dataset, record):
    if dataset.lower() in ["cora", "pubmed", "arxiv"]:
        platform = dataset.capitalize()
        node_type = "paper"
        relation_type = "citation"
    elif dataset.lower() in ["computers", "sports", "products"]:
        platform = "Amazon"
        node_type = "product"
        relation_type = "co-purchase" 

    task_desc = (
        "You are a helpful assistant with the ability to reason and search for information from a graph retriever."
        f"You are given a node classification task on the {platform} platform dataset."
        f"Using the provided {platform} {node_type}'s information and the dataset domain knowledge, "
        f"please choose the most likely category of this node ({node_type}) from the category list.\n"
    )
    reasoning_policy = (
        "You must conduct reasoning inside <think> and </think> first every time you get new information."
        "After reasoning, if you need extra information, you can search by <search> your query </search> to call a graph retriever." 
        "The retriever considers both the graph structure and the semantic similarity of your query, and returns the top relevant data inside <information> and </information>."
        "You can search as many times as you want."
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations."
        "For example, <answer>Movies</answer>. Here is the given information: \n"
    )

    node_text = record['text']
    node_info = f"The target {node_type}'s information: {node_text}\n"

    degree = record['degree']
    avg_degree = record['dataset_avg_degree']
    domain_info = (
        "The domain knowledge:"
        f"Each node represents a {node_type} and connected to other {node_type}s through {relation_type} relationships."
        f"The degree of target node is {degree}, while the average degree of the dataset is {avg_degree}.\n"
    )

    categories = CATEGORIES[dataset]
    category_list = f"The category list: {'; '.join(categories)}.\n"

    # final prompt
    prompt = task_desc + reasoning_policy + node_info + domain_info + category_list
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000) # float16 or bfloat16
    parser.add_argument("--model_id", type=str, default="/vast/jl11523/projects-local-models/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo") # /vast/jl11523/projects-local-models/Qwen2.5-32B-Instruct
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./results/output.json")
    parser.add_argument("--candidate_filter", type=str, default="Hop+PPR") # "Sem" or "Hop" or "PPR" or "Sem+Hop" or "Sem+PPR" or "Hop+PPR" or "Sem+Hop+PPR"
    parser.add_argument("--ranker", type=str, default="SemQA") # ranker can be "SemQ", "SemA", "SemQA", "WeightedSemQA", "StrucSemQA"(not clear yet)
    args = parser.parse_args()

    model_id = args.model_id
    dataset = args.dataset
    candidate_filter = args.candidate_filter
    ranker = args.ranker
    output_path = args.output_path
    corpus_path = f"dataset/{dataset}/{dataset}_corpus.json"
    if not os.path.exists(corpus_path):
        raise ValueError(f"Corpus file {corpus_path} does not exist. Please build the corpus first.")

    print("[candidate_filter]:", candidate_filter)
    print("[ranker]:", ranker)
    print("[dataset]:", dataset)
    print("[corpus_path]:", corpus_path)
    print("[model_id]:", model_id)
    print("[output_path]:", output_path)

    # Initialize the tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    # VLLM
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id) #, trust_remote_code=True)
    llm = LLM(
        model=model_id,                     # e.g., "/vast/.../Qwen2.5-32B-Instruct"
        dtype="bfloat16",                   # match your previous torch_dtype
        tensor_parallel_size=max(torch.cuda.device_count(), 1),
        # trust_remote_code=True,
        # optional performance knobs:
        max_model_len=8192,                 # adjust to your context length
        swap_space=8,                       # GB CPU swap for long prompts (optional)
    )
    # stop strings for the “search” turn
    STOP_STRINGS = ["</search>", " </search>", "</search>\n", " </search>\n",
                    "</search>\n\n", " </search>\n\n",
                    "</answer>", "</answer>\n", "</answer>\n\n"]

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
        stop=STOP_STRINGS,                 # vLLM will stop exactly at </search> variants
        # You do NOT need to specify EOS; vLLM handles it automatically
    )

    # model special tokens
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    with open(corpus_path, "r") as f:
        all_nodes = json.load(f)
    print("len(all_nodes):", len(all_nodes))
    # test ids
    with open(f'./dataset/{dataset}/{dataset}_test_ids.txt', 'r') as f:
        ids_read = f.read()
    ids_list = [x for x in ids_read.split(',')]

    out_fp = open(output_path, "w", encoding="utf-8")
    for node_id in tqdm(ids_list[args.start_idx:], total=len(ids_list), initial=args.start_idx, desc="Processing"):
        record = all_nodes[node_id] # node_id: str

        prompt = make_prompt(dataset, record)

        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    
        original_prompt = prompt
        search_cnt = 0
        whole_output = ""
        # Encode the chat-formatted prompt and move it to the correct device
        while True:
            # vLLM
            res = llm.generate([prompt], sampling_params)
            out = res[0].outputs[0]    # top hypothesis
            output_text = out.text         # vLLM returns only the generated continuation text

            stop_reason = getattr(out, "stop_reason", None)
            if stop_reason in STOP_STRINGS:
                output_text += stop_reason
            whole_output += output_text

            if "</answer>" in output_text:
                break

            tmp_query = get_query(output_text)
            # print(f"[Generation] tmp_query:", tmp_query)

            if tmp_query:
                search_cnt += 1
                if search_cnt == 1:
                    retrived_nb_ids = [node_id]
                if len(retrived_nb_ids) == 0: # no neighbor
                    search_results = 'No relevant information found.\n'
                else:
                    search_results, retrived_nb_ids = search(tmp_query, record, search_cnt, retrived_nb_ids, candidate_filter, ranker)
            else:
                search_results = ''

            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            whole_output += search_text

        infer_result = {
            "node_id": node_id,
            "pred": whole_output,
            "search_times": search_cnt,
            "true_label": record['label'],
            "prompt": original_prompt
        }
        out_fp.write(json.dumps(infer_result, ensure_ascii=False) + "\n")
        out_fp.flush()
        os.fsync(out_fp.fileno())
