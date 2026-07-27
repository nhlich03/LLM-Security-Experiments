from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.nn import functional as F
import argparse
import numpy as np
import os
import json
import torch.nn as nn


def parse_arguments():
    parser = argparse.ArgumentParser(description='args for sample.py')

    parser.add_argument('--model_path', type=str, default="Alpaca-7b")
    parser.add_argument('--input_path', type=str, default="datasets/safetybench/test_en.json")
    parser.add_argument('--output_path', type=str, default="./outputs/safeybench")
    parser.add_argument('--data_type', type=str, default="en")
    parser.add_argument('--postive_prompt', type=str, default="")
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument(
        "--data_range",
        type=str,
        default="[0:]",
        help="the range of data",
    )
    parser.add_argument('--alpha', type=float, default=0.7)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    return args

@torch.no_grad()
def get_results(options, logits, tokenizer):
    assert len(options) in [2,3,4], "options error!"
    if len(options) == 2:
        candidate=["A", "B"]
    elif len(options) == 3:
        candidate=["A", "B", "C"]
    elif len(options) == 4:
        candidate=["A", "B", "C", "D"]
    
    id=0
    for i, j in zip(tokenizer("A").input_ids, tokenizer("B").input_ids):
        if i!=j:
            break
        else:
            id+=1
    try:
        candidate_logit=[logits[0][tokenizer.get_vocab()[i]] for i in candidate]
    except:
        candidate_logit=[logits[0][tokenizer(i).input_ids[id]] for i in candidate]
    pred=candidate_logit.index(max(candidate_logit))
    
    return  pred
    
@torch.no_grad()
def contrastive_decoding(pos_prompt, neg_prompt, data, model, tokenizer, alpha: float=0.5, en: bool=True):
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query=data["prompt"]
    if en:
        pos_total_prompt = f"{pos_prompt}\nHuman:\n{query}\n\nAssistant:\nAnswer: ("
        neg_total_prompt = f"{neg_prompt}\nHuman:\n{query}\n\nAssistant:\nAnswer: ("
    else:
        pos_total_prompt = f"{pos_prompt}\nHuman:\n{query}\n\nAssistant:\n答案：("
        neg_total_prompt = f"{neg_prompt}\nHuman:\n{query}\n\nAssistant:\n答案：("

    pos_input_ids = tokenizer.encode(pos_total_prompt, return_tensors='pt').to(torch_device)
    neg_input_ids = tokenizer.encode(neg_total_prompt, return_tensors='pt').to(torch_device)

    pos_logits = model(pos_input_ids).logits[::, -1, :]
    neg_logits = model(neg_input_ids).logits[::, -1, :]
    logits=pos_logits-alpha*neg_logits
    probs = nn.functional.softmax(logits, dim=-1)    

    pos_result=get_results(data["options"], pos_logits, tokenizer)
    neg_result=get_results(data["options"], neg_logits, tokenizer)
    our_result=get_results(data["options"], logits, tokenizer)

    torch.cuda.empty_cache()

    return pos_result, neg_result, our_result

def construct_evaluate_prompts(path, en=True):
    
    with open(path) as f:
        data = json.load(f)
    
    
    res = []
    for d in tqdm(data):
        question = d['question']
        options = d['options']
        option_str = ''
        option_letters = ['(A)', '(B)', '(C)', '(D)']
        if len(options) > 4:
            print(d)
        for i, option in enumerate(options):
            option_str += f'{option_letters[i]} {option}\n'
        if en:
            prompt = f'Question: {question.strip()}\nOptions:\n{option_str}'
        else:
            prompt = f'问题：{question.strip()}\n选项：\n{option_str}'

        d['prompt'] = prompt
        res.append(d)
        
    return res

def main(args):
    en, zero_shot=True, True
    if args.data_type == "zh":
        en=False
        
    infra_data=construct_evaluate_prompts(args.input_path, en=en)

    data_min, _, data_max=args.data_range[1:-1].partition(":")
    if data_min =="":
        data_min=0
    if data_max =="":
        data_max=len(infra_data)
    data_min, data_max=int(data_min), int(data_max)

    pos_input = args.postive_prompt

    neg_input= args.negative_prompt
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,  trust_remote_code=True)
    
    print("loading the model")
    large_model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True, device_map="auto")

    pos_all, neg_all, ours_all={},{},{}
    alpha = args.alpha
    print("**"*10+"Start inference."+"**"*10)
    for id, data in tqdm(enumerate(infra_data[data_min:data_max])):

        pos_result, neg_result, our_result = contrastive_decoding(pos_input, neg_input, data, large_model, tokenizer, alpha=args.alpha, en=en)
        pos_all[str(data["id"])]=pos_result
        neg_all[str(data["id"])]=neg_result
        ours_all[str(data["id"])]=our_result

        if (id+1)%100==0:
            with open(f"{args.output_path}/pos_evaluation.json", "w", encoding="utf-8") as r:
                json.dump(pos_all, r, indent=2, ensure_ascii=False)
                
            with open(f"{args.output_path}/neg_evaluation.json", "w", encoding="utf-8") as r:
                json.dump(neg_all, r, indent=2, ensure_ascii=False)
                
            with open(f"{args.output_path}/ours_evaluation.json", "w", encoding="utf-8") as r:
                json.dump(ours_all, r, indent=2, ensure_ascii=False)
        
    print("**"*10+"Finish inference!"+"**"*10)   
    
    with open(f"{args.output_path}/pos_evaluation.json", "w", encoding="utf-8") as r:
        json.dump(pos_all, r, indent=2, ensure_ascii=False)
        
    with open(f"{args.output_path}/neg_evaluation.json", "w", encoding="utf-8") as r:
        json.dump(neg_all, r, indent=2, ensure_ascii=False)
        
    with open(f"{args.output_path}/ours_evaluation.json", "w", encoding="utf-8") as r:
        json.dump(ours_all, r, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

    
