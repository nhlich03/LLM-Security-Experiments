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

    parser.add_argument('--model_path', type=str, default="chinese-alpaca-2-7b")
    parser.add_argument('--input_path', type=str, default="datasets/cvalues/cvalues_responsibility_mc.jsonl")
    parser.add_argument('--output_path', type=str, default="./outputs/cvalues")
    parser.add_argument('--postive_prompt', type=str, default="")
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument('--alpha', type=float, default=0.7)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    return args

@torch.no_grad()
def get_results(logits, tokenizer):

    candidate=["1", "2"]
    id=0
    for i, j in zip(tokenizer("1").input_ids, tokenizer("2").input_ids):
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

    pos_total_prompt = f"{pos_prompt}\nHuman:\n{query}\n\nAssistant:\n答案：[回复"
    neg_total_prompt = f"{neg_prompt}\nHuman:\n{query}\n\nAssistant:\n答案：[回复"

    pos_input_ids = tokenizer.encode(pos_total_prompt, return_tensors='pt').to(torch_device)
    neg_input_ids = tokenizer.encode(neg_total_prompt, return_tensors='pt').to(torch_device)
    
    
    if pos_input_ids[0][-1]==tokenizer.eos_token_id:
        pos_logits = model(pos_input_ids[::,:-1]).logits[::, -1, :]
        neg_logits = model(pos_input_ids[::,:-1]).logits[::, -1, :]
    else:
        pos_logits = model(pos_input_ids).logits[::, -1, :]
        neg_logits = model(neg_input_ids).logits[::, -1, :]
    logits=pos_logits-alpha*neg_logits 
     
    probs = nn.functional.softmax(logits, dim=-1)    

    pos_result=get_results(pos_logits, tokenizer)
    neg_result=get_results(neg_logits, tokenizer)
    our_result=get_results(logits, tokenizer)

    torch.cuda.empty_cache()

    return pos_result, neg_result, our_result

def main(args):

    infra_data=[]
    with open(args.input_path, "r", encoding="utf-8") as r:
        for line in r.readlines():
            infra_data.append(json.loads(line))

    pos_input = args.postive_prompt

    neg_input= args.negative_prompt
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,  trust_remote_code=True)
    
    print("loading the model")
    large_model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True, device_map="auto")

    pos_all, neg_all, ours_all=[],[],[]
    alpha = args.alpha
    print("**"*10+"Start inference."+"**"*10)
    for id, data in tqdm(enumerate(infra_data)):

        label=int(data["label"][-1])-1

        pos_result, neg_result, our_result = contrastive_decoding(pos_input, neg_input, data, large_model, tokenizer, alpha=args.alpha)
        pos_all.append(pos_result==label)
        neg_all.append(neg_result==label)
        ours_all.append(our_result==label)
        
        score={
            "pos": len([1 for i in pos_all if i==True])/len(pos_all),
            "neg": len([1 for i in neg_all if i==True])/len(neg_all),
            "ours": len([1 for i in ours_all if i==True])/len(ours_all),
        }
        
        if (id+1)%10==0:

            all_results={
                "pos": pos_all,
                "neg": neg_all,
                "ours": ours_all,
            }
            with open(f"{args.output_path}/evaluation.json", "w") as r:
                json.dump(score, r, indent=4)
            with open(f"{args.output_path}/inference_results.json", "w") as r:
                json.dump(all_results, r, indent=4)
        
    print("**"*10+"Finish inference!"+"**"*10)   
    
    all_results={
        "pos": len([1 for i in pos_all if i==True])/len(pos_all),
        "neg": len([1 for i in neg_all if i==True])/len(neg_all),
        "ours": len([1 for i in ours_all if i==True])/len(ours_all),
    }
    with open(f"{args.output_path}/evaluation.json", "w") as r:
        json.dump(all_results, r, indent=4)
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

    
