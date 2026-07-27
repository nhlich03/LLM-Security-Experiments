# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import argparse
import fire
import torch
from torch import nn
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm
from lmdeploy import turbomind as tm
from transformers import AutoTokenizer, top_k_top_p_filtering
import re
import json
import warnings
warnings.filterwarnings("ignore")

os.environ['TM_LOG_LEVEL'] = 'ERROR'

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for sample.py')

    parser.add_argument('--model_path', type=str, default="Alpaca-7b/fastertransformer_model")
    parser.add_argument('--input_path', type=str, default="datasets/xstest/xstest_v2_prompts.csv")
    parser.add_argument('--output_path', type=str, default="./output/xstest")
    parser.add_argument('--postive_prompt', type=str, default="")
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument('--max_tokens', type=int, default=64)
    parser.add_argument(
        "--data_range",
        type=str,
        default="[0:]",
        help="the range of data",
    )
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--stop_word', type=str, default="</s>")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    return args

@torch.no_grad()
def directDecodeByGreedyDecoding(sys_prompt, query, generator, tokenizer, args):
    inputs = f"{sys_prompt}\nHuman:\n{query}\n\nAssistant:\n"
    input_ids = tokenizer.encode(inputs)
    for _ in range(args.max_tokens):
        logits = generator.decode(input_ids)
        logits = logits[: , -1, :]
        probs = nn.functional.softmax(logits, dim=-1)
        next_token = torch.argsort(probs, dim=-1, descending=True)[:, 0].unsqueeze(0)
        input_ids[-1] = next_token.item()
    res = tokenizer.decode(input_ids)
    res = re.split(args.stop_word, res)[0]
    res = re.split(r"Assistant:\n", res)[-1]
    return res

@torch.no_grad()
def directDecodeByConstractiveAware(sys_prompt, pos_prompt, neg_prompt, query, alpha, generator_pos, generator_neg, tokenizer, args):
    pos_total_prompt = f"{sys_prompt}{pos_prompt}\nHuman:\n{query}\n\nAssistant:\n"
    neg_total_prompt = f"{sys_prompt}{neg_prompt}\nHuman:\n{query}\n\nAssistant:\n"

    pos_input_ids = tokenizer.encode(pos_total_prompt)
    neg_input_ids = tokenizer.encode(neg_total_prompt)
    for _ in range(args.max_tokens):
        logits_P = generator_pos.decode(pos_input_ids)
        logits_pos = logits_P[:,-1,:]
        logits_N = generator_neg.decode(neg_input_ids)
        logits_neg = logits_N[:,-1,:]
        

        logits=logits_pos-alpha*logits_neg

        probs = nn.functional.softmax(logits, dim=-1)

        next_token = torch.argsort(probs, dim=-1, descending=True)[:, 0].unsqueeze(0)
            
        pos_input_ids[-1] = next_token.item()
        neg_input_ids[-1] = next_token.item()
    res = tokenizer.decode(pos_input_ids)
    res = re.split(args.stop_word, res)[0]
    res = re.split(r"Assistant:\n", res)[-1]
    return res

def main(args):

    print("**"*10+"Start loading model."+"**"*10)
    tokenizer_model_path = osp.join(args.model_path, 'triton_models', 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, trust_remote_code=True)
    tm_model = tm.TurboMind(args.model_path, eos_id=tokenizer.eos_token_id, tp=4)
    
    generator = tm_model.create_instance()
    generator_pos = tm_model.create_instance()
    generator_neg = tm_model.create_instance()
    
    infra_data=pd.read_csv(args.input_path)
    
    data_min, _, data_max=args.data_range[1:-1].partition(":")
    if data_min =="":
        data_min=0
    if data_max =="":
        data_max=len(infra_data)
    data_min, data_max=int(data_min), int(data_max)

    sys_prompt = ""
    pos_input = args.postive_prompt
    neg_input= args.negative_prompt
    
    pos_all, neg_all, ours_all=[],[],[]
    alpha = args.alpha
    print("**"*10+"Start inference."+"**"*10)
    for id, instruction in tqdm(enumerate(infra_data["prompt"][data_min:data_max])):

        pos=directDecodeByGreedyDecoding(sys_prompt=pos_input,
                                     query=instruction,
                                     generator = generator,
                                     tokenizer=tokenizer,
                                     args=args)
        pos_all.append({
            "id": id,
            "type": infra_data["type"][id],
            "instruciton":instruction,
            "output":pos
        })
        
        neg=directDecodeByGreedyDecoding(sys_prompt=neg_input,
                                     query=instruction,
                                     generator = generator,
                                     tokenizer=tokenizer,
                                     args=args)
        neg_all.append({
            "id": id,
            "type": infra_data["type"][id],
            "instruciton":instruction,
            "output":neg
        })

        ours=directDecodeByConstractiveAware(sys_prompt=sys_prompt,
                                    pos_prompt=pos_input,
                                    neg_prompt=neg_input,
                                    query=instruction,
                                    alpha=alpha,
                                    generator_pos = generator_pos,
                                    generator_neg = generator_neg,
                                    tokenizer=tokenizer,
                                     args=args)
        ours_all.append({
            "id": id,
            "type": infra_data["type"][id],
            "instruciton":instruction,
            "output":ours
        })
        if (id+1)%10==0:
            with open(f"{args.output_path}/pos_output.json", "w") as r:
                json.dump(pos_all, r, indent=4)
                
            with open(f"{args.output_path}/neg_output.json", "w") as r:
                json.dump(neg_all, r, indent=4)
                
            with open(f"{args.output_path}/ours_output.json", "w") as r:
                json.dump(ours_all, r, indent=4)
    
    print("**"*10+"Finish inference!"+"**"*10)
    with open(f"{args.output_path}/pos_output.json", "w") as r:
        json.dump(pos_all, r, indent=4)
        
    with open(f"{args.output_path}/neg_output.json", "w") as r:
        json.dump(neg_all, r, indent=4)
        
    with open(f"{args.output_path}/ours_output.json", "w") as r:
        json.dump(ours_all, r, indent=4)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)