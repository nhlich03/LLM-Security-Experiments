# ROSE
This is the official implementation of our ACL2024 (Findings) paper, "[ROSE Doesn't Do That: Boosting the Safety of Instruction-Tuned Large Language Models with Reverse Prompt Contrastive Decoding](https://aclanthology.org/2024.findings-acl.814.pdf)" (in Pytorch).

## Requirements and Installation

- PyTorch version >= 1.10.0
- Python version >= 3.8
- lmdeploy
- transformers
- openai == 0.28
- For evaluation, you'll also need an NVIDIA GPU and NCCL.

# Getting Started
Here, we introduce how to reproduce our experimental results in the paper.

## Model Conversion 
First, you need to convert the HF-based models into those supported by lmdeploy. More details can be found in [lmdeploy](https://github.com/InternLM/lmdeploy).

## Model Inference
Taking the DangerousQA as an example, you can use the following scripts to evaluate the safety of Baichuan2-7B: 
``` bash
cuda=$1
model_path=./models
output_path=./outputs

CUDA_VISIBLE_DEVICES=$cuda python3 ./src/dangerousqa_inference.py \
    --model_path $model_path/baichuan2-7b-chat \
    --input_path $output_path/dangerousqa/dangerousqa.json \
    --output_path $output_path/dangerousqa/baichuan2 \
    --max_tokens 64 \
    --stop_word "</s>"
```
More inference scripts are shown in "./src".

## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@inproceedings{zhong2024rose,
  title={ROSE Doesn't Do That: Boosting the Safety of Instruction-Tuned Large Language Models with Reverse Prompt Contrastive Decoding},
  author={Zhong, Qihuang and Ding, Liang and Liu, Juhua and Du, Bo and Tao, Dacheng},
  booktitle={ACL},
  year={2024}
}
```



