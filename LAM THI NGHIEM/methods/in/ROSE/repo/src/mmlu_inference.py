import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

import logging

logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        pos_input_ids = tokenizer(f"{args.postive_prompt}\n{prompt}", return_tensors="pt").input_ids.cuda()
        neg_input_ids = tokenizer(f"{args.negative_prompt}\n{prompt}", return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        pos_logits = model(input_ids=pos_input_ids).logits[::, -1, :]
        neg_logits = model(input_ids=neg_input_ids).logits[::, -1, :]
        logits=pos_logits-args.alpha*neg_logits
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[0][tokenizer.get_vocab()["A"]],
                        logits[0][tokenizer.get_vocab()["B"]],
                        logits[0][tokenizer.get_vocab()["C"]],
                        logits[0][tokenizer.get_vocab()["D"]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    logger.info("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):

    logger.info("Start Loading the model")

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    model = AutoModelForCausalLM.from_pretrained(args.model_path,trust_remote_code=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False,  trust_remote_code=True)

    model.eval()

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        logger.info("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        logger.info("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    logger.info("Average accuracy: {:.3f}".format(weighted_acc))
    with open(os.path.join(args.data_dir, "overall_result.txt"), "a") as w:
        w.write(f"{args.model_path}\t{weighted_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=4)
    parser.add_argument("--data_dir", "-d", type=str, default="mmlu/data/")
    parser.add_argument("--save_dir", "-s", type=str, default="output/mmlu_results")
    parser.add_argument("--model", "-m", type=str, default="vicuna-7b")
    parser.add_argument('--postive_prompt', type=str, default="")
    parser.add_argument('--negative_prompt', type=str, default="")
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument(
        "--model_path",
        type=str,
        default="vicuna-7b-delta-v1.1-convert",
    )
    args = parser.parse_args()
    main(args)