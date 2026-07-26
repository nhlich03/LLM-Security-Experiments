import argparse
import json
from tqdm import tqdm

import nltk
# Download when running for the first time
nltk.download('words')

from config import path_harmful, path_harmless, model_paths
from utils import load_model, load_ori_prompts, get_jailbreak_prompts
from utils import get_sentence_embeddings
from utils import interpret_difference_matrix


def interpret(model_name):
    # Load model
    model, tokenizer = load_model(model_name, model_paths)

    # Load data
    harmful_prompts, harmless_prompts = load_ori_prompts(
        path_harmful, path_harmless
    )
    print("Number of harmful prompts: {}".format(len(harmful_prompts)))
    print("Number of harmless prompts: {}".format(len(harmless_prompts)))
    jailbreaks = [
        "gcg",
        "autodan",
        "saa",
        "drattack",
        "pair",
        "puzzler",
        "ijp",
        "base64",
        "zulu",
    ]
    jailbreak_prompts = get_jailbreak_prompts(model_name, jailbreaks, split="all")
    for jailbreak in jailbreaks:
        print(f"Number of {jailbreak} prompts: {len(jailbreak_prompts[jailbreak])}")

    # Get embdddings for prompts
    harmful_embeddings = get_sentence_embeddings(
        harmful_prompts, model, model_name, tokenizer
    )
    harmless_embeddings = get_sentence_embeddings(
        harmless_prompts, model, model_name, tokenizer
    )

    embeddings_gcg = get_sentence_embeddings(
        jailbreak_prompts["gcg"], model, model_name, tokenizer
    )
    embeddings_puzzler = get_sentence_embeddings(
        jailbreak_prompts["puzzler"], model, model_name, tokenizer
    )
    embeddings_saa = get_sentence_embeddings(
        jailbreak_prompts["saa"], model, model_name, tokenizer
    )
    embeddings_autodan = get_sentence_embeddings(
        jailbreak_prompts["autodan"], model, model_name, tokenizer
    )
    embeddings_drattack = get_sentence_embeddings(
        jailbreak_prompts["drattack"], model, model_name, tokenizer
    )
    embeddings_pair = get_sentence_embeddings(
        jailbreak_prompts["pair"], model, model_name, tokenizer
    )
    embeddings_ijp = get_sentence_embeddings(
        jailbreak_prompts["ijp"], model, model_name, tokenizer
    )
    embeddings_base64 = get_sentence_embeddings(
        jailbreak_prompts["base64"], model, model_name, tokenizer
    )
    embeddings_zulu = get_sentence_embeddings(
        jailbreak_prompts["zulu"], model, model_name, tokenizer
    )

    # Interpret the embeddings
    # Save the results in interpre_results/{model_name}.txt
    with open("interpret_results/{}.txt".format(model_name), "a") as f:
        for layer_n in tqdm(range(len(harmful_embeddings))):
            f.write(
                "====================layer {}====================\n".format(layer_n)
            )
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                harmful_embeddings[layer_n],
                harmless_embeddings[layer_n],
            )
            f.write("Toxic harmful-harmless " + json.dumps(interpret_tokens) + "\n")
            f.write("------------------------------------------------\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_gcg[layer_n], harmless_embeddings[layer_n]
            )
            f.write("Toxic gcg-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_saa[layer_n], harmless_embeddings[layer_n]
            )
            f.write("Toxic saa-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_autodan[layer_n],
                harmless_embeddings[layer_n],
            )
            f.write("Toxic autodan-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_drattack[layer_n],
                harmless_embeddings[layer_n],
            )
            f.write("Toxic drattack-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_pair[layer_n], harmless_embeddings[layer_n]
            )
            f.write("Toxic pair-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_puzzler[layer_n],
                harmless_embeddings[layer_n],
            )
            f.write("Toxic puzzler-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_ijp[layer_n], harmless_embeddings[layer_n]
            )
            f.write("Toxic ijp-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_base64[layer_n],
                harmless_embeddings[layer_n],
            )
            f.write("Toxic base64-harmless " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_zulu[layer_n], harmless_embeddings[layer_n]
            )
            f.write("Toxic zulu-harmless " + json.dumps(interpret_tokens) + "\n")
            f.write("------------------------------------------------\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_gcg[layer_n], harmful_embeddings[layer_n]
            )
            f.write("Jailbreak gcg-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_saa[layer_n], harmful_embeddings[layer_n]
            )
            f.write("Jailbreak saa-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_autodan[layer_n],
                harmful_embeddings[layer_n],
            )
            f.write("Jailbreak autodan-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_drattack[layer_n],
                harmful_embeddings[layer_n],
            )
            f.write("Jailbreak drattack-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_pair[layer_n], harmful_embeddings[layer_n]
            )
            f.write("Jailbreak pair-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_puzzler[layer_n],
                harmful_embeddings[layer_n],
            )
            f.write("Jailbreak puzzler-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_ijp[layer_n], harmful_embeddings[layer_n]
            )
            f.write("Jailbreak ijp-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model,
                tokenizer,
                embeddings_base64[layer_n],
                harmful_embeddings[layer_n],
            )
            f.write("Jailbreak base64-harmful " + json.dumps(interpret_tokens) + "\n")
            interpret_tokens, _, _ = interpret_difference_matrix(
                model, tokenizer, embeddings_zulu[layer_n], harmful_embeddings[layer_n]
            )
            f.write("Jailbreak zulu-harmful " + json.dumps(interpret_tokens) + "\n")
            f.flush()


if __name__ == "__main__":
    # Get parameters
    parser = argparse.ArgumentParser(description="Interpret the model")
    parser.add_argument("--model", type=str, help="Taregt model")

    args = parser.parse_args()
    model_name = args.model

    # Show the tokens that are most assiciated with the toxic and jailbreak concepts
    # The results are saved in ./interpre_results
    interpret(model_name)
    print("{} Interpretation done.".format(model_name))

# An example for run this script on llama-2
# python interpret.py --model llama-2
