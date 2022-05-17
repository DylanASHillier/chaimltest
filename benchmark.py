from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from datasets import load_dataset
from torchmetrics import BLEUScore
import time
import numpy as np
import json
import torch

# test_dataset = 

def run_benchmarks():
    # load generation params from json
    generation_params = {}
    with open("generation_config.json") as f:
        generation_params = json.load(f)["generation_params"]

    models = [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
    ]
    metric = BLEUScore()
    # dataset = load_dataset("glue","qnli")["validation"]
    dataset = load_dataset("newsgroup","18828_sci.space")["train"]["text"] ## Dataset consisting in text
    for model_string in models:
        model = AutoModelForCausalLM.from_pretrained(model_string)
        tokenizer = AutoTokenizer.from_pretrained(model_string)
        model.config.max_length=200
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=1)
        print(f"evaluating {model_string}")
        start = time.time()
        run_loop(pipeline, dataset, generation_params, metric)
        end = time.time()
        print(f"{model_string} took {end-start} seconds")
        metric.reset()


def run_loop(pipeline, dataset, generation_params, metric):
    # questions = dataset["question"]
    # answers = dataset["sentence"]
    np.random.seed(123)
    random_indices = np.random.choice(len(dataset), size=1, replace=False)
    for i in random_indices:
        text =dataset[i]
        length = min(500,len(text))
        idx = length//2
        # generation_params["max_new_tokens"]=idx//2
        question, target = dataset[i][:idx], dataset[i][idx:length]
        source = f"{question}"
        answer = pipeline(source,return_full_text=False,**generation_params)
        answer = answer[0]["generated_text"]
        # print(f"output: {answer}")
        # print(f"target: {target}")
        metric([answer],[[target]])
    scores = metric.compute()
    print(f"scores were: {scores}")
        
if __name__ == '__main__':
    run_benchmarks()