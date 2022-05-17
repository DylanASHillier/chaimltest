from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from datasets import load_dataset
from torchmetrics import BLEUScore
import time
import numpy as np
import json

def run_benchmarks():
    # load generation params from json
    generation_params = {}
    with open("generation_config.json") as f:
        generation_params = json.load(f)

    models = [
        "facebook/opt-125m",
        # "facebook/opt-350m",
        "facebook/opt-1.3b",
        # "facebook/opt-2.7b",
        # "facebook/opt-6.7b",
        # "facebook/opt-13b",
        # "facebook/opt-30b",
    ]
    metric = BLEUScore()
    # dataset = load_dataset("glue","qnli")["validation"]
    dataset = load_dataset("newsgroup","18828_sci.space")["train"]["text"] ## Dataset consisting in text
    for model_string in models:
        model = AutoModelForCausalLM.from_pretrained(model_string)
        tokenizer = AutoTokenizer.from_pretrained(model_string)
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
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
    random_indices = np.random.choice(len(dataset), size=10, replace=False)
    for i in random_indices:
        idx = len(dataset[i])//2
        generation_params["max_new_tokens"]=idx//2
        question, target = dataset[i][:idx], dataset[i][idx:]
        source = f"{question}"
        answer = pipeline(source,return_full_text=False,generate_kwargs=generation_params)
        answer = answer[0]["generated_text"]
        # print(f"output: {output}")
        # print(f"target: {target}")
        metric([answer],[[target]])
    scores = metric.compute()
    print(scores)
        
if __name__ == '__main__':
    run_benchmarks()