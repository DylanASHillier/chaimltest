from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from torchmetrics import BLEUScore
import time
import numpy as np

def run_benchmarks():
    generation_params = {
        "max_new_tokens":100,
        "temperature":0.7,
        "do_sample":True,
        "top_p":0.9,
    }

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
        model.eval()
        print(f"evaluating {model_string}")
        start = time.time()
        run_loop(model, tokenizer, dataset, generation_params, metric)
        end = time.time()
        print(f"{model_string} took {end-start} seconds")
        metric.reset()


def run_loop(model, tokenizer, dataset, generation_params, metric):
    # questions = dataset["question"]
    # answers = dataset["sentence"]
    np.random.seed(123)
    random_indices = np.random.choice(len(dataset), size=10, replace=False)
    for i in random_indices:
        idx = len(dataset[i])//2
        generation_params["max_new_tokens"]=idx//2
        question, answer = dataset[i][:idx], dataset[i][idx:]
        source = f"{question}"
        tokenized_source = tokenizer(source, return_tensors="pt")
        output = model.generate(**tokenized_source, 
            **generation_params
        )
        output = tokenizer.decode(output[0])[len(source):]
        target = answer
        # print(f"output: {output}")
        # print(f"target: {target}")
        metric([output],[[target]])
    scores = metric.compute()
    print(scores)
        
if __name__ == '__main__':
    run_benchmarks()