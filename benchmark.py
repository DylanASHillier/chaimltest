from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from optimum.onnxruntime import ORTModelForCausalLM
from datasets import load_dataset
from torchmetrics import BLEUScore
import time
import numpy as np
import json
import torch
import argparse

def run_benchmarks(version):
    '''
    runs time benchmarks on the various different versions of a model
    arguments:
        version: string in "125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"
    assumes that this also the name of the folder where the models are stored.
    '''
    # load generation params from json
    generation_params = {}
    with open("generation_config.json") as f:
        generation_params = json.load(f)["generation_params"]
    generation_params["do_sample"]=False ## make deterministic

    onnx_models = [
        "opt",
        "oopt",
        "foopt",
        "qoopt",
    ]

    onnx_models = [f"{item}-{version}.onnx" for item in onnx_models]
    metric = BLEUScore()
    model_name = f"facebook/opt-{version}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = load_dataset("newsgroup","18828_sci.space")["train"]["text"] ## Dataset consisting in text
    for model_string in onnx_models:
        model = ORTModelForCausalLM.from_pretrained(version,file_name=model_string)
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
        print(f"evaluating {model_string}") 
        run_loop(pipeline, dataset, generation_params, metric)
        metric.reset()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
    print(f"evaluating baseline")
    run_loop(pipeline, dataset, generation_params, metric)
    metric.reset()


def run_loop(pipeline, dataset, generation_params, metric):
    # questions = dataset["question"]
    # answers = dataset["sentence"]
    np.random.seed(123)
    random_indices = np.random.choice(len(dataset), size=10, replace=False)
    tot_time = 0
    for i in random_indices:
        text =dataset[i]
        length = min(50,len(text))
        idx = length//2
        # generation_params["max_new_tokens"]=idx//2
        question, target = dataset[i][:idx], dataset[i][idx:length]
        source = f"{question}"
        start = time.time()
        answer = pipeline(source,return_full_text=False,generation_kwargs=generation_params)
        print(source,answer[0]["generated_text"])
        end = time.time()
        tot_time += end-start
        answer = answer[0]["generated_text"]
        # print(f"output: {answer}")
        # print(f"target: {target}")
        metric([answer],[[target]])
    print(tot_time)
    scores = metric.compute()
    print(f"scores were: {scores}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", default="125m",help="Name of the opt model version -- i.e. one of 125m, 30b etc.")
    return parser.parse_args()
        
if __name__ == '__main__':
    args = parse_args()
    run_benchmarks(args.model_version)