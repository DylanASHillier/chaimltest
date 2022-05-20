from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from optimum.onnxruntime import ORTModelForCausalLM
from datasets import load_dataset
from torchmetrics import BLEUScore
import time
import numpy as np
import json
import argparse
from memory_profiler import profile

onnx_models = [
    "opt",
    "oopt",
    "foopt",
    "qoopt",
]


def load_generation_params():
    with open("generation_config.json") as f:
        generation_params = json.load(f)["generation_params"]
    generation_params["do_sample"] = False
    return generation_params


def run_benchmarks(opt_version):
    '''
    runs time benchmarks on the various different versions of a model
    arguments:
        version: string in "125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"
    assumes that this also the name of the folder where the models are stored.
    '''
    generation_params = load_generation_params()
    onnx_model_strings = [f"{item}-{opt_version}.onnx" for item in onnx_models]
    model_name = f"facebook/opt-{opt_version}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = load_dataset("newsgroup", "18828_sci.space")["train"]["text"]
    inference_params = {
        "generation_params": generation_params,
    }
    num_samples = 10
    for model_string in onnx_model_strings:
        model = ORTModelForCausalLM.from_pretrained(
            opt_version, file_name=model_string)
        inference_params["pipeline"] = TextGenerationPipeline(
            model=model, tokenizer=tokenizer, device=-1)
        print(f"evaluating {model_string}")
        run_eval(pipeline, dataset, num_samples)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipeline = TextGenerationPipeline(
        model=model, tokenizer=tokenizer, device=-1)
    print(f"evaluating baseline")
    run_eval(inference_params, dataset, num_samples)


@profile
def run_inference(inference_params):
    pipeline = inference_params["pipeline"]
    return pipeline(inference_params["question"], return_full_text=False, generation_kwargs=inference_params["generation_params"])


def update_metrics(inference_params, metrics, target):
    start = time.time()
    answer = run_inference(inference_params)
    end = time.time()
    metrics["time_metric"] += end-start
    metrics["bleu_metric"].update([answer], [[target]])


def report_metrics(metrics):
    print(f"time: {metrics['time_metric']}")
    print(f"bleu: {metrics['bleu_metric'].compute()}")


def get_prefix_target(dataset, i):
    text = dataset[i]
    acceptable_length = 75
    length = min(acceptable_length, len(text))
    split_idx = length//3
    return dataset[i][:split_idx], dataset[i][split_idx:length]


def run_eval(inference_params, dataset, num_samples):
    metrics = {
        "bleu_metric": BLEUScore(),
        "time_metric": 0,
    }
    np.random.seed(123)
    random_indices = np.random.choice(
        len(dataset), size=num_samples, replace=False)
    for i in random_indices:
        prefix, target = get_prefix_target(dataset, i)
        inference_params["prefix"] = prefix
        update_metrics(inference_params, metrics, target)
    report_metrics(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", default="125m",
                        help="Name of the opt model version -- i.e. one of 125m, 30b etc.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_benchmarks(args.model_version)
