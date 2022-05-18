from transformers.onnx import OnnxConfigWithPast, export, validate_model_outputs
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Mapping, OrderedDict
import onnx
import torch
from optimum.onnxruntime import ORTConfig, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import argparse


class OPTOnnxConfig(OnnxConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int,str]]:
        common_inputs = OrderedDict(
            [
                ("input_ids", OrderedDict([(0, "input_ids")])),
                ("attention_mask", OrderedDict([(0, "attention_mask")])),
            ]
        )
        if self.use_past:
            for i in range(self.num_layers[0]):
                common_inputs[f"past_key_values.{i}.key"]= OrderedDict([(0,"batch"),(2,"past_sequence + sequence")])
                common_inputs[f"past_key_values.{i}.value"]= OrderedDict([(0,"batch"),(2,"past_sequence + sequence")])
        return common_inputs

def get_opt_config(model_name) -> OPTOnnxConfig:
    return OPTOnnxConfig(AutoConfig.from_pretrained(model_name), task="causal-lm")

def export_onxx(model_name, output_path):
    config = get_opt_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = get_opt_config(model_name)
    return export(tokenizer, model, config, config.default_onnx_opset, Path(output_path))

def quantize_from_hub(model_name):
    '''
    Does not currently work as opt has not been added to hub --> use in future to bypass bugs by me
    '''
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    quantizer = ORTQuantizer.from_pretrained(model_name, feature="causal-lm")
    # Quantize the model!
    quantizer.fit(model_name, output_dir=".", feature="causal-lm")
    # quantizer = ORTQuantizer(
    #     onnx_model_path="model.onnx",
    #     onnx_quantized_model_output_path="model-quantized.onnx",
    #     quantization_config=qconfig,
    # )

def quantize(model_name,onnx_path,output_path):
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    fake_name = "distilbert-base-uncased-finetuned-sst-2-english" ### Very Hacky solution
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    quantizer = ORTQuantizer.from_pretrained(fake_name, feature="sequence-classification")
    quantizer.feature = "causal-lm"
    quantizer.model = model
    quantizer.tokenizer = tokenizer

    # quantizer.fit(model_name, output_dir=".", feature="causal-lm")
    quantizer.export(Path(onnx_path), Path(output_path), qconfig, use_external_data_format=True)

def validate_model(onnx_config, onnx_outputs, model_name, onnx_filename):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    # onnx.checker.check_model(onnx.load(onnx_filename)) # doesn't work for overly large files...
    validate_model_outputs(onnx_config, tokenizer, base_model, Path(onnx_filename), onnx_outputs, 1e-3)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specific_model", default="opt-125m",help="Name of the model to export")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    specific_model = args.specific_model
    model_name = f"facebook/{specific_model}"
    onnx_name = f"{specific_model}.onnx"
    qonnx_name = f"{specific_model}q.onnx"
    config = get_opt_config(model_name)
    onnx_inputs, onnx_outputs = export_onxx(model_name, onnx_name)
    validate_model(config, onnx_outputs, model_name, onnx_name)
    quantize(model_name,onnx_name,qonnx_name)
    validate_model(config, onnx_outputs, model_name, qonnx_name)
    # quantize_from_hub("facebook/opt-125m")