from transformers.onnx import OnnxConfigWithPast, export, validate_model_outputs
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import onnx
import torch
from optimum.onnxruntime import ORTConfig, ORTQuantizer, ORTOptimizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
import argparse
from onnx_config import OPTOnnxConfig
from optimization_custom import optimize

model_folder = "../13b"

def get_opt_config(model_name) -> OPTOnnxConfig:
    return OPTOnnxConfig(AutoConfig.from_pretrained(model_name), task="causal-lm")

def export_onnx(model_name, output_path):
    config = get_opt_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = get_opt_config(model_name)
    return export(tokenizer, model, config, config.default_onnx_opset,
        Path(model_folder+"/"+output_path))

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
    quantizer.export(Path( onnx_path), Path( output_path), qconfig, use_external_data_format=True)

# def optimize(model_name, onnx_config, onnx_path, optimized_onnx_path):
#     oconfig = OptimizationConfig(optimization_level=99)
#     fake_name = "distilbert-base-uncased-finetuned-sst-2-english"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     optimizer = ORTOptimizer.from_pretrained(fake_name, feature="sequence-classification")
#     optimizer.feature = "causal-lm"
#     optimizer.model = model
#     optimizer.tokenizer = tokenizer
#     optimizer._onnx_config = onnx_config
#     optimizer._model_type = 'gptj'


#     optimizer.export(Path(model_folder + "/" + onnx_path),
#         Path(model_folder+ "/"+ optimized_onnx_path), oconfig)

def validate_model(onnx_config, onnx_outputs, model_name, onnx_filename, tol=1e-3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    # onnx.checker.check_model(onnx.load(onnx_filename)) # doesn't work for overly large files...
    validate_model_outputs(onnx_config, tokenizer, base_model,
        Path(model_folder+ "/"+ onnx_filename), onnx_outputs, tol)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specific_model", default="opt-125m",help="Name of the model to export")
    parser.add_argument("--quantize", action="store_true", default=False, help="Quantize the model")
    parser.add_argument("--optimize", action="store_true", default=False, help="Optimize the model")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    specific_model = args.specific_model
    model_name = f"facebook/{specific_model}"
    curr_onnx_name = f"{specific_model}.onnx"
    config = get_opt_config(model_name)
    onnx_inputs, onnx_outputs = export_onnx(model_name, curr_onnx_name)
    # validate_model(config, onnx_outputs, model_name, curr_onnx_name)
    if args.optimize:
        print("optimizing")
        oonnx_name = "o"+curr_onnx_name
        # optimize(model_name, config, curr_onnx_name,oonnx_name)
        # optimize(curr_onnx_name,oonnx_name, config, AutoConfig.from_pretrained(model_name))
        # validate_model(config, onnx_outputs, model_name, oonnx_name, tol=10)
        curr_onnx_name = oonnx_name
    if args.quantize:
        print("quantizing")
        qonnx_name = "q"+curr_onnx_name
        quantize(model_name, curr_onnx_name, qonnx_name)
        # validate_model(config, onnx_outputs, model_name, qonnx_name, tol=10)
        curr_onnx_name = qonnx_name

    # quantize_from_hub("facebook/opt-125m")