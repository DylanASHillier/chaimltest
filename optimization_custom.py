from transformers.onnx import OnnxConfigWithPast, export, validate_model_outputs
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import onnx
import torch
from optimum.onnxruntime import ORTConfig, ORTQuantizer, ORTOptimizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
import argparse
from onnx_config import OPTOnnxConfig
import onnxruntime
from onnxruntime.transformers.onnx_model_gpt2 import Gpt2OnnxModel
from onnxruntime.transformers.fusion_options import FusionOptions

## adapted from onnxruntime.transformers.optimizer
## Fusion not implemented

def optimize_by_onnxruntime(onnx_model_path, optimized_model_path, disabled_optimizers=[]):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = optimized_model_path

    kwargs = {}
    if disabled_optimizers:
        kwargs["disabled_optimizers"] = disabled_optimizers

    
    session = onnxruntime.InferenceSession(onnx_model_path,
                                               sess_options,
                                               providers=['CPUExecutionProvider'],
                                               **kwargs)
    return optimized_model_path

def optimize(onnx_model, optimized_model_path, config, base_config):
    optimize_by_onnxruntime(onnx_model, optimized_model_path)
    optimize_by_fusion(optimized_model_path, config.num_attention_heads, base_config.hidden_size)
    print("done onnxruntime optimization -- fusion not yet implemented")

def optimize_by_fusion(model_path,
                       num_heads: int = 0,
                       hidden_size: int = 0):
    """ Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.
    """
    model = onnx.load_model(model_path, format=None, load_external_data=True)
    optimizer_class = Gpt2OnnxModel

    optimization_options = FusionOptions('gpt2')

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()
    optimizer.save_model_to_file(f"f{model_path}", True)


