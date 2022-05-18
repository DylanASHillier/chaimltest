from transformers.onnx import OnnxConfigWithPast, export, validate_model_outputs
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Mapping, OrderedDict
import onnx


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
    return OPTOnnxConfig(AutoConfig.from_pretrained(model_name))

def export_onxx(model_name, output_path):
    config = get_opt_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = get_opt_config(model_name)
    export(tokenizer, model, config, config.default_onnx_opset, Path(output_path))

if __name__ == '__main__':
    # config = get_opt_config("facebook/opt-125m")
    # print(config.outputs)
    # export_onxx("facebook/opt-125m", "opt-125m.onnx")
    onnx.checker.check_model(onnx.load("opt-125m.onnx"))
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    onnx_config = get_opt_config(model_name)
    validate_model_outputs(onnx_config, tokenizer, base_model, Path("opt-125m.onnx"), onnx_config.outputs, onnx_config.atol_for_validation)