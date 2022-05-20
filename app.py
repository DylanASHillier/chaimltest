from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from flask import Flask, request, json
from optimum.onnxruntime import ORTModelForCausalLM

app = Flask(__name__)
app_params = {}
open("app_config.json") as f:
    app_params = json.load(f)
app_params=generation_params["app_params"]
version = app_params["opt_version"]

model_type = app_params["optimization_level"]
if model_type == 'baseline':
    model = AutoModelForCausalLM.from_pretrained(f"facebook/opt-{version}")
elif model_type == 'onnx':
    model = ORTModelForCausalLM.from_pretrained(version,file_name=f"opt-{version}.onnx")
elif model_type == 'onnxruntime':
    model = ORTModelForCausalLM.from_pretrained(version,file_name=f"oopt-{version}.onnx")
elif model_type == 'fusion':
    model = ORTModelForCausalLM.from_pretrained(version,file_name=f"foopt-{version}.onnx")
elif model_type == 'quantized':
    model = ORTModelForCausalLM.from_pretrained(version,file_name=f"qoopt-{version}.onnx")
tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{version}")
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)

generation_params = {}
with open("generation_config.json") as f:
    generation_params = json.load(f)
generation_params=generation_params["generation_params"]

@app.route('/', methods=['POST'])
def infer():
    '''
    Receives text input from post request and runs infernece on it.
    '''
    text = request.form['text']
    out = pipeline(
        text, 
        return_full_text=False,
        generation_kwargs=generation_params
    )
    out = out[0]["generated_text"]
    return out

if __name__=='__main__':
    while True:
        text = input("Enter a sentence: ")
        out = pipeline(
            text, 
            return_full_text=False,
            generation_kwargs=generation_params
        )
        out = out[0]["generated_text"]
        print(out)