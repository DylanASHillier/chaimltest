from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from flask import Flask, request, json
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-30b")
model.config.max_length=21

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)

generation_params = {}
# with open("generation_config.json") as f:
#     generation_params = json.load(f)
# generation_params=generation_params["generation_params"]

@app.route('/', methods=['POST'])
def infer():
    '''
    Receives text input from post request and runs infernece on it.
    '''
    text = request.form['text']
    out = pipeline(
        text, 
        return_full_text=False,
        **generation_params
    )
    out = out[0]["generated_text"]
    return out

if __name__=='__main__':
    while True:
        text = input("Enter a sentence: ")
        out = pipeline(
            text, 
            return_full_text=False,
            **generation_params
        )
        out = out[0]["generated_text"]
        print(out)