from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

@app.route('/', methods=['POST'])
def infer():
    '''
    Receives text input from post request and runs infernece on it.
    '''
    text = request.form['text']
    tokenized_text = tokenizer(text, return_tensors="pt")
    output = model.generate(**tokenized_text)

    out =  tokenizer.decode(output[0])
    print(out)
    return out

# if __name__=='__main__':
#     while True:
#         text_input = input("Enter a sentence: ")
#         tokenized_input = tokenizer(text_input, return_tensors="pt")
#         output = model.generate(**tokenized_input)
#         print(tokenizer.decode(output[0]))