import os
import sys
from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
model_name = None
model = None
tokenizer = None

def load_model(model_name):
    global model, tokenizer
    model = T5ForConditionalGeneration.from_pretrained(f"./models/{model_name}")
    tokenizer = T5Tokenizer.from_pretrained(f"./models/{model_name}")

def generate_response(input_text, max_length=50, num_beams=5, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, repetition_penalty=2.0):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/')
def index():
    model_options = sorted([f for f in os.listdir('./models') if os.path.isdir(os.path.join('./models', f))])
    return render_template('index.html', models=model_options)

@app.route('/select_model', methods=['POST'])
def select_model():
    global model_name
    model_name = request.json.get('model')
    load_model(model_name)
    return jsonify({'status': 'Model loaded', 'model': model_name})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('prompt')
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(debug=True, port=port)