from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-large-256"
# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(f"./{model_name}")
tokenizer = T5Tokenizer.from_pretrained(f"./{model_name}")

# Function to generate responses with better settings
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

# Main loop to test the chatbot
while True:
    user_input = input("Enter: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print(response)