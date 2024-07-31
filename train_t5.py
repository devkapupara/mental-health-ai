import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import torch

# Load the dataset
df = pd.read_csv("cleaned_mental_health_data.csv")

# Convert your data into a dictionary format
data = {
    "input_text": [],
    "target_text": []
}
for row in df["text"]:
    question, response = row.split("\n")
    question = question.replace("User: ", "")
    response = response.replace("Therapist: ", "")
    data["input_text"].append(question)
    data["target_text"].append(response)

# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data["input_text"], data["target_text"], test_size=0.2
)

# Create a DatasetDict object
dataset = DatasetDict({
    "train": Dataset.from_dict({"input_text": train_texts, "target_text": train_labels}),
    "test": Dataset.from_dict({"input_text": test_texts, "target_text": test_labels}),
})

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def preprocess_function(examples):
    inputs = [ex for ex in examples["input_text"]]
    targets = [ex for ex in examples["target_text"]]
    
    model_inputs = tokenizer(inputs, max_length=256, padding="max_length", truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, padding="max_length", truncation=True)
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Set up training arguments with matching save and evaluation strategies
training_args = TrainingArguments(
    use_cpu=True,
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=3e-4,  # Slightly higher learning rate for T5
    per_device_train_batch_size=4,  # Adjust batch size based on your GPU memory
    per_device_eval_batch_size=4,  # Adjust batch size based on your GPU memory
    num_train_epochs=5,  # Increased number of epochs
    weight_decay=0.01,
    save_total_limit=3,  # Limit the total number of checkpoints
    load_best_model_at_end=True,  # Load the best model at the end
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_t5_base_mps")
tokenizer.save_pretrained("./fine_tuned_t5_base_mps")