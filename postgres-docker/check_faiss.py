import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
import numpy as np

# Load the embedding model and tokenizer
embedding_model_name = "distilbert-base-uncased"  # Replace with your embedding model
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to("cuda")

# Generate a sample embedding
sample_text = "This is a sample sentence to determine embedding size."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
with torch.no_grad():
    outputs = embedding_model(**inputs)
    
# Assume the last hidden state is used for embeddings
embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Determine the dimension of the embedding
embedding_dimension = embedding.shape[1]
print(f"Embedding Dimension: {embedding_dimension}")

# Create the FAISS index with the determined dimension
index = faiss.IndexFlatL2(embedding_dimension)

# Adding a sample embedding to the FAISS index
index.add(embedding)

# Example: Searching the FAISS index
D, I = index.search(embedding, k=1)  # Searching for the nearest neighbor
print(f"Distances: {D}, Indices: {I}")