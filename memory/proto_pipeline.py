from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# Input Module that caches conversation history
class InputModule:
    def __init__(self):
        self.conversation_history = []

    def add_to_history(self, user_input):
        self.conversation_history.append(user_input)

    def get_context(self):
        return " ".join(self.conversation_history)


# Memory Module: Use a sentence transformer to embed sentences as a 384D vec space
# Update memory after each interaction to keep relevant information grouped spatially
class MemoryModule:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.embeddings = []
        self.texts = []

    def update_memory(self, text):
        embedding = self.model.encode(text)
        self.embeddings.append(embedding)
        self.texts.append(text)
        self._update_index()

    def _update_index(self):
        if self.embeddings:
            d = len(self.embeddings[0])
            self.index = faiss.IndexFlatL2(d)
            self.index.add(np.array(self.embeddings))

    def retrieve_relevant_info(self, query, top_k=3):
        if not self.embeddings:
            return []
        query_embedding = self.model.encode(query)
        D, I = self.index.search(query_embedding.reshape(1, -1), top_k)
        relevant_texts = [self.texts[i] for i in I[0] if i < len(self.texts)]
        return relevant_texts

# Response Generation Module
# Excellent at deciphering context and answering questions based off it
class ResponseGenerationModule:
    def __init__(self):
        self.qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')

    def generate_response(self, query, context):
        if not context:
            return "I don't have enough information to answer that."
        result = self.qa_pipeline(question=query, context=context)
        return result['answer']

# Integration time
input_module = InputModule()
memory_module = MemoryModule()
response_generation_module = ResponseGenerationModule()

# Sequence of inputs to test out memory
user_inputs = [
    "I am a 25-year-old male from the US. I enjoy volleyball and Japanese cuisine.",
    "I recently travelled to New York",
    "Michael met Holly at Dunder Miffline where she worked in the HR department.",
    "I have been feeling a bit down lately, can you help?",
    "Can you remind me what my favorite cuisine is?",
    "What did I say about my hobbies?",
    "Where did I travel recently?",
    "Who worked in human resources?",
    "Holly replaced Toby after he left for Costa Rica",
    "Toby is boring."
    "Although Toby seems like a nice guy, Michael really hates him.",
    "Why does Michael hate Toby?"
]

# Simulating the conversation
for user_input in user_inputs:
    input_module.add_to_history(user_input)
    memory_module.update_memory(user_input)
    query = user_input
    context = input_module.get_context()
    relevant_info = memory_module.retrieve_relevant_info(query)
    context_with_relevant_info = " ".join(relevant_info)
    response = response_generation_module.generate_response(query, context_with_relevant_info)
    print(f"User Query: {query}")
    print(f'Fetched context: {context_with_relevant_info}')
    print(f"Response: {response}")
    print("-" * 50)

# while True:
#     query = input("You: ")
#     input_module.add_to_history(query)
#     memory_module.update_memory(query)
#     context = input_module.get_context()
#     relevant_info = memory_module.retrieve_relevant_info(query)
#     context_with_relevant_info = " ".join(relevant_info)
#     response = response_generation_module.generate_response(query, context_with_relevant_info)
#     print(f"Response: {response}")
#     print("-" * 50)