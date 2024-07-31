# mental-health-ai
A Mental Health chatbot that is inteded as an auxillary tool for learning more about depression. This project was sponsored by DS-Pathways and Dandelion Fly NGO.

## Summary

- We scoured the internet for datasets on mental health dataset that was in a Q&A form to facilitate training. It was cleaned and combined into one single CSV file of about 5800 rows which you can find in this repo.
- T5 models of all configurations were trained on it with varying hyperparameters. Training and inference scripts provided.
- A basic webapp was built on top of the model to allow inferencing and have a chat with the model. This is implemented using Flask.
- Inside the `memory` folder, you can find a POC for circumventing the `context-length` limitation of our T5 model. It implements a pipeline for storing conversation history as embedded sentences and allows searching through it using semantic similarity. This is facilitated by the `faiss` library for indexing the vectors.

## Usage

- I highly recommend using some form of virtual environment using either Conda or `virtualenv`.
- Activate your environment and install all necessary modules using `pip install -r requirements.txt`
- This was tested on a Linux machine with GPU support. You will need to verify that your CUDA drivers are up to date and compatible with PyTorch. Go to PyTorch's official website to see the compatability matrix.
- Running the Flask app is easy for inferencing: `python app.py`. This will run the app on port 5000.
- The models are not uploaded here due to size limitation. Please request them and place it in the `models` folder before inferencing.

