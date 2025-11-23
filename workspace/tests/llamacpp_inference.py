#!/usr/bin/env python3

import os
import sys
from llama_cpp import Llama

# Define the path to GGUF model file
model_path = "/workspace/models/gguf/Llama-3.2-1B-Instruct-F16.gguf" 
if not os.path.isfile(model_path):
    print(f"Mising {model_path}. Please download it by running the demo Jupyter notebook.")
    sys.exit(1)

# Initialize the Llama model
llm = Llama(model_path=model_path, 
    n_gpu_layers=20, 
    n_ctx=2048, 
    verbose=True
)

# Define your prompt
prompt = "What is Python programming language?"

# Generate a completion
output = llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user","content": prompt}
    ]
)

# Print the generated text
print(f"\n\n{prompt}\n")
print(output['choices'][0]['message']['content'])
