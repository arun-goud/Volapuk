#!/usr/bin/env bash

# Define the path to GGUF model file
model_path="/workspace/models/gguf/Llama-3.2-1B-Instruct-F16.gguf" 

if [ -f "$model_path" ]; then
    echo "Missing ${model_path}. Please download it by running the demo Jupyter notebook."
    exit
fi

# Create modelfile
echo "FROM $model_path" > /workspace/tests/Modelfile

# Run model
ollama create test_ollama_model -f /workspace/tests/Modelfile
ollama run test_ollama_model
