#!/usr/bin/env bash

# Define the path to GGUF model file
model_path="/workspace/models/gguf/Llama-3.2-1B-Instruct-F16.gguf" 

if [ -f "$model_path" ]; then
    echo "Missing ${model_path}. Please download it by running the demo Jupyter notebook."
    exit
fi

# Run koboldcpp using GPU
koboldcpp --usecuda --model $model_path --gpulayers 20
