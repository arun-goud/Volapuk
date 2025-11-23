#!/usr/bin/env python3

from vllm import LLM, SamplingParams

# Define the path to GGUF model file
model_path = "/workspace/models/unsloth/Llama-3.2-1B-Instruct" 
if not os.path.isfiledir(model_path):
    print(f"Mising {model_path}. Please download it by running the demo Jupyter notebook.")
    sys.exit(1)

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.1,      # Controls randomness; higher values mean more random.
    top_p=0.9,            # Nucleus sampling; considers tokens with cumulative probability up to top_p.
    max_tokens=512        # Maximum number of tokens to generate per prompt.
)

# Define your input prompts
prompts = [
    "What is Python programming language?",
    "What is the capital of USA?",
    "Name the planets in the solar system.",
]

def main():
    # Initialize the vLLM engine
    llm = LLM(model=model_path, swap_space=1, enforce_eager=False)

    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated text
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
