#!/usr/bin/env python3

import os
import sys
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch
import pandas as pd
import glob

local_model = "/workspace/models/unsloth/Llama-3.2-1B-Instruct"
save_ft_model = "/workspace/ft_models"
fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    # Qwen3 new models
    "unsloth/Qwen3-4B-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    # Other very popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

def show_header(status):
    separator = '-'*len(status)
    print(f"\n{separator}\n{status}\n{separator}")


show_header("Reading dataset")
filenames = glob.glob("data/output/datasheet_*.txt")

final_filenames = [
    f"data/final/datasheet_{i}_qa_pairs_ft.json"
    for i in range(len(filenames))
]
conversations = pd.concat([
    pd.read_json(name) for name in final_filenames
]).reset_index(drop = True)

dataset = Dataset.from_pandas(conversations)
print(f"dataset[0] = \n{dataset[0]}")

show_header("Loading model weights and tokenizer")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = local_model,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = True, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
    gpu_memory_utilization = 0.7, 
    fast_inference = True,
    #enforce_eager = True,
)
print("Done.")

show_header("Setting up LoRA for PEFT")
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # Apply LoRA to attention, MLP layers
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)
print("Done.")

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, 
                tokenize = False, 
                add_generation_prompt = False
            ) for convo in convos]
    return { "text" : texts }

show_header("Applying chat template to dataset")
# Get our previous dataset and format it:
dataset = dataset.map(formatting_prompts_func, batched = True)
print(f"dataset[0] = \n{dataset[0]}")

show_header("Setting up SFT Trainer")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = 1024,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        gradient_checkpointing = False,
        warmup_steps = 20,
        max_steps = 250,
        max_grad_norm = 0.3,
        learning_rate = 2e-4,
        logging_steps = 1,
        bf16 = True,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear", # constant/linear/cosine
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

# Show current memory stats
show_header("Memory stats")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

show_header("Training")
trainer_stats = trainer.train()

show_header("Inference")
messages = [
    {"role": "user", "content": "What is Tiny Tapeout SKY 25a?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
print(f"Prompt: {messages[0]['content']}\nResponse: ")
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer,
                   max_new_tokens = 256, temperature = 0.1,
                   pad_token_id = tokenizer.pad_token_id)

messages = [
    {"role": "user", "content": "Roughly how many chip designs were taped on Tiny Tapeout SKY 25a?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
print(f"\nPrompt: {messages[0]['content']}\nResponse: ")
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer,
                   max_new_tokens = 256, temperature = 0.1,
                   pad_token_id = tokenizer.pad_token_id)

show_header("Saving model")
print(f"Saving LoRA to {save_ft_model}/lora_model")
model.save_pretrained(f"{save_ft_model}/lora_model")  # Local saving
tokenizer.save_pretrained(f"{save_ft_model}/lora_model")
print("Done.\n")

print(f"Saving merged model to {save_ft_model}/merged_model")
model.save_pretrained_merged(f"{save_ft_model}/merged_model", tokenizer, save_method = "merged_16bit", maximum_memory_usage = 0.5)
print("Done.\n")

print(f"Saving quantized GGUF to {save_ft_model}/gguf_model")
quant_method = "q4_k_m"
gguf_filename = f"{os.path.basename(local_model)}.{quant_method.upper()}.gguf"
model.save_pretrained_gguf(f"{save_ft_model}/gguf_model", tokenizer, quantization_method = quant_method)
if os.path.exists(gguf_filename):
    os.rename(gguf_filename, f"{save_ft_model}/gguf_model/{gguf_filename}")
print("Done.\n")
