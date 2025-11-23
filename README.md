# Volapuk

**Volapuk** (pronounced *Vo La Puke*) is a convenient Docker image that provides a ready-made development environment to experiment with Large Language Models (LLMs) and fine-tune Llama-style LLMs locally on a Windows PC equipped with an Nvidia RTX GeForce 30 series GPU.

The name Volapuk is a homage to the [world's first auxlang](https://en.wikipedia.org/wiki/Volap%C3%BCk). Partly inspired by another LLM fine-tuning environment [Kolo](https://github.com/MaxHastings/Kolo), Volapuk is composed of and named after the most popular tools being used, as of 2025, for serving and fine-tuning LLMs locally:

- <span style="color: violet">**V**</span> - **vLLM**, a high performance engine for serving LLMs and interacting with LLMs
- <span style="color: indigo">**O**</span> - **Ollama** for serving and interacting with LLMs
- <span style="color: blue">**L**</span> - **llama.cpp** and its python binding **llama-cpp-python** for serving, interacting with GGUF model format LLMs and for quantizing, converting model checkpoints to GGUF format
- <span style="color: green">**A**</span> - **Ampere** microarchitecture Nvidia GPU (GeForce RTX 30 series), such as, RTX 3050/3060/3070/3080/3090 with CUDA compute capability = 8.6
- <span style="color: gold">**P**</span> - **PyTorch**, a Deep Learning library and Machine Learning (ML) framework that vLLM and Unsloth rely upon
- <span style="color: orange">**U**</span> - **Unsloth**, a VRAM-efficient, opinionated LLM fine-tuning framework
- <span style="color: red">**K**</span> - **KoboldCpp**, a local UI front-end for serving and chatting with LLMs 

In addition, Volapuk includes:

- **Jupyter Lab** - An interactive python notebook environment
- **Synthetic Data Kit** - A library from Meta to generate synthetic Q&A pairs to be used as training dataset
- **uv** - A Rust-based python package resolver, installer and virtual environment manager
- **CUDA 12.8** - Toolkit containing CUDA libraries, runtime and **nvcc** (Nvidia C Compiler)



## Recommended System Requirements

- Operating System: Windows 11 with WSL2 running Ubuntu >= 20.04 LTS
- Containerization Software: Docker Desktop for Windows
- Graphics Card: Nvidia GeForce RTX 30* GPU with at least 12GB of VRAM and driver supporting CUDA 12.8 or later
- Memory: 16GB or more of system RAM
- Storage: 100GB or more

## Getting Started

### 1️⃣ Install Dependencies
Ensure [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) is installed. 

Ensure [Ubuntu 20.04 LTS](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/) or later has been installed as the distro in WSL2.

Ensure [Docker Desktop](https://docs.docker.com/get-docker/) is installed. Go to Docker Desktop's Settings &rarr; Resources &rarr; WSL integration &rarr; Check the option <b>Enable Integration with WSL distro</b>.

Ensure [Nvidia GeForce Game Ready Driver](https://www.nvidia.com/en-us/drivers/) is installed. 

### 2️⃣ Check CUDA Version Supported by Driver

Launch WSL2 Ubuntu terminal, run the following command and verify that the CUDA version reported is >= 12.8.

<pre style="padding-top: 0">
<code>
<b>$ nvidia-smi</b>
Fri Nov 21 15:22:16 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.02              Driver Version: 581.42         <mark>CUDA Version: 13.0</mark>     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        On  |   00000000:01:00.0  On |                  N/A |
| 36%   28C    P8              7W /  170W |     627MiB /  12288MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              23      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
</code>
</pre>

### 3️⃣ Build the Docker Image

To build the Docker image, run:

<pre style="padding-top: 0">
<code>
<b>$ ./build_volapuk.sh</b>
[+] Building 2496.7s (16/16) FINISHED                                                                    
.....
.....
</code>
</pre>

Check that the Docker image has been generated:

<pre style="padding-top: 0">
<code>
<b>$ docker image ls</b>
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
volapuk      latest    e520e5ee3300   4 hours ago   38.8GB
</code>
</pre>

### 4️⃣ Run the Docker Container

To start the container, run:

<pre style="padding-top: 0">
<code>
<b>$ ./run_volapuk.sh</b>
</code>
</pre>

Within the current work directory in WSL2 Ubuntu, where the above command is run, the subdirectory named `workspace` will be mounted as `/workspace` in the container.


## Application

### 🤖💬 A Llama chatbot trained on [Tiny Tapeout SKY 25a datasheet](https://tinytapeout.github.io/tinytapeout-sky-25a/datasheet.pdf)

### 1️⃣ Jupyter notebook approach

Launch the demo Jupyter notebook [TT_finetune_Llama3_2_1B.ipynb](workspace/TT_finetune_Llama3_2_1B.ipynb) by running the below command and opening http://127.0.0.1:8888/lab in a web browser:

<pre style="padding-top: 0">
<code>
<b>$ ./run_jupyter.sh</b>
</code>
</pre>

Step through the cells in the Jupyter notebook which will: 
- Download Llama-3.2-1B Instruct model 
- Generate synthetic data from Tiny Tapeout SKY 25a shuttle's datasheet
- Fine-tune the instruct model by training it on the synthetic data 
- Run inference using the fine-tuned model
- Save the fine-tuned model

The resulting Llama 3.2 1B LLM chatbot will be able to answer questions pertaining to Tiny Tapeout SKY 25a shuttle's datasheet.

### 2️⃣ Python script approach

Once the synthetic data has been generated fine-tune an LLM model on that data non-interactively using the below command:

<pre style="padding-top: 0">
<code>
<b>$ ./run_unsloth_lora_peft.py</b>
</code>
</pre>

## Testing LLM inference engines inside Docker container

The below tests will load Llama 3.2 1B Instruct model and run inference using respective engines.

### 1️⃣ Check using llama.cpp

<pre style="padding-top: 0">
<code>
<b>$ ./tests/llamacpp_inference.py</b>
</code>
</pre>

### 2️⃣ Check using vLLM

<pre style="padding-top: 0">
<code>
<b>$ ./tests/vllm_inference.py</b>
</code>
</pre>

### 3️⃣ Check using Ollama

Open a new terminal in WSL2 Ubuntu and run the below command which will connect to the running Docker container and start ollama server in it:

<pre style="padding-top: 0">
<code>
<b>$ workspace/tests/ollama_run_server.sh</b>
</code>
</pre>

In the original terminal where the Docker container's shell is accessible, run the below command to push an LLM model to the ollama server and chat with it:

<pre style="padding-top: 0">
<code>
<b>$ ./tests/ollama_chat.sh</b>
</code>
</pre>

Use `/exit` to exit the chat and return to the shell prompt.

### 4️⃣ Check using KoboldCpp

Run the below command and open http://localhost:5001 in a web browser to chat using the KoboldCpp UI

<pre style="padding-top: 0">
<code>
<b>$ ./tests/koboldcpp_inference.sh</b>
</code>
</pre>

## Known Issues

When saving the trained model to GGUF for the first time, Unsloth will clone `llama.cpp` repo into the current work dir, build it and utilize the resulting `llama-quantize` binary along with the model conversion python scripts to perform GGUF conversion.
If this step errors out or freezes then manually execute the below commands to build the `llama-quantize` binary:

```bash
$ MAX_JOBS=6
$ CUDA_ARCH="86"
$ cd llama.cpp
$ cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
$ LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH" cmake --build build --config Release -j ${MAX_JOBS}
$ ln -sf build/bin/llama-quantize llama-quantize
$ cd ..
```
Then rerun the LLM fine-tuning script and GGUF conversion will finish without erroring out.
