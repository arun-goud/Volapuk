FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG MAX_JOBS=6
ARG CUDA_ARCH="86"
ARG CUDA_VERSION="cu128"
ARG PYTHON_VERSION=3.12
ARG VLLM_VERSION=0.11.1
ARG OLLAMA_VERSION=0.13.0
ARG LLAMACPPPY_VERSION=0.3.16
#ARG PYTORCH_VERSION=2.9.0+cu128
ARG UNSLOTH_VERSION=2025.11.3
ARG KOBOLDCPP_VERSION=1.101.1
ARG SDK_VERSION=0.0.5
ARG BNB_VERSION=0.48.2

# Install essential packages
RUN apt update -y && \
    apt install -y build-essential git git-lfs wget curl libcurl4-openssl-dev cmake fontconfig && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Install uv
RUN curl -fLo uv-installer.sh https://astral.sh/uv/install.sh && \
    sh ./uv-installer.sh && rm -rf ./uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_NO_CACHE=1
ENV UV_NO_BUILD_ISOLATION=1
ENV UV_TORCH_BACKEND=${CUDA_VERSION}

# Install Python with version matching PYTHON_VERSION in virtual environment
RUN uv venv --python ${PYTHON_VERSION} --seed /opt/venv

# Use virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install vLLM
RUN uv pip install vllm==${VLLM_VERSION} --torch-backend=${UV_TORCH_BACKEND}

# Install Unsloth
RUN uv pip install "unsloth[${CUDA_VERSION}-ampere-torch290]==${UNSLOTH_VERSION}" bitsandbytes==${BNB_VERSION}

# Install Synthetic Data Kit
RUN uv pip install synthetic-data-kit==${SDK_VERSION}

# Install Jupyter Lab. Downgrade notebook from 7.5.0 to 6.4.12 to eliminate notebook.nbextensions ModuleNotFoundError.
RUN uv pip install jupyterlab jupyter_contrib_nbextensions ipywidgets && \
    uv pip install --upgrade notebook==6.4.12 && \
    uv run jupyter contrib nbextension install --user && \
    uv run jupyter nbextension enable --py widgetsnbextension

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=${OLLAMA_VERSION} sh

# Install KoboldCpp
RUN curl -fLo /usr/bin/koboldcpp "https://github.com/LostRuins/koboldcpp/releases/download/v${KOBOLDCPP_VERSION}/koboldcpp-linux-x64" && \
    chmod +x /usr/bin/koboldcpp

# Install llama.cpp and llama-cpp-python
# Set LLAMA_CPP_LIB_PATH as per https://github.com/abetlen/llama-cpp-python/issues/1070#issuecomment-2812505894
# Add /usr/local/cuda/compat to LD_LIBRARY_PATH during llama.cpp build to eliminate libcuda.so.1 not found error 
ENV PATH="/usr/src/llama-cpp-python/vendor/llama.cpp/build/bin:$PATH"
ENV LLAMA_CPP_LIB_PATH="/usr/src/llama-cpp-python/vendor/llama.cpp/build/bin"
# ENV LLAMA_CPP_LIB="/usr/src/llama-cpp-python/vendor/llama.cpp/build/bin/libllama.so"
RUN uv pip install poetry scikit-build-core && \
    git clone --recursive --branch "v${LLAMACPPPY_VERSION}" https://github.com/abetlen/llama-cpp-python.git /usr/src/llama-cpp-python && \
    cd /usr/src/llama-cpp-python/vendor/llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} && \
    LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH" \
    cmake --build build --config Release -j ${MAX_JOBS} && \
    cd /usr/src/llama-cpp-python && \
    CMAKE_ARGS="-DLLAMA_BUILD=OFF" uv pip install .


WORKDIR /workspace

# Expose necessary ports - koboldcpp (5001), vLLM (8000), llama.cpp (8080), jupyter (8888), ollama (11434)
EXPOSE 5001 8000 8080 8888 11434

CMD ["bash"]
