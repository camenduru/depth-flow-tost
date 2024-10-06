FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg libegl1-mesa libegl1

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    safetensors einops transformers scipy torchsde aiohttp kornia opencv-python matplotlib scikit-image imageio imageio-ffmpeg ffmpeg-python av fvcore ultralytics \
    omegaconf ftfy accelerate bitsandbytes sentencepiece protobuf diffusers pykalman segment_anything timm insightface addict onnxruntime onnxruntime-gpu yapf numpy==1.26.4 depthflow==0.6.0 && \
    git clone https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/akatz-ai/ComfyUI-Depthflow-Nodes /content/ComfyUI/custom_nodes/ComfyUI-Depthflow-Nodes && \
    git clone https://github.com/spacepxl/ComfyUI-Depth-Pro /content/ComfyUI/custom_nodes/ComfyUI-Depth-Pro && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/depthflow/resolve/main/depth_pro.fp16.safetensors -d /content/ComfyUI/models/depth/ml-depth-pro -o depth_pro.fp16.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py