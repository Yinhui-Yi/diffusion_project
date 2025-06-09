from modelscope import snapshot_download
from diffsynth.models.downloader import download_from_huggingface, download_from_modelscope

# download
snapshot_download(
    model_id="InstantX/FLUX.1-dev-Controlnet-Union-alpha",
    allow_file_pattern="diffusion_pytorch_model.safetensors",
    cache_dir="/root/models/controlnet"
)

# download_from_huggingface("enhanceaiteam/Flux-Uncensored-V2", "lora.safetensors", "models/lora")

