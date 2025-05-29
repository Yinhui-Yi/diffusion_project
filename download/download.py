from modelscope import snapshot_download

# download
snapshot_download(
    model_id="LiblibAI/FLUX.1-dev-ControlNet-Union-Pro-2.0",
    allow_file_pattern="diffusion_pytorch_model.safetensors",
    cache_dir="models"
)
