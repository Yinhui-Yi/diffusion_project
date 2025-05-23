

import torch
from modelscope import snapshot_download
from diffsynth import ModelManager, SDXLImagePipeline, download_models
import pickle
import numpy as np

rng = np.random.default_rng()  # 创建随机数生成器

def get_seed(random_seed=None):
    if not random_seed:
        random_seed = int(rng.integers(0, 2**32 - 1))
    return random_seed


def main():
    # download models
    download_models([
        "BluePencilXL_v200",
        "ControlNet_union_sdxl_promax",
        "SDXL_lora_zyd232_ChineseInkStyle_SDXL_v1_0",
        "IP-Adapter-SDXL"
    ])

    model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
    model_manager.load_models(["models/stable_diffusion_xl/bluePencilXL_v200.safetensors"])
    pipe = SDXLImagePipeline.from_model_manager(model_manager)
    torch.manual_seed(1)
    # prompt = "masterpiece, best quality, solo, long hair, wavy hair, silver hair, blue eyes, blue dress, medium breasts, dress, underwater, air bubble, floating hair, refraction, portrait,"
    prompt = "bride"
    negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw,"
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cfg_scale=6, num_inference_steps=60,
    )
    image.save("image.jpg")


if __name__ == "__main__":
    main()