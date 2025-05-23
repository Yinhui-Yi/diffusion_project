

import torch
from modelscope import snapshot_download
from diffsynth import ModelManager, FluxImagePipeline
import pickle
import numpy as np

rng = np.random.default_rng()  # 创建随机数生成器

def get_seed(random_seed=None):
    if not random_seed:
        random_seed = int(rng.integers(0, 2**32 - 1))
    return random_seed


def main():
    # download
    snapshot_download(
        model_id="MAILAND/majicflus_v1",
        allow_file_pattern="majicflus_v134.safetensors",
        cache_dir="models"
    )
    snapshot_download(
        model_id="black-forest-labs/FLUX.1-dev",
        allow_file_pattern=["ae.safetensors", "text_encoder/model.safetensors", "text_encoder_2/*"],
        cache_dir="models"
    )

    # set precision bfloat16
    model_manager = ModelManager(torch_dtype=torch.bfloat16)
    # load DiT with precision float8
    model_manager.load_models(
        ["models/MAILAND/majicflus_v1/majicflus_v134.safetensors"],
        torch_dtype=torch.float8_e4m3fn,
        device="cpu"
    )
    # load Text Encoder and VAE with precision bfloat16 
    model_manager.load_models(
        [
            "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
            "models/FLUX/FLUX.1-dev/text_encoder_2",
            "models/FLUX/FLUX.1-dev/ae.safetensors",
        ],
        torch_dtype=torch.bfloat16,
        device="cpu"
    )
    model_manager.load_lora("models/lora/ClothingAdjuster3.safetensors", lora_alpha=0.0)
    # 开启量化与显存管理
    pipe = FluxImagePipeline.from_model_manager(model_manager, device="cuda")
    pipe.enable_cpu_offload()
    pipe.dit.quantize()

    # 生图！
    # prompt = "In a cozy, softly lit bedroom, " \
    # "a naked wemon wear nothing stands in front of a full-length mirror, fucked by huzband's leader," \
    # " holding their phone at arm's length to take a selfie. The room is adorned with warm, pastel tones and soft, " \
    # "plush furnishings. Their legs are spread slightly, emphasizing the smooth, silky texture of the pantyhose. " \
    # "The screen of the phone glows, capturing the playful and confident expression on their face as they pose, " \
    # "ready to snap the perfect picture."

    # prompt = "naked women wearing black high heel sits and spread her legs, full body, supermarket, big smile,enjoy"
    # prompt = "Hands on chest, ragged dress, a little girl flips up her white transparent dress, age 10, full body, black lace panty, mini bra"
    
    # prompt = "photo of an asian girl standing in a modern apartment, wearing a sheer white lace dress with floral patterns, spread legs, revealing red lace panties underneath, natural light, soft shadows, full body shot, medium shot, realistic photography, slightly wide shot"
    
    prompt = "bride, full body"
    result = {
        "prompt": prompt,
        "images": []
    }
    for i in range(4):
        random_seed = get_seed(random_seed=None)
        image = pipe(
            prompt=prompt, seed=random_seed
        )
        result["images"].append((image, random_seed))
        with open("result.pkl", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    main()