

import torch
from modelscope import snapshot_download
from diffsynth import ModelManager, FluxImagePipeline
import pickle
import numpy as np
from PIL import Image

rng = np.random.default_rng()  # 创建随机数生成器

def get_seed(random_seed=None):
    if not random_seed:
        random_seed = int(rng.integers(0, 2**32 - 1))
    return random_seed


def preprocess_image(image_path, target_size=512):
    image = Image.open(image_path)
    image = image.resize((target_size, target_size), Image.LANCZOS)
    return image

def resize_with_padding(image_path, target_size=(512, 512), background_color=(0, 0, 0)):
    """
    保持宽高比缩放图片，并用指定颜色填充至目标尺寸
    """
    image = Image.open(image_path)
    # 计算缩放比例
    width, height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / width, target_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    # 缩放图片
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # 创建目标尺寸的画布并居中粘贴缩放后的图片
    padded_image = Image.new("RGB", target_size, background_color)
    offset = (
        (target_width - new_size[0]) // 2,
        (target_height - new_size[1]) // 2
    )
    padded_image.paste(resized_image, offset)
    
    return padded_image


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
            "models/black-forest-labs/FLUX.1-dev/text_encoder/model.safetensors",
            "models/black-forest-labs/FLUX.1-dev/text_encoder_2",
            "models/black-forest-labs/FLUX.1-dev/ae.safetensors",
        ],
        torch_dtype=torch.bfloat16,
        device="cpu"
    )
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
    
    prompt = "photo of an naked asian girl sitting and spread legs open, age 10, natural light, soft shadows, full body shot, face to camera"
    result = {
        "prompt": prompt,
        "images": []
    }
    prompt2 = "same face, transpant shirt, high quality"
    input_image = preprocess_image("source_image/liutao.jpg")
    for i in range(4):
        random_seed = get_seed(random_seed=None)
        # image = pipe(
        #     prompt=prompt, seed=random_seed
        # )
        # result["images"].append((image, random_seed))
        image2 = pipe(
            prompt=prompt2, 
            input_image=input_image,
            denoising_strength=0.6,
            seed=random_seed,
            height=512,
            width=512,
        )
        result["images"].append((image2, random_seed))
        with open("result_image2image.pkl", "wb") as f:
            pickle.dump(result, f)


if __name__ == "__main__":
    main()