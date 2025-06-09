

import torch
from modelscope import snapshot_download
from diffsynth import ModelManager, FluxImagePipeline, download_models, ControlNetConfigUnit
import pickle
import numpy as np

rng = np.random.default_rng()  # 创建随机数生成器

def get_seed(random_seed=None):
    if not random_seed:
        random_seed = int(rng.integers(0, 2**32 - 1))
    return random_seed


def create_module_pipe(lora=False):
    download_models(["Annotators:Depth"])
    # set precision bfloat16
    model_manager = ModelManager(torch_dtype=torch.bfloat16)
    # load DiT with precision float8
    model_manager.load_models(
        ["/root/models/MAILAND/majicflus_v1/majicflus_v134.safetensors"],
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
    model_manager.load_models(
        [
            "/root/models/controlnet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
        ],
        torch_dtype=torch.bfloat16,
        device="cpu"
    )
    # load controlnet
    if lora:
        model_manager.load_lora("/root/models/lora/Flux-Uncensored-V2/lora.safetensors", lora_alpha=5.0)
    # 开启量化与显存管理
    pipe = FluxImagePipeline.from_model_manager(
        model_manager, device="cuda",
        controlnet_config_units=[
            ControlNetConfigUnit(
                "canny", 
                "/root/models/controlnet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
                scale=0.9
            ),
            # ControlNetConfigUnit(
            #     "depth", 
            #     "/root/models/controlnet/InstantX/FLUX.1-dev-Controlnet-Union-alpha/diffusion_pytorch_model.safetensors",
            #     scale=0.3,    
            # ),
        ]
    )
    pipe.enable_cpu_offload()
    pipe.dit.quantize()

    return pipe


def raw_images(pipe, result, prompt=None, negative_prompt=None):
    if not prompt:
        prompt = "full body shot, high quality, a 10 years old girl smiles to camera, bride"
    if not negative_prompt:
        negative_prompt = ""
    images = []
    for i in range(2):
        num_inference_steps = 60
        random_seed = get_seed(random_seed=None)
        for i in range(1):
            image = pipe(
                prompt=prompt, 
                negative_prompt=negative_prompt,
                seed=random_seed,
                cfg_scale=7,
                num_inference_steps=num_inference_steps,
                height=1024,
                width=1024,
            )
            # result["images"].append((image, random_seed))
            images.append((image, random_seed))
            num_inference_steps += 10

    return images


def run_controlnet(pipe, result, input_images, prompt=None, negative_prompt=None):
    if not prompt:
        prompt = "full body shot, high quality, a 10 years old girl smiles to camera, naked"
    if not negative_prompt:
        negative_prompt = ""

    for input_image, random_seed in input_images:
        result["images"].append((input_image, random_seed))
        for i in range(2):
            num_inference_steps = 60
            random_seed = get_seed(random_seed=None)
            for i in range(2):
                image = pipe(
                    prompt=prompt, 
                    negative_prompt=negative_prompt,
                    seed=random_seed,
                    cfg_scale=7,
                    num_inference_steps=num_inference_steps,
                    controlnet_image=input_image,
                    input_image=input_image,
                    denoising_strength=0.5,
                )
                result["images"].append((image, random_seed))
                num_inference_steps += 20


def main():
    # create pipe
    pipe = create_module_pipe()

    result = {
        "images": []
    }
    # raw image
    input_images = raw_images(pipe, result)
    pipe = create_module_pipe(lora=True)
    run_controlnet(pipe, result, input_images)

    with open("result.pkl", "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    main()