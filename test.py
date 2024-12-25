from functools import partial

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer

import src.quant as q
from src.pipeline import OneDiffusionPipeline
from src.models import NextDiT

model_path = "./onediffusion/"

with torch.inference_mode():
    transformer = NextDiT.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    print("transformer loaded")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    text_encoder = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    print("text encoder loaded")
    tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )

    pipeline = OneDiffusionPipeline(
        transformer, vae, text_encoder, tokenizer, scheduler
    )
    pipeline.to(
        "cuda",
        # torch.bfloat16,
        {
            "transformer": partial(q.qfloatx, 3, 2),
            "text_encoder": partial(q.qint8),
        },
    )
    pipeline.offload = False
    # pipeline.to("cpu")
    torch.cuda.empty_cache()

    from PIL import Image

    print("gening")
    pipeline.__call__(
        [
            "[[text2img]] a cat looking at the camera from afar with a spectacular sixteen foot long tophat looking extremely perplexed",
            (
                "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                True,
            ),
        ],
        images_per_prompt=1,
        seed=1337,
        steps=32,
        # image=Image.open("frieren2.png"),
        size=(1024, 1024),
        cfg=4.5,
    )
