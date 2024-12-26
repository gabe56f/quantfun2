from functools import partial

import torch

import src.quant as q
from src.models import OneDiffusionPipeline

model_path = "./onediffusion/"

with torch.inference_mode():
    pipeline = OneDiffusionPipeline.from_pretrained(
        "./onediffusion/",
        dtype={
            "transformer": partial(q.qfloatx, 3, 2),  # fp6 (sign+3+2)
            "text_encoder": torch.bfloat16,
        },
        device="cpu",
    )

    # print(
    #     NextDiT.from_pretrained(
    #         model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    #     ).x_embedder.weight.data.sum()
    # )

    pipeline.offload = True
    pipeline.device = torch.device("cuda:0")
    # pipeline.to("cpu")
    torch.cuda.empty_cache()

    # from PIL import Image

    print("gening")
    images = pipeline(
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

    for i, image in enumerate(images):
        image.save(f"output_{i}.png")
