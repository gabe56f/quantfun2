from functools import partial

import torch

import src.quant as q
from src.models import OneDiffusionPipeline

model_path = "./onediffusion/"

with torch.inference_mode():
    pipeline = OneDiffusionPipeline.from_pretrained(
        "./onediffusion/",
        dtype={
            "transformer": partial(q.qflute4, 64),  # fp6 (sign+3+2)
            "text_encoder": torch.bfloat16,
        },
        device="cpu",
    )

    # print(
    #     NextDiT.from_pretrained(
    #         model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    #     ).x_embedder.weight.data.sum()
    # )

    # pipeline.offload = True
    # pipeline.device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    # pipeline.to("cuda:0")
    # pipeline.compile()

    # from PIL import Image

    # pipeline.scheduler = "heun"

    pipeline.postprocessors += "pixelize-contrast@128@8@4"

    def generate(n: str, cfg: str, scale: float = 4.5, seed=1337):
        pipeline.cfg = cfg

        print("gening")
        images = pipeline(
            [
                # "[[image_editing]] make the man be in a suit and tie with a top hat and a monocle",
                "[[text2img]] a cat looking at the camera from afar with a spectacular sixteen foot long tophat looking extremely perplexed",
                (
                    "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                    True,
                ),
            ],
            images_per_prompt=1,
            seed=seed,
            steps=32,
            # image=Image.open("kopp.png"),
            size=(1024, 1024),
            cfg=scale,
        )

        for i, image in enumerate(images):
            image.save(f"output_{n}_{i}_{cfg}.png")

    # generate("0", "cfg")
    # generate("0", "apg", 16)
    # generate("0", "mimic", 16)
    # for i in range(1337, 1340):
    #     generate(str(i), "mimic", 16, i)

    # for i in range(1337, 1340):
    #     generate(f"c{i}", "mimic", 16, i)

    pipeline.postprocessors.pixelize.mode = "kmeans"

    del pipeline.postprocessors

    for i in range(1337, 1340):
        generate(f"flute4_{i}", "mimic", 16, i)
