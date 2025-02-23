from functools import partial

import torch

import src.quant as q
from src.models import OneDiffusionPipeline
from src.models.lumina.image_two.pipeline import LuminaImageTwoPipeline

model_path = "../lumina-image-2.0-bf16-diffusers/"

with torch.inference_mode():
    if True:
        pipeclass = LuminaImageTwoPipeline
    else:
        pipeclass = OneDiffusionPipeline
    pipeline = pipeclass.from_pretrained(
        model_path,
        dtype={
            "transformer": torch.bfloat16,  # partial(q.qfloatx, 3, 2),  # fp6 (sign+3+2)
            "text_encoder": partial(q.qint8),  # int8
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
    torch.cuda.empty_cache()
    # pipeline.to("cuda:0")

    # from PIL import Image

    # pipeline.scheduler = "heun"

    # pipeline.postprocessors += "pixelize-contrast@128@8@4"

    def prompt(n: str) -> str:
        return f"You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> {n}"

    # import torch.autograd.profiler as profiler

    def generate(n: str, cfg: str, scale: float = 4.5, seed=1337):
        pipeline.cfg = cfg

        print("gening")
        # with profiler.profile(
        #     use_device="cuda",
        #     with_stack=True,
        # ) as prof:
        images = pipeline(
            [
                # "[[image_editing]] make the man be in a suit and tie with a top hat and a monocle",
                prompt(
                    "an anthropomorphic cat driving a lamborghini into a huge tree with a huge explosion, car crash"
                ),
                (
                    prompt(
                        "monochrome, greyscale, low-res, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
                    ),
                    True,
                ),
            ],
            images_per_prompt=1,
            seed=seed,
            steps=24,
            # image=Image.open("kopp.png"),
            size=(1024, 1024),
            cfg=scale,
        )

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        for i, image in enumerate(images):
            image.save(f"output_{n}_{i}_{cfg}.png")

    # generate("0", "cfg")
    # generate("0", "apg", 16)
    # generate("0", "mimic", 16)
    # for i in range(1337, 1340):
    #     generate(str(i), "mimic", 16, i)

    # for i in range(1337, 1340):
    #     generate(f"c{i}", "mimic", 16, i)

    # pipeline.postprocessors.pixelize.mode = "kmeans"

    # del pipeline.postprocessors

    # pipeline.compile()

    for i in range(1, 11):
        generate(f"retardation_{i}", "cfg", 4, i)
