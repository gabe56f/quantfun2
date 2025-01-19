import torch
from flute.tune import TuneTask, tune_tasks_legacy


def create_tasks(shapes, bits: list, sms: list):
    tasks = []
    for n, m in shapes:
        for batch in range(1, 3):
            for group in [64, 128]:
                for bit in bits:
                    for sm, dtype in sms:
                        tasks.append(
                            TuneTask(
                                M=batch,
                                N=n,
                                K=m,
                                num_bits=bit,
                                dtype=dtype,
                                group_size=group,
                                num_sms=sm,
                                device=torch.device("cuda:0"),
                            )
                        )
    return tasks


tune_tasks_legacy(
    create_tasks(
        shapes=[
            (768, 768),
            (2048, 2048),
            (3840, 3840),
            (768, 3072),
            (3072, 768),
            (2048, 5120),
            (5120, 2048),
            (1024, 3840),  # onediffusion final layer adaln modulation
            (3840, 64),  # onediffusion final layer
            (64, 3840),
            (3072, 64),  # flux.1 dev
            (64, 3072),
            (4096, 3072),
            (3840, 960),
            (2048, 960),
            (2048, 1024),
            # Llama feed-forward - onediffusion
            (3840, 10240),
            (10240, 3840),
            (1024, 15360),  # adaln modulation
            (256, 1024),
            (1024, 1024),
            (3072, 3072),
            (3072, 12288),
            (12288, 3072),
            (15360, 3072),
            (3072, 9216),
        ],
        bits=[2, 3, 4],
        sms=[(76, torch.float16), (28, torch.float16)],
    )
)
