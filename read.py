from enum import Enum
import re
from pathlib import Path

from accelerate import init_empty_weights
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
import diffusers.loaders.single_file_utils as sfu
from gguf.gguf_reader import GGUFReader
import torch

from src import quant as q


conversion_map = {
    "F16": (
        "bfloat16",
        torch.bfloat16,
    ),
    "F32": (
        "float32",
        torch.float32,
    ),
    "Q2": ("intx_2", q.qintx, torch.uint2),
    "Q3": (
        "int4",
        q.qint4,
    ),
    "Q4": (
        "int4",
        q.qint4,
    ),
    "Q5": ("floatx_2x2", q.qfloatx, 2, 2),
    "Q6": ("floatx_3x2", q.qfloatx, 3, 2),
    "Q7": (
        "float8",
        q.qfloat8,
    ),
    "Q8": (
        "float8",
        q.qfloat8,
    ),
}


class ModelType(Enum):
    SD3 = (
        "sd3",
        0,
        "model.diffusion_model.joint_blocks.0.context_block.adaLN_modulation.1.bias",
        r"(?<=joint_blocks\.)(.*?)\.",
        "",
    )
    SD35 = (
        "sd35large",
        -999,
        "model.diffusion_model.joint_blocks.37.x_block.mlp.fc1.weight",
        r"(?<=joint_blocks\.)(.*?)\.",
    )
    FLUX = (
        "flux",
        1,
        [
            "double_blocks.0.img_attn.norm.key_norm.scale",
            "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
        ],
        r"(?<=double_blocks\.)(.*?)\.",
    )

    @classmethod
    def from_keys(cls, keys) -> "ModelType":
        found = {}
        for key in keys:
            for model_type in cls:
                if key in model_type.value[2]:
                    found[model_type.value[1]] = model_type
                if f"model.diffusion_model.{key}" in model_type.value[2]:
                    found[model_type.value[1]] = model_type
        try:
            return min(found.items(), key=lambda x: x[0])[1]
        except:  # noqa
            return None

    def get_layer_count(self, keys) -> int:
        m = 0
        for key in keys:
            match = re.findall(self.value[3], key)
            if len(match) == 0:
                continue
            for _match in match:
                m = max(m, int(_match))
        return m + 1


def get_conversion_func(quant: str) -> any:
    base, *_ = quant.split("_")
    base = base.replace("I", "")
    dtype = conversion_map.get(base, ("bfloat16", torch.bfloat16))
    if isinstance(dtype[1], q.qdtype):
        return dtype[1](*dtype[2:])

    return lambda x: torch.nn.Parameter(
        x.weight.to(dtype=dtype[1]), requires_grad=False
    )


class Anon:
    weight: any


def load_gguf_dict(path: Path):
    reader = GGUFReader(path)

    state_dict = {}

    keys = [x.name for x in reader.tensors]
    model_type: ModelType = ModelType.from_keys(keys)
    # layer_count = model_type.get_layer_count(keys)

    # for tensor in reader.tensors:
    #     print(tensor.name)

    # print(path)
    # print(model_type)
    # print(layer_count)
    # print("----")
    # print("")

    for tensor in reader.tensors:
        state_dict[tensor.name] = torch.tensor(
            tensor.data, device="cpu", dtype=torch.bfloat16
        )

    if model_type == ModelType.FLUX:
        with init_empty_weights():
            flux = FluxTransformer2DModel(
                **{
                    "attention_head_dim": 128,
                    "axes_dims_rope": (16, 56, 56),
                    "guidance_embeds": True,
                    "in_channels": 64,
                    "joint_attention_dim": 4096,
                    "num_attention_heads": 24,
                    "num_layers": model_type.get_layer_count(keys),
                    "num_single_layers": 38,
                    "patch_size": 1,
                    "pooled_projection_dim": 768,
                }
            )
        # flux.to("cpu")
        state_dict = sfu.convert_flux_transformer_checkpoint_to_diffusers(state_dict)
        m, u = flux.load_state_dict(state_dict, assign=True)
        print(m, u)
        conv_map = {}
        for tensor in reader.tensors:
            k, _ = tensor.name.rsplit(".", 1)
            quantization_str = tensor.tensor_type.name
            t = get_conversion_func(quantization_str)
            conv_map[k] = t
        flux = q.quantize_model(flux, conv_map)

    # for tensor in reader.tensors:
    #     t = Anon()
    #     t.weight = torch.tensor(tensor.data, device="cpu", dtype=torch.bfloat16)
    #     quantization_str = tensor.tensor_type.name
    #     t = get_conversion_func(quantization_str)(t)
    #     if isinstance(t, torch.Tensor):
    #         t_ = Anon()
    #         t_.weight = t
    #         t = t_
    #     state_dict[tensor.name] = t.weight.to("cuda")


# load_gguf_dict(Path("/mnt/c/Users/kissi/Downloads/SmolLM2-135M-Instruct-Q3_K_M.gguf"))
# load_gguf_dict(Path("/mnt/c/sd/sd3.5_medium-Q5_1.gguf"))
# load_gguf_dict(Path("/mnt/c/sd/stableDiffusion35Large_q51.gguf"))
load_gguf_dict(Path("/mnt/c/sd/flux.1-lite-8B-alpha-Q5_K_S.gguf"))
