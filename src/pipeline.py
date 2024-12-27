from typing import Protocol, Optional, Tuple, List, Union, Dict, TypeVar, TYPE_CHECKING
import inspect
from pathlib import Path as _Path

import torch
from PIL import Image

from . import quant as q
from .quant import qdtype, quantize_model, _nil

if TYPE_CHECKING:
    from gguf import GGUFReader


Prompt = Union[str, Tuple[str, bool]]
Prompts = List[Prompt]
Images = List[Image.Image]
Pseudorandom = Union[torch.Generator, int]
Datatype = Union[torch.dtype, qdtype]
Path = Union[str, _Path]
T = TypeVar("T")

_CONV_MAP = {
    "F16": torch.bfloat16,
    "F32": torch.float32,
    "Q2": (q.qint4,),
    "Q3": (q.qint4,),
    "Q4": (q.qint4,),
    "Q5": (q.qfloatx, 2, 2),
    "Q6": (q.qfloatx, 3, 2),
    "Q7": (q.qfloat8,),
    "Q8": (q.qfloat8,),
}


class ImageSettings:
    crop: bool

    multiview: bool
    azimuths: List[int]
    elevations: List[int]
    distances: List[float]
    c2ws: List[torch.Tensor]
    intrinsics: torch.Tensor
    focal_length: float

    def __init__(
        self,
        crop: bool = True,
        multiview: bool = False,
        azimuths: List[int] = [0, 30, 60, 90],
        elevations: List[int] = [0, 0, 0, 0],
        distances: float = 1.7,
        c2ws: List[torch.Tensor] = None,
        intrinsics: torch.Tensor = None,
        focal_length: float = 1.3887,
    ):
        self.crop = crop
        self.multiview = multiview
        self.azimuths = azimuths
        self.elevations = elevations
        if not isinstance(distances, list) and not isinstance(distances, tuple):
            self.distances = [distances] * len(azimuths)
        else:
            self.distances = distances
        self.c2ws = c2ws
        self.intrinsics = intrinsics
        self.focal_length = focal_length


DEFAULT_SETTINGS = ImageSettings()


class Schedulerlike(Protocol):
    timesteps: torch.Tensor
    init_noise_sigma: torch.Tensor

    def set_timesteps(self, num_inference_steps: int, device: torch.device): ...

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
    ) -> tuple: ...


class Pipelinelike:
    def __init__(self) -> None:
        super().__setattr__("models", {})
        super().__setattr__("device", torch.device("cpu"))
        super().__setattr__("dtype", torch.bfloat16)
        super().__setattr__("offload", False)

    def __setattr__(self, __name: str, __value) -> None:
        skip_models = ["models", "device", "dtype", "offload"]
        if __name not in skip_models and hasattr(__value, "to"):
            self.models[__name] = __value
        super().__setattr__(__name, __value)

    @classmethod
    def from_pretrained(
        cls,
        file_or_folder: Path,
        device: torch.device = "cpu",
        dtype: Union[Datatype, Dict[str, Datatype]] = torch.bfloat16,
    ) -> "Pipelinelike": ...

    def prompt(
        self,
        prompts: Prompts,
        images_per_prompt: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, int]: ...

    def prepare_latents(
        self,
        batch_size: int,
        latent_channels: int = 4,
        height: int = 1024,
        width: int = 1024,
        generator: torch.Generator = None,
        latents: torch.Tensor = None,
        image: torch.Tensor = None,
    ) -> torch.Tensor: ...

    @classmethod
    def create_quantized_model_from_gguf(
        cls,
        meta_model: T,
        gguf_file: Union[Path, "GGUFReader"],
        device: torch.device = "cpu",
        quantization_device: torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        dtype: qdtype = None,
        override_dtype: bool = True,
        replace_map: Optional[Dict[str, str]] = None,
    ) -> T:
        from gguf import GGUFReader

        if not isinstance(gguf_file, GGUFReader):
            gguf_file = GGUFReader(gguf_file)

        if replace_map is None:
            replace_map = {}

        meta_model.to(dtype=torch_dtype)
        dtype = dtype()

        def repl(
            model: torch.nn.Linear,
            cur_fqn: str = "",
        ):
            _dtype = dtype
            model.to_empty(device=device, recurse=False)

            for k, param in model.named_parameters():
                tensor_name = f"{cur_fqn}{k}"
                # print(f"loading {tensor_name}")
                for old, new in replace_map.items():
                    tensor_name = tensor_name.replace(old, new)
                try:
                    # TODO: this is horribly bad, fix this
                    tensor = [x for x in gguf_file.tensors if x.name == tensor_name][0]

                    if override_dtype:
                        base, *_ = tensor.tensor_type.name.split("_")
                        base = base.replace("I", "")
                        _dtype = _CONV_MAP.get(base, torch.bfloat16)

                    data_tensor = torch.tensor(
                        tensor.data, device=device, dtype=torch_dtype
                    )

                    param.module_load(data_tensor)
                except:  # noqa
                    print(f"failed to load {tensor_name}")
                    pass

            if _nil(model, cur_fqn[:-1]):
                if isinstance(_dtype, torch.dtype):
                    model = model.to(dtype=_dtype, device=device)
                else:
                    model = _dtype(
                        model, device=device, quant_device=quantization_device
                    )
                return model
            else:
                for name, child in model.named_children():
                    new_child = repl(child, f"{cur_fqn}{name}.")
                    if new_child is not child:
                        setattr(model, name, new_child)
                return model

        repl(meta_model, "")
        return meta_model

    @classmethod
    def create_quantized_model_from_safetensors(
        cls,
        meta_model: T,
        safetensors_file: Path,
        device: torch.device = "cpu",
        quantization_device: torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        dtype: qdtype = None,
        replace_map: Optional[Dict[str, str]] = None,
    ) -> T:
        from safetensors import safe_open

        if replace_map is None:
            replace_map = {}

        meta_model.to(dtype=torch_dtype)
        dtype = dtype()

        with safe_open(safetensors_file, framework="pt") as f:

            def repl(
                model: torch.nn.Linear,
                cur_fqn: str = "",
            ):
                model.to_empty(device=device, recurse=False)
                for k, param in model.named_parameters():
                    tensor_name = f"{cur_fqn}{k}"
                    # print(f"loading {tensor_name}")
                    for old, new in replace_map.items():
                        tensor_name = tensor_name.replace(old, new)
                    try:
                        tensor = f.get_tensor(tensor_name)

                        param.module_load(tensor.to(dtype=torch_dtype))
                    except:  # noqa
                        print(f"failed to load {tensor_name}")
                        pass

                if _nil(model, cur_fqn[:-1]):
                    model = dtype(
                        model, device=device, quant_device=quantization_device
                    )
                    return model
                else:
                    for name, child in model.named_children():
                        new_child = repl(child, f"{cur_fqn}{name}.")
                        if new_child is not child:
                            setattr(model, name, new_child)
                    return model

            repl(meta_model, "")
        return meta_model

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[Dict[str, Datatype], Datatype]] = None,
    ) -> None:
        device = device or self.device
        skip_dtype = dtype is None
        dtype = dtype or self.dtype

        self.device = device
        if skip_dtype:
            for model in self.models.values():
                model.to(device=device)
            return

        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        elif isinstance(dtype, dict):
            # get the smallest dtype provided
            wm = {None: 999, torch.float32: 3, torch.float16: 2, torch.bfloat16: 1}
            s = None

            for v in dtype.values():
                if isinstance(v, torch.dtype):
                    if wm[s] > wm[v]:
                        s = v

            # if only qdtype is provided, use bfloat16
            if s is not None:
                self.dtype = s
            else:
                self.dtype = torch.bfloat16
        else:
            # qdtype == bfloat16
            # TODO: add fp16 override for older devices that don't support bf16
            self.dtype = torch.bfloat16

        if self.offload:
            device = "cpu"

        ddict = {}
        if isinstance(dtype, dict):
            ddict = dtype
        else:
            for k in self.models.keys():
                ddict[k] = dtype

        for name, model in self.models.items():
            if isinstance(dt := ddict[name], torch.dtype):
                model.to(device=device, dtype=dt)
            else:
                print(f"quantizing {name}")
                # TODO: override
                model.to(device=device, dtype=torch.bfloat16)
                quantize_model(model, dt(), device=device)
                print(f"done {model}")

    def decode_image(
        self,
        latents: torch.Tensor,
        height: int = 1024,
        width: int = 1024,
    ) -> List[Image.Image]: ...

    def encode_image(
        self,
        image: Images,
        image_settings: "ImageSettings",
        height: int = 1024,
        width: int = 1024,
    ) -> Optional[torch.Tensor]: ...

    def __call__(
        self,
        prompts: Prompts,
        image: Images = None,
        image_settings: "ImageSettings" = DEFAULT_SETTINGS,
        size: Tuple[int, int] = None,
        images_per_prompt: int = 1,
        steps: int = 24,
        cfg: float = 3.5,
        seed: Pseudorandom = 1337,
        eta: float = 0.0,
        denoise_mask: List[int] = [1, 0],
        noise_scale: float = 1.0,
        latents: torch.Tensor = None,
    ) -> Images: ...

    def prepare_extra_step_kwargs(self, **kwargs) -> Dict[str, any]:
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        extra_kwargs = {}
        for k, v in kwargs.items():
            if k in set(inspect.signature(self.scheduler.step).parameters.keys()):
                extra_kwargs[k] = v
        return extra_kwargs


def requires(load: str):
    def decorator(func):
        def inner(self, *args, **kwargs):
            if isinstance(self, Pipelinelike):
                if self.offload:
                    for name, model in self.models.items():
                        if name in load:
                            model.to(self.device)
                        else:
                            model.to("cpu")
                    torch.cuda.empty_cache()
            return func(self, *args, **kwargs)

        return inner

    return decorator


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b

    return mu


def retrieve_timesteps(
    scheduler: Schedulerlike,
    num_inference_steps: int,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    scheduler.set_timesteps(num_inference_steps, device, **kwargs)

    return scheduler.timesteps, num_inference_steps
