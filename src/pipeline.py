from typing import Protocol, Optional, Tuple, List, Union, Dict, TypeVar, TYPE_CHECKING
from pathlib import Path as _Path

import torch
from PIL import Image

from . import quant as q
from .quant import qdtype, quantize_model, _nil
from .misc.scheduling import Sampler, SAMPLERS
from .misc.guidance import Guidance, CFGS

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


class Postprocessors:
    def __init__(self) -> None:
        super().__setattr__("postprocessors", {})

    def __iadd__(self, other) -> "Postprocessors":
        update = {}
        if isinstance(other, dict):
            update = other
        elif isinstance(other, str):
            value = other.lower()
            if "pixelize" in value:
                from .misc.postprocessors import PixelOE

                poe = PixelOE()
                if "-" in value:
                    try:
                        mode, *others = value.replace("pixelize-", "").split("@")
                        poe.mode = mode
                        if len(others) == 3:
                            poe.target_size, poe.patch_size, poe.thickness = map(
                                int, others
                            )
                        elif len(others) == 2:
                            poe.target_size, poe.patch_size = map(int, others)
                        else:
                            poe.target_size = int(others[0])
                    except:  # noqa
                        pass

                update = {"pixelize": poe}
        self.postprocessors.update(update)
        return self

    def __isub__(self, other) -> "Postprocessors":
        if isinstance(other, str):
            self.postprocessors.pop(other, None)
        return self

    def setup(self, pipeline: "Pipelinelike") -> "Postprocessors":
        for _, postprocessor in self.postprocessors.items():
            postprocessor.setup(pipeline)
        return self

    def __call__(self, images: Images, width: int, height: int) -> Images:
        for _, postprocessor in self.postprocessors.items():
            images = postprocessor(images, width, height)
        return images

    def __del__(self) -> None:
        self.postprocessors.clear()

    def __getattr__(self, __name: str) -> "Postprocessor":
        if __name == "postprocessors":
            return super().__getattribute__("postprocessors")
        return super().__getattribute__("postprocessors").get(__name, None)


class Postprocessor(Protocol):
    def setup(self, pipeline: "Pipelinelike") -> "Postprocessor": ...
    def __call__(self, images: Images, width: int, height: int) -> Images: ...


class Pipelinelike:
    models: Dict[str, torch.nn.Module]

    device: torch.device
    dtype: torch.dtype
    offload: bool

    sampler: Sampler
    cfg: Guidance
    postprocessors: Postprocessors

    def __init__(self) -> None:
        from .misc.guidance import CFG

        super().__setattr__("models", {})
        super().__setattr__("sampler", None)
        super().__setattr__("device", torch.device("cpu"))
        super().__setattr__("dtype", torch.bfloat16)
        super().__setattr__("offload", False)
        super().__setattr__("cfg", CFG())
        super().__setattr__("postprocessors", Postprocessors())

    def __setattr__(self, __name: str, __value) -> None:
        skip_models = [
            "models",
            "device",
            "dtype",
            "postprocessors",
        ]
        if __name == "cfg":
            if isinstance(__value, str):
                value = __value.lower()
                if value == "disable":
                    self.cfg.disable = True
                elif value == "enable":
                    self.cfg.disable = False
                else:
                    cl = CFGS.get(value, None)
                    if cl is None:
                        raise ValueError(f"Invalid value for cfg: {__value}")
                    super().__setattr__("cfg", cl())
            else:
                super().__setattr__("cfg", __value)
            return
        elif __name == "sampler":
            if isinstance(__value, str):
                value = __value.lower()
                super().__setattr__("sampler", SAMPLERS.get(value, SAMPLERS["euler"])())
            else:
                super().__setattr__("sampler", __value)
            return
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
    @torch.no_grad()
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
    @torch.no_grad()
    def create_quantized_model_from_safetensors(
        cls,
        meta_model: T,
        safetensors_file: Path,
        device: torch.device = "cpu",
        quantization_device: torch.device = "cuda",
        dtype: qdtype = None,
        replace_map: Optional[Dict[str, str]] = None,
    ) -> T:
        from safetensors import safe_open

        if replace_map is None:
            replace_map = {}

        dtype, torch_dtype = dtype()
        meta_model.to(dtype=torch_dtype)

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
                        raise ValueError(f"failed to load {tensor_name}")

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

    def compile(
        self,
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
        mode: str = "max-autotune",
        options: dict = None,
        device: torch.device = "cuda:0",
    ) -> None:
        if self.offload:
            print("Offload and compile not supported, disabling offload")
            self.offload = False
            self.device = device

        self.to(device)

        for name, model in self.models.items():
            from diffusers import AutoencoderKL

            if isinstance(model, (AutoencoderKL)):
                continue

            setattr(
                self,
                name,
                torch.compile(
                    model,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                    backend=backend,
                    mode=mode,
                    options=options,
                ),
            )

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
