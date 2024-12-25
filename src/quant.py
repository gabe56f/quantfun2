from dataclasses import dataclass

import torch
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quant_api import (
    float8_weight_only,
    int4_weight_only,
    int8_weight_only,
    uintx_weight_only,
    fpx_weight_only,
)

aten = torch.ops.aten


def _get_to_kwargs(self: torch.nn.Module, *args, **kwargs) -> dict:
    device, dtype, _, memory_format = torch._C._nn._parse_to(*args, **kwargs)
    device = self.device if device is None else device
    dtype = self.dtype if dtype is None else dtype
    # memory_format = (
    #     self.memory_format if memory_format is None else torch.preserve_format
    # )

    kwargs = {
        "device": device,
        "dtype": dtype,
        "memory_format": torch.preserve_format,  # memory_format,
    }
    return kwargs


@dataclass
class qdtype:
    display: str
    apply_function: any

    def __call__(self, *args, **kwargs) -> any:
        return self.apply_function(*args, **kwargs)


def move_aqt(self: AffineQuantizedTensor, *args, **kwargs) -> AffineQuantizedTensor:
    kwargs = _get_to_kwargs(self, *args, **kwargs)
    device = kwargs["device"]

    tensor_impl = self.tensor_impl
    tensors = ["int_data", "zero_point", "packed_weight", "scale_and_zero", "scale"]
    for tensor in tensors:
        if hasattr(tensor_impl, tensor):
            if (x := getattr(tensor_impl, tensor)) is not None:
                x.to(device)

    return self.__class__(
        tensor_impl,
        self.block_size,
        self.shape,
        self.quant_min,
        self.quant_max,
        self.zero_point_domain,
    )


AffineQuantizedTensor.to = move_aqt


@AffineQuantizedTensor.implements(aten._has_compatible_shallow_copy_type.default)
def _(f, types, *args, **kwargs):
    return False


qintx = qdtype("intx", uintx_weight_only)
qint4 = qdtype("int4", int4_weight_only)
qint8 = qdtype("int8", int8_weight_only)
qfloatx = qdtype("floatx", fpx_weight_only)
qfloat8 = qdtype("float8", float8_weight_only)


def _nil(module: torch.nn.Module, fqn: str):
    return isinstance(module, torch.nn.Linear)


def _replace_with_custom_fn_if_matches_filter(
    model: torch.Tensor,
    replacement_map,
    filter_fn,
    cur_fqn: str = "",
    device=None,
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        if device is not None:
            model.to(device=device)  # move to device before quantization
        if isinstance(replacement_map, dict):
            model = replacement_map[cur_fqn](model)
        else:
            model = replacement_map(model)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child, replacement_map, filter_fn, f"{cur_fqn}{name}.", device
            )
            if new_child is not child:
                setattr(model, name, new_child)
        if device is not None:
            model.to(device=device)  # move parent module to device
        return model


def quantize_model(
    model,
    # initialize,
    dtype_map,
    device: torch.device = "cuda",
    skip=_nil,
) -> torch.nn.Module:
    # model = initialize()

    _replace_with_custom_fn_if_matches_filter(model, dtype_map, skip, device=device)
    return model
