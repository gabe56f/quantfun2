from dataclasses import dataclass
import types

import torch
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)
from torchao.quantization import quant_api as qa

aten = torch.ops.aten


@dataclass
class qdtype:
    display: str
    apply_function: any

    def __call__(self, *args, **kwargs) -> any:
        return self.apply_function(*args, **kwargs)


def patched_inserter(constructor, *, allow_requires_grad=False, **kwargs):
    def insert_subclass(lin, device=None, quant_device=None):
        requires_grad = allow_requires_grad and lin.weight.requires_grad
        if device is None:
            device = lin.weight.device
        if quant_device is None:
            quant_device = lin.weight.device
        tensor = constructor(lin.weight.to(device=quant_device), **kwargs).to(device)
        lin.weight = torch.nn.Parameter(
            tensor,
            requires_grad=requires_grad,
        )
        lin.extra_repr = types.MethodType(qa._linear_extra_repr, lin)
        return lin

    return insert_subclass


# TODO: produces unstable tensors -> black images
# look into this
def cublas_linear_only():
    from cublas_ops import CublasLinear

    def insert_cublas(lin: torch.nn.Linear, **_):
        linear = CublasLinear(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            dtype=lin.weight.dtype,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()
        return linear

    return insert_cublas


def bnb_int8_weight_only():
    from bitsandbytes.nn import Linear8bitLt

    def insert_bnb(lin: torch.nn.Linear, device=None, quant_device=None):
        lin.to(torch.float16)
        if device is None:
            device = lin.weight.device
        if quant_device is None:
            quant_device = "cuda:0"

        linear = Linear8bitLt(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()

        # quantize
        linear = linear.to(device=quant_device)
        linear = linear.to(device)
        return linear

    return insert_bnb


def bnb_int4_weight_only():
    from bitsandbytes.nn import LinearFP4

    # TODO: check for this
    from torch_bnb_fp4 import TorchFP4Linear

    def insert_bnb(lin: torch.nn.Linear, device=None, quant_device=None):
        lin.to(torch.float16)
        if device is None:
            device = lin.weight.device
        if quant_device is None:
            quant_device = "cuda:0"

        linear = LinearFP4(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()
        linear = linear.to(device=quant_device)

        linear = TorchFP4Linear.from_linear(linear, use_codebook_dequant=True)
        # fix t5 issue
        linear.weight = None

        # quantize
        linear = linear.to(device)
        return linear

    return insert_bnb


def bnb_nf4_weight_only():
    from bitsandbytes.nn import LinearNF4

    def insert_bnb(lin: torch.nn.Linear, device=None, quant_device=None):
        lin.to(torch.float16)
        if device is None:
            device = lin.weight.device
        if quant_device is None:
            quant_device = "cuda:0"

        linear = LinearNF4(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()

        # quantize
        linear = linear.to(device=quant_device)
        linear = linear.to(device)
        return linear

    return insert_bnb


qa._get_linear_subclass_inserter = patched_inserter


@AffineQuantizedTensor.implements(aten._has_compatible_shallow_copy_type.default)
def _(f, types, *args, **kwargs):
    return False


qint4 = qdtype("int4", qa.int4_weight_only)
qint8 = qdtype("int8", qa.int8_weight_only)
qfloat8 = qdtype("float8", qa.float8_weight_only)

qfloatx = qdtype("floatx_", qa.fpx_weight_only)
qintx = qdtype("intx_", qa.uintx_weight_only)

# cublas
qfloat16 = qdtype("float16", cublas_linear_only)

# bnb
qbint8 = qdtype("bint8", bnb_int8_weight_only)
qfloat4 = qdtype("float4", bnb_int4_weight_only)
qnfloat4 = qdtype("nf4", bnb_nf4_weight_only)


def _nil(module: torch.nn.Module, fqn: str):
    return isinstance(module, torch.nn.Linear)


def _replace_with_custom_fn_if_matches_filter(
    model: torch.nn.Linear,
    replacement_map,
    filter_fn,
    cur_fqn: str = "",
    device=None,
    quant_device=None,
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        if isinstance(replacement_map, dict):
            model = replacement_map[cur_fqn](
                model, device=device, quant_device=quant_device
            )
        else:
            model = replacement_map(model, device=device, quant_device=quant_device)
        return model
    else:
        for name, child in model.named_children():
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                replacement_map,
                filter_fn,
                f"{cur_fqn}{name}.",
                device,
                quant_device,
            )
            if new_child is not child:
                setattr(model, name, new_child)
        return model


def quantize_model(
    model,
    dtype_map,
    device: torch.device = "cuda",
    quantization_device: torch.device = "cuda",
    skip=_nil,
) -> torch.nn.Module:
    _replace_with_custom_fn_if_matches_filter(
        model, dtype_map, skip, device=device, quant_device=quantization_device
    )
    return model
