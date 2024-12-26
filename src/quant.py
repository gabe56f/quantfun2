from dataclasses import dataclass
import types

import torch
from torchao.dtypes.affine_quantized_tensor import (
    AffineQuantizedTensor,
)

# from torchao.dtypes import Layout, UintxLayout
# from torchao.dtypes.floatx import FloatxTensorCoreLayout
from torchao.quantization import quant_api as qa
from torchao.quantization import quant_primitives as qp

QUANTIZATION_DEVICE = torch.device("cuda:0")
_MOVE_TO = None

aten = torch.ops.aten


@dataclass
class qdtype:
    display: str
    apply_function: any

    def __call__(self, *args, **kwargs) -> any:
        return self.apply_function(*args, **kwargs)


def patched_inserter(constructor, *, allow_requires_grad=False, **kwargs):
    def insert_subclass(lin):
        requires_grad = allow_requires_grad and lin.weight.requires_grad
        tensor = constructor(lin.weight.to(QUANTIZATION_DEVICE), **kwargs).to(_MOVE_TO)
        lin.weight = torch.nn.Parameter(
            tensor,
            requires_grad=requires_grad,
        )
        lin.extra_repr = types.MethodType(qa._linear_extra_repr, lin)
        return lin

    return insert_subclass


def cublas_linear_only():
    from cublas_ops import CublasLinear

    def insert_cublas(lin: torch.nn.Linear):
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

    def insert_bnb(lin: torch.nn.Linear):
        lin.to(torch.float16)
        linear = Linear8bitLt(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()
        linear.to(QUANTIZATION_DEVICE)
        linear.to(_MOVE_TO)
        return linear

    return insert_bnb


def bnb_int4_weight_only():
    from bitsandbytes.nn import LinearFP4

    # TODO: check for this
    from torch_bnb_fp4 import TorchFP4Linear

    def insert_bnb(lin: torch.nn.Linear):
        lin.to(torch.float16)
        linear = LinearFP4(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()
        linear.to(QUANTIZATION_DEVICE)

        linear = TorchFP4Linear.from_linear(linear, use_codebook_dequant=True)
        linear.weight = None
        linear.to(_MOVE_TO)
        return linear

    return insert_bnb


def bnb_nf4_weight_only():
    from bitsandbytes.nn import LinearNF4

    def insert_bnb(lin: torch.nn.Linear):
        lin.to(torch.float16)
        linear = LinearNF4(
            lin.in_features,
            lin.out_features,
            lin.bias is not None,
            device=lin.weight.device,
        )
        linear.weight.data = lin.weight.clone().detach()
        if lin.bias is not None:
            linear.bias.data = lin.bias.clone().detach()
        linear.to(QUANTIZATION_DEVICE)
        linear.to(_MOVE_TO)
        return linear

    return insert_bnb


qa._get_linear_subclass_inserter = patched_inserter


@AffineQuantizedTensor.implements(aten._has_compatible_shallow_copy_type.default)
def _(f, types, *args, **kwargs):
    return False


qintx = qdtype("intx_", qa.uintx_weight_only)
qint4 = qdtype("int4", qa.int4_weight_only)
qint8 = qdtype("int8", qa.int8_weight_only)
qfloatx = qdtype("floatx_", qa.fpx_weight_only)
qfloat8 = qdtype("float8", qa.float8_weight_only)
qfloat16 = qdtype("float16", cublas_linear_only)
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
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        # if device is not None:
        #     model.to(device=device)  # move to device before quantization
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
        # if device is not None:
        #     model.to(device=device)  # move parent module to device
        return model


def quantize_model(
    model,
    # initialize,
    dtype_map,
    device: torch.device = "cuda",
    skip=_nil,
) -> torch.nn.Module:
    # model = initialize()

    global _MOVE_TO
    _MOVE_TO = device
    _replace_with_custom_fn_if_matches_filter(model, dtype_map, skip, device=device)
    return model
