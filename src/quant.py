from dataclasses import dataclass, field
from functools import partial
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
    args: list = field(default_factory=list)

    def __repr__(self) -> str:
        if "x" in self.display:
            display = self.display
            for arg in self.args:
                display = display.replace("x", str(arg), 1)
            display = display.replace("_", "")
            return display
        return self.display

    def __call__(self, *args, **kwargs) -> any:
        self.args = args
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


def flute_linear_only(
    group_size: int = 64, lql: bool = False, num_bits: int = 4, fake: bool = False
):
    try:
        from flute.integrations.base import FluteLinear
        from flute.integrations.learnable import LearnableQuantizedLinear
        from flute.nf_utils import nf_quantize_2, nf_quantize
        from flute.utils import make_qmap2_from_qmap

        try:
            from flute.tune import tune_and_pack
        except ImportError:
            return lambda x, **_: x

        def insert_flute(lin: torch.nn.Linear, device=None, quant_device=None):
            lin.to(torch.bfloat16)
            if fake:
                new_weight = nf_quantize_2(
                    W=lin.weight.to(device=quant_device),
                    num_bits=num_bits,
                    group_size=group_size,
                    dtype=lin.weight.dtype,
                )
                lin.weight = torch.nn.Parameter(
                    new_weight.to(device=quant_device), requires_grad=False
                )
            else:
                flute_dtype = lin.weight.dtype
                if lql:
                    return lin
                _, _Q, scales, qmap = nf_quantize(
                    W=lin.weight.to(device=quant_device),
                    num_bits=num_bits,
                    group_size=group_size,
                    # dtype=flute_dtype,
                )

                if not (_Q.to(dtype=torch.uint8) == _Q).all():
                    raise ValueError("Q should be uint8")

                example_inputs = torch.randn(
                    1, lin.in_features, dtype=flute_dtype, device=quant_device
                )
                Q, tune_metadata = tune_and_pack(
                    inputs=example_inputs,
                    weight=_Q.to(dtype=torch.uint8).T.contiguous(),
                    num_bits=num_bits,
                    group_size=group_size,
                )

                new_lin = FluteLinear(
                    lin.in_features,
                    lin.out_features,
                    num_bits=num_bits,
                    group_size=group_size,
                    template_id=tune_metadata.template_id,
                    workspace_lazy_init=False,
                    bias=lin.bias is not None,
                    device=quant_device,
                    dtype=flute_dtype,
                )
                scales = scales.view(new_lin.scales.shape)
                scales = scales.to(dtype=new_lin.scales.dtype)
                qmap = qmap.to(dtype=new_lin.tables.dtype)
                qmap2 = make_qmap2_from_qmap(qmap)

                new_lin.weight.copy_(Q)
                new_lin.scales.copy_(scales)
                new_lin.tables.copy_(qmap)
                new_lin.tables2.copy_(qmap2)
                if new_lin.bias is not None:
                    new_lin.bias.copy_(lin.bias)

                return new_lin

        return insert_flute

    except ImportError:
        return lambda x, **_: x


# TODO: produces unstable tensors -> black images
# look into this
def cublas_linear_only():
    try:
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

    except ImportError:

        def insert_cublas(lin: torch.nn.Linear, **_):
            return lin

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

    try:
        from torch_bnb_fp4 import TorchFP4Linear
    except ImportError:
        TorchFP4Linear = None

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

        if TorchFP4Linear is not None:
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

qflute4 = qdtype("qflute4", partial(flute_linear_only, num_bits=4))
qflute3 = qdtype("qflute3", partial(flute_linear_only, num_bits=3))
qflute2 = qdtype("qflute2", partial(flute_linear_only, num_bits=2))


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
            model = replacement_map[cur_fqn]()(
                model, device=device, quant_device=quant_device
            )
        else:
            model = replacement_map()(model, device=device, quant_device=quant_device)
        return model.to(device=device)
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
