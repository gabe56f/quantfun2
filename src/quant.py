from dataclasses import dataclass, field
from functools import partial
import types
from typing import Literal

import torch
from torch.sparse import to_sparse_semi_structured
from torchao.dtypes import (
    MarlinSparseLayout,
    Layout,
    TensorCoreTiledLayout,
    BlockSparseLayout,
    Int4CPULayout,
    Float8Layout,
    AffineQuantizedTensor,
    UintxLayout,
)
from torchao.dtypes.floatx import FloatxTensorCoreLayout
from torchao.dtypes.uintx.marlin_sparse_layout import MarlinSparseAQTTensorImpl
from torchao.dtypes.uintx.tensor_core_tiled_layout import TensorCoreTiledAQTTensorImpl
from torchao.dtypes.uintx.gemlite_layout import GemliteAQTTensorImpl
from torchao.sparsity.blocksparse import BlockSparseTensor
from torchao.quantization.quant_api import (
    MappingType,
    ZeroPointDomain,
)
from torchao.quantization import quant_api as qa

aten = torch.ops.aten


def sparse_marlin_to(self: MarlinSparseAQTTensorImpl, *args, **kwargs):
    device, *_ = torch._C._nn._parse_to(*args, **kwargs)
    return self.__class__(
        self.int_data.to(device),
        self.scale.to(device),
        self.zero_point.to(device),
        self.meta.to(device),
        self._layout,
        self.original_shape,
        self.group_size,
        self.num_bits,
    )


def tensor_core_to(self: TensorCoreTiledAQTTensorImpl, *args, **kwargs):
    device, *_ = torch._C._nn._parse_to(*args, **kwargs)
    return self.__class__(
        self.packed_weight.to(device),
        self.scale_and_zero.to(device),
        self.transposed,
        self._layout,
    )


def gemlite_to(self: GemliteAQTTensorImpl, *args, **kwargs):
    device, *_ = torch._C._nn._parse_to(*args, **kwargs)
    return self.__class__(
        self.packed_weight.to(device),
        self.scale.to(device),
        self.zero_point.to(device),
        self.gemlite_kwargs,
        self._layout,
    )


def blocksparse_to(self: BlockSparseTensor, *args, **kwargs):
    device, *_ = torch._C._nn._parse_to(*args, **kwargs)
    return self.__class__(
        shape=self.shape,
        blocksize=self.blocksize,
        bsr_crow_indices=self.bsr_crow_indices.to(device),
        bsr_col_indices=self.bsr_col_indices.to(device),
        bsr_values=self.bsr_values.to(device),
        requires_grad=False,
    )


GemliteAQTTensorImpl.to = gemlite_to
TensorCoreTiledAQTTensorImpl.to = tensor_core_to
MarlinSparseAQTTensorImpl.to = sparse_marlin_to
BlockSparseTensor.to = blocksparse_to


@BlockSparseTensor.implements(aten._has_compatible_shallow_copy_type.default)
@GemliteAQTTensorImpl.implements(aten._has_compatible_shallow_copy_type.default)
@TensorCoreTiledAQTTensorImpl.implements(aten._has_compatible_shallow_copy_type.default)
@MarlinSparseAQTTensorImpl.implements(aten._has_compatible_shallow_copy_type.default)
def _(f, types, *args, **kwargs):
    return False


@dataclass
class qdtype:
    display: str
    apply_function: any
    args: list = field(default_factory=list)
    supported_dtypes: list = field(default_factory=list)

    def __repr__(self) -> str:
        if "x" in self.display:
            display = self.display
            for arg in self.args:
                display = display.replace("x", str(arg), 1)
            display = display.replace("_", "")
            return display
        return self.display

    def __call__(self, *args, layout: str = "default", **kwargs) -> any:
        self.args = args

        if layout == "default":
            return self.apply_function(*args, **kwargs), self.supported_dtypes[0]
        else:
            if layout == "tensor-core":
                layout = TensorCoreTiledLayout(inner_k_tiles=8)
            elif layout == "marlin-sparse":
                layout = MarlinSparseLayout()
            elif isinstance(layout, str):
                if "block-sparse" in layout:
                    *_, block_size = layout.split("-")
                    try:
                        layout = BlockSparseLayout(int(block_size))
                    except ValueError:
                        layout = BlockSparseLayout()
            kwargs["layout"] = layout

            return self.apply_function(*args, **kwargs), self.supported_dtypes[0]


def linear_inserter(constructor, *args, allow_requires_grad=False, **kwargs):
    def insert_subclass(lin, device=None, quant_device=None):
        requires_grad = allow_requires_grad and lin.weight.requires_grad
        if device is None:
            device = lin.weight.device
        if quant_device is None:
            quant_device = lin.weight.device
        tensor = constructor(lin.weight.to(device=quant_device), *args, **kwargs).to(
            device
        )
        lin.weight = torch.nn.Parameter(
            tensor,
            requires_grad=requires_grad,
        )
        lin.extra_repr = types.MethodType(qa._linear_extra_repr, lin)
        return lin

    return insert_subclass


def _int4_weight_only_transform(
    weight: torch.Tensor,
    group_size: int = 128,
    layout: Layout = TensorCoreTiledLayout(inner_k_tiles=8),
    use_hqq: bool = False,
    zero_point_domain: ZeroPointDomain = ZeroPointDomain.NONE,
    **_,
) -> torch.Tensor:
    if weight.shape[-1] % group_size != 0:
        print(
            f"Skipping quantizing weight with int4 weight only quantization because the shape of weight {weight.shape} is not compatible with group_size {group_size}"
        )
        return weight

    mapping_type = MappingType.ASYMMETRIC
    preserve_zero = qa.LAYOUT_TO_PRESERVE_ZEROS.get(type(layout), None)
    zero_point_dtype = (
        weight.dtype if isinstance(layout, Int4CPULayout) else torch.bfloat16
    )

    if preserve_zero is None:
        print(
            f"Skipping quantizing weight because {type(layout)} type is not supported for int4."
        )
        return weight

    # nonlocal zero_point_domain
    if zero_point_domain == ZeroPointDomain.NONE:
        # the first value is the default one
        zero_point_domain = qa.LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)][0]
    else:
        assert (
            zero_point_domain in qa.LAYOUT_TO_ZERO_POINT_DOMAIN[type(layout)]
        ), f"Layout only support {qa.LAYOUT_TO_ZERO_POINT_DOMAIN[layout]}"

    # Sparse Marlin only supports symmetric quantization.
    if isinstance(layout, MarlinSparseLayout):
        mapping_type = MappingType.SYMMETRIC
        assert (
            group_size == 128 or group_size == weight.shape[-1]
        ), f"MarlinSparseLayout only supports 128 group size or per channel quantization, got {group_size}"

    return AffineQuantizedTensor.from_hp_to_intx(
        weight,
        mapping_type,
        block_size=(1, group_size),
        target_dtype=torch.int32,
        quant_min=0,
        quant_max=15,
        eps=1e-6,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=preserve_zero,
        zero_point_domain=zero_point_domain,
        _layout=layout,
        use_hqq=use_hqq,
    )


def _int8_weight_only_transform(
    weight: torch.Tensor, group_size: int = None, **_
) -> torch.Tensor:
    mapping_type = MappingType.SYMMETRIC
    eps = torch.finfo(torch.float32).eps
    zero_point_dtype = torch.int64
    if group_size is None:
        group_size = weight.shape[1]
    return AffineQuantizedTensor.from_hp_to_intx(
        weight,
        mapping_type,
        block_size=(1, group_size),
        target_dtype=torch.int8,
        eps=eps,
        zero_point_dtype=zero_point_dtype,
    )


def _float8_weight_only_transform(
    weight: torch.Tensor,
    weight_dtype: Literal["float8_e4m3fn", "float8_e5m2"] = torch.float8_e4m3fn,
    **_,
) -> torch.Tensor:
    if isinstance(weight_dtype, str):
        weight_dtype = getattr(torch, weight_dtype)

    return AffineQuantizedTensor.from_hp_to_floatx(
        input_float=weight,
        block_size=(1, weight.shape[1]),
        target_dtype=weight_dtype,
        scale_dtype=None,
        _layout=Float8Layout(mm_config=None),
    )


def _fpx_weight_only_transform(
    weight: torch.Tensor, ebits: int = 3, mbits: int = 2, **_
) -> torch.Tensor:
    assert weight.dim() == 2, f"floatx only works for 2-d Tensor, got: {weight.dim()}"
    out_dim, in_dim = weight.shape
    if (in_dim % 64 != 0) or (out_dim % 256 != 0):
        print(
            f"Skipping floatx quantization float{ebits + mbits + 1}_{ebits}_{mbits} because "
            f"the shape is not compatible with the kernel: in_dim={in_dim}, out_dim={out_dim} "
            "expected in_dim % 64 == 0 and out_dim % 256 == 0"
        )
        return weight

    _layout = FloatxTensorCoreLayout(ebits, mbits)
    return AffineQuantizedTensor.from_hp_to_fpx(weight, _layout)


def _uintx_weight_only_transform(
    weight: torch.Tensor, bits: int, group_size: int = 64, pack_dim: int = -1, **_
) -> torch.Tensor:
    dtype: torch.dtype = torch.getattr(f"uint{bits}")

    eps = torch.finfo(torch.float32).eps
    _layout = UintxLayout(dtype=dtype, pack_dim=pack_dim)

    return AffineQuantizedTensor.from_hp_to_intx(
        weight,
        MappingType.ASYMMETRIC,
        (1, group_size),
        target_dtype=dtype,
        quant_min=None,
        quant_max=None,
        eps=eps,
        zero_point_dtype=torch.int32,
        zero_point_domain=ZeroPointDomain.INT,
        preserve_zero=True,
        _layout=_layout,
        use_hqq=False,
    )


def _gemlite_uintx_weight_only_transform(
    weight: torch.Tensor,
    bits: Literal[4, 8] = 4,
    group_size: Literal[32, 64, 128, 256, 512, 1024, None] = 64,
    pack_width: Literal[8, 16, 32] = 32,
    contiguous: bool = None,
) -> torch.Tensor:
    use_hqq = bits == 4

    from torchao.dtypes.uintx.gemlite_layout import get_gemlite_aqt_kwargs

    use_hqq = True if bits == 4 else False
    return AffineQuantizedTensor.from_hp_to_intx(
        weight,
        **get_gemlite_aqt_kwargs(
            weight, group_size, bits, pack_width, contiguous, use_hqq
        ),
    )


def _block_sparse_transform(
    weight: torch.Tensor,
    group_size: int = 64,
    **_,
) -> torch.Tensor:
    return BlockSparseTensor.from_dense(weight, group_size)


@AffineQuantizedTensor.implements(aten._has_compatible_shallow_copy_type.default)
def _(f, types, *args, **kwargs):
    return False


qint4 = qdtype(
    "int4",
    partial(linear_inserter, _int4_weight_only_transform),
    supported_dtypes=[torch.bfloat16],
)
qint8 = qdtype(
    "int8",
    partial(linear_inserter, _int8_weight_only_transform),
    supported_dtypes=[torch.float16],
)
qfloat8 = qdtype(
    "float8",
    partial(linear_inserter, _float8_weight_only_transform),
    supported_dtypes=[torch.float16],
)

qfloatx = qdtype(
    "floatx_",
    partial(linear_inserter, _fpx_weight_only_transform),
    supported_dtypes=[torch.bfloat16, torch.float16],
)
qintx = qdtype(
    "intx_",
    partial(linear_inserter, _uintx_weight_only_transform),
    supported_dtypes=[torch.bfloat16, torch.float16],
)
qintx_gemlite = qdtype(
    "intxgemlite_",
    partial(linear_inserter, _gemlite_uintx_weight_only_transform),
    supported_dtypes=[torch.float16],
)


q_sparse = qdtype(
    "sparse",
    partial(linear_inserter, to_sparse_semi_structured),
    supported_dtypes=[torch.bfloat16, torch.float16],
)
q_block_sparse = qdtype(
    "block_sparse",
    partial(linear_inserter, _block_sparse_transform),
    supported_dtypes=[torch.float16],
)
# q_sparse_block = qdtype("sparse_block", qs.block_sparse_weight, sparse=True)


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
            converter, torch_dtype = replacement_map[cur_fqn]()
        else:
            converter, torch_dtype = replacement_map()

        return converter(
            model.to(dtype=torch_dtype), device=device, quant_device=quant_device
        ).to(device=device)
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
        if isinstance(replacement_map, dict):
            return model.to(replacement_map[cur_fqn]()[1])
        return model.to(replacement_map()[1])


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
