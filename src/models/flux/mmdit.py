import torch
from torch import nn

from .attn import _do_attn, rope_apply, SelfAttention, FluxTimestepEmbedder
from ..common.rmsnorm import RMSNorm
from ..common.utilities import modulate


def FluxModulation(dim: int, double: bool) -> nn.Sequential:
    if double:
        return nn.Sequential(
            nn.SiLU(),
            nn.Unflatten(1, (1, -1)),
            nn.Linear(dim, 6 * dim, bias=True),
        )
    else:
        return nn.Sequential(
            nn.SiLU(),
            nn.Unflatten(1, (1, -1)),
            nn.Linear(dim, 3 * dim, bias=True),
        )


class FluxRopeEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float,
        axes_dim: list,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.dtype = dtype

    @classmethod
    def rope(cls, pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
        omega = 1.0 / (theta**scale)
        out = torch.einsum("...n,d->...nd", pos, omega)
        out = torch.stack(
            [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
        )
        out = out.reshape(*out.shape[:-1], 2, 2)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_axes = x.shape[-1]
        emb = torch.cat(
            [
                self.rope(x[..., i], self.axes_dim[i], self.theta).type(self.dtype)
                for i in range(n_axes)
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.dtype = dtype
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = FluxModulation(hidden_size, True)
        self.img_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = FluxModulation(hidden_size, True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H
        self.do_clamp = dtype == torch.float16

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: torch.Tensor,
        precomputed_modulation: tuple[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if precomputed_modulation is not None:
            (shift_img1, scale_img1, gate_img1, shift_img2, scale_img2, gate_img2), (
                shift_txt1,
                scale_txt1,
                gate_txt1,
                shift_txt2,
                scale_txt2,
                gate_txt2,
            ) = precomputed_modulation
        else:
            shift_img1, scale_img1, gate_img1, shift_img2, scale_img2, gate_img2 = (
                self.img_mod(vec).chunk(6, dim=1)
            )
            shift_txt1, scale_txt1, gate_txt1, shift_txt2, scale_txt2, gate_txt2 = (
                self.txt_mod(vec).chunk(6, dim=1)
            )

        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, scale_img1) + shift_img1
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.reshape(
            *img_qkv.shape[:-1], self.K, self.H, -1
        ).permute(2, 0, 3, 1, 4)
        img_q = self.img_attn.q_norm(img_q)
        img_k = self.img_attn.k_norm(img_k)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, scale_txt1) + shift_txt1
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.reshape(
            *txt_qkv.shape[:-1], self.K, self.H, -1
        ).permute(2, 0, 3, 1, 4)
        txt_q = self.txt_attn.q_norm(txt_q)
        txt_k = self.txt_attn.k_norm(txt_k)

        q = torch.cat([img_q, txt_q], dim=2)
        k = torch.cat([img_k, txt_k], dim=2)
        v = torch.cat([img_v, txt_v], dim=2)

        q, k = rope_apply(q, k, freqs_cis)
        attn = _do_attn(q, k, v)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        img = img + gate_img1 * self.img_attn.proj(img_attn)
        img = img + gate_img2 * self.img_mlp(
            modulate(self.img_norm2(img), scale_img2) + shift_img2
        )

        txt = txt + gate_txt1 * self.txt_attn.proj(txt_attn)
        txt = txt + gate_txt2 * self.txt_mlp(
            modulate(self.txt_norm2(txt), scale_txt2) + shift_txt2
        )

        if self.do_clamp:
            img = img.clamp(-32000, 32000)
            txt = txt.clamp(-32000, 32000)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.dtype = dtype
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.linear_1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear_2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = FluxModulation(hidden_size, False)
        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H
        self.do_clamp = dtype == torch.float16

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: torch.Tensor,
        precomputed_modulation: torch.Tensor = None,
    ) -> torch.Tensor:
        if precomputed_modulation is not None:
            shift_mod, scale_mod, gate_mod = precomputed_modulation
        else:
            shift_mod, scale_mod, gate_mod = self.modulation(vec).chunk(3, dim=1)

        pre_norm = self.pre_norm(x)
        x_mod = modulate(pre_norm, scale_mod) + shift_mod
        qkv, mlp = torch.split(
            self.linear_1(x_mod), [self.hidden_dim * 3, self.mlp_hidden_dim], dim=-1
        )
        q, k, v = qkv.reshape(*qkv.shape[:-1], self.K, self.H, -1).permute(
            2, 0, 3, 1, 4
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = rope_apply(q, k, freqs_cis)
        attn = _do_attn(q, k, v)
        output = self.linear_2(torch.cat([attn, self.mlp_act(mlp)], dim=2))
        if self.do_clamp:
            out = (x + gate_mod * output).clamp(-32000, 32000)
        else:
            out = x + gate_mod * output
        return out


class FluxFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        precomputed_modulation: torch.Tensor = None,
    ) -> torch.Tensor:
        if precomputed_modulation is not None:
            shift, scale = precomputed_modulation
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        x = modulate(self.norm_final(x), scale[:, None, :]) + shift[:, None, :]
        x = self.linear(x)
        return x


class FluxTransformer2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_heads: int,
        axes_dim: list,
        theta: float,
        context_in_dim: int,
        double_block_depth: int,
        single_block_depth: int,
        mlp_ratio: float,
        qk_scale: float,
        vec_in_dim: int,
        time_in_dim: int = 256,  # 16, if chroma
        guidance_in_dim: int = 256,  # 16, if chroma
        chroma: bool = False,
        guidance_embed: bool = False,
        qkv_bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.chroma = chroma
        self.dtype = dtype
        self.out_channels = self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.guidance_embed = guidance_embed

        rope_dim = hidden_size // num_heads
        self.rope_embedder = FluxRopeEmbedder(
            dim=rope_dim,
            theta=theta,
            axes_dim=axes_dim,
            dtype=self.dtype,
        )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size, bias=True)
        self.time_in = FluxTimestepEmbedder(self.hidden_size, time_in_dim)
        self.vector_in = FluxTimestepEmbedder(self.hidden_size, vec_in_dim)
        if self.guidance_embed:
            self.guidance_in = FluxTimestepEmbedder(self.hidden_size, guidance_in_dim)
        else:
            self.guidance_in = nn.Identity()
        if self.chroma:
            self.modulation_in = FluxTimestepEmbedder(self.hidden_size, 32)
        else:
            self.modulation_in = nn.Identity()
        self.distilled_guidance_layer = nn.Identity()

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    dtype,
                )
                for _ in range(double_block_depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    qk_scale,
                    dtype,
                )
                for _ in range(single_block_depth)
            ]
        )

        self.final_layer = FluxFinalLayer(self.hidden_size, 1, self.out_channels)

    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        guidance: torch.Tensor = None,
        skip_double: list = [],
        skip_single: list = [],
    ) -> torch.Tensor:
        assert img.ndim == 3, "Input image must have shape (B, H, W)"
        assert txt.ndim == 3, "Input text must have shape (B, L, D)"

        img = self.img_in(img)
        txt = self.txt_in(txt)

        if self.chroma:
            modulation_index_length = 344
            distilled_timestep = self.time_in(timesteps)
            distilled_guidance = self.guidance_in(guidance)

            modulation_index = self.modulation_in(
                torch.arange(
                    modulation_index_length, device=img.device, dtype=img.dtype
                )
            )
            modulation_index = modulation_index.unsqueeze(0).expand(img.shape[0], 1, 1)

            timestep_guidance = (
                torch.cat([distilled_timestep, distilled_guidance], dim=1)
                .unsqueeze(1)
                .repeat(1, modulation_index_length, 1)
            )
            vec = torch.cat([timestep_guidance, modulation_index], dim=-1)

            vec = self.distilled_guidance_layer(vec)
            precomputed_vec = {}

            idx = 0
            for i in range(len(self.single_blocks)):
                precomputed_vec[f"s{i}"] = (
                    vec[:, idx : idx + 1, :],
                    vec[:, idx + 1 : idx + 2, :],
                    vec[:, idx + 2 : idx + 3, :],
                )
                idx += 3
            for k in ["i", "t"]:
                for i in range(len(self.double_blocks)):
                    precomputed_vec[f"{k}{i}"] = (
                        vec[:, idx : idx + 1, :],
                        vec[:, idx + 1 : idx + 2, :],
                        vec[:, idx + 2 : idx + 3, :],
                        vec[:, idx + 3 : idx + 4, :],
                        vec[:, idx + 4 : idx + 5, :],
                        vec[:, idx + 5 : idx + 6, :],
                    )
                    idx += 6

            precomputed_vec["f"] = (
                vec[:, idx : idx + 1, :].squeeze(1),
                vec[: idx + 1 : idx + 2, :].squeeze(1),
            )

        else:
            vec = self.time_in(timesteps)
            if self.guidance_embed:
                vec = vec + self.guidance_in(guidance)
            vec = vec + self.vector_in(y)
            precomputed_vec = {}

        ids = torch.cat([txt_ids, img_ids], dim=1)
        rope = self.rope_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            if i in skip_double:
                continue

            img_mod = precomputed_vec.get(f"i{i}", None)
            txt_mod = precomputed_vec.get(f"t{i}", None)
            if img_mod is not None and txt_mod is not None:
                img, txt = block(
                    img, txt, None, rope, precomputed_modulation=[img_mod, txt_mod]
                )
            else:
                img, txt = block(img, txt, vec, rope)

        img = torch.cat([txt, img], dim=1)
        for i, block in enumerate(self.single_blocks):
            if i in skip_single:
                continue

            mod = precomputed_vec.get(f"s{i}", None)
            if mod is not None:
                img = block(img, None, rope, precomputed_modulation=mod)
            else:
                img = block(img, vec, rope)

        img = img[:, txt.shape[1] :, ...]

        final_mod = precomputed_vec.get("f", None)
        if final_mod is not None:
            img = self.final_layer(img, None, precomputed_modulation=final_mod)
        else:
            img = self.final_layer(img, vec)
        return img
