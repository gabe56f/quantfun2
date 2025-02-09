import math

import torch
from torch import nn
import numpy as np

from ...onediff.attn import apply_merged_rotary_embedding, do_attn, ATTN
from ...onediff.model import (
    RMSNorm,
    TimestepEmbedder,
    LlamaFeedForward,
    TransformerFinalLayer,
)
from ...onediff.utils import modulate


class LuminaTimestepEmbedder(TimestepEmbedder):
    def __init__(self, hidden_size: int, frequency_embeddings_size: int = 256):
        super().__init__(hidden_size, frequency_embeddings_size, True)

    def timestep_embedding(
        self, t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class JointAttention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int = None, qk_norm: bool = False
    ):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def apply_rotary_embedding(x_in: torch.Tensor, freqs_cis: torch.Tensor):
        with torch.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_merged_rotary_embedding(xq, xk, freqs_cis)

        softmax_scale = math.sqrt(1 / self.head_dim)
        if ATTN == "sdpa":
            n_rep = self.n_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
        output = do_attn(
            self,
            xq,
            xk,
            xv,
            x_mask,
            lambda x: x.bool()
            .view(bsz, 1, 1, seqlen)
            .expand(-1, self.n_heads, seqlen, -1),
            bsz,
            seqlen,
            softmax_scale,
        )
        output = output.flatten(-2)
        return self.wo(output)


class JointTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = LlamaFeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            hidden_dim_type="lumina",
        )
        self.layer_id = layer_id
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    min(dim, 1024),
                    4 * dim,
                    bias=True,
                ),
            )
        else:
            self.adaLN_modulation = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor = None,
    ):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                adaln_input
            ).chunk(4, dim=-1)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa.unsqueeze(1)),
                    x_mask,
                    freqs_cis,
                )
            )
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp.unsqueeze(1)))
            )
        else:
            x = x + self.attention_norm2(
                self.attention(self.attention_norm1(x), x_mask, freqs_cis)
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


def precompute_freqs_cis(
    dim: list[int], end: list[int], theta: float = 10000.0
) -> list[torch.Tensor]:
    freqs_cis = []
    for d, e in zip(dim, end):
        freqs = 1.0 / (
            theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
        )
        timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        freqs_cis.append(freqs_cis_i)
    return freqs_cis


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 10000.0,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (1, 512, 512),
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = precompute_freqs_cis(
            self.axes_dims, self.axes_lens, theta=self.theta
        )

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            index = (
                ids[:, :, i : i + 1]
                .repeat(1, 1, self.freqs_cis[i].shape[-1])
                .to(torch.int64)
            )
            result.append(
                torch.gather(
                    self.freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1),
                    dim=1,
                    index=index,
                )
            )
        return torch.cat(result, dim=-1)


class LuminaDiT(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: int = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: float = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (1, 512, 512),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = LuminaTimestepEmbedder(min(dim, 1024))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True)
        )
        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(
            dim, eps=norm_eps, elementwise_affine=False, bias=True
        )
        # self.norm_final = RMSNorm(dim, eps=norm_eps)
        self.final_layer = TransformerFinalLayer(
            dim, patch_size * patch_size, self.out_channels, bias=True
        )

        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(axes_dims=axes_dims, axes_lens=axes_lens)
        self.dim = dim
        self.n_heads = n_heads

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def unpatchify(
        self,
        x: torch.Tensor,
        img_size: list[tuple[int, int]],
        cap_size: list[int],
        return_tensor: bool = False,
    ) -> list[torch.Tensor]:
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )
        if return_tensor:
            return torch.stack(imgs, dim=0)
        return imgs

    def patchify_and_embed(
        self,
        x: list[torch.Tensor] | torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device
        l_effective_cap_len = cap_mask.sum(dim=1).to(torch.int32).tolist()
        img_sizes = [(img.size(1), img.size(2)) for img in x]
        l_effective_img_len = [(H // pH) * (W // pW) for H, W in img_sizes]

        max_seq_len = max(
            (
                cap_len + img_len
                for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len)
            )
        )
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(
            bsz, max_seq_len, 3, dtype=torch.int32, device=device
        )

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW
            position_ids[i, :cap_len, 0] = torch.arange(
                cap_len, dtype=torch.int32, device=device
            )
            position_ids[i, cap_len : cap_len + img_len, 0] = cap_len
            row_ids = (
                torch.arange(H_tokens, dtype=torch.int32, device=device)
                .view(-1, 1)
                .repeat(1, W_tokens)
                .flatten()
            )
            col_ids = (
                torch.arange(W_tokens, dtype=torch.int32, device=device)
                .view(1, -1)
                .repeat(H_tokens, 1)
                .flatten()
            )
            position_ids[i, cap_len : cap_len + img_len, 1] = row_ids
            position_ids[i, cap_len : cap_len + img_len, 2] = col_ids
        freqs_cis = self.rope_embedder(position_ids)
        cap_freqs_cis_shape = list(freqs_cis.shape)
        cap_freqs_cis_shape[1] = cap_feats.shape[1]
        cap_freqs_cis = torch.zeros(
            *cap_freqs_cis_shape, dtype=freqs_cis.dtype, device=device
        )

        img_freqs_cis_shape = list(freqs_cis.shape)
        img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(
            *img_freqs_cis_shape, dtype=freqs_cis.dtype, device=device
        )

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len : cap_len + img_len]

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        flat_x = []
        for i in range(bsz):
            img = x[i]
            C, H, W = img.size()
            img = (
                img.view(C, H // pH, pH, W // pW, pW)
                .permute(1, 3, 2, 4, 0)
                .flatten(2)
                .flatten(0, 1)
            )
            flat_x.append(img)
        x = flat_x
        padded_img_embed = torch.zeros(
            bsz, max_img_len, x[0].shape[-1], device=device, dtype=x[0].dtype
        )
        padded_img_mask = torch.zeros(bsz, max_img_len, device=device, dtype=torch.bool)
        for i in range(bsz):
            padded_img_embed[i, : l_effective_img_len[i]] = x[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        padding_img_embed = self.x_embedder(padded_img_embed)
        for layer in self.noise_refiner:
            padding_img_embed = layer(
                padding_img_embed, padded_img_mask, img_freqs_cis, t
            )

        mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)
        padded_full_embed = torch.zeros(
            bsz, max_seq_len, self.dim, device=device, dtype=x[0].dtype
        )
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]

            mask[i, : cap_len + img_len] = True
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len : cap_len + img_len] = padding_img_embed[
                i, :img_len
            ]
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
    ) -> torch.Tensor:
        t = self.t_embedder(t)
        adaln_input = t
        cap_feats = self.cap_embedder(cap_feats)

        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x, cap_feats, cap_mask, t
        )
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)
        return x

    @staticmethod
    def get_replace_map() -> dict:
        return {
            y: x
            for x, y in {
                "norm_out.linear_1": "final_layer.adaLN_modulation.1",
                "norm_out.linear_2": "final_layer.linear",
                ".attn.to_k.": ".attention.wk.",
                ".attn.to_q.": ".attention.wq.",
                ".attn.to_v.": ".attention.wv.",
                ".attn.to_out.0.": ".attention.wo.",
                ".attn.": ".attention.",
                ".feed_forward.linear_": ".feed_forward.w",
                "time_caption_embed.caption_embedder": "cap_embedder",
                "time_caption_embed.timestep_embedder.linear_1": "t_embedder.mlp.0",
                "time_caption_embed.timestep_embedder.linear_2": "t_embedder.mlp.2",
                "time_caption_embed.timestep_embedder": "t_embedder",
                "norm1.linear": "adaLN_modulation.1",
                ".norm2": ".attention_norm2",
                "context_refiner.0.norm1.weight": "context_refiner.0.attention_norm1.weight",
                "context_refiner.1.norm1.weight": "context_refiner.1.attention_norm1.weight",
                "norm_q": "q_norm",
                "norm_k": "k_norm",
                "norm1.norm": "attention_norm1",
            }.items()
        }

    @staticmethod
    def get_key_from_diffusers(key: str) -> str:
        key = key.replace(".attn.to_k.", ".attention.wk.")
        key = key.replace(".attn.to_q.", ".attention.wq.")
        key = key.replace(".attn.to_v.", ".attention.wv.")
        key = key.replace(".attn.to_out.0.", ".attention.wo.")
        key = key.replace(".attn.", ".attention.")
        key = key.replace(".feed_forward.linear_", ".feed_forward.w")
        key = key.replace("time_caption_embed.caption_embedder", "cap_embedder")
        key = key.replace(
            "time_caption_embed.timestep_embedder.linear_1", "t_embedder.mlp.0"
        )
        key = key.replace(
            "time_caption_embed.timestep_embedder.linear_2", "t_embedder.mlp.2"
        )
        key = key.replace("time_caption_embed.timestep_embedder", "t_embedder")
        key = key.replace(".norm1.linear", ".adaLN_modulation.1")
        key = key.replace(".norm1.norm", ".attention_norm1")
        key = key.replace(".norm2", ".attention_norm2")
        key = key.replace(".norm1.weight", ".attention_norm1.weight")
        key = key.replace("norm_out.linear_1", "final_layer.adaLN_modulation.1")
        key = key.replace("norm_out.linear_2", "final_layer.linear")
        key = key.replace("norm_q", "q_norm")
        key = key.replace("norm_k", "k_norm")

        return key

    @staticmethod
    def transform_state_dict(state_dict: dict) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            key = LuminaDiT.get_key_from_diffusers(key)
            new_state_dict[key] = value
        return new_state_dict
