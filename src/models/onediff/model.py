from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import numpy as np
import torch
from torch import nn

from ..common.attn import Attention
from ..common.rmsnorm import RMSNorm
from ..common.nextdit import TimestepEmbedder, TransformerFinalLayer, LlamaFeedForward
from ..common.utilities import modulate


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        n_layers: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        y_dim: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.attention = Attention(
            layer_id,
            n_layers,
            dim,
            n_heads,
            n_kv_heads,
            qk_norm,
            y_dim=y_dim,
            max_position_embeddings=max_position_embeddings,
        )
        self.feed_forward = LlamaFeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 4 * dim),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.BoolTensor,
        freqs_cis: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.BoolTensor,
        adaln_input: torch.Tensor = None,
    ) -> torch.Tensor:
        if adaln_input is not None:
            scales_gates = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = scales_gates.chunk(4, dim=-1)
            x = x + torch.tanh(gate_msa) * self.attention_norm2(
                self.attention(
                    modulate(self.attention_norm1(x), scale_msa),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                )
            )
            x = x + torch.tanh(gate_mlp) * self.ffn_norm2(
                self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp))
            )
        else:
            x = x + self.attention_norm2(
                self.attention(
                    self.attention_norm1(x),
                    x_mask,
                    freqs_cis,
                    self.attention_y_norm(y),
                    y_mask,
                )
            )
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class NextDiT(ModelMixin, ConfigMixin):
    """
    Diffusion model with a Transformer backbone for joint image-video training.
    """

    @register_to_config
    def __init__(
        self,
        input_size=(1, 32, 32),
        patch_size=(1, 2, 2),
        in_channels=16,
        hidden_size=4096,
        depth=32,
        num_heads=32,
        num_kv_heads=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        pred_sigma=False,
        caption_channels=4096,
        qk_norm=False,
        norm_type="rms",
        model_max_length=120,
        rotary_max_length=384,
        rotary_max_length_t=None,
        **ignore,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.pred_sigma = pred_sigma
        self.caption_channels = caption_channels
        self.qk_norm = qk_norm
        self.norm_type = norm_type
        self.model_max_length = model_max_length
        self.rotary_max_length = rotary_max_length
        self.rotary_max_length_t = rotary_max_length_t
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        self.x_embedder = nn.Linear(np.prod(self.patch_size) * in_channels, hidden_size)

        self.t_embedder = TimestepEmbedder(min(hidden_size, 1024))
        self.y_embedder = nn.Sequential(
            nn.LayerNorm(caption_channels, eps=1e-6),
            nn.Linear(caption_channels, min(hidden_size, 1024)),
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_id=i,
                    n_layers=depth,
                    dim=hidden_size,
                    n_heads=num_heads,
                    n_kv_heads=self.num_kv_heads,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                    norm_eps=norm_eps,
                    qk_norm=qk_norm,
                    y_dim=caption_channels,
                    max_position_embeddings=rotary_max_length,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = TransformerFinalLayer(
            hidden_size=hidden_size,
            num_patches=np.prod(patch_size),
            out_channels=self.out_channels,
        )

        assert (
            hidden_size // num_heads
        ) % 6 == 0, "3d rope needs head dim to be divisible by 6"

        with torch.device("cpu"):
            self.freqs_cis = self.precompute_freqs_cis(
                hidden_size // num_heads,
                self.rotary_max_length,
                end_t=self.rotary_max_length_t,
            )

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        # self.freqs_cis = self.freqs_cis.to(*args, **kwargs)
        return self

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        end_t: int = None,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scale_watershed: float = 1.0,
        timestep: float = 1.0,
    ):
        if timestep < scale_watershed:
            linear_factor = scale_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scale_factor

        theta = theta * ntk_factor
        freqs = (
            1.0
            / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim))
            / linear_factor
        )

        timestep = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis = torch.exp(1j * freqs)

        if end_t is not None:
            freqs_t = (
                1.0
                / (theta ** (torch.arange(0, dim, 6)[: (dim // 6)] / dim))
                / linear_factor
            )
            timestep_t = torch.arange(end_t, dtype=torch.float32)
            freqs_t = torch.outer(timestep_t, freqs_t).float()
            freqs_cis_t = torch.exp(1j * freqs_t)
            freqs_cis_t = freqs_cis_t.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)
        else:
            end_t = end
            freqs_cis_t = freqs_cis.view(end_t, 1, 1, dim // 6).repeat(1, end, end, 1)

        freqs_cis_h = freqs_cis.view(1, end, 1, dim // 6).repeat(end_t, 1, end, 1)
        freqs_cis_w = freqs_cis.view(1, 1, end, dim // 6).repeat(end_t, end, 1, 1)
        freqs_cis = torch.cat([freqs_cis_t, freqs_cis_h, freqs_cis_w], dim=-1).view(
            end_t, end, end, -1
        )
        return freqs_cis

    def forward(
        self,
        samples,
        timesteps,
        encoder_hidden_states,
        encoder_attention_mask,
        scale_factor: float = 1.0,  # scale_factor for rotary embedding
        scale_watershed: float = 1.0,  # scale_watershed for rotary embedding
    ):
        if samples.ndim == 4:  # B C H W
            samples = samples[:, None, ...]  # B F C H W

        precomputed_freqs_cis = None
        if scale_factor != 1 or scale_watershed != 1:
            precomputed_freqs_cis = self.precompute_freqs_cis(
                self.hidden_size // self.num_heads,
                self.rotary_max_length,
                end_t=self.rotary_max_length_t,
                scale_factor=scale_factor,
                scale_watershed=scale_watershed,
                timestep=torch.max(timesteps.cpu()).item(),
            )

        if len(timesteps.shape) == 5:
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        elif len(timesteps.shape) == 1:
            timesteps = timesteps[:, None, None, None, None].expand_as(samples)
            t, *_ = self.patchify(timesteps, precomputed_freqs_cis)
            timesteps = t.mean(dim=-1)
        samples, T, H, W, freqs_cis = self.patchify(samples, precomputed_freqs_cis)
        samples = self.x_embedder(samples)
        t = self.t_embedder(timesteps)

        encoder_attention_mask_float = encoder_attention_mask[..., None].float()
        encoder_hidden_states_pool = (
            encoder_hidden_states * encoder_attention_mask_float
        ).sum(dim=1) / (encoder_attention_mask_float.sum(dim=1) + 1e-8)
        encoder_hidden_states_pool = encoder_hidden_states_pool.to(samples.dtype)
        y = self.y_embedder(encoder_hidden_states_pool)
        y = y.unsqueeze(1).expand(-1, samples.size(1), -1)

        adaln_input = t + y

        for block in self.layers:
            samples = block(
                samples,
                None,
                freqs_cis,
                encoder_hidden_states,
                encoder_attention_mask,
                adaln_input,
            )

        samples = self.final_layer(samples, adaln_input)
        samples = self.unpatchify(samples, T, H, W)

        return samples

    def patchify(self, x, precompute_freqs_cis=None):
        # pytorch is C, H, W
        B, T, C, H, W = x.size()
        pT, pH, pW = self.patch_size
        x = x.view(B, T // pT, pT, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
        x = x.reshape(B, -1, pT * pH * pW * C)
        if precompute_freqs_cis is None:
            freqs_cis = (
                self.freqs_cis[: T // pT, : H // pH, : W // pW]
                .reshape(-1, *self.freqs_cis.shape[3:])[None]
                .to(x.device)
            )
        else:
            freqs_cis = (
                precompute_freqs_cis[: T // pT, : H // pH, : W // pW]
                .reshape(-1, *precompute_freqs_cis.shape[3:])[None]
                .to(x.device)
            )
        return x, T // pT, H // pH, W // pW, freqs_cis

    def unpatchify(self, x, T, H, W):
        B = x.size(0)
        C = self.out_channels
        pT, pH, pW = self.patch_size
        x = x.view(B, T, H, W, pT, pH, pW, C)
        x = x.permute(0, 1, 4, 7, 2, 5, 3, 6)
        x = x.reshape(B, T * pT, C, H * pH, W * pW)
        return x
