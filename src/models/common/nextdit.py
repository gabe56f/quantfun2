import torch
from torch import nn

from ...kernels import ffn_forward


class TimestepEmbedder(nn.Module):
    def __init__(
        self, hidden_size: int, frequency_embeddings_size: int = 256, bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embeddings_size = frequency_embeddings_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embeddings_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )

    def timestep_embedding(
        self, t: torch.Tensor, dim: int, max_period: int = 10_000
    ) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(max_period, dtype=t.dtype, device=t.device))
            * torch.arange(0, half, dtype=t.dtype)
            / half
        ).to(t.device)
        args = t[:, :, None] * frequencies[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embeddings = torch.cat(
                [embeddings, torch.zeros_like(embeddings[:, :, :1])], dim=-1
            )
        return embeddings

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        timestep_frequencies = self.timestep_embedding(
            t, self.frequency_embeddings_size
        )
        timestep_frequencies = timestep_frequencies.to(torch.bfloat16)
        # timestep_frequencies = timestep_frequencies.to(self.mlp[0].weight.dtype)
        return self.mlp(timestep_frequencies)


class TransformerFinalLayer(nn.Module):
    def __init__(
        self, hidden_size: int, num_patches: int, out_channels: int, bias: bool = False
    ):
        super().__init__()
        self.lumina = bias
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, num_patches * out_channels, bias=bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), hidden_size, bias=bias),
        )
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # scale = self.adaLN_modulation(c)
        # x = modulate(self.norm_final(x), scale.unsqueeze(1) if self.lumina else scale)
        # x = self.linear(x)

        return ffn_forward(
            x,
            c,
            self.adaLN_modulation[1].weight,
            self.adaLN_modulation[1].bias,
            self.linear.weight,
            self.linear.bias,
            self.lumina,
            self.hidden_size,
            self.num_patches,
            self.out_channels,
        )


class LlamaFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float = None,
        zeros_initialize: bool = True,
        dtype: torch.dtype = torch.float32,
        hidden_dim_type: str = "onediff",
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.zeros_initialize = zeros_initialize
        self.dtype = dtype

        if hidden_dim_type == "onediff":
            hidden_dim_calculated = int(2 * self.hidden_dim / 3)
            if self.ffn_dim_multiplier is not None:
                hidden_dim_calculated = int(
                    self.ffn_dim_multiplier * hidden_dim_calculated
                )
            self.hidden_dim_calculated = self.multiple_of * (
                (hidden_dim_calculated + self.multiple_of - 1) // self.multiple_of
            )
        else:
            if self.ffn_dim_multiplier is not None:
                hidden_dim_calculated = int(self.ffn_dim_multiplier * self.hidden_dim)
            hidden_dim_calculated = self.multiple_of * (
                (self.hidden_dim + self.multiple_of - 1) // self.multiple_of
            )
        self.hidden_dim = hidden_dim_calculated

        self.w1 = nn.Linear(dim, hidden_dim_calculated, bias=False)
        self.w2 = nn.Linear(hidden_dim_calculated, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim_calculated, bias=False)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
