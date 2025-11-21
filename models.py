import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, List, Union


class MLP(nn.Module):
    def __init__(self, dim_input, dim_embd, dim_out):
        super().__init__()

        self.ff = nn.Sequential(nn.Linear(dim_input, dim_embd),
                                nn.GELU(),
                                nn.Linear(dim_embd, dim_embd // 2),
                                nn.GELU(),
                                nn.Linear(dim_embd // 2, dim_out)
                                )

    def forward(self, x):
        return self.ff(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.lin1(x)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.norm2(out)
        out = out + residual
        out = F.gelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, dim_input, dim_embd, dim_out, n_blocks=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(dim_input, dim_embd)

        self.blocks = nn.Sequential(
            *[ResidualBlock(dim_embd, dropout=dropout) for _ in range(n_blocks)]
        )

        self.output_proj = nn.Linear(dim_embd, dim_out)

    def forward(self, x):
        x = self.input_proj(x)
        x = F.gelu(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x



class LearnableFourierEmbedding(nn.Module):
    """
    Learnable Fourier features for multi-D inputs, with OPTIONAL grouping.

    Input:
        x : Tensor [B, M]   (multi-D features; include time yourself if desired)

    Output:
        pe : Tensor [B, out_dim]

    Args:
        x_dim:                 M (dimension of input)
        fourier_dim:           total Fourier feature dim D (must be even). The output before MLP is D.
        group_sizes:           optional list of ints that partition the input dims (sum must equal x_dim).
                               Example: if x = [t | kin(3) | misc(5)], pass group_sizes=[1,3,5].
                               If None, uses a single group over all dims.
        group_fourier_dims:    optional list of per-group D_g (each even) that sum to fourier_dim.
                               If None, D is split evenly across groups (must divide evenly).
        gamma:                 scalar or list per group. Gaussian-kernel width for N(0, gamma^-2) init of W_r.
        learn_W:               if False, W_r is fixed (random features).
        use_mlp:               add a small MLP after Fourier features.
        mlp_hidden:            hidden size for the MLP.
        out_dim:               final output dim (defaults to fourier_dim if no MLP).
    """
    def __init__(self,
                 x_dim: int,
                 fourier_dim: int = 128,
                 group_sizes: Optional[List[int]] = None,
                 group_fourier_dims: Optional[List[int]] = None,
                 gamma: Union[float, List[float]] = 1.0,
                 ):
        super().__init__()

        assert fourier_dim % 2 == 0, "fourier_dim must be even (sin/cos pairs)."

        # groups over the input dimensions
        if group_sizes is None:
            group_sizes = [x_dim]
        else:
            assert sum(group_sizes) == x_dim, "group_sizes must sum to x_dim."

        self.group_sizes = group_sizes
        self.num_groups = len(group_sizes)

        # allocate Fourier dims per group
        if group_fourier_dims is None:
            assert fourier_dim % self.num_groups == 0, "fourier_dim must be divisible by num_groups (or specify group_fourier_dims)."
            per = fourier_dim // self.num_groups
            group_fourier_dims = [per] * self.num_groups

        assert sum(group_fourier_dims) == fourier_dim, "sum(group_fourier_dims) must equal fourier_dim."

        for d in group_fourier_dims:
            assert d % 2 == 0, "each group_fourier_dim must be even."

        self.group_fourier_dims = group_fourier_dims
        self.fourier_dim = fourier_dim

        # gamma per group
        if isinstance(gamma, (float, int)):
            gammas = [float(gamma)] * self.num_groups
        else:
            assert len(gamma) == self.num_groups, "gamma list must match num_groups."
            gammas = [float(g) for g in gamma]
        self.gammas = gammas

        # one W_r per group: shape [D_g/2, group_input_dim]
        self.Wr = nn.ParameterList()
        for gin, Dg, g in zip(self.group_sizes, self.group_fourier_dims, self.gammas):
            Wr = nn.Parameter(torch.empty(Dg // 2, gin))
            nn.init.normal_(Wr, mean=0.0, std=(1.0 / g))   # kernel-aware init
            Wr.requires_grad_(True)
            self.Wr.append(Wr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M]
        """
        assert x.dim() == 2 and x.size(-1) == sum(self.group_sizes), "x last dim must equal x_dim."
        chunks = torch.split(x, self.group_sizes, dim=-1)

        feats = []
        for chunk, Wr, Dg in zip(chunks, self.Wr, self.group_fourier_dims):
            proj = F.linear(chunk, Wr)
            scale = math.sqrt(Dg / 2.0)
            c = torch.cos(proj)
            s = torch.sin(proj)
            feats.append(torch.cat([c, s], dim=-1) / scale)  # [B, Dg]

        return torch.cat(feats, dim=-1)  # [B, fourier_dim]