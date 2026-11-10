"""Token encoder for TERS cubes: CNN to an 8 x 8 token grid, transformer, CLS readout."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GRID = 8


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _norm(cout)
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _norm(cout)))

    def forward(self, x: Tensor) -> Tensor:
        h = F.silu(self.n1(self.c1(x)))
        return F.silu(self.n2(self.c2(h)) + self.skip(x))


def coord_channels(x: Tensor) -> Tensor:
    """x, y and radius in [-1, 1] as three channels, shape (B, 3, H, W)."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([xx, yy, (xx ** 2 + yy ** 2).sqrt()]).expand(b, -1, -1, -1)


class TersTokenEncoder(nn.Module):
    """Cube (B, C, 64, 64) -> vector (B, out_dim) and 64 tokens (B, 64, 8 * width)."""

    def __init__(self, in_channels: int, out_dim: int, width: int = 32, spec_dim: int = 128,
                 n_layers: int = 2, n_heads: int = 4, cond_drop: float = 0.0) -> None:
        super().__init__()
        self.token_dim = d = 8 * width
        self.cond_drop = cond_drop
        self.spectral = nn.Sequential(
            nn.Conv2d(in_channels, spec_dim, 1), _norm(spec_dim), nn.SiLU(),
            nn.Conv2d(spec_dim, spec_dim, 1), _norm(spec_dim), nn.SiLU())
        self.stem = nn.Sequential(
            nn.Conv2d(spec_dim + 3, width, 3, padding=1, bias=False), _norm(width), nn.SiLU(),
            ResBlock(width, width, 1))
        self.stages = nn.Sequential(
            ResBlock(width, 2 * width, 2), ResBlock(2 * width, 4 * width, 2), ResBlock(4 * width, d, 2))
        self.pos = nn.Parameter(torch.zeros(1, GRID * GRID, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.tok = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (vector, tokens)."""
        h = self.spectral(x)
        h = self.stages(self.stem(torch.cat([h, coord_channels(h)], 1)))
        if h.shape[-2:] != (GRID, GRID):
            raise ValueError(f'expected a {GRID * 8}x{GRID * 8} frame, got {tuple(x.shape[-2:])}')
        t = h.flatten(2).transpose(1, 2) + self.pos
        t = self.tok(torch.cat([self.cls.expand(len(t), -1, -1), t], 1))
        vec = self.head(t[:, 0])
        if self.training and self.cond_drop > 0:
            vec = vec * (torch.rand(len(vec), 1, device=vec.device) >= self.cond_drop)
        return vec, t[:, 1:]

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)[0]
