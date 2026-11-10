"""Cross-attention from DMT node features to TERS tokens, added in front of DMT blocks."""

import threading
from functools import partial

import torch
from torch import Tensor, nn

from .encoder import TersTokenEncoder

# DataParallel runs each replica in its own thread, so a thread-local gives every
# replica its own tokens and its own (device-local) adapter copies.
_current = threading.local()


class CrossAttnAdapter(nn.Module):
    """h + tanh(gate) * attention(h -> tokens). The gate starts at 0, so it starts as identity."""

    def __init__(self, d_node: int, d_tok: int, n_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_node)
        self.attn = nn.MultiheadAttention(d_node, n_heads, kdim=d_tok, vdim=d_tok, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: Tensor, tokens: Tensor, node_mask: Tensor) -> Tensor:
        """h (B, N, d_node), tokens (B, T, d_tok), node_mask (B, N, 1)."""
        out, _ = self.attn(self.norm(h), tokens, tokens, need_weights=False)
        return h + torch.tanh(self.gate) * out * node_mask


class TersTokenXAttn(nn.Module):
    """Token encoder whose tokens feed one adapter per DMT block in `block_ids`."""

    def __init__(self, trunk: TersTokenEncoder, d_node: int, block_ids: list[int], n_heads: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.adapters = nn.ModuleDict(
            {str(i): CrossAttnAdapter(d_node, trunk.token_dim, n_heads) for i in block_ids})

    def forward(self, x: Tensor) -> Tensor:
        vec, _current.tokens = self.trunk.encode(x)
        _current.adapters = self.adapters
        return vec


def _adapt_nodes(block: nn.Module, args: tuple, block_id: str) -> tuple:
    """Forward pre-hook of an EquivariantMixBlock: args are (pos, h, edge_attr, edge_index, node_mask, ...)."""
    tokens = _current.tokens
    h, node_mask = args[1], args[4]
    b = tokens.shape[0]
    h = _current.adapters[block_id](h.reshape(b, -1, h.shape[-1]), tokens, node_mask.reshape(b, -1, 1))
    return (args[0], h.flatten(0, 1), *args[2:])


def attach(dmt: nn.Module, block_ids: list[int]) -> None:
    for i in block_ids:
        getattr(dmt, f'e_block_{i}').register_forward_pre_hook(partial(_adapt_nodes, block_id=str(i)))
