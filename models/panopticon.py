# Adapted from https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/models/panopticon.py
# Changed: PanopticonPE is renamed SpmPatchEmbed, and ChnEmb learns one row per frequency bin.
import torch
import torch.nn as nn
from torch import Tensor


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class SpmPatchEmbed(nn.Module):
    """
    General class defining a patch embedding that takes in arbitrary channel
    dimension and outputs a fixed dimension embedding. This class handles
    the tokenization and projections. The attributed self.chnfus handles
    the channel fusion, i.e. the cross attention over channels.
    """

    def __init__(
            self,
            attn_dim: int,
            embed_dim: int,
            patch_size: int,
            chnfus_cfg: dict = {},
        ):
        super().__init__()

        self.conv3d = Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus = ChnAttn(**chnfus_cfg, dim=attn_dim)
        self.proj = nn.Linear(attn_dim, embed_dim)

    def forward(self, x_dict: dict) -> Tensor:
        x = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        w,h = x.shape[-2:]
        mask = x_dict.get('spec_masks', None)

        x = self.conv3d(x)
        x = self.chnfus(x, chn_ids=chn_ids, mask=mask) # B,L,D
        x = self.proj(x)

        return x, h, w


class Conv3dWrapper(nn.Module):
    """ Channel-wise patchification and projection, essentially wrapper around
        1 x P x P conv3d """
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        patch_size = make_2tuple(patch_size)
        patch_CHW = (1, *patch_size)
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)

    def forward(self, x: Tensor):
        x = self.conv3d(x.unsqueeze(1)).squeeze(1) # B D C Hp Wp
        return x.flatten(-2).permute(0, 2, 3, 1)     # B C L D


class ChnAttn(nn.Module):
    """
    Cross attention over channels with channel embeddings to reduce any number
    of channels to a fixed dimension. Inspired by
        https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/arch.py#L185
    """

    def __init__(self,
            dim: int,
            chnemb_cfg: dict = {},
            attn_cfg: dict = {},
            layer_norm: bool = False,
        ):
        """
        Args:
            dim (int): Dimension of the channel attention.
            chnemb_cfg (dict): Key-value pairs for the channel embedding.
            attn_cfg (dict): Key-value pairs for the channel attention.
            layer_norm (bool, optional): Whether to apply layer norm after
                channel attention. Defaults to False.
        """
        super().__init__()

        self.chnemb = ChnEmb(**chnemb_cfg, embed_dim=dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn = CrossAttnNoQueryProj(dim=dim, **attn_cfg)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, chn_ids: Tensor, mask: Tensor = None) -> Tensor:
        """
        Args:
            x (Tensor): Image tensor of shape (B, C, L, D)
            chn_ids (Tensor): Channel IDs tensor of shape (B,C) or (B,C,2) if
                stds of the SRFs curves are included, see ChnEmb.
            mask (Tensor, optional): Mask tensor of shape (B,C) indicating
                which channels have been masked out. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, L, D) independent of the input
                channel dimension C.
        """
        B, C, L, D = x.shape

        # add embeddings
        chn_embs = self.chnemb(chn_ids) # B,C,D
        x += chn_embs.unsqueeze(2)

        # abstract away L
        x = x.permute(0, 2, 1, 3).flatten(0, 1) # BL,C,D
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, L, -1).flatten(0, 1) # BL,C

        query = self.query.expand(x.shape[0], -1, -1) # BL,1,D
        assert query.shape == (x.shape[0], 1, x.shape[-1]), \
            f'Expected query to have shape: {x.shape[0], 1, x.shape[-1]}, but got shape: {query.shape}'

        x = self.xattn(query, x, x, key_padding_mask=mask)
        x = x.reshape(B, L, D)

        if hasattr(self, 'layer_norm'):
            return self.layer_norm(x)

        return x


# CHANGED: one learned embedding per frequency bin instead of a wavelength code. chn_ids are
# wavenumbers in cm^-1; a mode's bin is wavenumber // bin_width.
class ChnEmb(torch.nn.Module):

    def __init__(self, embed_dim: int, n_bins: int = 100, bin_width: float = 40.0, init_std: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_bins = n_bins
        self.bin_width = bin_width
        self.embed = nn.Embedding(n_bins, embed_dim)
        nn.init.normal_(self.embed.weight, std=init_std)

    def forward(self, input: Tensor) -> Tensor:
        bins = (input / self.bin_width).long().clamp(max=self.n_bins - 1)
        return self.embed(bins)                         # (B, C) wavenumbers -> (B, C, D)


class CrossAttnNoQueryProj(nn.Module):
    """ Cross Attention without query projection and final projection

    Comment: While doing the final refactor before the release, we noticed that
        we project from patches to 2304 with the conv3d and then again have the
        key & value projections from 2304 to 2304 without non-linearity in-between.
        Hence, the key & value projections are redundant and could be removed,
        significantly reducing the number of parameters. However, this is how
        the paper results were generated and we keep it for reproducibility.
        If you plan on further developing panopticon, please remove the key & value
        projections!
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
        ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.scale = head_dim ** -0.5

        self.inproj_q = nn.Identity() # no projection since query is a parameter itself
        self.inproj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask=None):
        """ q: (B, Nq, D), kv: (B, Nkv, D), key_padding_mask: (B, Nkv)"""

        B, Nq, D = q.shape
        q = self.inproj_q(q).reshape(B, Nq, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        B, Nkv, D = k.shape # same as v.shape
        k = self.inproj_k(k).reshape(B, Nkv, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = self.inproj_v(v).reshape(B, Nkv, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)  # shape: (B, num_heads, Nq, Nkv)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nkv)
            attn = attn.masked_fill(key_padding_mask, float("-inf"))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return x


