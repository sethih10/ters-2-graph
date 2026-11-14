"""TERS encoder: mode maps and their wavenumbers -> SpmPatchEmbed tokens -> timm ViT -> (B, output_dim).

Usage:
    encoder = SpmEncoder.from_config(config, output_dim=256)
    z = encoder(dict(imgs=imgs, chn_ids=chn_ids, spec_masks=spec_masks))
    # imgs (B, M, 64, 64) mode maps, chn_ids (B, M) wavenumbers in cm^-1, spec_masks (B, M) True on
    # padding: a batch of datasets.augmentations.ChnPad outputs.
"""
import timm
from torch import nn

from .spm_patch_embed import SpmPatchEmbed

VIT_MODELS = {'tiny': 'vit_tiny_patch16_224', 'small': 'vit_small_patch16_224'}


class SpmEncoder(nn.Module):
    """Mode maps (B, M, img_size, img_size) with their wavenumbers -> (B, output_dim).

    SpmPatchEmbed replaces the timm ViT's patch embedding, as in torchgeo (torchgeo/models/panopticon.py).
    """

    @classmethod
    def from_config(cls, config, output_dim):
        m = config.model
        return cls(output_dim=output_dim,
                   vit=m.ters_vit,
                   patch_size=getattr(m, 'ters_mode_patch', 4),
                   chn_std=getattr(m, 'ters_chn_std', 0.25))

    def __init__(self, output_dim=256, vit='tiny', img_size=64, patch_size=4, n_bins=100,
                 attn_dim=256, attn_heads=8, chn_std=0.25):
        super().__init__()
        if vit not in VIT_MODELS:
            raise ValueError(f'vit must be one of {sorted(VIT_MODELS)}, got {vit!r}')
        self.img_size = img_size
        self.vit = timm.create_model(VIT_MODELS[vit], pretrained=False, img_size=img_size,
                                     patch_size=patch_size, num_classes=0,
                                     class_token=False, global_pool='avg')
        self.vit.patch_embed = nn.Identity()  # set after create_model so timm's init skips SpmPatchEmbed
        self.pe = SpmPatchEmbed(attn_dim=attn_dim, embed_dim=self.vit.embed_dim, patch_size=patch_size,
                                chnfus_cfg=dict(chnemb_cfg=dict(n_bins=n_bins, init_std=chn_std),
                                                attn_cfg=dict(num_heads=attn_heads)))
        self.head = nn.Linear(self.vit.embed_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, context):
        size = tuple(context['imgs'].shape[-2:])
        if size != (self.img_size, self.img_size):
            raise ValueError(f'expected {self.img_size} x {self.img_size} maps, got {size}')
        tokens, _, _ = self.pe(context)
        return self.out_norm(self.head(self.vit(tokens)))
