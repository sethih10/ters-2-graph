"""Self-check of the token encoders on CPU with random inputs.

    python -m models.ters_tokens.check
"""

import os
import tempfile
import threading

os.environ.setdefault('DATA_ROOT', tempfile.gettempdir())

import torch
from ml_collections import ConfigDict
from torch import Tensor, nn

from configs.qm9s_ters_frozen import get_config as v1_config
from configs.qm9s_ters_frozen_token import get_config as token_config
from configs.qm9s_ters_frozen_token_xattn import get_config as xattn_config
from models.dmt import DMT
from models.ema import ExponentialMovingAverage
from models.ters_tokens.export import TRAINABLE, export

Inputs = tuple[tuple[Tensor, ...], dict[str, Tensor | None]]


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_inputs(config: ConfigDict, batch: int, atoms: int = 6) -> Inputs:
    node_dim = config.data.atom_types + int(config.model.include_fc_charge)
    node_mask = torch.ones(batch, atoms, 1)
    node_mask[0, -2:] = 0
    pair_mask = node_mask * node_mask.transpose(1, 2) * (1 - torch.eye(atoms))
    edge_x = torch.rand(batch, atoms, atoms, config.model.edge_ch)
    edge_x = (edge_x + edge_x.transpose(1, 2)) * pair_mask.unsqueeze(-1)
    args = (torch.rand(batch), torch.randn(batch, atoms, 3 + node_dim) * node_mask,
            node_mask, pair_mask.reshape(-1, 1))
    kwargs = dict(context=torch.randn(batch, config.model.ters_in_channels, 64, 64), edge_x=edge_x,
                  noise_level=torch.randn(batch), cond_x=None, cond_edge_x=None)
    return args, kwargs


def run(model: nn.Module, inputs: Inputs, self_cond: bool = False) -> tuple[Tensor, Tensor]:
    args, kwargs = inputs
    if self_cond:
        cond_x, cond_edge_x = model(*args, **kwargs)
        kwargs = {**kwargs, 'cond_x': cond_x, 'cond_edge_x': cond_edge_x}
    return model(*args, **kwargs)


def same(a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor]) -> bool:
    return all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b))


def token_weights(xattn: DMT) -> dict[str, Tensor]:
    """State dict of the `token` model that shares its weights with `xattn`."""
    return {k.replace('cond_encoder.trunk.', 'cond_encoder.'): v
            for k, v in xattn.state_dict().items() if '.adapters.' not in k}


def set_gates(model: DMT, value: float) -> None:
    for adapter in model.cond_encoder.adapters.values():
        nn.init.constant_(adapter.gate, value)


def check_default_path() -> None:
    model = DMT(v1_config())
    assert type(model.cond_encoder).__name__ == 'TersEncoder'


def check_token_encoder() -> None:
    config = token_config()
    encoder = DMT(config).cond_encoder
    vec, tokens = encoder.encode(torch.randn(2, 100, 64, 64))
    assert vec.shape == (2, config.model.nf) and tokens.shape == (2, 64, 256)
    assert n_params(encoder) == 2_955_680
    try:
        encoder(torch.randn(2, 100, 32, 32))
    except ValueError:
        return
    raise AssertionError('a 32 px frame must be rejected')


def check_identity_at_init() -> None:
    xattn, token = DMT(xattn_config()).eval(), DMT(token_config()).eval()
    token.load_state_dict(token_weights(xattn))
    assert n_params(xattn.cond_encoder.adapters) == 2_109_448
    inputs = make_inputs(xattn_config(), batch=3)
    with torch.no_grad():
        for self_cond in (False, True):
            assert same(run(xattn, inputs, self_cond), run(token, inputs, self_cond))
        set_gates(xattn, 0.5)
        assert not same(run(xattn, inputs), run(token, inputs))


def check_padding_and_block_selection() -> None:
    config = xattn_config()
    config.model.ters_adapter_every_k = 2
    model = DMT(config)
    assert list(model.cond_encoder.adapters) == ['0', '2', '4', '6']
    set_gates(model, 1.0)
    adapter = model.cond_encoder.adapters['0']
    h, mask = torch.randn(2, 5, 256), torch.ones(2, 5, 1)
    mask[0, 3:] = 0
    out = adapter(h, torch.randn(2, 64, 256), mask)
    assert torch.equal(out[0, 3:], h[0, 3:]) and not torch.equal(out[0, :3], h[0, :3])


def check_gradients_and_names() -> None:
    config = xattn_config()
    model, token = DMT(config), DMT(token_config())
    new = set(model.state_dict()) - set(token.state_dict())
    assert all(k.startswith('cond_encoder.') for k in new)
    pred, edge_pred = run(model, make_inputs(config, batch=2))
    (pred.square().mean() + edge_pred.square().mean()).backward()
    assert all(a.gate.grad.abs().item() > 0 for a in model.cond_encoder.adapters.values())


def check_threads_keep_their_own_tokens() -> None:
    config = xattn_config()
    model = DMT(config).eval()
    set_gates(model, 0.5)
    batches = [make_inputs(config, batch=b) for b in (2, 3, 4)]
    with torch.no_grad():
        expected = [run(model, inputs) for inputs in batches]
        got: list = [None] * len(batches)

        def work(i: int) -> None:
            for _ in range(3):
                got[i] = run(model, batches[i])

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(batches))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert all(g is not None and same(g, e) for g, e in zip(got, expected))


def check_warm_start_and_reload() -> None:
    source = DMT(token_config())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'checkpoint.pth')
        torch.save({'model': {f'module.{k}': v for k, v in source.state_dict().items()}}, path)
        config = xattn_config()
        config.model.ters_encoder_init = path
        model = DMT(config)
    trunk = model.cond_encoder.trunk.state_dict()
    assert all(torch.equal(v, trunk[k]) for k, v in source.cond_encoder.state_dict().items())
    DMT(xattn_config()).load_state_dict(model.state_dict(), strict=True)


def check_export_gives_a_standard_checkpoint() -> None:
    frozen = DMT(xattn_config())
    for name, p in frozen.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE))
    ema = ExponentialMovingAverage(frozen.parameters(), decay=0.999)
    for s in ema.shadow_params:
        s.add_(1.0)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'frozen.pth'), os.path.join(tmp, 'standard.pth')
        torch.save({'model': {f'module.{k}': v for k, v in frozen.state_dict().items()},
                    'ema': ema.state_dict(), 'step': 7}, src)
        export(src, dst)
        ckpt = torch.load(dst, weights_only=False)
    model = DMT(xattn_config())
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt['model'].items()}, strict=True)
    standard_ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    standard_ema.load_state_dict(ckpt['ema'])
    standard_ema.copy_to(model.parameters())
    for (name, got), want in zip(model.named_parameters(), frozen.parameters()):
        assert torch.equal(got, want + 1.0 if name.startswith(TRAINABLE) else want), name


def main() -> None:
    torch.manual_seed(0)
    checks = [check_default_path, check_token_encoder, check_identity_at_init,
              check_padding_and_block_selection, check_gradients_and_names,
              check_threads_keep_their_own_tokens, check_warm_start_and_reload,
              check_export_gives_a_standard_checkpoint]
    for check in checks:
        check()
        print(f'ok  {check.__name__}')


if __name__ == '__main__':
    main()
