"""`infra/modal/living_upscale.py`'s stand-ins, against what they stand in for: FlashVSR's
block-sparse attention in PyTorch against dense attention masked token by token, and the
torch norms in place of apex's against the maths apex computes. Loaded with a fake `modal`
(only what the module touches at import)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

UPSCALE = Path(__file__).resolve().parents[3] / "infra" / "modal" / "living_upscale.py"


class _Chain:
    """Any attribute, called with anything, returns itself (the image builder's chain)."""

    def __getattr__(self, name: str) -> object:
        return lambda *args, **kwargs: self


def _fake_modal() -> types.ModuleType:
    fake = types.ModuleType("modal")
    decorator = lambda **kwargs: lambda f: f
    fake.App = lambda name: types.SimpleNamespace(  # type: ignore[attr-defined]
        function=decorator, local_entrypoint=decorator
    )
    fake.is_local = lambda: True  # type: ignore[attr-defined]
    fake.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())  # type: ignore[attr-defined]
    fake.Secret = types.SimpleNamespace(from_name=lambda *a, **k: object())  # type: ignore[attr-defined]
    fake.Image = _Chain()  # type: ignore[attr-defined]
    return fake


@pytest.fixture()
def upscale(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setitem(sys.modules, "modal", _fake_modal())
    spec = importlib.util.spec_from_file_location("living_upscale_under_test", UPSCALE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    for name in ("apex", "apex.normalization"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("lq", "lk", "gather"),
    [(128 * 5, 128 * 7, 1 << 28), (128 * 4 + 50, 128 * 6 + 17, 1 << 14)],
)
def test_block_sparse_attention_is_dense_attention_on_the_mask(
    upscale: types.ModuleType, lq: int, lk: int, gather: int
) -> None:
    torch.manual_seed(0)
    heads, dim, blk = 3, 16, 128
    nq, nk = -(-lq // blk), -(-lk // blk)
    q, k, v = (torch.randn(n, heads, dim) for n in (lq, lk, lk))
    mask = torch.rand(1, heads, nq, nk) < 0.4
    mask[..., 0] = True
    upscale.GATHER_BYTES = gather  # small: the query blocks go through in several passes
    got = upscale.block_sparse_attn_func(q, k, v, None, None, None, None, mask, lq, lk, 0.0)
    token = mask[0].repeat_interleave(blk, 1).repeat_interleave(blk, 2)[:, :lq, :lk]
    want = F.scaled_dot_product_attention(
        q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2), attn_mask=token
    ).permute(1, 0, 2)
    assert got.shape == q.shape
    assert torch.allclose(got, want, atol=1e-5)


def test_a_query_block_that_selects_nothing_reads_nothing(upscale: types.ModuleType) -> None:
    q, k, v = (torch.randn(256, 2, 8) for _ in range(3))
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 1, 1] = False  # head 1, query block 1: no key block
    got = upscale.block_sparse_attn_func(q, k, v, None, None, None, None, mask, 256, 256, 0.0)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[128:, 1]) == 0


def test_apex_norms_are_apex_maths_with_loadable_parameters(upscale: types.ModuleType) -> None:
    upscale._apex_norms()
    from apex.normalization import FusedLayerNorm, FusedRMSNorm

    rms = FusedRMSNorm(normalized_shape=8, elementwise_affine=True, eps=1e-5)
    assert list(rms.state_dict()) == ["weight"]
    with torch.no_grad():
        rms.weight.copy_(torch.linspace(0.5, 2.0, 8))
    x = torch.randn(4, 8).to(torch.bfloat16)
    y = rms(x)
    assert y.dtype == torch.bfloat16
    xf = x.float()
    want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * rms.weight
    assert torch.allclose(y.float(), want, atol=2e-2)

    ln = FusedLayerNorm(normalized_shape=8, elementwise_affine=True, eps=1e-5)
    assert sorted(ln.state_dict()) == ["bias", "weight"]
    assert FusedRMSNorm(8, elementwise_affine=False).weight is None
