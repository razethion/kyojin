"""Exact differential qualification of dual-pointer versus concatenated key storage."""
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HIP GPU")
@pytest.mark.parametrize("nw,npool", [(0, 65), (33, 0), (31, 67), (513, 4097)])
def test_dual_pool_bitexact(nw, npool):
    from exllamav3.modules.dsv41_fn import ds41_pf as pf
    gen = torch.Generator(device="cuda").manual_seed(20261103)
    q = torch.randn(1, 17, 64, 512, device="cuda", generator=gen).half()
    kv = torch.randn(1, nw, 512, device="cuda", generator=gen).half()
    pool = torch.randn(1, npool, 512, device="cuda", generator=gen).half()
    sink = torch.randn(64, device="cuda", generator=gen)
    idx = torch.randint(-1, nw+npool+7, (1, 17, 65), device="cuda", generator=gen, dtype=torch.int32)
    idx[:, 0] = -1
    idx[:, 1, :4] = torch.tensor([-1, max(0,nw-1), nw, nw+npool+3], device="cuda")
    cat = pf.sparse_attention(q, torch.cat([kv, pool], dim=1), sink, idx, 512**-0.5)
    dual = pf.sparse_attention(q, kv, sink, idx, 512**-0.5, pool)
    assert torch.equal(cat, dual)
    assert torch.isfinite(dual).all()
