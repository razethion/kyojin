"""Independent matrix references and NaN guard storage for MLA tile tails."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
parser.add_argument('--expect-broken', action='store_true')
args = parser.parse_args()
os.environ['EXL3_MLA_DEC_VALU'] = '0'
os.environ['EXL3_MLA_PF_FAST'] = '0'
import torch
from exllamav3.modules.attention_fn.mla_triton import mla_absorb, mla_unfold
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.manual_seed(413)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
cases = [(64, 64, 64, 17)] if args.expect_broken else [
    (latent, nope, value, rows) for latent, nope, value in ((64,64,64),(96,64,128),(128,128,128),(512,128,128))
    for rows in (17,250)]
for latent, nope, value, rows in cases:
    heads = 4
    q = (torch.randn(rows, heads, nope, device='cuda:0') * .1).half()
    padded = ((latent + 127)//128+1)*128
    uk_backing = torch.full((padded, heads*nope), float('nan'), dtype=torch.float16, device='cuda:0')
    uk = uk_backing[:latent]; uk.copy_((torch.randn_like(uk)*.1).half())
    absorbed = mla_absorb(q, uk, heads, nope)
    gold_a = torch.stack([q[:,h].float() @ uk[:,h*nope:(h+1)*nope].float().T for h in range(heads)]).half()
    latent_backing = torch.full((heads*rows*latent+128,), float('nan'), dtype=torch.float16, device='cuda:0')
    latent_values = latent_backing[:heads*rows*latent].view(heads,rows,latent)
    latent_values.copy_((torch.randn_like(latent_values)*.1).half())
    uv_backing = torch.full((padded, heads*value), float('nan'), dtype=torch.float16, device='cuda:0')
    uv = uv_backing[:latent]; uv.copy_((torch.randn_like(uv)*.1).half())
    unfolded = mla_unfold(latent_values, uv, value)
    gold_u = torch.stack([latent_values[h].float() @ uv[:,h*value:(h+1)*value].float() for h in range(heads)],dim=1).half()
    finite_a, finite_u = bool(torch.isfinite(absorbed).all()), bool(torch.isfinite(unfolded).all())
    close_a = finite_a and bool(torch.allclose(absorbed, gold_a, rtol=.002, atol=.0002))
    close_u = finite_u and bool(torch.allclose(unfolded, gold_u, rtol=.002, atol=.0002))
    record = dict(latent=latent,nope=nope,value=value,rows=rows,absorb_finite=finite_a,
                  unfold_finite=finite_u,absorb_close=close_a,unfold_close=close_u)
    records.append(record); print(json.dumps(record),flush=True)
    if args.expect_broken: assert not(close_a and close_u), 'expected original tile bug did not reproduce'
    else:
        torch.testing.assert_close(absorbed,gold_a,rtol=.002,atol=.0002)
        torch.testing.assert_close(unfolded,gold_u,rtol=.002,atol=.0002)
Path(args.out).write_text(json.dumps(dict(passed=True,expect_broken=args.expect_broken,records=records),indent=2)+'\n')
