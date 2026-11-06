"""Opt-in GPU regression for padded skinny rows; exact integer-valued oracle.

NaN-sentinel outputs catch missing column writes independently of allocator
contents. Compare both launch modes and W2 policies over padded row classes.
"""
import json
import torch
from exllamav3.ext import exllamav3_ext as ext

def main():
    torch.manual_seed(923)
    cases=0
    try:
        for rows in (1,7,8,9,12,13,16,17,24,25,32):
            for width in (64,96,128):
                for inner in (129,256):
                    x=torch.randint(-2,3,(rows,inner),device='cuda',dtype=torch.int32).half()
                    w=torch.randint(-2,3,(inner,width),device='cuda',dtype=torch.int32).half()
                    expected=x.float()@w.float()
                    for dtype in (torch.float16,torch.float32):
                        for onepass in (0,1):
                            for w2 in (0,1):
                                ext.skinny_set_onepass(onepass)
                                ext.skinny_set_w2(w2)
                                output=torch.full((rows,width),float('nan'),device='cuda',dtype=dtype)
                                ext.hgemm(x,w,output)
                                if not torch.equal(output.float(),expected):
                                    bad=(output.float()!=expected)
                                    raise AssertionError(json.dumps(dict(rows=rows,n=width,k=inner,dtype=str(dtype),onepass=onepass,w2=w2,
                                                                        nonfinite=int((~torch.isfinite(output)).sum()),bad=int(bad.sum()))))
                                cases+=1
    finally:
        ext.skinny_set_onepass(1)
        ext.skinny_set_w2(1)
    print(json.dumps({'test':'skinny-padded-exact-oracle','passed':cases,'torch':torch.__version__}),flush=True)

if __name__=='__main__':main()
