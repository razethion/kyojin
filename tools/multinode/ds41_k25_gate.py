"""Real source-20 uniform K2.5 routed experts: exact outputs and launch timing."""
import argparse,json,time,os
from pathlib import Path
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--result',required=True);args=ap.parse_args()
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.60,0)
    from exllamav3.dsv41.pack import load
    from exllamav3.modules.dsv41_fn.dsv41_moe import UnifiedMoE
    model=load(args.model,layer_span=(20,21),mtp=False,device='cuda:0',prefill_chunk=256,log=lambda s:print(s,flush=True))
    be=model.moe_backends[20]
    assert len(be.store.groups)==1 and be.store.groups[0].k_gu==2.5 and be.uni is None
    candidate=UnifiedMoE(be.store,be.buffers,be.outputs[:1],be.H,be.I,float(be.dcfg.swiglu_limit),be.device)
    records=[];gen=torch.Generator().manual_seed(20261102)
    for i in range(12):
        x=(torch.randn(1,be.H,generator=gen)*(.2 if i%2 else 1)).half().cuda()
        sel=torch.randperm(384,generator=gen)[:6][None].cuda()
        wh=torch.rand(1,6,generator=gen).half().cuda();wh=wh/wh.sum()
        gsel=be.store.lut[:,sel[0]].reshape(1,1,6)
        native=be.decode_parts(x,sel,wh,gsel)[0].clone()
        new=candidate.run(x,sel,wh)[0][:1].clone()
        assert native.shape==new.shape
        same=torch.equal(native,new);err=float((native-new).abs().max())
        records.append(dict(case=i,exact=same,max_abs=err))
        assert same,records[-1]
    def timing(fn):
        for _ in range(8):fn()
        torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();a.record()
        for _ in range(100):fn()
        b.record();b.synchronize()
        return dict(gpu_ms=a.elapsed_time(b)/100,wall_ms=(time.perf_counter()-start)*10)
    old=timing(lambda:be.decode_parts(x,sel,wh,gsel));new=timing(lambda:candidate.run(x,sel,wh))
    result=dict(pass_all=True,cases=records,native=old,candidate=new,speedup=old['wall_ms']/new['wall_ms'])
    Path(args.result).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
