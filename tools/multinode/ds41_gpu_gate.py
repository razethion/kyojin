"""Actual checkpoint first-four-block native/full versus cut-at-source HIP gate."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from exllamav3.dsv41.pack import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--result',required=True)
    ap.add_argument('--skip-engram',action='store_true',help='attention/MoE slice gate while original ngram is downloading')
    args=ap.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.60,0)
    start=time.time()
    from exllamav3.dsv41.engram import MemoryEngramStore
    model=load(args.model,layers=4,mtp=False,device='cuda:0',prefill_chunk=256,log=lambda s:print(s,flush=True),
               engram_store=MemoryEngramStore([],[]) if args.skip_engram else None)
    if args.skip_engram:
        model.engram_layout=None;model.engram_store=None
        for w in model.lw.values():w.engram=None
    a,b=copy.copy(model),copy.copy(model)
    a.layer_span,b.layer_span=(0,2),(2,4)
    a.lw={l:w for l,w in model.lw.items() if l<2};b.lw={l:w for l,w in model.lw.items() if l>=2}
    b.engram_layout=None;b.engram_store=None
    sa,sb,sf=[m.new_state(1,3072) for m in (a,b,model)]
    ids=torch.randint(100,10000,(1,2700),generator=torch.Generator().manual_seed(20261102))
    records=[]
    def forward(pos,n,journal=False,keep=None):
        if journal:
            for st in (sa,sb,sf):st.begin_block(pos,n)
        x=ids[:,pos:pos+n]
        serial_state=sf.fork() if journal else None
        lf,_=model.forward_stage(x,pos,sf,all_logits=True)
        carry,_=a.forward_stage(x,pos,sa,all_logits=True)
        lb,_=b.forward_stage(x,pos,sb,carry,all_logits=True)
        torch.cuda.synchronize()
        err=(lf-lb).abs().max().item();same=torch.equal(lf,lb)
        rec=dict(past=pos,rows=n,keep=keep,max_abs=err,exact=same,argmax_equal=torch.equal(lf.argmax(-1),lb.argmax(-1)))
        records.append(rec);print(json.dumps(rec),flush=True)
        if journal:
            serial=[]
            accepted_state=None
            for j in range(n):
                logits,_=model.forward_stage(x[:,j:j+1],pos+j,serial_state,all_logits=True)
                serial.append(logits)
                if j+1==keep:accepted_state=serial_state.fork()
            serial=torch.cat(serial,dim=1)
            rec.update(serial_argmax_equal=torch.equal(lf.argmax(-1),serial.argmax(-1)),
                       serial_max_abs=(lf-serial).abs().max().item(),serial_exact=torch.equal(lf,serial))
            print(json.dumps(dict(serial_diagnostic=rec)),flush=True)
        Path(args.result).write_text(json.dumps(dict(records=records,elapsed=time.time()-start),indent=2))
        if not same:raise RuntimeError('native source-aligned split logits differ')
        if journal:
            for stage,st in zip((a,b,model),(sa,sb,sf)):
                if keep<n:st.rollback(keep)
                else:st.end_block()
                st.length=pos+keep
                if not all(torch.equal(st.win[l],accepted_state.win[l]) for l in range(*stage.layer_span)):
                    raise RuntimeError('accepted-prefix window cache differs from serial')
                for l in st.comp_kv:
                    end=(pos+keep)//stage.cfg.ratio(l)
                    if not torch.equal(st.comp_kv[l][:,:end],accepted_state.comp_kv[l][:,:end]):
                        raise RuntimeError('accepted compressed pool differs from serial')
                    if not torch.equal(st.idx_k[l][:,:end],accepted_state.idx_k[l][:,:end]):
                        raise RuntimeError('accepted index pool differs from serial')
                for l in st.comp_kv_state:
                    if not torch.equal(st.comp_kv_state[l],accepted_state.comp_kv_state[l]) or not torch.equal(st.comp_score_state[l],accepted_state.comp_score_state[l]):
                        raise RuntimeError('accepted open compressor group differs from serial')
            rec['accepted_cache_exact']=True
    for p in range(0,2048,256):forward(p,256)
    pos=2048
    for rows,keep in ((1,1),(3,1),(5,2),(6,3),(4,4),(5,1),(6,6)):
        forward(pos,rows,True,keep);pos+=keep
    # HIP materialization/collapse uses the very same hc_cn body on both sides.
    from exllamav3.dsv41.serve_ops import HipHC
    from exllamav3.dsv41.config import DSV41Config
    cfg=model.cfg;hc=HipHC(cfg,torch.float16)
    for rows in (1,3,6,256):
        h=torch.randn(1,rows,4,cfg.hidden_size,device='cuda',dtype=torch.float32)
        y=torch.randn(1,rows,cfg.hidden_size,device='cuda',dtype=torch.float16)
        pre=torch.randn(1,rows,4,device='cuda');post=torch.randn_like(pre)
        comb=torch.randn(1,rows,4,4,device='cuda');w=torch.ones(cfg.hidden_size,device='cuda')
        x=hc.new(h.clone());z=hc.new(h.clone());x.push(y,post,comb);z.push(y,post,comb)
        expected=x.cn(pre,w);incoming=z.apply().view(1,rows,4,-1).clone();actual=hc.new(incoming).cn(pre,w)
        if not torch.equal(expected,actual):raise RuntimeError('HIP mHC stage boundary differs')
    result=dict(pass_all=True,records=records,hip_boundary_rows=[1,3,6,256],elapsed=time.time()-start,
                serial_all_exact=all(r.get('serial_exact',True) for r in records),
                serial_all_argmax_equal=all(r.get('serial_argmax_equal',True) for r in records),
                allocated_bytes=torch.cuda.memory_allocated(),engram_omitted=args.skip_engram)
    Path(args.result).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
