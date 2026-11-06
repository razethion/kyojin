"""Actual layers20..24: ratio1, shared KV/index and two-level candidates."""
import argparse
import json
from pathlib import Path
import time
import torch
from exllamav3.dsv41.pack import load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--result',required=True)
    a=ap.parse_args();torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.60,0);began=time.time()
    m=load(a.model,layer_span=(20,25),mtp=False,prefill_chunk=256,device='cuda:0',log=lambda s:print(s,flush=True))
    st=m.new_state(1,17408)
    gen=torch.Generator().manual_seed(20261103)
    ids=torch.randint(100,10000,(1,17100),generator=gen)
    # Deterministic external mHC inputs qualify this owned span independently
    # of any earlier trunk/Engram output. This is not a response-quality test.
    def incoming(pos,n):
        g=torch.Generator().manual_seed(20261103+pos)
        h=torch.randn(1,n,4,m.cfg.hidden_size,generator=g).float()
        pre=torch.softmax(torch.randn(1,n,4,generator=g),-1)
        return h,pre
    for p in range(0,16640,256):
        m.forward_stage(ids[:,p:p+256],p,st,incoming(p,256))
        if p%2048==0:print(json.dumps(dict(prefilled=p+256)),flush=True)
    records=[];pos=16640
    for n,keep in ((1,1),(3,1),(5,2),(6,3),(4,4),(5,1),(6,6)):
        serial=st.fork();h,pre=incoming(pos,n)
        st.begin_block(pos,n)
        out,_=m.forward_stage(ids[:,pos:pos+n],pos,st,(h,pre),all_logits=True)
        hs=[];ps=[]
        accepted_state=None
        for j in range(n):
            one,_=m.forward_stage(ids[:,pos+j:pos+j+1],pos+j,serial,(h[:,j:j+1],pre[:,j:j+1]),all_logits=True)
            hs.append(one[0]);ps.append(one[1])
            if j+1==keep:accepted_state=serial.fork()
        ref=(torch.cat(hs,1),torch.cat(ps,1))
        rec=dict(past=pos,rows=n,keep=keep,exact=all(torch.equal(x,y) for x,y in zip(out,ref)),
                 max_abs=[(x-y).abs().max().item() for x,y in zip(out,ref)])
        records.append(rec);print(json.dumps(rec),flush=True)
        Path(a.result).write_text(json.dumps(dict(records=records,complete=False),indent=2))
        if not rec['exact']:raise RuntimeError('shared/candidate span serial-vs-verify differs')
        if keep<n:st.rollback(keep)
        else:st.end_block()
        st.length=pos+keep;pos+=keep
        if not all(torch.equal(st.win[l],accepted_state.win[l]) for l in range(*m.layer_span)):
            raise RuntimeError('shared span accepted window cache differs')
        for l in st.comp_kv:
            if not torch.equal(st.comp_kv[l][:,:pos],accepted_state.comp_kv[l][:,:pos]) or not torch.equal(st.idx_k[l][:,:pos],accepted_state.idx_k[l][:,:pos]):
                raise RuntimeError('shared span accepted pool differs')
        rec['accepted_cache_exact']=True
    result=dict(pass_all=True,records=records,elapsed=time.time()-began,allocated_bytes=torch.cuda.memory_allocated(),
                candidate_pool_rows=16640,candidate_keep_rows=m.cfg.candidate_topk_blocks*m.cfg.candidate_block_size)
    Path(a.result).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
