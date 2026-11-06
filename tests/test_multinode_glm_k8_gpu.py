"""Real GLM K8 attention: bounds, row parity, independent FP64 error, A/B speed."""
import argparse,gc,json,math,os,statistics
from pathlib import Path
ap=argparse.ArgumentParser()
for n in ('model','out'):ap.add_argument('--'+n,required=True)
args=ap.parse_args()
os.environ.update(EXL3_DENSE_K8='1',EXL3_VERIFY_FUSE='1',EXL3_VERIFY_FUSE_PAIR='0',EXL3_GEMV_R_DEC1='1')
import torch
from exllamav3 import Config,Model
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant import exl3 as q
torch.set_grad_enabled(False);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.8,0)
cfg=Config.from_directory(args.model);models=[Model.from_config(cfg),Model.from_config(cfg,component='mtp')]
shapes={(4096,24576),(8192,4096),(4096,1536),(1536,16384),(4096,512),(16384,4096),(1536,4096)}
selected={}
for model in models:
    for m in model:
        if isinstance(m,Linear) and '.self_attn.' in m.key and (m.in_features,m.out_features) in shapes:
            selected.setdefault((m.in_features,m.out_features),m)
assert set(selected)==shapes
dev=torch.device('cuda:0')
scratch=torch.full((q.DEC_SCRATCH_FLOATS+256,),12345.,device=dev)
counters=torch.full((q.DEC_COUNTERS+256,),12345,dtype=torch.int32,device=dev)
scratch[128:-128].zero_();counters[128:-128].zero_()
q._dec_workspaces[str(dev)]=(scratch[128:-128],counters[128:-128])
def guards():
    assert (scratch[:128]==12345).all() and (scratch[-128:]==12345).all()
    assert (counters[:128]==12345).all() and (counters[-128:]==12345).all()
    assert counters[128:-128].count_nonzero()==0
def had(x):
    shape=x.shape;y=x.reshape(-1,128)
    for size in (1,2,4,8,16,32,64):
        p=y.reshape(-1,128//(2*size),2,size)
        a,b=p[:,:,0,:],p[:,:,1,:]
        y=torch.stack((a+b,a-b),dim=2).reshape(-1,128)
    return y.reshape(shape)/math.sqrt(128)
records=[];flush=torch.zeros(64<<20,device=dev)
try:
    for shape,m in sorted(selected.items()):
        m.load(dev)
        try:
            inner=m.inner
            assert inner.K==8 and inner.mul1 and not inner.mcg and inner.dec_ok
            w=inner.get_inner_weight_tensor().double()
            quality=[]
            for seed in (1234,5678):
                gen=torch.Generator(device=dev).manual_seed(seed)
                x=(torch.randn(4,shape[0],generator=gen,device=dev)*(1 if seed==1234 else 8)).half()
                x[1]=x[0];x[3]=x[2];saved=x.clone()
                gold=had((had(x.double()*inner.suh.double())@w))*inner.svh.double()
                for rows in (1,2,3,4):
                    for dtype in (torch.float16,torch.float32):
                        old=inner.forward(x[:rows],{'reconstruct':True},dtype).clone()
                        got=inner.forward(x[:rows],{'dflash_verify':True},dtype).clone()
                        repeat=inner.forward(x[:rows],{'dflash_verify':True},dtype).clone()
                        serial=torch.cat([inner.forward(x[i:i+1],{},dtype) for i in range(rows)],dim=0)
                        torch.cuda.synchronize();guards()
                        assert torch.equal(got,repeat) and torch.equal(got,serial),'K8 row/repeat parity'
                        assert torch.equal(x,saved) and torch.isfinite(got).all()
                        reference=gold[:rows];den=reference.norm().clamp_min(1e-30)
                        old_error=float((old.double()-reference).norm()/den)
                        new_error=float((got.double()-reference).norm()/den)
                        assert new_error<.002 and new_error<=old_error*1.25+1e-6,(m.key,old_error,new_error)
                        quality.append(dict(rows=rows,seed=seed,dtype=str(dtype),old_fp64_rel=old_error,
                                            new_fp64_rel=new_error,serial_exact=True))
            del w
            timing=[]
            for rows in (1,3):
                sample=x[:rows];values={False:[],True:[]}
                for candidate in (False,True):
                    for _ in range(3):inner.forward(sample,{'dflash_verify':True} if candidate else {'reconstruct':True},torch.float32)
                for iteration in range(7):
                    for candidate in ((False,True) if iteration%2==0 else (True,False)):
                        flush.add_(1);first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                        first.record();inner.forward(sample,{'dflash_verify':True} if candidate else {'reconstruct':True},torch.float32)
                        last.record();last.synchronize();values[candidate].append(first.elapsed_time(last))
                old_ms,new_ms=(statistics.median(values[k]) for k in (False,True))
                timing.append(dict(rows=rows,old_ms=old_ms,new_ms=new_ms,speedup=old_ms/new_ms))
            record=dict(key=m.key,shape=shape,quality=quality,timing=timing)
            records.append(record)
            print(json.dumps(dict(key=m.key,shape=shape,comparisons=len(quality),
                worst_new=max(v['new_fp64_rel'] for v in quality),timing=timing)),flush=True)
            Path(args.out).write_text(json.dumps(dict(complete=False,records=records),indent=2)+'\n')
        finally:
            m.unload();gc.collect();torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(dict(complete=True,records=records),indent=2)+'\n')
finally:
    guards()
