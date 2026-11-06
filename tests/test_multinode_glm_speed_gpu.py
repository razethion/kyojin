"""One-load exact-output gate and latency A/B for GLM device routing/graphs."""
import argparse,json,os,time,statistics
from pathlib import Path
ap=argparse.ArgumentParser()
for n in ('model','out'):ap.add_argument('--'+n,required=True)
ap.add_argument('--k8-graphs',action='store_true')
args=ap.parse_args()
os.environ.update(EXL3_KVWIN='0',EXL3_BLOCK_GRAPH='0',EXL3_VERIFY_FUSE='1',EXL3_VERIFY_FUSE_PAIR='0')
if args.k8_graphs:os.environ['EXL3_DENSE_K8']='1'
import torch
from exllamav3 import Config,Model
from tools.multinode.speculative import SpeculativeGlmStage
from exllamav3.modules import block_graph as bg
torch.set_grad_enabled(False);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.85,0);torch.manual_seed(7341)
cfg=Config.from_directory(args.model);model=Model.from_config(cfg)
stage=SpeculativeGlmStage(model,0,model.first_block_idx+8,speculation_depth=2,
    device='cuda:0',model_id='verified-sc4.10',source_id='speedgate',
    max_rows=256,context=512,sessions=1,cache_bits=8)
stage.load();records=[];oracle={};components=[];trace_components=False
originals=[]
for block in model.modules:
    for name in ('attn','mlp'):
        part=getattr(block,name,None)
        if part is None:continue
        fn=part.forward;originals.append((part,fn))
        def timed(*a,_fn=fn,_key=block.key+'.'+name,_type=type(part).__name__,**kw):
            if not trace_components:return _fn(*a,**kw)
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
            first.record();value=_fn(*a,**kw);last.record()
            components.append((_key,_type,first,last));return value
        part.forward=timed
prefix=torch.randint(200,500,(1,64))
inputs={w:[torch.randint(200,500,(1,w)) for _ in range(24)] for w in (1,3)}
def forward(ids,state,history):
    params=dict(attn_mode='flash_attn',cache=stage.cache,last_tokens_only=ids.shape[1],
        recurrent_states=[state],block_table=torch.arange(2,dtype=torch.int32).view(1,2),
        cache_seqlens=torch.tensor([state.position],dtype=torch.int32),
        recurrent_history=history,dflash_verify=history)
    output=model.forward(ids,params)
    pending=params.pop('hc_pending',None)
    if pending is not None:stage.flush_pending(output,pending)
    return output
try:
    for variant in (('verify','device','graphs') if args.k8_graphs else ('verify','decode1','rm3','glue')):
        bg.purge();bg.BLOCK_GRAPH_ENABLED=variant=='graphs'
        bg.BLOCK_GRAPH_MLA=2;bg.BLOCK_GRAPH_VERIFY=True
        os.environ['EXL3_DEC_MOE_UNION_DEV']='1' if args.k8_graphs and variant in ('device','graphs') else '0'
        os.environ['EXL3_GEMV_R_DEC1']='1' if args.k8_graphs or variant!='verify' else '0'
        os.environ['EXL3_MOEDEC1_RM3']='1' if variant in ('rm3','glue') else '0'
        os.environ['KYOJIN_HIP_GLUE']='1' if variant=='glue' else '0'
        for width in (1,3):
            with torch.inference_mode():state=stage.cache.get_new_state()
            forward(prefix,state,False)
            latencies=[];gpu=[];cpu=[]
            for index,ids in enumerate(inputs[width]):
                trace_components=variant=='verify' and width==3 and index>=16
                start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();start=time.perf_counter();start_event.record()
                output=forward(ids,state,width>1)
                enqueued=time.perf_counter();end_event.record();torch.cuda.synchronize();end=time.perf_counter()
                actual=output.cpu().clone();assert torch.isfinite(actual).all()
                key=(width,index)
                if variant=='verify':oracle[key]=actual
                else:torch.testing.assert_close(actual,oracle[key],rtol=0,atol=0,msg=f'{variant} width{width} step{index}')
                if index>=8:
                    latencies.append(1000*(end-start));cpu.append(1000*(enqueued-start));gpu.append(start_event.elapsed_time(end_event))
            state.free()
            record=dict(variant=variant,rows=width,exact_steps=24,wall_ms=statistics.mean(latencies),
                enqueue_ms=statistics.mean(cpu),gpu_span_ms=statistics.mean(gpu),graphs=bg.global_stats())
            if components:
                record['component_gpu_ms']={key:dict(type=kind,ms=statistics.mean(a.elapsed_time(b)
                    for k,_,a,b in components if k==key)) for key,kind,_,_ in components}
                components=[]
            records.append(record);print(json.dumps(record),flush=True)
            Path(args.out).write_text(json.dumps(dict(complete=False,records=records),indent=2)+'\n')
    Path(args.out).write_text(json.dumps(dict(complete=True,records=records,
        peak_allocated=torch.cuda.max_memory_allocated()),indent=2)+'\n')
finally:
    for part,fn in originals:part.forward=fn
    bg.purge();stage.model.unload()
