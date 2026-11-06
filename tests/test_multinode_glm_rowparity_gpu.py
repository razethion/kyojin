"""Native GLM serial-token vs three-row arithmetic, with real divergence inputs.

No network/stage transaction is used for these forwards. Also records which
native fused expert paths the4.10 pack actually qualifies for.
"""
import argparse,json,os,time
from pathlib import Path
ap=argparse.ArgumentParser()
for name in ('model','cases','out'):ap.add_argument('--'+name,required=True)
args=ap.parse_args()
os.environ['EXL3_BLOCK_GRAPH']='0';os.environ['EXL3_KVWIN']='0'
import torch
from exllamav3 import Config,Model,Tokenizer
from tools.multinode.speculative import SpeculativeGlmStage
from exllamav3.modules.quant.exl3 import VERIFY_FUSE
from exllamav3.modules.quant import exl3 as quant
from exllamav3.modules import mlp as mlp_module
torch.set_grad_enabled(False);torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.85,0)
cfg=Config.from_directory(args.model);tok=Tokenizer.from_config(cfg)
from jinja2.sandbox import ImmutableSandboxedEnvironment
template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,
    extensions=['jinja2.ext.loopcontrols']).from_string((Path(args.model)/'chat_template.jinja').read_text())
model=Model.from_config(cfg)
frontier=os.environ.get('GLM_AUDIT_FRONTIER')=='1'
long_context=os.environ.get('GLM_AUDIT_LONG')=='1'
context=2560 if long_context else 512
pages=context//256
stage=SpeculativeGlmStage(model,0,model.first_block_idx+8,speculation_depth=3 if frontier else 2,
    device='cuda:0',model_id='verified-sc4.10',source_id='audit',
    max_rows=256,context=context,sessions=2,cache_bits=8)
stage.load()
report=dict(native_only=True,cases=[],routing=[])
for block in model.modules:
    mlp=getattr(block,'mlp',None)
    if mlp is None:continue
    projections=getattr(mlp,'gates',[])
    first=getattr(projections[0],'inner',None) if projections else None
    record=dict(key=block.key,type=type(mlp).__name__,
        routed_bits=getattr(first,'K',None),mcg=getattr(first,'mcg',None),
        fused_decode=getattr(mlp,'support_dec_moe',None),
        grouped_decode=getattr(mlp,'support_hip_grouped',None),
        grouped_prefill=getattr(mlp,'support_hip_prefill',None),
        hidden=getattr(mlp,'hidden_size',None),intermediate=getattr(mlp,'intermediate_size_padded',None))
    report['routing'].append(record);print(json.dumps(record),flush=True)

def forward(ids,state,slot,history=False):
    params=dict(attn_mode='flash_attn',cache=stage.cache,last_tokens_only=ids.shape[1],
        recurrent_states=[state],block_table=torch.arange(slot*pages,(slot+1)*pages,dtype=torch.int32).view(1,pages),
        cache_seqlens=torch.tensor([state.position],dtype=torch.int32),
        recurrent_history=history,dflash_verify=history)
    value=model.forward(ids,params)
    pending=params.pop('hc_pending',None)
    if pending is not None:stage.flush_pending(value,pending)
    assert torch.isfinite(value).all()
    return value

trace=None;counts={};rows_now=0;originals=[]
def normalize(value,rows):
    if not isinstance(value,torch.Tensor) or value.ndim<2:return None
    if value.shape[0]==1 and value.ndim>=3 and value.shape[1]==rows:return value[0]
    if value.shape[0]==rows:return value
    if value.shape[0]%rows==0:return value.reshape(rows,-1,*value.shape[1:])
    return None
for module in (part for top in model.modules for part in top):
    original=module.forward;originals.append((module,original))
    def wrapped(*values,_fn=original,_key=module.key,**kw):
        n=None
        if trace is not None:
            n=counts.get(_key,0);counts[_key]=n+1
            value=normalize(values[0],rows_now) if values else None
            if value is not None:trace[_key+'#'+str(n)+':input']=value.detach().cpu().clone()
        output=_fn(*values,**kw)
        if trace is not None:
            value=normalize(output,rows_now)
            if value is not None:
                trace[_key+'#'+str(n)]=value.detach().cpu().clone()
        return output
    module.forward=wrapped
try:
    pair_check=mlp_module._vf_plain_exl3
    for case in json.loads(Path(args.cases).read_text())['cases']:
        text=template.render(messages=[dict(role='user',content=case['query'])],tools=None,add_generation_prompt=True)
        assert text.endswith('<think>');text+='\n</think>\n\n'
        prompt=tok.encode(text,add_bos=False,encode_special_tokens=True)
        prefix=case['prefix_tokens'];window=torch.tensor([case['window']],dtype=torch.int64)
        if long_context:
            target=case['long_past']
            gen=torch.Generator().manual_seed(2051)
            padding=torch.randint(200,500,(1,target-len(prefix)-prompt.shape[1]),generator=gen)
            prompt=torch.cat((prompt,padding),dim=1)
        arms=[('base',False,True,False,False),('verify',True,True,False,False),
              ('verify-loop',True,True,True,False),('verify-nopair',True,True,False,True),
              ('verify-nopair-loop',True,True,True,True)]
        if frontier:arms=[('verify-nopair',True,True,False,True)]
        for arm,fused,history,loop,nopair in arms:
            VERIFY_FUSE['on']=fused
            quant.DEC_GEMV_R_LOOP_MIN_K=0 if loop else 99
            mlp_module._vf_plain_exl3=(lambda *a:False) if nopair else pair_check
            with torch.inference_mode():states=[stage.cache.get_new_state() for _ in range(2)]
            for slot,state in enumerate(states):
                for offset in range(0,prompt.shape[1],256):forward(prompt[:,offset:offset+256],state,slot)
                for token in prefix:forward(torch.tensor([[token]]),state,slot)
            serial=[]
            for index in range(window.shape[1]):
                trace={};counts={};rows_now=1
                forward(window[:,index:index+1],states[0],0,False)
                serial.append(trace);trace=None
            trace={};counts={};rows_now=window.shape[1]
            forward(window,states[1],1,history)
            batched=trace;trace=None
            differences=[];exact=0
            for key,value in batched.items():
                if any(key not in part for part in serial):continue
                reference=torch.cat([part[key] for part in serial],dim=0)
                if reference.shape!=value.shape:continue
                if torch.equal(reference,value):exact+=1;continue
                delta=value.float()-reference.float()
                differences.append(dict(key=key,max_abs=float(delta.abs().max()),
                    relative_rms=float(delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-20)),
                    changed=int(torch.count_nonzero(delta)),elements=value.numel(),shape=list(value.shape)))
            record=dict(prompt=case['prompt'],arm=arm,history=history,verify_fuse=fused,window=case['window'],long_past=case.get('long_past'),
                prefix_tokens=len(prefix),native_exact_modules=exact,differences=differences)
            report['cases'].append(record)
            print(json.dumps(dict(prompt=case['prompt'],arm=arm,history=history,verify_fuse=fused,exact=exact,
                different=len(differences),first=differences[:4])),flush=True)
            for state in states:state.free()
            Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
finally:
    for module,original in originals:module.forward=original
    stage.model.unload()
