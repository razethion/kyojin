"""Matched target-only 6.05 solo/cluster experiment, separate from production MTP."""
import argparse,json,os,time,hashlib
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('role',choices=('preflight','metadata','solo','cluster'))
ap.add_argument('--model',required=True);ap.add_argument('--state',required=True)
ap.add_argument('--env',required=True);ap.add_argument('--manifest',required=True)
ap.add_argument('--token-file');ap.add_argument('--tokens',type=int,default=128)
ap.add_argument('--striped',action='store_true')
ap.add_argument('--host',default='127.0.0.1',help='worker address or local SSH-tunnel endpoint')
ap.add_argument('--port',type=int,default=9032)
ap.add_argument('--aux-host',help='explicit second worker address for --striped')
ap.add_argument('--aux-port',type=int,default=9033)
args=ap.parse_args()
if args.striped and not args.aux_host:ap.error('--striped requires --aux-host')
for key,value in json.loads(Path(args.env).read_text()).items():
    if key not in ('EXL3_ROOT','EXL3_VENV','EXL3_ROCM_SDK'):os.environ[key]=value
os.environ['EXL3_KVWIN']='0'
sidecar=Path(args.model)/'hc_gs_sidecar.safetensors'
if os.environ.get('EXL3_GR_GS','0')!='0' and sidecar.exists():
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest()=='473e7547adaad190788634556ce6b32614bdb280ccedadbed8789d39b59dcc8f'
    os.environ['EXL3_GR_GS_SIDECAR']=str(sidecar)
import torch
from exllamav3 import Config,Model,Cache,Tokenizer
from exllamav3.cache import CacheLayer_quant
from safetensors.torch import save_file,load_file
from tools.multinode.identity import verify
from tools.multinode.stage import load_stage,source_fingerprint
from tools.multinode.worker import CachedPipeline,StageClient

PROMPTS=[
 'Write a long, vivid story about a lighthouse keeper discovering a forgotten island. Continue for at least 800 words.',
 'Write a detailed practical guide to planning a community garden, covering its first year. Continue for at least 800 words.',
 'Explain how a small city could improve public transportation over five years, with concrete examples. Continue for at least 800 words.']
state=Path(args.state);state.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_grad_enabled(False)
cfg=Config.from_directory(args.model)
tok=Tokenizer.from_config(cfg)
encoded=[tok.encode('<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_bos=False,encode_special_tokens=True) for prompt in PROMPTS]
# Exercise the reporting path before loading 98GB of weights.
assert isinstance(tok.decode(encoded[0])[0],str)
if args.role=='preflight':
    print(json.dumps({'preflight':True,'prompt_lengths':[ids.shape[1] for ids in encoded],
                      'text_hash':hashlib.sha256(tok.decode(encoded[0])[0].encode()).hexdigest()}));raise SystemExit
if args.role=='metadata':
    model=Model.from_config(cfg)
    report={'architecture':cfg.architecture,'hidden':cfg.hidden_size,'streams':cfg.hc_mult,
            'modules':[module.key for module in model.modules],'prompt_lengths':[ids.shape[1] for ids in encoded],
            'source_id':source_fingerprint(Path.cwd()),'gpu_initialized':torch.cuda.is_initialized()}
    (state/'metadata.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit
assert 1<=args.tokens<=256
identity=verify(args.model,args.manifest) if args.role=='solo' else None
print(json.dumps({'checkpoint_verified':identity,'phase':'load','role':args.role}),flush=True)
torch.cuda.set_per_process_memory_fraction(.94,0)
pipeline=None;model=None;cache=None
if args.role=='solo':
    model=Model.from_config(cfg);cache=Cache(model,max_num_tokens=2048,max_batch_size=1,layer_type=CacheLayer_quant,k_bits=8,v_bits=8)
    model.load(device='cuda:0',progressbar=False)
else:
    # 24 blocks per node: embedding,expand,PLE and blocks0..23 form0:27.
    model=load_stage(args.model,0,27,'cuda:0',args.manifest,max_rows=256,cache_context=2048,cache_sessions=1,cache_bits=8,experimental_qwen=True)
    identity=model.manifest['model_id']
    class Local:
        manifest=model.manifest
        def call(self,op,values):return model.execute(op,values)
        def close(self):model.model.unload()
    auxiliary=(args.aux_host,args.aux_port) if args.striped else None
    remote=StageClient(args.host,args.port,bytes.fromhex(Path(args.token_file).read_text().strip()),timeout=60,auxiliary=auxiliary)
    pipeline=CachedPipeline([Local(),remote])
print(json.dumps({'phase':'loaded','role':args.role,'allocated':torch.cuda.memory_allocated()}),flush=True)
oracle={};runs=[]
if args.role=='cluster':oracle=load_file(str(state/'solo-oracle.bin'))
try:
    for run_index,prompt_index in enumerate((0,0,1,2)):
        warmup=run_index==0;limit=24 if warmup else args.tokens
        ids=encoded[prompt_index];current=ids;past=0;tokens=[];name=f'run{run_index}'
        if pipeline:pipeline.open(name)
        else:
            with torch.inference_mode():recurrent=cache.get_new_state()
        begin=time.perf_counter();first=None;step_ms=[]
        for step in range(limit):
            before=time.perf_counter()
            if pipeline:logits=pipeline.forward(name,current)
            else:
                params={'attn_mode':'flash_attn','cache':cache,'last_tokens_only':1,
                        'block_table':torch.arange(8,dtype=torch.int32).view(1,8),
                        'cache_seqlens':torch.tensor([past],dtype=torch.int32),'recurrent_states':[recurrent]}
                logits=model.forward(current,params).cpu()
            assert torch.isfinite(logits).all(),(run_index,step)
            next_id=logits[:,-1,:tok.actual_vocab_size].argmax(-1,keepdim=True).cpu()
            tokens.append(int(next_id.item()));past+=current.shape[1];current=next_id
            torch.cuda.synchronize()
            now=time.perf_counter()
            if first is None:first=now
            else:step_ms.append((now-before)*1000)
            if not warmup and step<8:
                key=f'prompt{prompt_index}_step{step}'
                if args.role=='solo':oracle[key]=logits.clone()
                else:
                    if not torch.equal(logits,oracle[key]):
                        save_file({'actual':logits.contiguous(),'expected':oracle[key].contiguous()},str(state/'failed-logits.bin'))
                        print(json.dumps({'mismatch':key,'max_abs_diff':(logits-oracle[key]).abs().max().item(),
                                          'actual_token':int(next_id.item()),'expected_token':int(oracle[key][:,-1,:tok.actual_vocab_size].argmax().item())}),flush=True)
                    torch.testing.assert_close(logits,oracle[key],rtol=0,atol=0,msg=key)
        elapsed=time.perf_counter()-first
        if pipeline:pipeline.release(name)
        else:recurrent.free()
        record={'warmup':warmup,'prompt_index':prompt_index,'prompt_tokens':ids.shape[1],
                'generated_tokens':tokens,'ttft_ms':(first-begin)*1000,
                'decode_tokens_per_second':(limit-1)/elapsed,'step_ms':step_ms,
                'text_sha256':hashlib.sha256(tok.decode(torch.tensor([tokens]))[0].encode()).hexdigest()}
        runs.append(record)
        if args.role=='cluster' and not warmup:
            expected=json.loads((state/'solo-result.json').read_text())['runs'][run_index]['generated_tokens']
            assert tokens==expected,f'Greedy output diverged for prompt{prompt_index}'
        print(json.dumps({key:value for key,value in record.items() if key not in ('generated_tokens','step_ms')}),flush=True)
    report={'role':args.role,'mtp':False,'cache_bits':8,'cache_tokens':2048,'sessions':1,'model_id':identity,
            'source_id':source_fingerprint(Path.cwd()),'runs':runs,'peak_allocated':torch.cuda.max_memory_allocated()}
    if pipeline:report['boundary_flushes']=model.boundary_flushes
    (state/f'{args.role}-result.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.role=='solo':save_file(oracle,str(state/'solo-oracle.bin'))
finally:
    if pipeline:pipeline.close()
    elif model:model.unload()
