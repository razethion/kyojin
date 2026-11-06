"""Opt-in greedy MTP parity/performance gate; never replaces a serving endpoint."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('role',choices=('preflight','solo','cluster','worker'))
for name in ('model','manifest','env','state'):ap.add_argument('--'+name,required=True)
ap.add_argument('--token-file');ap.add_argument('--tokens',type=int,default=128)
ap.add_argument('--host',default='127.0.0.1',help='worker bind address or coordinator peer address')
ap.add_argument('--port',type=int,default=9032)
ap.add_argument('--usb4-interface',help='worker only: explicit interface for a direct USB4 bind')
args=ap.parse_args()
for key,value in json.loads(Path(args.env).read_text()).items():
    if key not in ('EXL3_ROOT','EXL3_VENV','EXL3_ROCM_SDK'):os.environ[key]=value
os.environ['EXL3_KVWIN']='0'
sidecar=Path(args.model)/'hc_gs_sidecar.safetensors'
assert hashlib.sha256(sidecar.read_bytes()).hexdigest()=='473e7547adaad190788634556ce6b32614bdb280ccedadbed8789d39b59dcc8f'
os.environ['EXL3_GR_GS_SIDECAR']=str(sidecar)
assert os.environ.get('EXL3_DRAFT2S')==str(Path(args.model)/'draft2s.st')
import torch
from safetensors.torch import load_file,save_file
from exllamav3 import Config,Tokenizer,Generator,Job
from exllamav3.generator.sampler import ArgmaxSampler
from tools.multinode.stage import load_stage,source_fingerprint
from tools.multinode.speculative import SpeculativePipeline,greedy_prefix
from tools.multinode.worker import StageClient
from tools.qwen import spec_policy
torch.set_num_threads(4);torch.set_grad_enabled(False)
state=Path(args.state);state.mkdir(parents=True,exist_ok=True)
cfg=Config.from_directory(args.model);tok=Tokenizer.from_config(cfg)
prompts=[
 'Write a long, vivid story about a lighthouse keeper discovering a forgotten island. Continue for at least 800 words.',
 'Write a detailed practical guide to planning a community garden, covering its first year. Continue for at least 800 words.',
 'Explain how a small city could improve public transportation over five years, with concrete examples. Continue for at least 800 words.']
encoded=[tok.encode('<|im_start|>user\n'+p+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_bos=False,encode_special_tokens=True) for p in prompts]
assert [x.shape[1] for x in encoded]==[38,38,40]
assert isinstance(tok.decode(encoded[0])[0],str) and 16<=args.tokens<=256
if args.role=='preflight':print(json.dumps({'preflight':True,'lengths':[x.shape[1] for x in encoded]}));raise SystemExit
torch.cuda.set_per_process_memory_fraction(.94,0)
if args.role=='worker':
    import sys
    from tools.multinode.worker import main
    sys.argv=['worker','--model',args.model,'--weights-manifest',args.manifest,'--modules','27:53',
              '--device','cuda:0','--host',args.host,'--port',str(args.port),
              '--once','--accept-timeout','900','--token-file',args.token_file,'--cache-context','2048',
              '--cache-sessions','1','--cache-bits','8','--max-rows','256','--experimental-qwen','--speculation-depth','7']
    if args.usb4_interface:sys.argv+=['--usb4-interface',args.usb4_interface]
    main();print(json.dumps({'worker_peak':torch.cuda.max_memory_allocated()}),flush=True);raise SystemExit
end=53 if args.role=='solo' else 27
stage=load_stage(args.model,0,end,'cuda:0',args.manifest,max_rows=256,cache_context=2048,
                 cache_sessions=1,cache_bits=8,experimental_qwen=True,speculation_depth=7)
print(json.dumps({'loaded':True,'role':args.role,'allocated':torch.cuda.memory_allocated(),'model_id':stage.manifest['model_id']}),flush=True)
report={'role':args.role,'mtp':True,'greedy_only':True,'ngram_lookup':False,'cache_tokens':2048,'sessions':1,
        'model_id':stage.manifest['model_id'],'source_id':stage.manifest['source_id'],'native':[],'adapted':[]}
def save():
    (state/f'{args.role}-mtp-progress.json').write_text(json.dumps(report,indent=2)+'\n')

def native_run(which,prompt_index,limit,warmup):
    model=stage.model;kwargs={}
    if which=='mtp':kwargs=dict(draft_model=stage.draft_model,draft_cache=stage.draft_cache,num_draft_tokens=7)
    generator=Generator(model=model,cache=stage.cache,tokenizer=tok,max_batch_size=1,**kwargs)
    uninstall=None
    if which=='mtp':uninstall=spec_policy.install(generator,stage.draft_model,model,maxd=7,ngram=None,rule=spec_policy.TierRule(maxd=7))
    trace=[];original_forward=model.forward
    def traced_forward(input_ids,params=None):
        if params is not None and params.get('recurrent_history'):
            assert input_ids.device.type=='cpu'
            trace.append({'past':int(params['cache_seqlens'][0]),'ids':input_ids[0].tolist()})
        return original_forward(input_ids,params)
    model.forward=traced_forward
    job=Job(input_ids=encoded[prompt_index],max_new_tokens=limit,sampler=ArgmaxSampler(),stop_conditions=[])
    generator.enqueue(job);tokens=[];first=None;first_batch=0;begin=time.perf_counter()
    while generator.num_remaining_jobs():
        chunk=[]
        for item in generator.iterate():
            ids=item.get('token_ids')
            if ids is not None and ids.numel():chunk.extend(ids.flatten().tolist())
        tokens.extend(chunk)
        if chunk and first is None:
            torch.cuda.synchronize();first=time.perf_counter();first_batch=len(chunk)
    torch.cuda.synchronize();end=time.perf_counter()
    record={'arm':which,'warmup':warmup,'prompt':prompt_index,'tokens':tokens,'ttft_ms':1000*(first-begin),
            'first_batch_tokens':first_batch,'tps':(len(tokens)-first_batch)/(end-first),
            'accepted_drafts':job.accepted_draft_tokens,'rejected_drafts':job.rejected_draft_tokens,'verify_trace':trace}
    model.forward=original_forward
    if uninstall:uninstall()
    generator.clear_prompt_ckpts();del generator,job;gc.collect()
    return record

class Local:
    manifest=stage.manifest
    def call(self,op,values):return stage.execute(op,values)
    def close(self):pass
pipeline=None;oracle={}

def row_diagnostic():
    """Same prefix snapshot: native R1 vs R2 row0; tracing is outside timings."""
    cache=stage.cache;model=stage.model
    with torch.inference_mode():rec=cache.get_new_state()
    prompt=encoded[0][:,:-1];past=prompt.shape[1]
    def params(history):
        return dict(attn_mode='flash_attn',cache=cache,block_table=torch.arange(8,dtype=torch.int32).view(1,8),
                    cache_seqlens=torch.tensor([rec.position],dtype=torch.int32),recurrent_states=[rec],
                    recurrent_history=history,dflash_verify=history,last_tokens_only=8)
    model.forward(prompt,params(False))
    with torch.inference_mode():snapshot=rec.stash()
    traces=[]
    def traced(ids,history):
        captured=[];old=[]
        for module in model.modules:
            original=module.forward;old.append((module,original))
            def forward(*a,_fn=original,_key=module.key,**kw):
                output=_fn(*a,**kw)
                if isinstance(output,torch.Tensor):captured.append((_key,output[:,0].detach().cpu().clone()))
                return output
            module.forward=forward
        try:out=model.forward(ids,params(history)).cpu()
        finally:
            for module,original in old:module.forward=original
        traces.append(captured);return out
    try:
        current=encoded[0][:,-1:];out=traced(current,False)
        next_id=out[:,-1,:tok.actual_vocab_size].argmax(-1,keepdim=True)
        with torch.inference_mode():rec.position=past;rec.last_history=0;rec.unstash(snapshot)
        batched=traced(torch.cat((current,next_id),dim=1),True)
        changes=[]
        for (key,a),(other,b) in zip(*traces):
            assert key==other
            if not torch.equal(a,b):changes.append({'module':key,'max_abs_diff':float((a.float()-b.float()).abs().max())})
        result={'first_differing_module':changes[0] if changes else None,'different_modules':len(changes),
                'max_logit_diff':float((out[:,0]-batched[:,0]).abs().max()),
                'argmax_equal':bool(torch.equal(out[:,0,:tok.actual_vocab_size].argmax(-1),batched[:,0,:tok.actual_vocab_size].argmax(-1)))}
        print(json.dumps({'native_row_diagnostic':result}),flush=True);return result
    finally:rec.free();cache.reset_states()

try:
    if args.role=='solo':
        report['native_row_diagnostic']=row_diagnostic();save()
        for arm in ('plain','mtp'):
            for index,prompt in enumerate((0,0,1,2)):
                record=native_run(arm,prompt,24 if index==0 else args.tokens,index==0)
                report['native'].append(record);save()
                print(json.dumps({k:v for k,v in record.items() if k not in ('tokens','verify_trace')}),flush=True)
        plain=[r['tokens'] for r in report['native'] if r['arm']=='plain' and not r['warmup']]
        mtp=[r['tokens'] for r in report['native'] if r['arm']=='mtp' and not r['warmup']]
        report['native_plain_vs_mtp_first_difference']=[next((i for i,(a,b) in enumerate(zip(p,m)) if a!=b),None) for p,m in zip(plain,mtp)]
        # Preserve existing native verifier behavior; do not call it R1-lossless
        # when the native baseline itself differs. Exact native-MTP parity below
        # is required after matching its prefill boundary and verification rows.
        stage.cache.reset_states()
        # The generator attached MTP to the same full target; restore explicit binding.
        stage.draft_model.attach_to(stage.binding)
        pipeline=SpeculativePipeline([Local()])
    else:
        previous=json.loads((state/'solo-mtp-result.json').read_text())
        assert previous['model_id']==stage.manifest['model_id']
        oracle=load_file(str(state/'mtp-oracle.bin'))
        remote=StageClient(args.host,args.port,bytes.fromhex(Path(args.token_file).read_text().strip()),timeout=90)
        pipeline=SpeculativePipeline([Local(),remote])
    for run_index,(phase,prompt_index,limit) in enumerate([('warmup',0,24)]+[('check',p,48) for p in range(3)]+[('speed',p,args.tokens) for p in range(3)]):
        name=f'run{run_index}';pipeline.open(name);begin=time.perf_counter();prompt=encoded[prompt_index]
        prefix=prompt[:,:-1]
        pipeline.prepare(name,prefix);pipeline.commit(name,prefix.shape[1])
        tokens=[];current=prompt[:,-1:];first=None;first_batch=0;rounds=[]
        while len(tokens)<limit:
            remaining=limit-len(tokens);depth=7;past=pipeline.sessions.items[name].length
            proposals=pipeline.draft(name,current,depth)['tokens'] if depth else torch.empty((1,0),dtype=torch.int64)
            ids=torch.cat((current,proposals),dim=1);probe=phase=='check' and len(rounds)<4
            output=pipeline.prepare(name,ids,speculative=bool(depth),return_logits=probe)
            predictions=output['tokens'][0].tolist();drafts=proposals[0].tolist()
            decision=greedy_prefix(drafts,predictions,remaining)
            accepted,committed=decision['accepted'],decision['committed'];pipeline.commit(name,committed)
            emitted=decision['emitted'];tokens.extend(emitted)
            current=torch.tensor([[tokens[-1]]],dtype=torch.int64)
            rounds.append({'past':past,'current':int(ids[0,0]),'drafts':drafts,'predictions':predictions,'accepted':accepted,'committed':committed})
            if first is None:
                torch.cuda.synchronize();first=time.perf_counter();first_batch=len(emitted)
            if probe:
                key=f'p{prompt_index}_r{len(rounds)-1}'
                if args.role=='solo':oracle[key]=output['logits'].clone()
                else:torch.testing.assert_close(output['logits'],oracle[key],rtol=0,atol=0,msg=key)
        torch.cuda.synchronize();end=time.perf_counter();pipeline.release(name)
        record={'phase':phase,'prompt':prompt_index,'tokens':tokens,'rounds':rounds,
                'ttft_ms':1000*(first-begin),'first_batch_tokens':first_batch,'tps':(len(tokens)-first_batch)/(end-first),
                'accepted_drafts':sum(min(r['accepted'],r['committed']-1) for r in rounds),
                'verified_drafts':sum(len(r['drafts']) for r in rounds)}
        report['adapted'].append(record);save()  # preserve offending tokens/schedule before an assertion
        if args.role=='solo' and phase!='warmup':
            assert tokens==mtp[prompt_index][:limit],f'adapter differs from native MTP {phase}:{prompt_index}'
            native_trace=[r for r in report['native'] if r['arm']=='mtp' and not r['warmup'] and r['prompt']==prompt_index][0]['verify_trace']
            actual_trace=[{'past':r['past'],'ids':[r['current']]+r['drafts']} for r in rounds]
            assert actual_trace==native_trace[:len(actual_trace)],f'adapter verifier schedule differs {phase}:{prompt_index}'
        if args.role=='cluster':
            expected=previous['adapted'][run_index]
            assert tokens==expected['tokens'] and rounds==expected['rounds'],f'cluster MTP round/token divergence {phase}:{prompt_index}'
        print(json.dumps({k:v for k,v in record.items() if k not in ('tokens','rounds')}),flush=True)
    report['peak_allocated']=torch.cuda.max_memory_allocated();report['boundary_flushes']=stage.boundary_flushes
    (state/f'{args.role}-mtp-result.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.role=='solo':save_file(oracle,str(state/'mtp-oracle.bin'))
except BaseException as error:
    report['failure']={'type':type(error).__name__,'message':str(error),'run':locals().get('run_index'),
                       'partial_tokens':locals().get('tokens',[]),'partial_rounds':locals().get('rounds',[])}
    save()
    raise
finally:
    if pipeline:pipeline.close()
    if stage.draft_model:stage.draft_model.unload()
    stage.model.unload()
