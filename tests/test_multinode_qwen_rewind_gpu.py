"""Real Qwen GDN/PLE/QSA accepted-prefix rollback versus native recurrence."""
import argparse,json,os,time,hashlib
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--manifest',required=True)
ap.add_argument('--env',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
for key,value in json.loads(Path(args.env).read_text()).items():
    if key not in ('EXL3_ROOT','EXL3_VENV','EXL3_ROCM_SDK'):os.environ[key]=value
os.environ['EXL3_KVWIN']='0'
sidecar=Path(args.model)/'hc_gs_sidecar.safetensors'
assert hashlib.sha256(sidecar.read_bytes()).hexdigest()=='473e7547adaad190788634556ce6b32614bdb280ccedadbed8789d39b59dcc8f'
os.environ['EXL3_GR_GS_SIDECAR']=str(sidecar)
import torch
from tools.multinode.stage import load_stage
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.90,0)
stage=load_stage(args.model,0,8,'cuda:0',args.manifest,max_rows=256,cache_context=512,
                 cache_sessions=2,cache_bits=8,experimental_qwen=True,speculation_depth=7)
print(json.dumps({'loaded':True,'allocated':torch.cuda.memory_allocated(),'manifest':stage.manifest}),flush=True)
torch.manual_seed(7281);records=[]
cases=[(17,2,a) for a in range(1,3)]+[(17,3,a) for a in range(1,4)]+[(250,8,a) for a in range(1,9)]
try:
    for index,(prefix,width,accepted) in enumerate(cases):
        name=f'case{index}';stage.execute('open',{'session':name})
        with torch.inference_mode():reference=stage.cache.get_new_state()
        assert stage.states[name].slot!=reference.slot
        def native(ids,past,history):
            params=dict(attn_mode='flash_attn',cache=stage.cache,last_tokens_only=ids.shape[1],
                        block_table=torch.arange(2,4,dtype=torch.int32).view(1,2),
                        cache_seqlens=torch.tensor([past],dtype=torch.int32),recurrent_states=[reference],
                        recurrent_history=history,dflash_verify=history)
            with torch.inference_mode():
                output=stage.model.forward(ids,params)
                pending=params.pop('hc_pending',None)
                if pending is not None:stage.flush_pending(output,pending)
            return output.cpu()
        def prepared(ids,past,step,history):
            return stage.execute('prepare',dict(session=name,step=step,past=past,input_ids=ids,activation=None,
                         last_tokens_only=ids.shape[1],speculative=history,return_logits=False)).cpu()
        prompt=torch.randint(200,400,(1,prefix));window=torch.randint(200,400,(1,width))
        actual=prepared(prompt,0,0,False);expected=native(prompt,0,False)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        stage.execute('commit',dict(session=name,step=0,accepted=prefix))
        actual=prepared(window,prefix,1,True);expected=native(window,prefix,True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        stage.execute('commit',dict(session=name,step=1,accepted=accepted))
        with torch.inference_mode():reference.rewind(width-accepted)
        assert stage.states[name].position==reference.position==prefix+accepted
        assert stage.states[name].last_history==reference.last_history==0
        follow=torch.tensor([[299]])
        actual=prepared(follow,prefix+accepted,2,False);expected=native(follow,prefix+accepted,False)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        stage.execute('commit',dict(session=name,step=2,accepted=1))
        stage.execute('release',{'session':name});reference.free()
        record=dict(prefix=prefix,verify_rows=width,accepted=accepted,rejected=width-accepted,exact=True)
        records.append(record);print(json.dumps(record),flush=True)
    report=dict(passed=len(records),cases=records,source_id=stage.manifest['source_id'],
                model_id=stage.manifest['model_id'],peak_allocated=torch.cuda.max_memory_allocated())
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
finally:stage.model.unload()
